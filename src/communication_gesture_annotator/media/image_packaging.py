# SPDX-License-Identifier: Apache-2.0
"""Deterministic temporal selection and contact-sheet model inputs.

The canonical analysis grid remains frame based.  This module only changes how
those frame images are transported to a vision model: every source image keeps
an explicit row-major cell mapping, timestamp, and digest.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4
from .ffmpeg import (
    MediaRuntimeError,
    _escape_image2_sequence_pattern,
    _resolve_executable,
    file_sha256,
)
from ..rights import (
    RIGHTS_CONTRACT_VERSION,
    combine_rights,
    rights_sidecar_path,
    validate_rights,
    write_rights_sidecar,
)

IMAGE_PACKAGING_CONTRACT_VERSION = "0.1.0"
IMAGE_PACKAGING_RENDERER_VERSION = "ffmpeg-contact-sheet-v0.2"
TARGET_IMAGE_PACKAGING_RENDERER_VERSION = "ffmpeg-target-contact-sheet-v0.1"
GLOBAL_IMAGE_PACKAGING_RENDERER_VERSION = "ffmpeg-global-frame-labels-v0.1"
_MANIFEST_NAME = "image-packages.json"
_RIGHTS_MANIFEST_NAME = "rights-manifest.json"


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all((character in "0123456789abcdefABCDEF" for character in value))
    )


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class FrameImage:
    """One extracted analysis image with timestamp-safe source identity."""

    sample_index: int
    start_us: int
    end_us: int
    path: Path = field(repr=False)
    sha256: str = ""
    source_gap_reason: str | None = None
    rights: Mapping[str, Any] | None = None
    selection_anchor_us: int | None = None

    def __init__(
        self,
        *,
        sample_index: int,
        start_us: int,
        end_us: int,
        path: str | Path,
        sha256: str,
        source_gap_reason: str | None = None,
        rights: Mapping[str, Any] | None = None,
        selection_anchor_us: int | None = None,
    ) -> None:
        sample_index = _integer(sample_index, "sample_index", minimum=0)
        start_us = _integer(start_us, "start_us", minimum=0)
        end_us = _integer(end_us, "end_us", minimum=1)
        if end_us <= start_us:
            raise ValueError("frame image end_us must be greater than start_us")
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError("frame image path must be non-blank")
        if not _is_sha256(sha256):
            raise ValueError("frame image sha256 must be a hexadecimal SHA-256")
        if source_gap_reason is not None and (
            not isinstance(source_gap_reason, str) or not source_gap_reason.strip()
        ):
            raise ValueError("source_gap_reason must be a non-blank string or null")
        normalized_rights = validate_rights(rights) if rights is not None else None
        if selection_anchor_us is not None:
            selection_anchor_us = _integer(selection_anchor_us, "selection_anchor_us", minimum=0)
            if not start_us <= selection_anchor_us < end_us:
                raise ValueError("selection anchor must lie inside its frame interval")
        object.__setattr__(self, "sample_index", sample_index)
        object.__setattr__(self, "start_us", start_us)
        object.__setattr__(self, "end_us", end_us)
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(self, "sha256", sha256.lower())
        object.__setattr__(self, "source_gap_reason", source_gap_reason)
        object.__setattr__(self, "rights", normalized_rights)
        object.__setattr__(self, "selection_anchor_us", selection_anchor_us)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrameImage":
        required = {"sample_index", "start_us", "end_us", "path", "sha256"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(
                "frame image mapping is missing required fields: " + ", ".join(missing)
            )
        return cls(
            sample_index=value["sample_index"],
            start_us=value["start_us"],
            end_us=value["end_us"],
            path=value["path"],
            sha256=value["sha256"],
            source_gap_reason=value.get("source_gap_reason"),
            rights=value.get("rights"),
            selection_anchor_us=value.get("selection_anchor_us"),
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "start_us": self.start_us,
            "end_us": self.end_us,
            "source_sha256": self.sha256,
            **(
                {"selection_anchor_us": self.selection_anchor_us}
                if self.selection_anchor_us is not None
                else {}
            ),
            **(
                {"source_gap_reason": self.source_gap_reason}
                if self.source_gap_reason is not None
                else {}
            ),
        }


def _coerce_frames(
    values: Sequence[FrameImage | Mapping[str, Any]], *, require_contiguous_time: bool
) -> tuple[FrameImage, ...]:
    if not values:
        raise ValueError("at least one frame image is required")
    frames = tuple(
        (
            value if isinstance(value, FrameImage) else FrameImage.from_mapping(value)
            for value in values
        )
    )
    for previous, current in zip(frames, frames[1:]):
        if current.sample_index <= previous.sample_index:
            raise ValueError("frame image sample indices must be strictly increasing")
        if current.start_us < previous.end_us:
            raise ValueError("frame image intervals must not overlap")
        if require_contiguous_time and current.start_us != previous.end_us:
            raise ValueError("fixed-period selection requires contiguous frame coverage")
    return frames


@dataclass(frozen=True, slots=True)
class ContactSheetLayout:
    """Configurable rectangular layout for row-major model images."""

    columns: int = 10
    rows: int = 10
    tile_width_px: int = 160
    tile_height_px: int = 90
    gutter_px: int = 2
    label_mode: Literal["ordinal", "none", "target_time", "global_sample"] = "ordinal"
    jpeg_quality: int = 85

    def __post_init__(self) -> None:
        for name in ("columns", "rows"):
            value = _integer(getattr(self, name), name, minimum=1)
            if value > 100:
                raise ValueError(f"{name} may not exceed 100")
        if self.columns * self.rows > 400:
            raise ValueError("contact sheet may not exceed 400 cells")
        for name in ("tile_width_px", "tile_height_px"):
            value = _integer(getattr(self, name), name, minimum=16)
            if value > 4096 or value % 2:
                raise ValueError(f"{name} must be even and no greater than 4096")
        _integer(self.gutter_px, "gutter_px", minimum=0)
        if self.gutter_px > 128:
            raise ValueError("gutter_px may not exceed 128")
        if self.label_mode not in {"ordinal", "none", "target_time", "global_sample"}:
            raise ValueError("unsupported label_mode")
        if self.label_mode == "target_time" and (
            self.tile_height_px < 64 or self.tile_width_px < 320
        ):
            raise ValueError("target/time labels require tiles at least 320 by 64 pixels")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer in [1, 100]")
        if self.output_width_px > 32768 or self.output_height_px > 32768:
            raise ValueError("contact-sheet dimensions exceed the supported limit")

    @property
    def capacity(self) -> int:
        return self.columns * self.rows

    @property
    def output_width_px(self) -> int:
        return self.columns * self.tile_width_px + (self.columns - 1) * self.gutter_px

    @property
    def output_height_px(self) -> int:
        return self.rows * self.tile_height_px + (self.rows - 1) * self.gutter_px

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": IMAGE_PACKAGING_CONTRACT_VERSION,
            "kind": "contact_sheet",
            "columns": self.columns,
            "rows": self.rows,
            "capacity": self.capacity,
            "tile_width_px": self.tile_width_px,
            "tile_height_px": self.tile_height_px,
            "gutter_px": self.gutter_px,
            "label_mode": self.label_mode,
            "jpeg_quality": self.jpeg_quality,
            "output_width_px": self.output_width_px,
            "output_height_px": self.output_height_px,
            "cell_order": "row_major",
        }


@dataclass(frozen=True, slots=True)
class ImagePackageCell:
    ordinal: int
    row: int
    column: int
    frame: FrameImage
    visible_label: str | None

    def provenance(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "row": self.row,
            "column": self.column,
            "visible_label": self.visible_label,
            **self.frame.provenance(),
        }


@dataclass(frozen=True, slots=True)
class ImagePackagePlan:
    package_id: str
    sequence_index: int
    layout: ContactSheetLayout
    cells: tuple[ImagePackageCell, ...]
    rights: Mapping[str, Any] | None

    @property
    def filename(self) -> str:
        suffix = self.package_id.rsplit("-", 1)[-1][:16]
        return f"contact-sheet-{self.sequence_index:04d}-{suffix}.jpg"

    def provenance(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "sequence_index": self.sequence_index,
            "cell_count": len(self.cells),
            "cells": [cell.provenance() for cell in self.cells],
        }


def _combined_frame_rights(frames: Sequence[FrameImage]) -> dict[str, Any] | None:
    present = [frame.rights for frame in frames if frame.rights is not None]
    if present and len(present) != len(frames):
        raise ValueError("frame rights must be present on every image or on none")
    if not present:
        return None
    return combine_rights(*present)


def plan_image_packages(
    values: Sequence[FrameImage | Mapping[str, Any]],
    layout: ContactSheetLayout | None = None,
    *,
    maximum_images: int | None = None,
    visible_labels: Mapping[int, str] | None = None,
) -> tuple[ImagePackagePlan, ...]:
    """Plan lossless row-major contact sheets without reading or writing images."""
    frames = _coerce_frames(values, require_contiguous_time=False)
    selected_layout = layout or ContactSheetLayout()
    if selected_layout.label_mode in {"target_time", "global_sample"}:
        if visible_labels is None or set(visible_labels) != {f.sample_index for f in frames}:
            raise ValueError("target/time labels must cover exactly the delivered frames")
        for label in visible_labels.values():
            if (
                not isinstance(label, str)
                or not label
                or len(label) > 70
                or any(
                    (
                        c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .-"
                        for c in label
                    )
                )
            ):
                raise ValueError("invalid target/time label")
    elif visible_labels is not None:
        raise ValueError("explicit labels require target_time layout")
    if maximum_images is not None:
        maximum_images = _integer(maximum_images, "maximum_images", minimum=1)
    package_count = (len(frames) + selected_layout.capacity - 1) // selected_layout.capacity
    if maximum_images is not None and package_count > maximum_images:
        raise ValueError(
            f"contact-sheet plan requires {package_count} images; budget is {maximum_images}"
        )
    rights = _combined_frame_rights(frames)
    plans: list[ImagePackagePlan] = []
    for sequence_index, offset in enumerate(range(0, len(frames), selected_layout.capacity)):
        portion = frames[offset : offset + selected_layout.capacity]
        cells = tuple(
            (
                ImagePackageCell(
                    ordinal=ordinal,
                    row=ordinal // selected_layout.columns,
                    column=ordinal % selected_layout.columns,
                    frame=frame,
                    visible_label=visible_labels[frame.sample_index]
                    if visible_labels is not None
                    else str(ordinal)
                    if selected_layout.label_mode == "ordinal"
                    else None,
                )
                for ordinal, frame in enumerate(portion)
            )
        )
        identity = _canonical_digest(
            {
                "contract_version": IMAGE_PACKAGING_CONTRACT_VERSION,
                "renderer_version": _renderer_version(selected_layout),
                "layout": selected_layout.to_dict(),
                "cells": [cell.provenance() for cell in cells],
                "rights": rights,
            }
        )
        plans.append(
            ImagePackagePlan(
                package_id=f"image-package-{identity[:20]}",
                sequence_index=sequence_index,
                layout=selected_layout,
                cells=cells,
                rights=rights,
            )
        )
    return tuple(plans)


@dataclass(frozen=True, slots=True)
class RenderedImagePackage:
    package_id: str
    sequence_index: int
    path: Path = field(repr=False)
    sha256: str
    width_px: int
    height_px: int
    cells: tuple[ImagePackageCell, ...]
    rights: Mapping[str, Any] | None

    def trace_reference(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "sequence_index": self.sequence_index,
            "sha256": self.sha256,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "cell_count": len(self.cells),
            "cells": [cell.provenance() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class ImagePackageSet:
    package_set_id: str
    layout: ContactSheetLayout
    images: tuple[RenderedImagePackage, ...]
    manifest_path: Path
    rights: Mapping[str, Any] | None

    def image_paths(self) -> list[Path]:
        return [image.path for image in self.images]

    def trace_refs(self) -> list[dict[str, Any]]:
        return [image.trace_reference() for image in self.images]


def _renderer_version(layout: ContactSheetLayout) -> str:
    if layout.label_mode == "global_sample":
        return GLOBAL_IMAGE_PACKAGING_RENDERER_VERSION
    return (
        TARGET_IMAGE_PACKAGING_RENDERER_VERSION
        if layout.label_mode == "target_time"
        else IMAGE_PACKAGING_RENDERER_VERSION
    )


def _target_label_filters(plans: Sequence[ImagePackagePlan]) -> list[str]:
    """Keep labels in a reserved top band; generated text is strictly validated."""
    layout = plans[0].layout
    font_size = min(22, max(10, layout.tile_width_px // 29))
    band = font_size + 8
    font_option = ""
    for font in (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ):
        if font.is_file():
            escaped = font.as_posix().replace(":", "\\:").replace("'", "'\\''")
            font_option = f":fontfile='{escaped}'"
            break
    filters = [
        f"scale=w={layout.tile_width_px}:h={layout.tile_height_px - band}:force_original_aspect_ratio=decrease:flags=lanczos",
        f"pad={layout.tile_width_px}:{layout.tile_height_px}:(ow-iw)/2:{band}+(oh-{band}-ih)/2:color=black",
    ]
    for n, cell in enumerate((cell for plan in plans for cell in plan.cells)):
        label = cell.visible_label
        color = "yellow" if label and label.startswith("TARGET ") else "white"
        filters.append(
            f"drawtext=text='{label}':expansion=none{font_option}:x=4:y=3:fontcolor={color}:fontsize={font_size}:enable='eq(n,{n})'"
        )
    return filters


def _build_contact_sheet_batch_command(
    plans: Sequence[ImagePackagePlan], input_pattern: Path, output_pattern: Path, *, ffmpeg: str
) -> list[str]:
    """Build one finite FFmpeg process for every sheet in a request."""
    if not plans:
        raise ValueError("contact-sheet batch cannot be empty")
    layout = plans[0].layout
    if any((plan.layout != layout for plan in plans)):
        raise ValueError("contact-sheet batch layouts must match")
    filters = [
        f"scale=w={layout.tile_width_px}:h={layout.tile_height_px}:force_original_aspect_ratio=decrease:flags=lanczos",
        f"pad={layout.tile_width_px}:{layout.tile_height_px}:(ow-iw)/2:(oh-ih)/2:color=black",
    ]
    if layout.label_mode in {"target_time", "global_sample"}:
        filters = _target_label_filters(plans)
    if layout.label_mode == "ordinal":
        font_size = max(10, min(32, layout.tile_height_px // 8))
        filters.append(
            "drawtext=text='%{eif\\:mod(n\\,"
            + str(layout.capacity)
            + f")\\:d}}':x=4:y=4:fontcolor=white:fontsize={font_size}:box=1:boxcolor=black@0.65"
        )
    filters.append(
        f"tile=layout={layout.columns}x{layout.rows}:nb_frames={layout.capacity}:margin=0:padding={layout.gutter_px}:color=black"
    )
    ffmpeg_qscale = max(2, min(31, round(31 - (layout.jpeg_quality - 1) * 29 / 99)))
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-framerate",
        "1",
        "-start_number",
        "0",
        "-i",
        _escape_image2_sequence_pattern(input_pattern, "%08d"),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        ",".join(filters),
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(len(plans)),
        "-q:v",
        str(ffmpeg_qscale),
        _escape_image2_sequence_pattern(output_pattern, "%04d"),
    ]


def _run_packaging(command: Sequence[str]) -> None:
    try:
        subprocess.run(
            list(command), check=True, capture_output=True, text=True, encoding="utf-8", timeout=120
        )
    except OSError as exc:
        raise MediaRuntimeError("contact-sheet renderer could not be started") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaRuntimeError(
            f"contact-sheet rendering failed with exit status {exc.returncode}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaRuntimeError(
            "contact-sheet rendering exceeded the 120-second safety limit"
        ) from exc


def _stage_batch_input_sequence(plans: Sequence[ImagePackagePlan], staging: Path) -> Path:
    inputs = staging / "inputs"
    inputs.mkdir()
    for ordinal, cell in enumerate((cell for plan in plans for cell in plan.cells)):
        shutil.copyfile(cell.frame.path, inputs / f"frame-{ordinal:08d}.jpg")
    return inputs / "frame-%08d.jpg"


def _validate_plan_sequence(
    plans: Sequence[ImagePackagePlan],
) -> tuple[ContactSheetLayout, tuple[FrameImage, ...], dict[str, Any] | None]:
    if not plans:
        raise ValueError("at least one image package plan is required")
    if [plan.sequence_index for plan in plans] != list(range(len(plans))):
        raise ValueError("image package plan sequence must be contiguous from zero")
    layout = plans[0].layout
    if any((plan.layout != layout for plan in plans)):
        raise ValueError("all image package plans must use one layout")
    frames = tuple((cell.frame for plan in plans for cell in plan.cells))
    if len({frame.sample_index for frame in frames}) != len(frames):
        raise ValueError("image package plans may not duplicate sample indices")
    if any((len(plan.cells) < 1 or len(plan.cells) > layout.capacity for plan in plans)):
        raise ValueError("image package plan cell count is invalid")
    rights = _combined_frame_rights(frames)
    if any((plan.rights != rights for plan in plans)):
        raise ValueError("image package plan rights disagree with their source frames")
    return (layout, frames, rights)


def _source_states(
    frames: Sequence[FrameImage], destination: Path
) -> dict[Path, tuple[int, int, str]]:
    states: dict[Path, tuple[int, int, str]] = {}
    destination_resolved = destination.resolve()
    for frame in frames:
        source = frame.path.resolve()
        if source.is_relative_to(destination_resolved):
            raise ValueError("contact-sheet output must be separate from source images")
        if source in states:
            if states[source][2] != frame.sha256:
                raise ValueError("one source path has conflicting declared hashes")
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        stat = source.stat()
        digest = file_sha256(source)
        if digest != frame.sha256:
            raise ValueError("source image bytes do not match declared SHA-256")
        states[source] = (stat.st_size, stat.st_mtime_ns, digest)
    return states


def _verify_source_states(states: Mapping[Path, tuple[int, int, str]]) -> None:
    for source, (size, modified_ns, digest) in states.items():
        if not source.is_file():
            raise MediaRuntimeError("source image changed during contact-sheet rendering")
        stat = source.stat()
        if stat.st_size != size or stat.st_mtime_ns != modified_ns or file_sha256(source) != digest:
            raise MediaRuntimeError("source image changed during contact-sheet rendering")


def _owned_outputs(destination: Path) -> list[Path]:
    if not destination.is_dir():
        return []
    owned = [
        *destination.glob("contact-sheet-*.jpg"),
        *destination.glob("contact-sheet-*.jpg.rights.json"),
        destination / _MANIFEST_NAME,
        destination / _RIGHTS_MANIFEST_NAME,
    ]
    return sorted({path for path in owned if path.exists()}, key=lambda path: path.name)


def _commit_staged_outputs(
    staged_to_target: Sequence[tuple[Path, Path]], *, stale: Sequence[Path]
) -> None:
    token = uuid4().hex
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for target in stale:
            backup = target.with_name(f".{target.name}.{token}.backup")
            target.replace(backup)
            backups.append((target, backup))
        for staged, target in staged_to_target:
            staged.replace(target)
            committed.append(target)
    except Exception:
        for target in committed:
            target.unlink(missing_ok=True)
        for target, backup in backups:
            if backup.exists():
                backup.replace(target)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def render_image_packages(
    plans: Sequence[ImagePackagePlan],
    output_dir: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    overwrite: bool = False,
    reuse_existing: bool = False,
) -> ImagePackageSet:
    """Render and atomically commit one complete contact-sheet package set."""
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be Boolean")
    layout, frames, rights = _validate_plan_sequence(plans)
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir()):
        raise ValueError("image package output must be a directory")
    stale = _owned_outputs(destination)
    if stale and (not overwrite):
        raise FileExistsError(stale[0])
    states = _source_states(frames, destination)
    executable = _resolve_executable(ffmpeg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    package_set_id = (
        "image-package-set-"
        + _canonical_digest(
            {
                "contract_version": IMAGE_PACKAGING_CONTRACT_VERSION,
                "renderer_version": _renderer_version(layout),
                "layout": layout.to_dict(),
                "package_ids": [plan.package_id for plan in plans],
                "rights": rights,
            }
        )[:20]
    )
    if reuse_existing:
        from .packaging_resume import restore_packages
        restored = restore_packages(plans, destination, layout, rights, package_set_id)
        if restored is not None:
            _verify_source_states(states)
            return restored
    with TemporaryDirectory(prefix=".image-packages-", dir=destination.parent) as staging_text:
        staging = Path(staging_text)
        rendered: list[RenderedImagePackage] = []
        staged_to_target: list[tuple[Path, Path]] = []
        input_pattern = _stage_batch_input_sequence(plans, staging)
        rendered_pattern = staging / "rendered-%04d.jpg"
        command = _build_contact_sheet_batch_command(
            plans, input_pattern, rendered_pattern, ffmpeg=executable
        )
        _run_packaging(command)
        staged_images = sorted(staging.glob("rendered-*.jpg"))
        if len(staged_images) != len(plans):
            raise MediaRuntimeError("contact-sheet renderer produced an unexpected image count")
        for plan, staged_image in zip(plans, staged_images, strict=True):
            if not staged_image.is_file() or staged_image.stat().st_size == 0:
                raise MediaRuntimeError("contact-sheet renderer produced no usable image")
            digest = file_sha256(staged_image)
            target = destination / plan.filename
            rendered.append(
                RenderedImagePackage(
                    package_id=plan.package_id,
                    sequence_index=plan.sequence_index,
                    path=target.resolve(),
                    sha256=digest,
                    width_px=layout.output_width_px,
                    height_px=layout.output_height_px,
                    cells=plan.cells,
                    rights=rights,
                )
            )
            staged_to_target.append((staged_image, target))
            if rights is not None:
                staged_sidecar = write_rights_sidecar(staged_image, rights)
                staged_to_target.append((staged_sidecar, rights_sidecar_path(target)))
        _verify_source_states(states)
        manifest_document = {
            "schema_version": IMAGE_PACKAGING_CONTRACT_VERSION,
            "package_set_id": package_set_id,
            "renderer_version": _renderer_version(layout),
            "layout": layout.to_dict(),
            "package_count": len(rendered),
            "source_sample_count": len(frames),
            "rights": rights,
            "packages": [
                {
                    **plan.provenance(),
                    "filename": image.path.name,
                    "sha256": image.sha256,
                    "width_px": image.width_px,
                    "height_px": image.height_px,
                }
                for plan, image in zip(plans, rendered, strict=True)
            ],
        }
        staged_manifest = staging / _MANIFEST_NAME
        staged_manifest.write_text(
            json.dumps(manifest_document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staged_to_target.append((staged_manifest, destination / _MANIFEST_NAME))
        if rights is not None:
            staged_rights_manifest = staging / _RIGHTS_MANIFEST_NAME
            staged_rights_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": RIGHTS_CONTRACT_VERSION,
                        "rights": rights,
                        "artifacts": [
                            {"filename": image.path.name, "sha256": image.sha256}
                            for image in rendered
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            staged_to_target.append((staged_rights_manifest, destination / _RIGHTS_MANIFEST_NAME))
        destination.mkdir(parents=True, exist_ok=True)
        current_stale = _owned_outputs(destination)
        if current_stale and (not overwrite):
            raise FileExistsError(current_stale[0])
        targets = {target.resolve() for _, target in staged_to_target}
        if len(targets) != len(staged_to_target):
            raise ValueError("image package output targets are not unique")
        _commit_staged_outputs(staged_to_target, stale=current_stale if overwrite else ())
    return ImagePackageSet(
        package_set_id=package_set_id,
        layout=layout,
        images=tuple(rendered),
        manifest_path=(destination / _MANIFEST_NAME).resolve(),
        rights=rights,
    )
