# SPDX-License-Identifier: Apache-2.0
"""Hash-bound PNG recovery for image decoding and request-size failures."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
import re
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4
from ..export.json_export import read_json, write_json
from ..media.ffmpeg import _resolve_executable, file_sha256
from ..media.lossless_images import optimize_png, resize_png, transcode
from ..rights import combine_rights, read_rights_sidecar, write_rights_sidecar

POLICY_VERSION = "joint-activity-lossless-png-recovery-v0.1"
SIZE_POLICY_VERSION = "probability-lossless-png-size-recovery-v0.1"
SPATIAL_SIZE_POLICY_VERSION = "probability-spatial-png-size-recovery-v0.1"
SPATIAL_SIZE_PERCENTAGES = (85, 70)


def release_checkpointed_images(run_dir: Path, request_id: str) -> None:
    """Keep retry decisions, but release PNGs only after a durable answer exists."""
    from ..storage import minimal_enabled, release_cache_files
    if not minimal_enabled():
        return
    root = ImageRecovery.request_directory(run_dir, request_id)
    release_cache_files(root, [])
    if not root.is_dir():
        return
    for directory in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{64}", directory.name) or not directory.is_dir():
            continue
        release_cache_files(directory, [])
        images = sorted(p for p in directory.iterdir() if re.fullmatch(
            r"image-[0-9]{3,}\.png(?:\.rights\.json)?", p.name))
        release_cache_files(root, [directory / "images.json",
            directory / "images.json.rights.json", *images])


def digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


class ImageRecovery:
    """Record the representation decision before conversion or model dispatch."""

    def __init__(
        self, run_dir: Path, request_id: str, identity: Mapping[str, Any],
        *, optimize_size: bool = False, scale_percent: int = 100,
        preserve_ordinals: Sequence[int] = (),
    ) -> None:
        self.run_dir = run_dir
        self.optimize_size = optimize_size
        if scale_percent != 100 and (not optimize_size or scale_percent not in SPATIAL_SIZE_PERCENTAGES):
            raise ValueError("unsupported image recovery scale")
        self.scale_percent = scale_percent
        self.preserve_ordinals = tuple(sorted(set(preserve_ordinals)))
        self.policy_version = SIZE_POLICY_VERSION if optimize_size else POLICY_VERSION
        self.identity = dict(identity)
        if scale_percent != 100:
            self.policy_version = SPATIAL_SIZE_POLICY_VERSION
            self.identity["spatial_transform"] = {
                "policy_version": self.policy_version, "scale_percent": scale_percent,
                "preserve_ordinals": list(self.preserve_ordinals), "interpolation": "INTER_AREA",
            }
        self.input_sha256 = digest(self.identity)
        self.directory = self.request_directory(run_dir, request_id) / self.input_sha256
        self.decision = {
            "policy_version": self.policy_version,
            "request_id": request_id,
            "input_sha256": self.input_sha256,
            "trigger": "request_too_large" if optimize_size else "backend_internal_error",
            "maximum_representation_fallbacks": 1,
        }

    @staticmethod
    def request_directory(run_dir: Path, request_id: str) -> Path:
        return run_dir / "model-image-recovery" / sha256(request_id.encode()).hexdigest()[:24]

    def saved(self) -> bool:
        path = self.directory / "decision.json"
        if not path.exists():
            return False
        if read_json(path) != self.decision:
            raise ValueError("image recovery decision does not match its request")
        return True

    def record(self) -> None:
        if not self.saved():
            write_json(self.directory / "decision.json", self.decision)

    def size_rejected(self) -> bool:
        """Skip a representation already rejected by this endpoint/input pair."""
        path = self.directory / "size-rejected.json"
        if not path.exists():
            return False
        if read_json(path) != {"policy_version": self.policy_version, "input_sha256": self.input_sha256}:
            raise ValueError("size rejection does not match its request")
        return True

    def record_size_rejection(self) -> None:
        write_json(self.directory / "size-rejected.json", {
            "policy_version": self.policy_version, "input_sha256": self.input_sha256,
        })

    def prepare(
        self, images: Sequence[Path], *, maximum_bytes: int, maximum_width: int, maximum_height: int
    ) -> tuple[list[Path], dict[str, Any]]:
        if not self.saved():
            raise ValueError("image recovery decision must be recorded before preparation")
        original_records = self.identity["images"]
        if len(images) != len(original_records) or any(
            (
                file_sha256(path) != original["sha256"]
                for path, original in zip(images, original_records, strict=True)
            )
        ):
            raise ValueError("image recovery source identity changed")
        labels = []
        preprocessing = self.run_dir / "preprocessing.json"
        if preprocessing.exists():
            labels.append(read_json(preprocessing)["source"]["rights"])
        for path in images:
            label = read_rights_sidecar(path)
            if label is not None:
                labels.append(label)
            packages = path.parent / "image-packages.json"
            if packages.exists():
                label = read_json(packages).get("rights")
                if label is not None:
                    labels.append(label)
        if not labels:
            raise ValueError("image recovery requires inherited source rights")
        rights = combine_rights(*labels)
        manifest_path = self.directory / "images.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if (
                read_rights_sidecar(manifest_path) != rights
                or manifest.get("rights") != rights
                or manifest.get("policy_version") != self.policy_version
                or (manifest.get("input_sha256") != self.input_sha256)
                or (len(manifest.get("images", [])) != len(images))
            ):
                raise ValueError("image recovery manifest does not match its request")
            outputs = []
            for ordinal, (item, source) in enumerate(
                zip(manifest["images"], original_records, strict=True)
            ):
                path = self.directory / f"image-{ordinal:03d}.png"
                if (
                    item["image_ordinal"] != ordinal
                    or item["source_sha256"] != source["sha256"]
                    or file_sha256(path) != item["sha256"]
                    or (read_rights_sidecar(path) != rights)
                    or (item["decoded_pixels_identical"] != (self.scale_percent == 100 or ordinal in self.preserve_ordinals))
                    or (item["width_px"] > maximum_width)
                    or (item["height_px"] > maximum_height)
                ):
                    raise ValueError("cached PNG recovery image failed validation")
                outputs.append(path)
            if sum((path.stat().st_size for path in outputs)) > maximum_bytes:
                raise ValueError("PNG recovery exceeds maximum_image_bytes_per_request")
            return (outputs, manifest)
        ffmpeg, ffprobe = (_resolve_executable("ffmpeg"), _resolve_executable("ffprobe"))
        version = (
            subprocess.run([ffmpeg, "-version"], capture_output=True, check=True, timeout=10)
            .stdout.decode("utf-8", errors="replace")
            .splitlines()[0]
        )
        if self.optimize_size:
            import cv2

            version = f"OpenCV {cv2.__version__}: PNG compression 9, filtered strategy"
            if self.scale_percent != 100:
                version += f", INTER_AREA resize to {self.scale_percent}% (except preserved images)"
        outputs, records = ([], [])
        total = 0
        for ordinal, (source, original) in enumerate(zip(images, original_records, strict=True)):
            dimensions = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    str(source.resolve()),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
            stream = json.loads(dimensions.stdout)["streams"][0]
            width, height = (stream["width"], stream["height"])
            if width > maximum_width or height > maximum_height:
                raise ValueError("PNG recovery image exceeds configured dimensions")
            resized = self.scale_percent != 100 and ordinal not in self.preserve_ordinals
            source_width, source_height = width, height
            png, pixel_sha256 = (
                resize_png(source, width, height, self.scale_percent) if resized else
                optimize_png(source, width, height)
                if self.optimize_size else transcode(source, width, height, ffmpeg)
            )
            if resized:
                width, height = (max(1, (value * self.scale_percent + 50) // 100) for value in (width, height))
            total += len(png)
            if total > maximum_bytes:
                raise ValueError("PNG recovery exceeds maximum_image_bytes_per_request")
            target = self.directory / f"image-{ordinal:03d}.png"
            temporary = self.directory / f".{target.name}.{uuid4().hex}.tmp"
            try:
                temporary.write_bytes(png)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            write_rights_sidecar(target, rights, overwrite=True)
            outputs.append(target)
            records.append(
                {
                    "image_ordinal": ordinal,
                    "source_sha256": original["sha256"],
                    "sha256": sha256(png).hexdigest(),
                    "rgb24_sha256": pixel_sha256,
                    "width_px": width,
                    "height_px": height,
                    "decoded_pixels_identical": not resized,
                    **({"source_width_px": source_width, "source_height_px": source_height,
                        "scale_percent": self.scale_percent, "interpolation": "INTER_AREA"} if resized else {}),
                    "path": target.relative_to(self.run_dir).as_posix(),
                }
            )
        manifest = {
            "policy_version": self.policy_version,
            "input_sha256": self.input_sha256,
            "rights": rights,
            "decoder_version": version,
            "pixel_comparison": (
                "exact output PNG raster; spatially resized source pixels differ"
                if self.scale_percent != 100 else
                "exact OpenCV RGB8 pixel equality"
                if self.optimize_size else "exact FFmpeg rgb24 byte equality"
            ),
            "images": records,
        }
        write_json(manifest_path, manifest)
        write_rights_sidecar(manifest_path, rights, overwrite=True)
        return (outputs, manifest)


def recovery_trace_context(
    context: Mapping[str, Any], manifest: Mapping[str, Any], parent_request_id: str
) -> dict[str, Any]:
    result = deepcopy(dict(context))
    previous = result.get("image_refs", [])
    refs = []
    for item in manifest["images"]:
        index = item["image_ordinal"]
        ref = deepcopy(previous[index]) if index < len(previous) else {"kind": "reencoded_image"}
        if not item["decoded_pixels_identical"]:
            # Ordinals/times are stable; pixel coordinates refer to the delivered
            # raster. Keep the original geometry alongside the transform.
            ref["source_geometry"] = {key: deepcopy(ref[key]) for key in ("width_px", "height_px", "cells") if key in ref}
            sx = item["width_px"] / item["source_width_px"]
            sy = item["height_px"] / item["source_height_px"]
            for cell in ref.get("cells", []):
                if "cell_xyxy_pixels" in cell:
                    cell["cell_xyxy_pixels"] = [round(value * (sx if n % 2 == 0 else sy))
                        for n, value in enumerate(cell["cell_xyxy_pixels"])]
                if "rgb_size_px" in cell:
                    cell["rgb_size_px"] = [round(value * scale) for value, scale in zip(cell["rgb_size_px"], (sx, sy), strict=True)]
        ref.update({key: item[key] for key in ("image_ordinal", "sha256", "width_px", "height_px")})
        ref["source_encoded_sha256"] = item["source_sha256"]
        refs.append(ref)
    result["image_refs"] = refs
    result["image_reencoding"] = {
        **deepcopy(dict(manifest)),
        "parent_request_id": parent_request_id,
    }
    return result
