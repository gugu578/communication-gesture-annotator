# SPDX-License-Identifier: Apache-2.0
"""Clean exact-frame person crops and compact, labelled Phase2 image packets."""

from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from ..rights import validate_rights, write_rights_sidecar, read_rights_sidecar
from ..export.json_export import write_json
from .image_packaging import ContactSheetLayout

VERSION = "clean-compact-phase2-rgb-v0.1"


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_rgb(frame: Mapping[str, Any]) -> Any:
    import cv2
    import numpy as np

    content = Path(frame["path"]).read_bytes()
    if sha256(content).hexdigest() != frame["sha256"]:
        raise ValueError("Phase2 RGB source hash mismatch")
    rgb = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError("Phase2 RGB source is not a decodable image")
    return rgb


def _cache_png(
    directory: Path,
    identity: Mapping[str, Any],
    render: Callable[[], Any],
    rights: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    import cv2

    key = _digest(identity)
    path, manifest = (directory / f"{key}.png", directory / f"{key}.json")
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            saved.get("input_sha256") != key
            or saved.get("input") != identity
            or saved.get("sha256") != sha256(path.read_bytes()).hexdigest()
            or (read_rights_sidecar(path) != rights)
        ):
            raise ValueError("Phase2 RGB cache identity or image hash mismatch")
        return (path, saved)
    rgb = render()
    ok, encoded = cv2.imencode(".png", rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("Phase2 RGB PNG encoding failed")
    payload = encoded.tobytes()
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    write_rights_sidecar(path, rights, overwrite=True)
    saved = {
        "input_sha256": key,
        "input": dict(identity),
        "sha256": sha256(payload).hexdigest(),
        "width_px": int(rgb.shape[1]),
        "height_px": int(rgb.shape[0]),
    }
    write_json(manifest, saved)
    return (path, saved)


class CleanPhase2Frames:
    """Lazily reuse clean RGB, never attempt to erase a rendered skeleton.

    The stored pose focus crop already includes the body/hand padding. Reapply
    that rectangle to its verified, unpainted source frame. Ambiguous/unverified
    detections retain full-scene RGB. Original preprocessing is never modified.
    """

    def __init__(
        self,
        frames: Mapping[int, Mapping[str, Any]],
        focus_frames: Sequence[Mapping[str, Any]],
        directory: Path,
        rights: Mapping[str, Any],
    ) -> None:
        self.source = frames
        self.focus = {int(f["sample_index"]): f for f in focus_frames}
        self.frames = {i: dict(f) for i, f in frames.items()}
        self.directory = directory
        self.rights = validate_rights(rights)
        self.prepared: set[int] = set()

    def prepare(self, indices: Sequence[int]) -> None:
        from ..storage import minimal_enabled
        if minimal_enabled():
            self.release_except(set(indices))
        for index in indices:
            if index in self.prepared:
                continue
            source = self.source[index]
            focus = self.focus.get(index, {})
            transform = focus.get("transform", {})
            selected = source.get("selected_source_frame")
            verified = (
                bool(selected)
                and selected == focus.get("selected_source_frame")
                and (focus.get("pose_rgb_pairing") == "verified_same_source_frame")
                and (focus.get("conditioning_mode") == "pose_guided_crop")
            )
            crop = transform.get("crop_xyxy_pixels") if verified else None
            identity = {
                "policy_version": VERSION,
                "source_sha256": source["sha256"],
                "crop_xyxy_pixels": crop,
                "source_dimensions": [transform.get("input_width"), transform.get("input_height")]
                if crop is not None
                else None,
                "rights": self.rights,
            }

            def render() -> Any:
                rgb = _read_rgb(source)
                if crop is None:
                    return rgb
                if (
                    not isinstance(crop, list)
                    or len(crop) != 4
                    or any((type(v) is not int for v in crop))
                    or ([rgb.shape[1], rgb.shape[0]] != identity["source_dimensions"])
                ):
                    raise ValueError("Phase2 crop does not match its exact source geometry")
                x1, y1, x2, y2 = crop
                if not (0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
                    raise ValueError("Phase2 crop lies outside source RGB")
                return rgb[y1:y2, x1:x2].copy()

            if sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Phase2 RGB source hash mismatch")
            path, saved = _cache_png(self.directory / "frames", identity, render, self.rights)
            self.frames[index] = {
                **source,
                "path": str(path.resolve()),
                "sha256": saved["sha256"],
                "byte_size": path.stat().st_size,
                "rights": self.rights,
                "width_px": saved["width_px"],
                "height_px": saved["height_px"],
                "instantaneous_skeleton": False,
                "conditioning_mode": "clean_person_crop" if crop else "clean_full_frame",
                "phase2_rgb_policy_version": VERSION,
                "crop_xyxy_pixels": crop,
                "clean_source_sha256": source["sha256"],
            }
            self.prepared.add(index)


    def release_except(self, keep: set[int]) -> None:
        """Bound derived crops to the current context; original RGB is retained."""
        from ..storage import release_image
        retained = {Path(self.frames[i]["path"]).resolve() for i in self.prepared & keep}
        for index in self.prepared - keep:
            self.frames[index] = dict(self.source[index])
        self.prepared.intersection_update(keep)
        cache = self.directory / "frames"
        # Also handles crops left by an interrupted or previously compact run.
        from ..storage import release_cache_files
        release_cache_files(cache, [])
        for path in cache.glob("*.png"):
            if path.resolve() not in retained:
                release_image(path, cache)

def compact_sheets(
    frames: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
    labels: Mapping[int, str],
    layout: ContactSheetLayout,
    directory: Path,
    rights: Mapping[str, Any],
    maximum_images: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Pack aspect-preserving cells with variable widths and a small label band.

    No 16:9 canvas is restored around a portrait crop. Empty final cells are
    omitted. A multirow sheet may retain only the small difference in row widths.
    """
    import cv2
    import numpy as np

    if not indices or len(set(indices)) != len(indices) or set(labels) != set(indices):
        raise ValueError("compact Phase2 labels must cover unique delivered samples")
    if (len(indices) + layout.capacity - 1) // layout.capacity > maximum_images:
        raise ValueError("compact Phase2 images exceed the image budget")
    rights = validate_rights(rights)
    paths, references = ([], [])
    font = cv2.FONT_HERSHEY_SIMPLEX
    band = min(42, layout.tile_height_px // 3)
    for sequence, offset in enumerate(range(0, len(indices), layout.capacity)):
        cells, tiles = ([], [])
        for ordinal, index in enumerate(indices[offset : offset + layout.capacity]):
            frame = frames[index]
            rgb = _read_rgb(frame)
            scale = min(
                layout.tile_width_px / rgb.shape[1], (layout.tile_height_px - band) / rgb.shape[0]
            )
            width, height = (
                max(1, round(rgb.shape[1] * scale)),
                max(1, round(rgb.shape[0] * scale)),
            )
            label = labels[index]
            words = label.split()
            lines = [" ".join(words[:2]), " ".join(words[2:])]
            font_scale = min(0.5, band / 80)
            text_width = max((cv2.getTextSize(t, font, font_scale, 1)[0][0] for t in lines))
            tile_width = min(layout.tile_width_px, max(width, text_width + 8))
            tile = np.full((height + band, tile_width, 3), 16, np.uint8)
            left = (tile_width - width) // 2
            tile[band:, left : left + width] = cv2.resize(
                rgb, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
            color = (0, 255, 255) if label.startswith("TARGET") else (255, 255, 255)
            for line, text in enumerate(lines):
                size = cv2.getTextSize(text, font, font_scale, 1)[0][0]
                cv2.putText(
                    tile,
                    text,
                    (4, (line + 1) * band // 2 - 4),
                    font,
                    font_scale * min(1.0, (tile_width - 8) / max(1, size)),
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cells.append(
                {
                    "ordinal": ordinal,
                    "row": ordinal // layout.columns,
                    "column": ordinal % layout.columns,
                    "sample_index": index,
                    "start_us": frame["start_us"],
                    "end_us": frame["end_us"],
                    "source_sha256": frame["sha256"],
                    "visible_label": label,
                    "rgb_size_px": [width, height],
                }
            )
            tiles.append(tile)
        row_widths = [
            sum((t.shape[1] for t in tiles[a : a + layout.columns]))
            + layout.gutter_px * (len(tiles[a : a + layout.columns]) - 1)
            for a in range(0, len(tiles), layout.columns)
        ]
        row_heights = [
            max((t.shape[0] for t in tiles[a : a + layout.columns]))
            for a in range(0, len(tiles), layout.columns)
        ]
        canvas = np.full(
            (sum(row_heights) + layout.gutter_px * (len(row_heights) - 1), max(row_widths), 3),
            16,
            np.uint8,
        )
        x = y = 0
        for n, (cell, tile) in enumerate(zip(cells, tiles, strict=True)):
            if n and n % layout.columns == 0:
                y += row_heights[cell["row"] - 1] + layout.gutter_px
                x = 0
            canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
            cell["cell_xyxy_pixels"] = [x, y, x + tile.shape[1], y + tile.shape[0]]
            x += tile.shape[1] + layout.gutter_px
        identity = {
            "policy_version": VERSION,
            "layout": layout.to_dict(),
            "cells": cells,
            "rights": rights,
        }
        path, saved = _cache_png(directory, identity, lambda: canvas, rights)
        paths.append(path.resolve())
        references.append(
            {
                "kind": "contact_sheet",
                "image_ordinal": sequence,
                "sequence_index": sequence,
                "package_id": "clean-rgb-" + saved["input_sha256"][:24],
                "sha256": saved["sha256"],
                "width_px": saved["width_px"],
                "height_px": saved["height_px"],
                "cell_count": len(cells),
                "cells": cells,
                "renderer_version": VERSION,
            }
        )
    return (paths, references)
