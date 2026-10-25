# SPDX-License-Identifier: Apache-2.0
"""Restore a contact-sheet package with the exact same rendered bytes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..rights import read_rights_sidecar
from .ffmpeg import file_sha256


def restore_packages(plans: Sequence[Any], destination: Path, layout: Any,
                     rights: Mapping[str, Any] | None, package_set_id: str) -> Any:
    from .image_packaging import (
        _MANIFEST_NAME, _renderer_version, ImagePackageSet, RenderedImagePackage,
    )
    manifest_path = destination / _MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (saved.get("package_set_id") != package_set_id
            or saved.get("layout") != layout.to_dict()
            or saved.get("rights") != rights
            or saved.get("renderer_version") != _renderer_version(layout)
            or len(saved.get("packages", [])) != len(plans)):
        return None
    images = []
    for plan, record in zip(plans, saved["packages"], strict=True):
        if any(record.get(k) != v for k, v in plan.provenance().items()):
            return None
        if record.get("filename") != plan.filename:
            return None
        path = destination / plan.filename
        if (not path.is_file() or file_sha256(path) != record.get("sha256")
                or record.get("width_px") != layout.output_width_px
                or record.get("height_px") != layout.output_height_px):
            return None
        if rights is not None and read_rights_sidecar(path) != rights:
            raise ValueError("cached contact sheet rights changed")
        images.append(RenderedImagePackage(package_id=plan.package_id,
            sequence_index=plan.sequence_index, path=path.resolve(),
            sha256=record["sha256"], width_px=record["width_px"],
            height_px=record["height_px"], cells=plan.cells, rights=rights))
    return ImagePackageSet(package_set_id=package_set_id, layout=layout,
        images=tuple(images), manifest_path=manifest_path.resolve(), rights=rights)
