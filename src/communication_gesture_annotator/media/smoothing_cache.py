# SPDX-License-Identifier: Apache-2.0
"""Atomic, independently versioned cache for validated source-rate smoothing."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import gzip
import json
import math
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4
from ..export.json_export import read_json, write_json
from ..rights import validate_rights, write_rights_sidecar, read_rights_sidecar
from .ffmpeg import file_sha256

CACHE_VERSION = "source-rate-smoothing-cache-v0.1"


def validate_cache(
    directory: Path, identity: Mapping[str, Any], rights: Mapping[str, Any]
) -> dict[str, Any]:
    expected_files = {"manifest.json", "people.jsonl.gz", "people.jsonl.gz.rights.json"}
    if (
        directory.is_symlink()
        or directory.is_junction()
        or {p.name for p in directory.iterdir()} != expected_files
        or any(
            (p.is_symlink() or p.is_junction() or (not p.is_file()) for p in directory.iterdir())
        )
    ):
        raise ValueError("smoothing cache must contain regular committed files")
    manifest = read_json(directory / "manifest.json")
    data = directory / "people.jsonl.gz"
    if (
        not isinstance(manifest, dict)
        or manifest.get("policy_version") != CACHE_VERSION
        or manifest.get("input") != dict(identity)
        or (manifest.get("rights") != validate_rights(rights))
        or (type(manifest.get("person_count")) is not int)
        or (manifest["person_count"] < 0)
        or (manifest.get("data") != {"sha256": file_sha256(data), "byte_size": data.stat().st_size})
        or (read_rights_sidecar(data) != validate_rights(rights))
    ):
        raise ValueError("smoothing cache identity, data or rights mismatch")
    return manifest


def load_smoothing(
    directory: Path, people: Sequence[Any], identity: Mapping[str, Any], rights: Mapping[str, Any]
) -> bool:
    if not directory.exists():
        return False
    manifest = validate_cache(directory, identity, rights)
    if manifest["person_count"] != len(people):
        raise ValueError("smoothing cache person count mismatch")
    statuses = {
        "smoothed",
        "raw_fallback_insufficient_support",
        "raw_fallback_asymmetric_support",
        "raw_fallback_singular_fit",
        "mixed_coordinate_status",
        None,
    }
    with gzip.open(directory / "people.jsonl.gz", "rt", encoding="utf-8") as handle:
        for person in people:
            row = json.loads(next(handle, "null"))
            expected = [
                person.source_frame_index,
                person.track_id,
                person.segment_id,
                person.canonical_time_us,
            ]
            if (
                not isinstance(row, list)
                or len(row) != 7
                or row[:4] != expected
                or any((type(a) is not type(b) for a, b in zip(row[:4], expected)))
            ):
                raise ValueError("smoothing cache observation identity mismatch")
            points, states, support = row[4:]
            if not all((isinstance(v, list) and len(v) == 133 for v in (points, states, support))):
                raise ValueError("smoothing cache keypoint dimensions mismatch")
            for raw_point, point, state, metadata in zip(
                person.points, points, states, support, strict=True
            ):
                if point is not None and (
                    not isinstance(point, list)
                    or len(point) != 2
                    or any((type(v) not in (int, float) or not math.isfinite(v) for v in point))
                ):
                    raise ValueError("smoothing cache coordinate is invalid")
                if not isinstance(state, (str, type(None))) or state not in statuses:
                    raise ValueError("smoothing cache state is invalid")
                if (state is None) != (metadata is None) or (
                    metadata is not None
                    and (not isinstance(metadata, dict) or set(metadata) != {"0", "1"})
                ):
                    raise ValueError("smoothing cache support is invalid")
                if state is None and (tuple(point) if point is not None else None) != raw_point:
                    raise ValueError("smoothing cache changed an unsmoothed point")
                if metadata is not None:
                    if point is None or raw_point is None:
                        raise ValueError("smoothing cache invented missing support")
                    for axis in metadata.values():
                        keys = {
                            "status",
                            "support_count",
                            "left_support_count",
                            "right_support_count",
                            "outlier_count",
                            "half_window_us",
                            "support_observation_time_min_us",
                            "support_observation_time_max_us",
                        }
                        if not isinstance(axis, dict) or set(axis) != keys:
                            raise ValueError("smoothing cache axis support is invalid")
                        if (
                            axis["status"] not in statuses - {None, "mixed_coordinate_status"}
                            or (state != "mixed_coordinate_status" and axis["status"] != state)
                            or any(
                                (type(axis[k]) is not int or axis[k] < 0 for k in keys - {"status"})
                            )
                            or (axis["half_window_us"] != identity["half_window_us"])
                            or (axis["outlier_count"] > axis["support_count"])
                            or (
                                not person.canonical_time_us - identity["half_window_us"]
                                < axis["support_observation_time_min_us"]
                                <= person.canonical_time_us
                                <= axis["support_observation_time_max_us"]
                                < person.canonical_time_us + identity["half_window_us"]
                            )
                        ):
                            raise ValueError("smoothing cache axis support contradicts its policy")
            person.points = [tuple(p) if p is not None else None for p in points]
            person.smoothing_statuses = states
            person.smoothing_support = support
        if next(handle, None) is not None:
            raise ValueError("smoothing cache contains extra observations")
    return True


def save_smoothing(
    directory: Path, people: Sequence[Any], identity: Mapping[str, Any], rights: Mapping[str, Any]
) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / (".smoothing-" + uuid4().hex + ".tmp")
    staging.mkdir()
    try:
        data = staging / "people.jsonl.gz"
        with data.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for person in people:
                row = [
                    person.source_frame_index,
                    person.track_id,
                    person.segment_id,
                    person.canonical_time_us,
                    person.points,
                    person.smoothing_statuses,
                    person.smoothing_support,
                ]
                handle.write(
                    (
                        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    ).encode()
                )
        write_rights_sidecar(data, rights)
        write_json(
            staging / "manifest.json",
            {
                "policy_version": CACHE_VERSION,
                "input": dict(identity),
                "rights": validate_rights(rights),
                "person_count": len(people),
                "data": {"sha256": file_sha256(data), "byte_size": data.stat().st_size},
            },
        )
        validate_cache(staging, identity, rights)
        if directory.exists():
            validate_cache(directory, identity, rights)
        else:
            staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def copy_smoothing_caches(
    source_pose: Path, target_pose: Path, raw_manifest_sha256: str, rights: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Copy only complete compatible smoothing checkpoints, never staging output."""
    source = source_pose / "smoothing-cache"
    copied = []
    if not source.exists():
        return copied
    if source.is_symlink() or source.is_junction():
        raise ValueError("smoothing cache root must not be a symlink")
    for directory in sorted(source.iterdir()):
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        manifest = read_json(directory / "manifest.json")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("input"), dict):
            raise ValueError("smoothing cache manifest is invalid")
        identity = manifest.get("input", {})
        if identity.get("raw_pose_manifest_sha256") != raw_manifest_sha256:
            continue
        validate_cache(directory, identity, rights)
        target = target_pose / "smoothing-cache" / directory.name
        target.mkdir(parents=True)
        for name in ("manifest.json", "people.jsonl.gz", "people.jsonl.gz.rights.json"):
            shutil.copyfile(directory / name, target / name)
        validate_cache(target, identity, rights)
        copied.append(
            {
                "relative_path": f"pose/smoothing-cache/{directory.name}",
                "manifest_sha256": file_sha256(target / "manifest.json"),
                "policy_version": CACHE_VERSION,
            }
        )
    return copied
