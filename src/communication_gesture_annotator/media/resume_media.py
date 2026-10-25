# SPDX-License-Identifier: Apache-2.0
"""Restore existing media on resume without decoding or rewriting the recording."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..domain import AnalysisSample
from ..export.json_export import read_json, write_json
from ..rights import read_rights_sidecar, validate_rights
from .ffmpeg import file_sha256


def restore_proxy(
    root: Path, source: Path, *, config: Any, rights: Mapping[str, Any],
    include_audio: bool, start_us: int, end_us: int | None,
) -> dict[str, Any] | None:
    """Validate the retained proxy; missing media can follow the normal rebuild path."""
    manifest_path, proxy_path = root / "preprocessing.json", root / "model-proxy.mp4"
    if not manifest_path.is_file() or not proxy_path.is_file():
        return None
    document = read_json(manifest_path)
    saved_source, proxy = document["source"], document["proxy"]
    if (saved_source["sha256"] != file_sha256(source)
            or saved_source["byte_size"] != source.stat().st_size
            or validate_rights(saved_source["rights"]) != validate_rights(rights)):
        raise ValueError("resume preprocessing belongs to a different source video")
    # Exact scope comparison is made against the recorded, grid-rounded bounds.
    from .processing_scope import processing_scope
    scope = processing_scope(saved_source["duration_us"], start_us=start_us,
                             end_us=end_us, period_us=config.sample_period_us)
    actual = document.get("processing_scope") or proxy.get("processing_scope")
    if actual is None:
        actual = {"start_us": 0, "end_us": saved_source["duration_us"]}
    if (actual["start_us"], actual["end_us"]) != (scope["start_us"], scope["end_us"]):
        raise ValueError("resume preprocessing uses a different processing time range")
    if (proxy["sampling_rate_hz"] != config.sample_rate_hz
            or proxy["maximum_width"] != config.maximum_proxy_width
            or proxy["audio_requested"] != include_audio):
        raise ValueError("resume preprocessing configuration changed")
    if (proxy["relative_path"] != "model-proxy.mp4"
            or proxy["sha256"] != file_sha256(proxy_path)
            or proxy["byte_size"] != proxy_path.stat().st_size
            or read_rights_sidecar(proxy_path) != proxy["rights"]):
        raise ValueError("resume proxy does not match its saved manifest")
    pose_path = root / "pose" / "manifest.json"
    if pose_path.is_file():
        expected = read_json(pose_path)["input"].get("preprocessing_sha256")
        if expected is not None and expected != file_sha256(manifest_path):
            raise ValueError("resume preprocessing manifest differs from its pose checkpoint")
    return document


def _frame_settings(proxy: Path, maximum_width: int, jpeg_quality: int) -> dict[str, Any]:
    return {"version": "analysis-frame-resume-v0.1", "proxy_sha256": file_sha256(proxy),
            "maximum_width": maximum_width, "jpeg_quality": jpeg_quality}


def remember_frame_settings(proxy: Path, destination: Path, maximum_width: int, jpeg_quality: int) -> None:
    """Keep cache settings separate so existing media/pose identities do not change."""
    write_json(destination / "extraction-settings.json", _frame_settings(proxy, maximum_width, jpeg_quality))


def restore_frames(proxy: Path, samples: Any, destination: Path, *,
                   maximum_width: int = 640, jpeg_quality: int = 85) -> list[dict[str, Any]] | None:
    """Read and validate the saved frame inventory, including scoped placeholders."""
    if not (destination / "rights-manifest.json").is_file():
        return None
    samples = [AnalysisSample(**s) if isinstance(s, Mapping) else s for s in samples]
    if any(not (destination / f"sample-{s.index:08d}.jpg").is_file() for s in samples):
        return None
    if destination.resolve() != (proxy.parent / "analysis-frames").resolve():
        return None
    settings_path = destination / "extraction-settings.json"
    if settings_path.is_file():
        if read_json(settings_path) != _frame_settings(proxy, maximum_width, jpeg_quality):
            raise ValueError("resume analysis-frame extraction settings or proxy changed")
    elif maximum_width != 640 or jpeg_quality != 85:
        # Older caches did not record extraction options. Non-default encodings
        # must follow ordinary regeneration and checkpoint identity validation.
        return None
    from .preprocessing_reuse import _validate_analysis_frames
    records, _, _ = _validate_analysis_frames(proxy.parent, samples, read_rights_sidecar(proxy))
    return [dict(record) for record in records]
