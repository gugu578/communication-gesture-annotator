# SPDX-License-Identifier: Apache-2.0
"""Ordered low-resolution frame extraction from the verified model proxy."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4
from .ffmpeg import (
    MediaRuntimeError,
    _escape_image2_sequence_pattern,
    _resolve_executable,
    file_sha256,
)
from ..rights import RIGHTS_CONTRACT_VERSION, read_rights_sidecar


def _sample_field(sample: object, name: str) -> int:
    value = sample.get(name) if isinstance(sample, Mapping) else getattr(sample, name, None)
    if type(value) is not int:
        raise ValueError(f"analysis sample {name} must be an integer")
    return value


def _validated_samples(samples: Sequence[object]) -> list[tuple[int, int, int]]:
    if not samples:
        raise ValueError("at least one analysis sample is required")
    validated: list[tuple[int, int, int]] = []
    for position, sample in enumerate(samples):
        index = _sample_field(sample, "index")
        start_us = _sample_field(sample, "start_us")
        end_us = _sample_field(sample, "end_us")
        if index != position:
            raise ValueError("analysis samples must be ordered and contiguous from index zero")
        if start_us < 0 or end_us <= start_us:
            raise ValueError("analysis sample intervals must be positive and non-negative")
        if position == 0 and start_us != 0:
            raise ValueError("the complete analysis grid must begin at zero")
        if validated and start_us != validated[-1][2]:
            raise ValueError("analysis sample intervals must be contiguous")
        duration_us = end_us - start_us
        if duration_us > 100000 or (position < len(samples) - 1 and duration_us != 100000):
            raise ValueError("analysis samples must describe the fixed 10 Hz grid")
        source_gap_reason = (
            sample.get("source_gap_reason")
            if isinstance(sample, Mapping)
            else getattr(sample, "source_gap_reason", None)
        )
        if source_gap_reason is not None and (not isinstance(source_gap_reason, str)):
            raise ValueError("analysis sample source_gap_reason must be a string or null")
        validated.append((index, start_us, end_us))
    return validated


def _run_extraction(command: Sequence[str]) -> None:
    try:
        subprocess.run(list(command), check=True, capture_output=True, text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        generic_markers = (
            "Terminating thread with return code",
            "Task finished with error code",
            "Conversion failed!",
        )
        useful = [line for line in lines if not any((marker in line for marker in generic_markers))]
        selected = (useful or lines)[-6:]
        detail = " | ".join(selected)[:2000] if selected else str(exc)
        raise MediaRuntimeError(f"analysis-frame extraction failed: {detail}") from exc


def extract_analysis_frames(
    proxy_path: str | Path,
    samples: Sequence[object],
    output_dir: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    maximum_width: int = 640,
    jpeg_quality: int = 85,
    overwrite: bool = False,
    resume: bool = False,
    reuse_from: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract exactly one ordered JPEG for every complete-grid analysis sample.

    The input must be the already verified fixed-10-Hz model proxy. Extraction is
    staged outside ``output_dir`` and no result is committed until FFmpeg has
    produced the exact expected file set.
    """
    proxy = Path(proxy_path)
    destination = Path(output_dir)
    if not proxy.is_file():
        raise MediaRuntimeError("verified model proxy does not exist")
    proxy_rights = read_rights_sidecar(proxy)
    if destination.resolve() == proxy.resolve():
        raise ValueError("the model proxy may not be overwritten")
    if destination.exists() and (not destination.is_dir()):
        raise ValueError("analysis-frame output must be a directory")
    if type(maximum_width) is not int or maximum_width < 2:
        raise ValueError("maximum_width must be an integer of at least two pixels")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer in [1, 100]")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be Boolean")
    grid = _validated_samples(samples)
    from .processing_scope import OUTSIDE

    selected_positions = [
        i
        for i, sample in enumerate(samples)
        if (
            sample.get("source_gap_reason")
            if isinstance(sample, Mapping)
            else getattr(sample, "source_gap_reason", None)
        )
        != OUTSIDE
    ]
    if not selected_positions or selected_positions != list(
        range(selected_positions[0], selected_positions[-1] + 1)
    ):
        raise ValueError("processing range must contain one contiguous analysis interval")
    scoped = len(selected_positions) != len(grid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    targets = [destination / f"sample-{index:08d}.jpg" for index, _, _ in grid]
    rights_manifest_target = destination / "rights-manifest.json"
    if any((target.resolve() == proxy.resolve() for target in targets)):
        raise ValueError("the model proxy may not be overwritten")
    existing = [target for target in targets if target.exists()]
    if rights_manifest_target.exists():
        existing.append(rights_manifest_target)
    if existing and (not overwrite):
        raise FileExistsError(f"analysis frame already exists: {existing[0]}")
    unexpected: list[Path] = []
    if destination.exists():
        expected_names = {target.name for target in targets}
        unexpected = [
            path for path in destination.glob("sample-*.jpg") if path.name not in expected_names
        ]
        if unexpected and (not overwrite):
            raise ValueError("analysis-frame output contains files from a different sample grid")
    if resume:
        from .resume_media import restore_frames
        restored = restore_frames(proxy, samples, destination,
            maximum_width=maximum_width, jpeg_quality=jpeg_quality)
        if restored is not None:
            return restored
    executable = _resolve_executable(ffmpeg)
    with TemporaryDirectory(prefix=".analysis-frames-", dir=destination.parent) as staging_text:
        staging = Path(staging_text)
        pattern = staging / "frame-%08d.jpg"
        ffmpeg_qscale = max(2, min(31, round(31 - (jpeg_quality - 1) * 29 / 99)))
        final_sample = samples[selected_positions[-1]]
        final_gap_reason = (
            final_sample.get("source_gap_reason")
            if isinstance(final_sample, Mapping)
            else getattr(final_sample, "source_gap_reason", None)
        )
        video_filters = []
        if final_gap_reason == "no_decodable_frame_in_sample":
            video_filters.append("tpad=stop_mode=clone:stop=1")
        video_filters.append(f"scale=w='min({maximum_width},iw)':h=-2:flags=lanczos")
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(proxy),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            ",".join(video_filters),
            "-fps_mode",
            "passthrough",
            "-q:v",
            str(ffmpeg_qscale),
            "-frames:v",
            str(len(selected_positions)),
            "-start_number",
            "0",
            _escape_image2_sequence_pattern(pattern, "%08d"),
        ]
        reused = False
        if reuse_from is not None:
            from .scoped_reuse import reuse_analysis_slice

            reused = reuse_analysis_slice(
                reuse_from,
                proxy.parent,
                selected_positions,
                staging,
                maximum_width=maximum_width,
                jpeg_quality=jpeg_quality,
            )
        if not reused:
            _run_extraction(command)
        actual = sorted(staging.glob("frame-*.jpg"))
        expected_names = [
            f"frame-{position:08d}.jpg" for position in range(len(selected_positions))
        ]
        if len(actual) != len(selected_positions):
            raise MediaRuntimeError(
                f"verified 10 Hz proxy frame extraction remained incompatible with the analysis grid after bounded tail adjustment: expected {len(selected_positions)}, extracted {len(actual)}"
            )
        if [path.name for path in actual] != expected_names:
            raise MediaRuntimeError(
                "analysis-frame extraction produced noncontiguous output numbering"
            )
        if scoped:
            import cv2
            import numpy as np

            blank = staging / "outside-range.jpg"
            ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
            if not ok:
                raise MediaRuntimeError("could not encode excluded-range placeholder")
            blank.write_bytes(encoded.tobytes())
            selected_files = dict(zip(selected_positions, actual, strict=True))
            expanded = []
            for i in range(len(grid)):
                if i in selected_files:
                    expanded.append(selected_files[i])
                else:
                    empty = staging / f"excluded-{i:08d}.jpg"
                    try:
                        empty.hardlink_to(blank)
                    except OSError:
                        empty.write_bytes(blank.read_bytes())
                    expanded.append(empty)
            actual = expanded
        destination.mkdir(parents=True, exist_ok=True)
        digests = [file_sha256(path) for path in actual]
        staged_rights_manifest: Path | None = None
        if proxy_rights is not None:
            staged_rights_manifest = staging / "rights-manifest.json"
            staged_rights_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": RIGHTS_CONTRACT_VERSION,
                        "rights": proxy_rights,
                        "artifacts": [
                            {"filename": target.name, "sha256": digest}
                            for target, digest in zip(targets, digests, strict=True)
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        token = uuid4().hex
        backups: list[tuple[Path, Path]] = []
        committed: list[Path] = []
        try:
            for stale in unexpected:
                backup = destination / f".{stale.name}.{token}.backup"
                stale.replace(backup)
                backups.append((stale, backup))
            for target in targets:
                if not target.exists():
                    continue
                if not overwrite:
                    raise FileExistsError(f"analysis frame already exists: {target}")
                backup = destination / f".{target.name}.{token}.backup"
                target.replace(backup)
                backups.append((target, backup))
            if rights_manifest_target.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"analysis-frame rights manifest already exists: {rights_manifest_target}"
                    )
                backup = destination / f".{rights_manifest_target.name}.{token}.backup"
                rights_manifest_target.replace(backup)
                backups.append((rights_manifest_target, backup))
            for extracted, target in zip(actual, targets, strict=True):
                extracted.replace(target)
                committed.append(target)
            if staged_rights_manifest is not None:
                staged_rights_manifest.replace(rights_manifest_target)
                committed.append(rights_manifest_target)
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
        result: list[dict[str, Any]] = []
        for (index, start_us, end_us), sample, target, digest in zip(
            grid, samples, targets, digests, strict=True
        ):
            source_gap_reason = (
                sample.get("source_gap_reason")
                if isinstance(sample, Mapping)
                else getattr(sample, "source_gap_reason", None)
            )
            result.append(
                {
                    "sample_index": index,
                    "start_us": start_us,
                    "end_us": end_us,
                    "path": str(target.resolve()),
                    "sha256": digest,
                    **(
                        {"source_gap_reason": source_gap_reason}
                        if source_gap_reason is not None
                        else {}
                    ),
                    **({"rights": proxy_rights} if proxy_rights is not None else {}),
                }
            )
    from .resume_media import remember_frame_settings
    remember_frame_settings(proxy, destination, maximum_width, jpeg_quality)
    return result
