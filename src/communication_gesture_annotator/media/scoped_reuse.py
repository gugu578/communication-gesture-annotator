# SPDX-License-Identifier: Apache-2.0
"""Reuse selected preprocessing evidence without importing the surrounding run."""

from __future__ import annotations
import json
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from ..config import PosePreprocessingConfig
from ..export.json_export import read_json, write_json
from ..rights import read_rights_sidecar, validate_rights
from ..storage import compact_enabled, data_path, open_data, output_data_path
from .ffmpeg import file_sha256
from .processing_scope import OUTSIDE, excluded_observation

LOG = logging.getLogger(__name__)


def _unavailable(stage: str, exc: Exception) -> bool:
    LOG.debug(
        "stage=preprocessing event=scoped_reuse status=recomputing component=%s reason=%s",
        stage,
        str(exc),
    )
    return False


def reuse_analysis_slice(
    old: Path,
    new: Path,
    indices: Sequence[int],
    staging: Path,
    *,
    maximum_width: int,
    jpeg_quality: int,
) -> bool:
    """Validate selected images before copying any; outside files need not exist."""
    try:
        previous, current = (
            read_json(old / "preprocessing.json"),
            read_json(new / "preprocessing.json"),
        )
        if previous["source"]["sha256"] != current["source"]["sha256"]:
            raise ValueError("different source video")
        if previous["proxy"]["sampling_rate_hz"] != current["proxy"]["sampling_rate_hz"]:
            raise ValueError("different analysis rate")
        if previous["proxy"]["maximum_width"] != current["proxy"]["maximum_width"]:
            raise ValueError("different proxy dimensions")
        index = (
            read_json(old / "analysis-index.json")
            if (old / "analysis-index.json").is_file()
            else {}
        )
        plan = (
            read_json(old / "inference-plan.json")
            if (old / "inference-plan.json").is_file()
            else {}
        )
        inference = index.get("configuration", {}).get("inference", plan.get("model_limits", {}))
        if (
            inference.get("frame_width") != maximum_width
            or inference.get("jpeg_quality") != jpeg_quality
        ):
            raise ValueError("image extraction settings unavailable or changed")
        rights = read_json(old / "analysis-frames" / "rights-manifest.json")
        if validate_rights(rights["rights"]) != validate_rights(current["proxy"]["rights"]):
            raise ValueError("different image rights")
        digests = {item["filename"]: item["sha256"] for item in rights["artifacts"]}
        files = []
        for i in indices:
            if previous["proxy"]["frame_mapping"][i] != current["proxy"]["frame_mapping"][i]:
                raise ValueError("selected source-time mapping differs")
            if previous["proxy"]["frame_mapping"][i].get("source_gap_reason") == OUTSIDE:
                raise ValueError("requested frames absent from previous processing range")
            path = old / "analysis-frames" / f"sample-{i:08d}.jpg"
            if file_sha256(path) != digests.get(path.name):
                raise ValueError("selected image hash mismatch")
            files.append(path)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return _unavailable("analysis_frames", exc)
    for position, path in enumerate(files):
        shutil.copyfile(path, staging / f"frame-{position:08d}.jpg")
    LOG.debug(
        "stage=preprocessing event=scoped_reuse status=completed component=analysis_frames count=%d",
        len(files),
    )
    return True


def reuse_pose_slice(
    old: Path,
    staging: Path,
    frames: Sequence[Mapping[str, Any]],
    source_sha256: str,
    config: PosePreprocessingConfig,
    rights: Mapping[str, Any],
    *,
    ffmpeg: str,
) -> bool:
    """Reuse native observations, then rebuild derived motion at the new edges."""
    from .pose import POSE_CONTRACT_VERSION

    try:
        manifest = read_json(old / "pose" / "manifest.json")
        if (
            manifest["input"]["source_sha256"] != source_sha256
            or manifest["configuration"] != config.to_dict()
            or validate_rights(manifest["rights"]) != validate_rights(rights)
        ):
            raise ValueError("pose source, configuration or rights changed")
        observation_path, overlay = (
            data_path(old / "pose" / "observations.jsonl"),
            old / "pose" / "overlay.mp4",
        )
        for name, path in (
            ("observations", observation_path),
            *(() if compact_enabled() else (("overlay", overlay),)),
        ):
            if file_sha256(path) != manifest["artifacts"][name]["sha256"]:
                raise ValueError("saved pose artifact hash mismatch")
            if read_rights_sidecar(path) != validate_rights(rights):
                raise ValueError("saved pose rights mismatch")
        selected = [i for i, f in enumerate(frames) if f.get("processing_status") != OUTSIDE]
        first, stop = (selected[0], selected[-1] + 1)
        old_span = manifest["input"].get(
            "processed_frame_span", [0, manifest["input"]["source_frame_count"]]
        )
        if not old_span[0] <= first < stop <= old_span[1]:
            raise ValueError("requested pose interval missing")
        selected_rows = {}
        with open_data(observation_path) as handle:
            for i, line in enumerate(handle):
                if first <= i < stop:
                    row = json.loads(line)
                    frame = frames[i]
                    expected_frame = {
                        k: v
                        for k, v in frame.items()
                        if k not in {"source_frame_index", "analysis_sample_indices"}
                    }
                    if (
                        row.get("source_frame") != expected_frame
                        or row.get("pose_inference_attempted") is not True
                    ):
                        raise ValueError("saved pose timeline differs or inference is missing")
                    selected_rows[i] = row
                if i >= stop - 1:
                    break
        if len(selected_rows) != stop - first:
            raise ValueError("selected native pose rows missing")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return _unavailable("whole_body_pose", exc)
    width, height = (manifest["input"]["frame_width"], manifest["input"]["frame_height"])
    output_path = output_data_path(staging / "observations.jsonl")
    with open_data(output_path, "w") as output:
        for i, frame in enumerate(frames):
            row = selected_rows.get(i) or excluded_observation(frame, width, height)
            output.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
    try:
        if not compact_enabled():
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-i",
                    str(overlay),
                    "-an",
                    "-vf",
                    f"trim=start_frame={first - old_span[0]}:end_frame={stop - old_span[0]},setpts=PTS-STARTPTS",
                    "-frames:v",
                    str(stop - first),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    str(staging / "overlay.mp4"),
                ],
                check=True,
                capture_output=True,
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        (staging / "overlay.mp4").unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return _unavailable("whole_body_pose", exc)
    people = [p for r in selected_rows.values() for p in r["people"]]
    write_json(
        staging / ".worker-result.json",
        {
            "schema_version": POSE_CONTRACT_VERSION,
            "frame_count": len(frames),
            "processed_frame_count": len(selected_rows),
            "processed_frame_span": [first, stop],
            "frame_width": width,
            "frame_height": height,
            "frames_with_people": sum((bool(r["people"]) for r in selected_rows.values())),
            "person_observation_count": len(people),
            "track_count": len({p["track_id"] for p in people}),
            "overlay_frame_rate": manifest["input"]["overlay_frame_rate"],
            "overlay_timing_semantics": manifest["input"]["overlay_timing_semantics"],
            "runtime": manifest["runtime"],
            "models": manifest["models"],
            "reuse": {
                "policy_version": "selected-native-pose-reuse-v0.1",
                "source_manifest_sha256": file_sha256(old / "pose" / "manifest.json"),
                "source_observations_sha256": file_sha256(observation_path),
                "selected_frame_span": [first, stop],
                "derived_motion_recomputed": True,
            },
        },
    )
    LOG.debug(
        "stage=preprocessing event=scoped_reuse status=completed component=whole_body_pose count=%d",
        len(selected_rows),
    )
    return True
