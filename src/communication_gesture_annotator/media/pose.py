# SPDX-License-Identifier: Apache-2.0
"""Isolated diagnostic whole-body pose preprocessing orchestration."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4
from ..config import PosePreprocessingConfig
from ..domain import AnalysisSample
from ..export.json_export import read_json, write_json
from ..storage import compact_enabled, data_path, open_data, overlay_omitted, relative_data_path
from ..rights import read_rights_sidecar, rights_sidecar_path, validate_rights, write_rights_sidecar
from .ffmpeg import MediaRuntimeError, file_sha256

_LOGGER = logging.getLogger(__name__)
POSE_CONTRACT_VERSION = "0.2.0"
_COCO_WHOLEBODY_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    *(f"face-{index}" for index in range(68)),
    "left_hand_root",
    *(f"left_thumb{index}" for index in range(1, 5)),
    *(f"left_forefinger{index}" for index in range(1, 5)),
    *(f"left_middle_finger{index}" for index in range(1, 5)),
    *(f"left_ring_finger{index}" for index in range(1, 5)),
    *(f"left_pinky_finger{index}" for index in range(1, 5)),
    "right_hand_root",
    *(f"right_thumb{index}" for index in range(1, 5)),
    *(f"right_forefinger{index}" for index in range(1, 5)),
    *(f"right_middle_finger{index}" for index in range(1, 5)),
    *(f"right_ring_finger{index}" for index in range(1, 5)),
    *(f"right_pinky_finger{index}" for index in range(1, 5)),
)


@dataclass(frozen=True, slots=True)
class PosePreprocessingResult:
    """Paths and validated manifest for one complete pose stage."""

    manifest: Mapping[str, Any]
    directory: Path
    manifest_path: Path
    observations_path: Path
    observations_rights_path: Path
    overlay_path: Path
    reused: bool = False


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _artifact_record(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MediaRuntimeError(f"pose preprocessing did not create {path.name}")
    return {
        "relative_path": relative_data_path(path, relative_path),
        "sha256": file_sha256(path),
        "byte_size": path.stat().st_size,
    }


def _validate_sample_grid(samples: Sequence[AnalysisSample]) -> None:
    if not samples:
        raise ValueError("pose preprocessing requires at least one analysis sample")
    previous_end = 0
    for position, sample in enumerate(samples):
        if sample.index != position:
            raise ValueError("pose preprocessing requires zero-based ordered samples")
        if sample.start_us != previous_end:
            raise ValueError("pose preprocessing requires a contiguous sample grid")
        previous_end = sample.end_us


def _source_frame_payload(
    source_frames: Sequence[Mapping[str, Any]], samples: Sequence[AnalysisSample]
) -> list[dict[str, Any]]:
    """Validate source-frame order and attach exact 10 Hz cross-references."""
    if not source_frames:
        raise ValueError("pose preprocessing requires at least one source video frame")
    _validate_sample_grid(samples)
    known_ids: dict[str, int] = {}
    previous_time: int | None = None
    for position, frame in enumerate(source_frames):
        if not isinstance(frame, Mapping):
            raise ValueError("pose source-frame timeline entries must be objects")
        frame_id = frame.get("frame_id")
        if frame_id != f"source-frame:{position}":
            raise ValueError("pose preprocessing requires ordered source-frame IDs")
        if frame.get("decode_index") != position:
            raise ValueError(
                "pose preprocessing requires a timestamp for every decoded source frame"
            )
        start_us = frame.get("canonical_time_us")
        end_us = frame.get("canonical_end_us")
        if (
            type(start_us) is not int
            or type(end_us) is not int
            or end_us <= start_us
            or (previous_time is not None and start_us < previous_time)
        ):
            raise ValueError("pose source-frame timeline is invalid or non-monotonic")
        time_base = frame.get("time_base")
        if (
            not isinstance(time_base, Mapping)
            or type(time_base.get("numerator")) is not int
            or type(time_base.get("denominator")) is not int
            or (time_base["numerator"] <= 0)
            or (time_base["denominator"] <= 0)
            or (type(frame.get("presentation_timestamp")) is not int)
        ):
            raise ValueError("pose source-frame timestamp provenance is incomplete")
        known_ids[str(frame_id)] = position
        previous_time = start_us
    analysis_indices: list[list[int]] = [[] for _ in source_frames]
    active_start = 0
    for sample in samples:
        while (
            active_start < len(source_frames)
            and int(source_frames[active_start]["canonical_end_us"]) <= sample.start_us
        ):
            active_start += 1
        expected_references: list[str] = []
        position = active_start
        while (
            position < len(source_frames)
            and int(source_frames[position]["canonical_time_us"]) < sample.end_us
        ):
            if sample.start_us < int(source_frames[position]["canonical_end_us"]):
                expected_references.append(str(source_frames[position]["frame_id"]))
            position += 1
        if tuple(expected_references) != tuple(sample.source_references):
            raise ValueError(
                "analysis grid source references do not match half-open frame coverage"
            )
        for frame_id in sample.source_references:
            position = known_ids.get(frame_id)
            if position is None:
                raise ValueError("analysis grid refers to an unknown decoded source frame")
            analysis_indices[position].append(sample.index)
    payload: list[dict[str, Any]] = []
    for position, frame in enumerate(source_frames):
        item = dict(frame)
        item["source_frame_index"] = position
        item["analysis_sample_indices"] = analysis_indices[position]
        payload.append(item)
    return payload


def _resolve_ffmpeg(value: str) -> str:
    candidate = Path(value)
    if candidate.parent != Path(".") or candidate.is_absolute():
        if candidate.is_file():
            return str(candidate.resolve())
        raise MediaRuntimeError(f"media executable does not exist: {value}")
    resolved = shutil.which(value)
    if resolved is None:
        raise MediaRuntimeError(
            f"{value} was not found; install a reviewed FFmpeg runtime or pass its path"
        )
    return resolved


def _read_worker_result(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise MediaRuntimeError("pose worker returned an invalid result")
    required = {
        "schema_version",
        "frame_count",
        "frame_width",
        "frame_height",
        "frames_with_people",
        "person_observation_count",
        "track_count",
        "overlay_frame_rate",
        "overlay_timing_semantics",
        "runtime",
        "models",
    }
    if (
        set(value) - {"processed_frame_count", "processed_frame_span", "reuse"} != required
        or value.get("schema_version") != POSE_CONTRACT_VERSION
    ):
        raise MediaRuntimeError("pose worker returned an unsupported result contract")
    for name in (
        "frame_count",
        "frame_width",
        "frame_height",
        "frames_with_people",
        "person_observation_count",
        "track_count",
    ):
        if type(value.get(name)) is not int or value[name] < 0:
            raise MediaRuntimeError("pose worker returned invalid count metadata")
    if not isinstance(value.get("runtime"), dict) or not isinstance(value.get("models"), dict):
        raise MediaRuntimeError("pose worker omitted runtime or model provenance")
    if "processed_frame_span" in value:
        span = value["processed_frame_span"]
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any((type(x) is not int for x in span))
            or (not 0 <= span[0] < span[1] <= value["frame_count"])
            or (value.get("processed_frame_count") != span[1] - span[0])
        ):
            raise MediaRuntimeError("pose worker returned an invalid processing range")
    rate = value.get("overlay_frame_rate")
    if (
        not isinstance(rate, dict)
        or type(rate.get("numerator")) is not int
        or type(rate.get("denominator")) is not int
        or (rate["numerator"] <= 0)
        or (rate["denominator"] <= 0)
        or (not isinstance(value.get("overlay_timing_semantics"), str))
    ):
        raise MediaRuntimeError("pose worker returned invalid overlay timing metadata")
    return value


def _validate_observations(path: Path, source_frames: Sequence[Mapping[str, Any]]) -> None:
    count = 0
    try:
        with open_data(path) as handle:
            for expected, line in enumerate(handle):
                if expected >= len(source_frames):
                    raise ValueError("pose observations contain extra source frames")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("pose observation must be an object")
                source_frame = source_frames[expected]
                mirrored_source_frame = {
                    key: item
                    for key, item in source_frame.items()
                    if key not in {"source_frame_index", "analysis_sample_indices"}
                }
                if (
                    value.get("schema_version") != POSE_CONTRACT_VERSION
                    or value.get("source_frame") != mirrored_source_frame
                    or value.get("source_frame_index") != expected
                    or (value.get("source_frame_id") != source_frame.get("frame_id"))
                    or (value.get("decode_index") != source_frame.get("decode_index"))
                    or (
                        value.get("presentation_timestamp")
                        != source_frame.get("presentation_timestamp")
                    )
                    or (value.get("time_base") != source_frame.get("time_base"))
                    or (value.get("canonical_time_us") != source_frame.get("canonical_time_us"))
                    or (value.get("canonical_end_us") != source_frame.get("canonical_end_us"))
                    or (
                        value.get("analysis_sample_indices")
                        != source_frame.get("analysis_sample_indices")
                    )
                    or (
                        value.get("pose_inference_attempted")
                        is not (source_frame.get("processing_status") != "outside_processing_range")
                    )
                    or (
                        source_frame.get("processing_status") == "outside_processing_range"
                        and value.get("people") != []
                    )
                    or (type(value.get("pose_evidence_available")) is not bool)
                    or (value.get("pose_evidence_available") != bool(value.get("people")))
                    or (not isinstance(value.get("decoded_frame"), dict))
                    or (type(value["decoded_frame"].get("width")) is not int)
                    or (type(value["decoded_frame"].get("height")) is not int)
                    or (value["decoded_frame"].get("width", 0) <= 0)
                    or (value["decoded_frame"].get("height", 0) <= 0)
                    or (
                        value["decoded_frame"].get("sample_aspect_ratio")
                        != {"numerator": 1, "denominator": 1}
                    )
                    or (not isinstance(value.get("people"), list))
                ):
                    raise ValueError("pose observation does not match its decoded source frame")
                count += 1
            if count != len(source_frames):
                raise ValueError("pose observations do not cover every decoded source frame")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pose observations are not valid UTF-8 JSONL") from exc


def _manifest_result(
    directory: Path, manifest: Mapping[str, Any], *, reused: bool
) -> PosePreprocessingResult:
    return PosePreprocessingResult(
        manifest=dict(manifest),
        directory=directory,
        manifest_path=directory / "manifest.json",
        observations_path=data_path(directory / "observations.jsonl"),
        observations_rights_path=rights_sidecar_path(data_path(directory / "observations.jsonl")),
        overlay_path=directory / "overlay.mp4",
        reused=reused,
    )


def _validate_committed_pose(
    directory: Path,
    *,
    source_path: Path,
    raw_source_frames: Sequence[Mapping[str, Any]],
    source_frames: Sequence[Mapping[str, Any]],
    samples: Sequence[AnalysisSample],
    expected_source_sha256: str,
    expected_source_byte_size: int,
    preprocessing_sha256: str,
    rights: Mapping[str, Any],
    config: PosePreprocessingConfig,
    reused: bool,
) -> PosePreprocessingResult:
    manifest_path = directory / "manifest.json"
    document = read_json(manifest_path)
    if not isinstance(document, dict):
        raise ValueError("pose manifest must be an object")
    if (
        document.get("schema_version") != POSE_CONTRACT_VERSION
        or document.get("status") != "completed"
        or document.get("consumed_by_model_stages") != []
    ):
        raise ValueError("pose manifest is incomplete or unsupported")
    input_record = document.get("input")
    if not isinstance(input_record, dict):
        raise ValueError("pose manifest input provenance is missing")
    if (
        input_record.get("source_sha256") != file_sha256(source_path)
        or input_record.get("source_sha256") != expected_source_sha256
        or input_record.get("source_byte_size") != expected_source_byte_size
        or (source_path.stat().st_size != expected_source_byte_size)
        or (input_record.get("preprocessing_sha256") != preprocessing_sha256)
        or (input_record.get("source_frame_count") != len(source_frames))
        or (input_record.get("analysis_sample_count") != len(samples))
        or (
            input_record.get("source_frame_timeline_sha256")
            != _canonical_sha256(list(raw_source_frames))
        )
        or (input_record.get("source_frame_crosswalk_sha256") != _canonical_sha256(source_frames))
        or (
            input_record.get("analysis_grid_sha256")
            != _canonical_sha256([asdict(sample) for sample in samples])
        )
        or (document.get("configuration_sha256") != _canonical_sha256(config.to_dict()))
    ):
        raise ValueError("pose checkpoint belongs to different inputs or configuration")
    observations = data_path(directory / "observations.jsonl")
    observations_rights = rights_sidecar_path(observations)
    overlay = directory / "overlay.mp4"
    overlay_rights = rights_sidecar_path(overlay)
    _validate_observations(observations, source_frames)
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("pose manifest artifact provenance is missing")
    for name, path in (
        ("observations", observations),
        ("observations_rights", observations_rights),
        ("overlay", overlay),
        ("overlay_rights", overlay_rights),
    ):
        record = artifacts.get(name)
        if name in {"overlay", "overlay_rights"} and overlay_omitted(document) and (record is None):
            continue
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or (record.get("byte_size") != path.stat().st_size)
        ):
            raise ValueError(f"pose {name} artifact does not match its manifest")
    expected_rights = validate_rights(document.get("rights"))
    if expected_rights != validate_rights(rights):
        raise ValueError("pose checkpoint belongs to different media rights")
    if read_rights_sidecar(observations) != expected_rights:
        raise ValueError("pose observations rights sidecar is missing or stale")
    if not overlay_omitted(document) and read_rights_sidecar(overlay) != expected_rights:
        raise ValueError("pose overlay rights sidecar is missing or stale")
    return _manifest_result(directory, document, reused=reused)


def _commit_directory(staging: Path, target: Path) -> None:
    """Replace one narrowly scoped pose directory with rollback."""
    backup = target.with_name(f".{target.name}.{uuid4().hex}.backup")
    moved_existing = False
    try:
        if target.exists():
            target.replace(backup)
            moved_existing = True
        staging.replace(target)
    except OSError:
        if target.exists():
            shutil.rmtree(target)
        if moved_existing and backup.exists():
            backup.replace(target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _run_worker(request_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "communication_gesture_annotator.media.pose_worker",
        "--request",
        str(request_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    final_detail = ""
    assert process.stderr is not None
    for raw_line in process.stderr:
        line = raw_line.strip()
        if not line:
            continue
        final_detail = line
        if line.startswith("PROGRESS "):
            _LOGGER.debug("stage=pose event=inference status=running %s", line[9:])
        else:
            _LOGGER.warning("stage=pose event=worker_message status=reported")
    return_code = process.wait()
    if return_code != 0:
        suffix = f": {final_detail}" if final_detail else ""
        raise MediaRuntimeError(f"whole-body pose worker failed{suffix}")


def run_pose_preprocessing(
    source_path: str | Path,
    source_frames: Sequence[Mapping[str, Any]],
    samples: Sequence[AnalysisSample],
    run_dir: str | Path,
    *,
    config: PosePreprocessingConfig,
    rights: Mapping[str, Any],
    expected_source_sha256: str,
    expected_source_byte_size: int,
    preprocessing_sha256: str,
    ffmpeg: str = "ffmpeg",
    overwrite: bool = False,
    resume: bool = False,
    reuse_from: Path | None = None,
) -> PosePreprocessingResult:
    """Produce isolated RTMW3D observations and a diagnostic overlay video.

    The worker is a terminating child process so ONNX/CUDA resources cannot be
    retained by the later LLM pipeline.  Its outputs are deliberately not
    appended to the preprocessing mapping passed to model stages.
    """
    source = Path(source_path).resolve()
    root = Path(run_dir).resolve()
    target = root / "pose"
    if not source.is_file():
        raise FileNotFoundError(source)
    if (
        not isinstance(expected_source_sha256, str)
        or len(expected_source_sha256) != 64
        or any((character not in "0123456789abcdef" for character in expected_source_sha256))
    ):
        raise ValueError("expected_source_sha256 must be a lowercase SHA-256 digest")
    if type(expected_source_byte_size) is not int or expected_source_byte_size <= 0:
        raise ValueError("expected_source_byte_size must be a positive integer")
    if (
        not isinstance(preprocessing_sha256, str)
        or len(preprocessing_sha256) != 64
        or any((character not in "0123456789abcdef" for character in preprocessing_sha256))
    ):
        raise ValueError("preprocessing_sha256 must be a lowercase SHA-256 digest")
    preprocessing_path = root / "preprocessing.json"
    if not preprocessing_path.is_file() or file_sha256(preprocessing_path) != preprocessing_sha256:
        raise ValueError("preprocessing_sha256 does not match preprocessing.json")
    preprocessing_document = read_json(preprocessing_path)
    if not isinstance(preprocessing_document, Mapping):
        raise ValueError("preprocessing.json must be an object")
    preprocessing_source = preprocessing_document.get("source")
    preprocessing_proxy = preprocessing_document.get("proxy")
    if (
        not isinstance(preprocessing_source, Mapping)
        or not isinstance(preprocessing_proxy, Mapping)
        or preprocessing_source.get("sha256") != expected_source_sha256
        or (preprocessing_source.get("byte_size") != expected_source_byte_size)
        or (
            _canonical_sha256(preprocessing_source.get("video_frames"))
            != _canonical_sha256(list(source_frames))
        )
        or (
            _canonical_sha256(preprocessing_proxy.get("frame_mapping"))
            != _canonical_sha256([asdict(sample) for sample in samples])
        )
        or (validate_rights(preprocessing_source.get("rights")) != validate_rights(rights))
    ):
        raise ValueError("pose inputs do not match preprocessing.json")
    initial_stat = source.stat()
    initial_sha256 = file_sha256(source)
    if (
        initial_sha256 != expected_source_sha256
        or initial_stat.st_size != expected_source_byte_size
    ):
        raise ValueError("source video does not match preprocessing provenance")
    prepared_source_frames = _source_frame_payload(source_frames, samples)
    effective_rights = validate_rights(rights)
    if target.exists():
        if resume:
            return _validate_committed_pose(
                target,
                source_path=source,
                raw_source_frames=source_frames,
                source_frames=prepared_source_frames,
                samples=samples,
                expected_source_sha256=expected_source_sha256,
                expected_source_byte_size=expected_source_byte_size,
                preprocessing_sha256=preprocessing_sha256,
                rights=effective_rights,
                config=config,
                reused=True,
            )
        if not overwrite:
            raise FileExistsError(target)
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".pose.{uuid4().hex}.tmp"
    staging.mkdir()
    request_path = staging / ".worker-request.json"
    try:
        write_json(
            request_path,
            {
                "schema_version": POSE_CONTRACT_VERSION,
                "source_path": str(source),
                "output_directory": str(staging),
                "ffmpeg": _resolve_ffmpeg(ffmpeg),
                "source_frames": prepared_source_frames,
                "configuration": config.to_dict(),
                **({"storage_mode": "compact"} if compact_enabled() else {}),
            },
        )
        reused_slice = False
        if reuse_from is not None:
            from .scoped_reuse import reuse_pose_slice

            reused_slice = reuse_pose_slice(
                reuse_from,
                staging,
                prepared_source_frames,
                expected_source_sha256,
                config,
                effective_rights,
                ffmpeg=_resolve_ffmpeg(ffmpeg),
            )
        if not reused_slice:
            _run_worker(request_path)
        final_stat = source.stat()
        if (
            final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
            or file_sha256(source) != initial_sha256
            or (not preprocessing_path.is_file())
            or (file_sha256(preprocessing_path) != preprocessing_sha256)
        ):
            raise MediaRuntimeError(
                "source video or preprocessing manifest changed during pose preprocessing; staged output was discarded"
            )
        request_path.unlink(missing_ok=True)
        worker_result_path = staging / ".worker-result.json"
        worker = _read_worker_result(worker_result_path)
        worker_result_path.unlink(missing_ok=True)
        if worker["frame_count"] != len(prepared_source_frames):
            raise MediaRuntimeError(
                "pose worker did not produce exactly one observation per decoded source frame"
            )
        observations = data_path(staging / "observations.jsonl")
        overlay = staging / "overlay.mp4"
        _validate_observations(observations, prepared_source_frames)
        write_rights_sidecar(observations, effective_rights, overwrite=False)
        if not compact_enabled():
            write_rights_sidecar(overlay, effective_rights, overwrite=False)
        observations_rights = rights_sidecar_path(observations)
        overlay_rights = rights_sidecar_path(overlay)
        manifest = {
            **({"diagnostic_overlay": "omitted-compact-storage-v0.1"} if compact_enabled() else {}),
            "schema_version": POSE_CONTRACT_VERSION,
            "status": "completed",
            "stage": "preprocessing",
            "implementation": "rtmlib-rtmw3d-wholebody-subprocess-v0.2",
            "input": {
                "source_path_recorded": False,
                "source_container_suffix": source.suffix.lower(),
                "source_sha256": expected_source_sha256,
                "source_byte_size": expected_source_byte_size,
                "preprocessing_sha256": preprocessing_sha256,
                "source_frame_count": len(prepared_source_frames),
                "analysis_sample_count": len(samples),
                "analysis_sample_rate_hz": 10,
                "source_frame_timeline_sha256": _canonical_sha256(list(source_frames)),
                "source_frame_crosswalk_sha256": _canonical_sha256(prepared_source_frames),
                "analysis_grid_sha256": _canonical_sha256([asdict(sample) for sample in samples]),
                "frame_width": worker["frame_width"],
                "frame_height": worker["frame_height"],
                "frame_position_mapping": "decoded source video frame position N maps to source-frame:N and retains its probed raw PTS, time base, and canonical interval",
                "analysis_grid_mapping": "each source-frame observation lists every overlapping canonical 10 Hz analysis sample; pose inference itself is not downsampled",
                "overlay_frame_rate": worker["overlay_frame_rate"],
                "overlay_frame_position_mapping": "overlay frame position N corresponds to source-frame:N; use the observation row, not overlay playback time, for exact source PTS",
                "overlay_timing_semantics": worker["overlay_timing_semantics"],
                **(
                    {
                        "processed_frame_span": worker["processed_frame_span"],
                        "overlay_frame_position_mapping": "overlay position N corresponds to processed_frame_span[0] + N",
                    }
                    if "processed_frame_span" in worker
                    else {}
                ),
            },
            "configuration": config.to_dict(),
            "configuration_sha256": _canonical_sha256(config.to_dict()),
            "runtime": worker["runtime"],
            "models": worker["models"],
            "keypoint_layout": {
                "name": "COCO-WholeBody-133",
                "count": 133,
                "keypoint_names": list(_COCO_WHOLEBODY_KEYPOINT_NAMES),
                "body_indices": [0, 16],
                "foot_indices": [17, 22],
                "face_indices": [23, 90],
                "left_hand_indices": [91, 111],
                "right_hand_indices": [112, 132],
            },
            "coordinate_systems": {
                "image_xy_px": "FFmpeg display-oriented, square-pixel source-frame canvas with even dimensions and origin at top left",
                "image_xy_normalized": "image_xy_px divided by frame width and height",
                "model_xyz": "RTMW3D crop-local x/y and monocular model-relative z; not metric world coordinates",
                "model_xyz_normalized": "x/pose_input_width, y/pose_input_height, z/configured_model_depth_range",
            },
            "tracking_semantics": "track_id is a deterministic short-gap greedy bounding-box IoU association for diagnostic continuity; it is not biometric identity and is not reliable across camera cuts",
            "coverage": {
                "processed_frame_count": worker.get("processed_frame_count", worker["frame_count"]),
                "source_frame_count": worker["frame_count"],
                "analysis_sample_count": len(samples),
                "frames_with_people": worker["frames_with_people"],
                "frames_without_people": worker.get("processed_frame_count", worker["frame_count"])
                - worker["frames_with_people"],
                "frames_not_processed": worker["frame_count"]
                - worker.get("processed_frame_count", worker["frame_count"]),
                "person_observation_count": worker["person_observation_count"],
                "track_count": worker["track_count"],
            },
            "artifacts": {
                "observations": _artifact_record(observations, "pose/observations.jsonl"),
                "observations_rights": _artifact_record(
                    observations_rights, "pose/observations.jsonl.rights.json"
                ),
                "overlay": None
                if compact_enabled()
                else _artifact_record(overlay, "pose/overlay.mp4"),
                "overlay_rights": None
                if compact_enabled()
                else _artifact_record(overlay_rights, "pose/overlay.mp4.rights.json"),
            },
            "rights": effective_rights,
            "consumed_by_model_stages": [],
            **({"scoped_reuse": worker["reuse"]} if worker.get("reuse") else {}),
        }
        write_json(staging / "manifest.json", manifest)
        _commit_directory(staging, target)
        return _validate_committed_pose(
            target,
            source_path=source,
            raw_source_frames=source_frames,
            source_frames=prepared_source_frames,
            samples=samples,
            expected_source_sha256=expected_source_sha256,
            expected_source_byte_size=expected_source_byte_size,
            preprocessing_sha256=preprocessing_sha256,
            rights=effective_rights,
            config=config,
            reused=reused_slice,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
