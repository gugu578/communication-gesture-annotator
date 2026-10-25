# SPDX-License-Identifier: Apache-2.0
"""Derived, timestamp-safe pose conditioning for the 10 Hz analysis grid.

The raw pose stage is intentionally immutable.  This module reads its validated
source-rate observations, smooths image-normalized body and hand coordinates
inside uninterrupted track segments, aggregates compact sample-local features,
and creates one deterministic focus JPEG for each existing analysis JPEG.
"""

from __future__ import annotations
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import gc
import json
import math
from numbers import Real
from pathlib import Path
from time import perf_counter
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4
from ..domain import AnalysisSample
from ..export.json_export import read_json, write_json
from ..storage import (
    compact_enabled,
    data_path,
    open_data,
    output_data_path,
    overlay_omitted,
    relative_data_path,
)
from ..rights import (
    RIGHTS_CONTRACT_VERSION,
    read_rights_sidecar,
    rights_sidecar_path,
    validate_rights,
    write_rights_sidecar,
)
from .ffmpeg import (
    MediaRuntimeError,
    _escape_image2_sequence_pattern,
    _resolve_executable,
    file_sha256,
)
from .pose import POSE_CONTRACT_VERSION, PosePreprocessingResult
from .pose_smoothing import robust_local_polynomial_series
from .smoothing_cache import load_smoothing, save_smoothing
from .camera_cuts import POLICY as CAMERA_CUT_POLICY, scan_source, segment_observations
from .native_pose import (
    NATIVE_POSE_CONTRACT_VERSION,
    NATIVE_POSE_PACKET_POLICY_VERSION,
    SOURCE_FRAME_SELECTION_POLICY_VERSION,
    iter_native_records,
    native_pose_packet,
    select_source_observation,
    PosePacketCapacityError,
)

POSE_CONDITIONING_CONTRACT_VERSION = "0.2.0"
POSE_SMOOTHING_POLICY_VERSION = "robust-local-polynomial-camera-segments-v0.2"
POSE_AGGREGATION_POLICY_VERSION = "half-open-overlap-camera-segments-v0.2"
POSE_FOCUS_RENDERER_VERSION = "opencv-exact-source-instantaneous-v0.2.1"
POSE_CONDITIONED_OVERLAY_VERSION = "fixed-10hz-h264-diagnostic-v0.1"
SMOOTHING_HALF_WINDOW_US = 100000
SMOOTHING_POLYNOMIAL_DEGREE = 2
SMOOTHING_MINIMUM_SUPPORT = 5
SMOOTHING_MINIMUM_SIDE_SUPPORT = 2
CROP_PADDING_FRACTION = 0.15
JPEG_QUALITY = 95
_BODY_INDICES = tuple(range(17))
_LEFT_HAND_INDICES = (9, *range(91, 112))
_RIGHT_HAND_INDICES = (10, *range(112, 133))
_SMOOTHED_KEYPOINT_INDICES = tuple(
    sorted(set(_BODY_INDICES + _LEFT_HAND_INDICES + _RIGHT_HAND_INDICES))
)
_BODY_CONNECTIONS = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def _hand_connections(root: int) -> tuple[tuple[int, int], ...]:
    connections: list[tuple[int, int]] = []
    for finger_start in (root + 1, root + 5, root + 9, root + 13, root + 17):
        connections.append((root, finger_start))
        connections.extend(((index, index + 1) for index in range(finger_start, finger_start + 3)))
    return tuple(connections)


_HAND_CONNECTIONS = _hand_connections(91) + _hand_connections(112)
_DISCONTINUITY_KEYS = frozenset(
    {
        "continuity_break",
        "discontinuity",
        "discontinuity_reason",
        "is_discontinuity",
        "scene_discontinuity",
        "source_gap_reason",
        "timestamp_discontinuity",
    }
)


@dataclass(frozen=True, slots=True)
class PoseConditioningResult:
    """Validated paths and in-memory records for one conditioning bundle."""

    manifest: Mapping[str, Any]
    directory: Path
    manifest_path: Path
    features_path: Path
    features_rights_path: Path
    focus_frames_directory: Path
    focus_rights_manifest_path: Path
    conditioned_overlay_path: Path
    conditioned_overlay_rights_path: Path
    feature_records: tuple[Mapping[str, Any], ...]
    focus_frame_records: tuple[Mapping[str, Any], ...]
    reused: bool = False
    native_feature_records: tuple[Mapping[str, Any], ...] = ()
    exact_frame_records: tuple[Mapping[str, Any], ...] = ()
    _packet_cache: OrderedDict[tuple[int, int], tuple[dict[str, Any], int]] = field(
        default_factory=OrderedDict, compare=False, repr=False
    )

    @property
    def native_features_path(self) -> Path:
        return data_path(self.directory / "native-features.jsonl")

    def temporal_packet(
        self, start_us: int, end_us: int, *, maximum_chars: int | None = None
    ) -> dict[str, Any]:
        """Source-rate evidence and canonical summaries for one request window."""
        if type(start_us) is not int or type(end_us) is not int or end_us <= start_us:
            raise ValueError("pose packet interval must be a positive integer-microsecond interval")
        if maximum_chars is not None and (type(maximum_chars) is not int or maximum_chars <= 0):
            raise ValueError("pose packet maximum_chars must be positive")
        key = (start_us, end_us)
        cached = self._packet_cache.get(key)
        if cached is None:
            native = self.manifest.get("artifacts", {}).get("native_features", {})
            packet = native_pose_packet(
                self.native_feature_records,
                self.feature_records,
                start_us=start_us,
                end_us=end_us,
                artifact_sha256=str(native.get("sha256", "")),
            )
            size = len(json.dumps(packet, separators=(",", ":"), allow_nan=False))
            if size <= 8000000:
                self._packet_cache[key] = (packet, size)
                while (
                    len(self._packet_cache) > 8
                    or sum((v[1] for v in self._packet_cache.values())) > 8000000
                ):
                    self._packet_cache.popitem(last=False)
        else:
            packet, size = cached
            self._packet_cache.move_to_end(key)
        if maximum_chars is not None and size > maximum_chars:
            raise PosePacketCapacityError(
                "native pose packet exceeds budget; split the temporal request with overlap; minimum-window overflow is a capacity failure"
            )
        return deepcopy(packet)

    def model_input_identity(self) -> dict[str, Any]:
        """Return concise, hash-bound identity material for allowed model calls."""
        artifacts = self.manifest.get("artifacts")
        inputs = self.manifest.get("input")
        policy = self.manifest.get("policy")
        if not isinstance(artifacts, Mapping) or not isinstance(inputs, Mapping):
            raise ValueError("pose conditioning manifest lineage is incomplete")
        if not isinstance(policy, Mapping):
            raise ValueError("pose conditioning manifest policy is incomplete")
        features = artifacts.get("analysis_features")
        native = artifacts.get("native_features")
        focus_rights = artifacts.get("focus_frames_rights_manifest")
        raw_manifest = inputs.get("raw_pose_manifest")
        raw_observations = inputs.get("raw_pose_observations")
        for label, record, path in (
            ("analysis features", features, self.features_path),
            ("native features", native, self.native_features_path),
            ("focus-frame rights manifest", focus_rights, self.focus_rights_manifest_path),
        ):
            if (
                not isinstance(record, Mapping)
                or not path.is_file()
                or record.get("sha256") != file_sha256(path)
                or (record.get("byte_size") != path.stat().st_size)
            ):
                raise ValueError(f"pose conditioning {label} was modified")
        if not self.manifest_path.is_file():
            raise ValueError("pose conditioning manifest is missing")
        current_manifest = read_json(self.manifest_path)
        if not isinstance(current_manifest, Mapping) or dict(current_manifest) != dict(
            self.manifest
        ):
            raise ValueError("pose conditioning manifest was modified")
        if not isinstance(raw_manifest, Mapping) or not isinstance(raw_observations, Mapping):
            raise ValueError("pose conditioning raw-pose lineage is incomplete")
        return {
            "contract_version": POSE_CONDITIONING_CONTRACT_VERSION,
            "smoothing_policy_version": policy.get("smoothing_policy_version"),
            "aggregation_policy_version": policy.get("aggregation_policy_version"),
            "focus_renderer_version": policy.get("focus_renderer_version"),
            "conditioned_overlay_version": policy.get("conditioned_overlay_version"),
            "conditioning_manifest_sha256": file_sha256(self.manifest_path),
            "analysis_features_sha256": features["sha256"],
            "native_features_sha256": native["sha256"],
            "native_packet_policy_version": NATIVE_POSE_PACKET_POLICY_VERSION,
            "focus_frames_rights_manifest_sha256": focus_rights["sha256"],
            "raw_pose_manifest_sha256": raw_manifest.get("sha256"),
            "raw_pose_observations_sha256": raw_observations.get("sha256"),
        }


@dataclass(slots=True)
class _SmoothedPerson:
    source_frame_index: int
    source_frame_id: str
    canonical_time_us: int
    canonical_end_us: int
    track_id: int
    segment_id: str
    bbox: tuple[float, float, float, float]
    detector_confidence: float | None
    points: list[tuple[float, float] | None]
    confidences: tuple[float | None, ...]
    smoothing_statuses: list[str | None]
    smoothing_support: list[dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class _MotionEdge:
    segment_id: str
    start_us: int
    end_us: int
    delta_time_us: int
    displacements: Mapping[str, float | None]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _artifact_record(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise MediaRuntimeError(f"pose conditioning did not create {path.name}")
    return {
        "relative_path": relative_data_path(path, relative_path),
        "sha256": file_sha256(path),
        "byte_size": path.stat().st_size,
    }


def _is_real(value: Any) -> bool:
    return isinstance(value, Real) and (not isinstance(value, bool)) and math.isfinite(float(value))


def _probability_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not _is_real(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"pose {label} must be a finite probability or null")
    return float(value)


def _finite_scalar_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not _is_real(value):
        raise ValueError(f"pose {label} must be a finite scalar or null")
    return float(value)


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    rounded = round(float(value), 9)
    return 0.0 if rounded == 0.0 else rounded


def _overlap_us(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _covered_duration_us(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(((start, end) for start, end in intervals if end > start))
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = (start, end)
    return total + current_end - current_start


def _validate_samples(samples: Sequence[AnalysisSample]) -> None:
    if not samples:
        raise ValueError("pose conditioning requires at least one analysis sample")
    previous_end = 0
    for position, sample in enumerate(samples):
        if not isinstance(sample, AnalysisSample):
            raise ValueError("pose conditioning samples must be AnalysisSample values")
        if sample.index != position or sample.start_us != previous_end:
            raise ValueError("pose conditioning requires a zero-based contiguous analysis grid")
        duration_us = sample.end_us - sample.start_us
        if duration_us > 100000 or (position < len(samples) - 1 and duration_us != 100000):
            raise ValueError("pose conditioning requires the fixed 10 Hz grid")
        previous_end = sample.end_us


def _frame_identity_payload(
    frame_records: Sequence[Mapping[str, Any]], samples: Sequence[AnalysisSample]
) -> list[dict[str, Any]]:
    if len(frame_records) != len(samples):
        raise ValueError("pose conditioning requires exactly one JPEG per sample")
    payload: list[dict[str, Any]] = []
    for sample, frame in zip(samples, frame_records, strict=True):
        if not isinstance(frame, Mapping):
            raise ValueError("pose conditioning frame records must be objects")
        path_value = frame.get("path")
        digest = frame.get("sha256")
        if (
            frame.get("sample_index") != sample.index
            or frame.get("start_us") != sample.start_us
            or frame.get("end_us") != sample.end_us
            or (not isinstance(path_value, (str, Path)))
            or (not isinstance(digest, str))
            or (len(digest) != 64)
        ):
            raise ValueError("analysis JPEG record does not match its sample")
        source_gap_reason = frame.get("source_gap_reason")
        if source_gap_reason != sample.source_gap_reason:
            raise ValueError("analysis JPEG source-gap provenance does not match")
        path = Path(path_value).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        if file_sha256(path) != digest:
            raise ValueError("analysis JPEG does not match its recorded SHA-256")
        if "rights" in frame:
            validate_rights(frame["rights"])
        payload.append(
            {
                "sample_index": sample.index,
                "start_us": sample.start_us,
                "end_us": sample.end_us,
                "sha256": digest,
                "byte_size": path.stat().st_size,
                "source_gap_reason": sample.source_gap_reason,
            }
        )
    return payload


def _validate_frame_rights(
    frame_records: Sequence[Mapping[str, Any]], rights: Mapping[str, Any]
) -> None:
    expected = validate_rights(rights)
    directories: set[Path] = set()
    for frame in frame_records:
        path = Path(str(frame["path"])).resolve()
        directories.add(path.parent)
        if "rights" in frame and validate_rights(frame["rights"]) != expected:
            raise ValueError("analysis JPEG belongs to different media rights")
        sidecar = rights_sidecar_path(path)
        if sidecar.exists() and read_rights_sidecar(path) != expected:
            raise ValueError("analysis JPEG rights sidecar belongs to different rights")
    for directory in directories:
        manifest_path = directory / "rights-manifest.json"
        if not manifest_path.exists():
            continue
        document = read_json(manifest_path)
        if (
            not isinstance(document, Mapping)
            or document.get("schema_version") != RIGHTS_CONTRACT_VERSION
            or validate_rights(document.get("rights")) != expected
            or (not isinstance(document.get("artifacts"), list))
        ):
            raise ValueError("analysis JPEG rights manifest is invalid")
        records = {
            item.get("filename"): item.get("sha256")
            for item in document["artifacts"]
            if isinstance(item, Mapping)
        }
        for frame in frame_records:
            path = Path(str(frame["path"])).resolve()
            if path.parent == directory and records.get(path.name) != frame["sha256"]:
                raise ValueError("analysis JPEG rights manifest is incomplete or stale")


def _raw_paths(pose_result: PosePreprocessingResult) -> tuple[Path, ...]:
    document = pose_result.manifest
    paths = [
        pose_result.manifest_path,
        pose_result.observations_path,
        pose_result.observations_rights_path,
        *(
            ()
            if overlay_omitted(dict(document))
            else (pose_result.overlay_path, rights_sidecar_path(pose_result.overlay_path))
        ),
    ]
    artifacts = document.get("artifacts")
    if isinstance(artifacts, Mapping):
        for record in artifacts.values():
            if isinstance(record, Mapping) and isinstance(record.get("relative_path"), str):
                candidate = pose_result.directory.parent / record["relative_path"]
                paths.append(candidate)
    return tuple(dict.fromkeys((path.resolve() for path in paths)))


def _snapshot(paths: Sequence[Path]) -> dict[Path, tuple[int, str]]:
    snapshot: dict[Path, tuple[int, str]] = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"raw pose artifact is missing: {path.name}")
        snapshot[path] = (path.stat().st_size, file_sha256(path))
    return snapshot


def _validate_raw_pose(
    pose_result: PosePreprocessingResult,
    samples: Sequence[AnalysisSample],
    rights: Mapping[str, Any],
    run_root: Path,
) -> tuple[dict[str, Any], dict[Path, tuple[int, str]]]:
    if not isinstance(pose_result, PosePreprocessingResult):
        raise TypeError("pose_result must be a PosePreprocessingResult")
    pose_directory = pose_result.directory.resolve()
    if pose_directory != (run_root / "pose").resolve():
        raise ValueError("raw pose result does not belong to run_dir/pose")
    expected_paths = {
        "manifest": pose_directory / "manifest.json",
        "observations": data_path(pose_directory / "observations.jsonl"),
        "observations_rights": rights_sidecar_path(
            data_path(pose_directory / "observations.jsonl")
        ),
        "overlay": pose_directory / "overlay.mp4",
    }
    if (
        pose_result.manifest_path.resolve() != expected_paths["manifest"].resolve()
        or pose_result.observations_path.resolve() != expected_paths["observations"].resolve()
        or pose_result.observations_rights_path.resolve()
        != expected_paths["observations_rights"].resolve()
        or (pose_result.overlay_path.resolve() != expected_paths["overlay"].resolve())
    ):
        raise ValueError("raw pose result uses noncanonical artifact paths")
    document = read_json(pose_result.manifest_path)
    if not isinstance(document, dict) or document != dict(pose_result.manifest):
        raise ValueError("raw pose manifest differs from PosePreprocessingResult")
    if (
        document.get("schema_version") != POSE_CONTRACT_VERSION
        or document.get("status") != "completed"
        or document.get("consumed_by_model_stages") != []
    ):
        raise ValueError("raw pose manifest is incomplete or unsupported")
    expected_rights = validate_rights(rights)
    if validate_rights(document.get("rights")) != expected_rights:
        raise ValueError("raw pose result belongs to different media rights")
    if read_rights_sidecar(pose_result.observations_path) != expected_rights:
        raise ValueError("raw pose observation rights are missing or stale")
    manifest_input = document.get("input")
    if (
        not isinstance(manifest_input, Mapping)
        or manifest_input.get("analysis_sample_count") != len(samples)
        or manifest_input.get("analysis_sample_rate_hz") != 10
        or (
            manifest_input.get("analysis_grid_sha256")
            != _canonical_sha256([asdict(sample) for sample in samples])
        )
    ):
        raise ValueError("raw pose result belongs to a different analysis grid")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("raw pose manifest artifact provenance is missing")
    for name, path in (
        ("observations", pose_result.observations_path),
        ("observations_rights", pose_result.observations_rights_path),
        ("overlay", pose_result.overlay_path),
        ("overlay_rights", rights_sidecar_path(pose_result.overlay_path)),
    ):
        record = artifacts.get(name)
        if name in {"overlay", "overlay_rights"} and overlay_omitted(document) and (record is None):
            continue
        if (
            not isinstance(record, Mapping)
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or (record.get("byte_size") != path.stat().st_size)
        ):
            raise ValueError(f"raw pose {name} artifact is missing or tampered")
    raw_snapshot = _snapshot(_raw_paths(pose_result))
    return (document, raw_snapshot)


def _has_discontinuity(row: Mapping[str, Any]) -> bool:
    containers = [row]
    source_frame = row.get("source_frame")
    if isinstance(source_frame, Mapping):
        containers.append(source_frame)
    for container in containers:
        for key in _DISCONTINUITY_KEYS:
            value = container.get(key)
            if value is True or (isinstance(value, str) and value.strip()):
                return True
    return False


def _validate_person(
    value: Any, *, source_frame_id: str
) -> tuple[
    int,
    tuple[float, float, float, float],
    float | None,
    list[tuple[float, float] | None],
    tuple[float | None, ...],
]:
    if not isinstance(value, Mapping):
        raise ValueError("raw pose person observation must be an object")
    track_id = value.get("track_id")
    bbox_value = value.get("bbox_xyxy_normalized")
    if type(track_id) is not int or track_id < 0:
        raise ValueError("raw pose track_id must be a non-negative integer")
    if (
        not isinstance(bbox_value, list)
        or len(bbox_value) != 4
        or (not all((_is_real(item) for item in bbox_value)))
    ):
        raise ValueError("raw pose normalized bounding box is invalid")
    bbox = tuple((float(item) for item in bbox_value))
    if (
        bbox[0] < 0.0
        or bbox[1] < 0.0
        or bbox[2] > 1.0
        or (bbox[3] > 1.0)
        or (bbox[2] <= bbox[0])
        or (bbox[3] <= bbox[1])
    ):
        raise ValueError("raw pose normalized bounding box is outside the image")
    detector_confidence = _probability_or_none(
        value.get("detector_confidence"), "detector_confidence"
    )
    raw_points = value.get("keypoints_image_xy_normalized")
    raw_confidences = value.get("keypoint_confidences")
    if (
        not isinstance(raw_points, list)
        or len(raw_points) != 133
        or (not isinstance(raw_confidences, list))
        or (len(raw_confidences) != 133)
    ):
        raise ValueError(f"raw pose keypoint arrays are invalid for {source_frame_id}")
    points: list[tuple[float, float] | None] = []
    confidences: list[float | None] = []
    for point, confidence in zip(raw_points, raw_confidences, strict=True):
        checked_confidence = _finite_scalar_or_none(confidence, "keypoint confidence")
        confidences.append(checked_confidence)
        if point == [None, None]:
            points.append(None)
        elif (
            isinstance(point, list) and len(point) == 2 and all((_is_real(axis) for axis in point))
        ):
            points.append((float(point[0]), float(point[1])))
        else:
            raise ValueError("raw pose normalized keypoint is invalid")
    return (track_id, bbox, detector_confidence, points, tuple(confidences))


def _read_observations(
    pose_result: PosePreprocessingResult,
    samples: Sequence[AnalysisSample],
    raw_manifest: Mapping[str, Any],
    *,
    smooth: bool = True,
) -> tuple[list[dict[str, Any]], list[_SmoothedPerson]]:
    rows: list[dict[str, Any]] = []
    expected_count = raw_manifest["input"].get("source_frame_count")
    try:
        with open_data(pose_result.observations_path) as handle:
            for position, line in enumerate(handle):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("raw pose observation must be an object")
                start_us = value.get("canonical_time_us")
                end_us = value.get("canonical_end_us")
                source_frame_id = value.get("source_frame_id")
                indices = value.get("analysis_sample_indices")
                decoded = value.get("decoded_frame")
                people = value.get("people")
                if (
                    value.get("schema_version") != POSE_CONTRACT_VERSION
                    or value.get("source_frame_index") != position
                    or value.get("decode_index") != position
                    or (source_frame_id != f"source-frame:{position}")
                    or (type(start_us) is not int)
                    or (type(end_us) is not int)
                    or (end_us <= start_us)
                    or (not isinstance(indices, list))
                    or any((type(index) is not int for index in indices))
                    or (indices != sorted(set(indices)))
                    or (not isinstance(decoded, Mapping))
                    or (type(decoded.get("width")) is not int)
                    or (type(decoded.get("height")) is not int)
                    or (decoded["width"] <= 0)
                    or (decoded["height"] <= 0)
                    or (
                        value.get("pose_inference_attempted")
                        is not (
                            value.get("source_frame", {}).get("processing_status")
                            != "outside_processing_range"
                        )
                    )
                    or (
                        value.get("source_frame", {}).get("processing_status")
                        == "outside_processing_range"
                        and people != []
                    )
                    or (not isinstance(people, list))
                    or (value.get("pose_evidence_available") != bool(people))
                ):
                    raise ValueError("raw pose observation contract is invalid")
                if rows and start_us <= rows[-1]["canonical_time_us"]:
                    raise ValueError("raw pose timestamps must be strictly increasing")
                expected_indices = [
                    sample.index
                    for sample in samples
                    if _overlap_us(start_us, end_us, sample.start_us, sample.end_us) > 0
                ]
                if indices != expected_indices:
                    raise ValueError(
                        "raw pose analysis_sample_indices do not match half-open overlap"
                    )
                checked_people = []
                seen_tracks: set[int] = set()
                for person in people:
                    checked = _validate_person(person, source_frame_id=source_frame_id)
                    if checked[0] in seen_tracks:
                        raise ValueError("raw pose frame repeats a track_id")
                    seen_tracks.add(checked[0])
                    checked_people.append(checked)
                row = dict(value)
                row["_checked_people"] = checked_people
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("raw pose observations are not valid UTF-8 JSONL") from exc
    if type(expected_count) is not int or len(rows) != expected_count:
        raise ValueError("raw pose observations do not have exact source-frame coverage")
    for sample in samples:
        references = tuple(
            (
                row["source_frame_id"]
                for row in rows
                if _overlap_us(
                    row["canonical_time_us"],
                    row["canonical_end_us"],
                    sample.start_us,
                    sample.end_us,
                )
                > 0
            )
        )
        if references != tuple(sample.source_references):
            raise ValueError("raw pose source-frame coverage differs from the analysis sample")
    people_by_track: dict[int, list[_SmoothedPerson]] = defaultdict(list)
    previous_by_track: dict[int, _SmoothedPerson] = {}
    segment_numbers: Counter[int] = Counter()
    for row in rows:
        for track_id, bbox, detector_confidence, points, confidences in row["_checked_people"]:
            previous = previous_by_track.get(track_id)
            break_segment = (
                previous is None
                or previous.source_frame_index + 1 != row["source_frame_index"]
                or previous.canonical_end_us < row["canonical_time_us"]
                or (previous.canonical_time_us >= row["canonical_time_us"])
                or _has_discontinuity(row)
                or _has_discontinuity(rows[previous.source_frame_index])
            )
            if break_segment:
                segment_number = segment_numbers[track_id]
                segment_numbers[track_id] += 1
            else:
                segment_number = int(previous.segment_id.rsplit("-", 1)[1])
            item = _SmoothedPerson(
                source_frame_index=row["source_frame_index"],
                source_frame_id=row["source_frame_id"],
                canonical_time_us=row["canonical_time_us"],
                canonical_end_us=row["canonical_end_us"],
                track_id=track_id,
                segment_id=f"track-{track_id}-segment-{segment_number}",
                bbox=bbox,
                detector_confidence=detector_confidence,
                points=points,
                confidences=confidences,
                smoothing_statuses=[None] * 133,
                smoothing_support=[None] * 133,
            )
            people_by_track[track_id].append(item)
            previous_by_track[track_id] = item
    people_flat = [
        person for track_id in sorted(people_by_track) for person in people_by_track[track_id]
    ]
    for row in rows:
        del row["_checked_people"]
    if smooth:
        _smooth_people(people_flat)
    return (rows, people_flat)


def _smooth_people(
    people: Sequence[_SmoothedPerson], *, progress: Callable[[int, int], None] | None = None
) -> None:
    completed = 0
    total = len(people) * len(_SMOOTHED_KEYPOINT_INDICES) * 2
    by_segment: dict[str, list[_SmoothedPerson]] = defaultdict(list)
    for person in people:
        by_segment[person.segment_id].append(person)
    for segment in by_segment.values():
        segment.sort(key=lambda item: item.source_frame_index)
        for keypoint_index in _SMOOTHED_KEYPOINT_INDICES:
            run: list[_SmoothedPerson] = []

            def smooth_run() -> None:
                nonlocal completed
                if not run:
                    return
                timestamps = [item.canonical_time_us for item in run]
                confidences = [float(item.confidences[keypoint_index] or 0.0) for item in run]
                for axis in (0, 1):
                    values = [float(item.points[keypoint_index][axis]) for item in run]
                    estimates = robust_local_polynomial_series(
                        timestamps,
                        values,
                        confidences,
                        half_window_us=SMOOTHING_HALF_WINDOW_US,
                        degree=SMOOTHING_POLYNOMIAL_DEGREE,
                        minimum_support=SMOOTHING_MINIMUM_SUPPORT,
                        minimum_side_support=SMOOTHING_MINIMUM_SIDE_SUPPORT,
                    )
                    for item, (estimate, support_min, support_max) in zip(
                        run, estimates, strict=True
                    ):
                        current = item.points[keypoint_index]
                        assert current is not None
                        updated = list(current)
                        updated[axis] = estimate.value
                        item.points[keypoint_index] = (updated[0], updated[1])
                        existing = item.smoothing_statuses[keypoint_index]
                        item.smoothing_statuses[keypoint_index] = (
                            estimate.status
                            if existing in {None, estimate.status}
                            else "mixed_coordinate_status"
                        )
                        support = item.smoothing_support[keypoint_index]
                        if support is None:
                            support = {}
                            item.smoothing_support[keypoint_index] = support
                        support[str(axis)] = {
                            "status": estimate.status,
                            "support_count": estimate.support_count,
                            "left_support_count": estimate.left_support_count,
                            "right_support_count": estimate.right_support_count,
                            "outlier_count": estimate.outlier_count,
                            "half_window_us": SMOOTHING_HALF_WINDOW_US,
                            "support_observation_time_min_us": support_min,
                            "support_observation_time_max_us": support_max,
                        }
                    completed += len(run)
                    if progress:
                        progress(completed, total)

            for person in segment:
                point = person.points[keypoint_index]
                confidence = person.confidences[keypoint_index]
                if point is None or confidence is None or confidence <= 0.0:
                    smooth_run()
                    run = []
                    completed += 2
                else:
                    run.append(person)
            smooth_run()
    if progress:
        progress(total, total)


def _category_displacement(
    left: _SmoothedPerson, right: _SmoothedPerson, indices: Sequence[int]
) -> float | None:
    distances: list[float] = []
    for index in indices:
        left_point = left.points[index]
        right_point = right.points[index]
        left_confidence = left.confidences[index]
        right_confidence = right.confidences[index]
        if (
            left_point is None
            or right_point is None
            or left_confidence is None
            or (right_confidence is None)
            or (left_confidence <= 0.0)
            or (right_confidence <= 0.0)
        ):
            continue
        distances.append(math.dist(left_point, right_point))
    return math.fsum(distances) / len(distances) if distances else None


def _motion_edges(people: Sequence[_SmoothedPerson]) -> dict[str, list[_MotionEdge]]:
    by_segment: dict[str, list[_SmoothedPerson]] = defaultdict(list)
    for person in people:
        by_segment[person.segment_id].append(person)
    result: dict[str, list[_MotionEdge]] = {}
    categories = {
        "body": _BODY_INDICES,
        "left_hand": _LEFT_HAND_INDICES,
        "right_hand": _RIGHT_HAND_INDICES,
    }
    for segment_id, segment in by_segment.items():
        ordered = sorted(segment, key=lambda item: item.canonical_time_us)
        edges: list[_MotionEdge] = []
        for left, right in zip(ordered, ordered[1:]):
            delta_us = right.canonical_time_us - left.canonical_time_us
            if delta_us <= 0:
                raise ValueError("pose motion requires positive actual timestamp deltas")
            edges.append(
                _MotionEdge(
                    segment_id=segment_id,
                    start_us=left.canonical_time_us,
                    end_us=right.canonical_time_us,
                    delta_time_us=delta_us,
                    displacements={
                        name: _category_displacement(left, right, indices)
                        for name, indices in categories.items()
                    },
                )
            )
        result[segment_id] = edges
    return result


def _motion_summary(
    edges: Sequence[_MotionEdge], sample: AnalysisSample, category: str
) -> dict[str, Any]:
    contributions: list[tuple[_MotionEdge, int, float]] = []
    for edge in edges:
        displacement = edge.displacements.get(category)
        overlap = _overlap_us(edge.start_us, edge.end_us, sample.start_us, sample.end_us)
        if displacement is None or overlap <= 0:
            continue
        speed = displacement * 1000000.0 / edge.delta_time_us
        contributions.append((edge, overlap, speed))
    if not contributions:
        return {
            "interval_count": 0,
            "covered_duration_us": 0,
            "actual_delta_time_us_min": None,
            "actual_delta_time_us_max": None,
            "path_length_normalized": None,
            "mean_speed_normalized_per_s": None,
            "max_speed_normalized_per_s": None,
        }
    covered = sum((overlap for _edge, overlap, _speed in contributions))
    path_length = math.fsum(
        (speed * overlap / 1000000.0 for _edge, overlap, speed in contributions)
    )
    mean_speed = math.fsum((speed * overlap for _edge, overlap, speed in contributions)) / covered
    deltas = [edge.delta_time_us for edge, _overlap, _speed in contributions]
    return {
        "interval_count": len(contributions),
        "covered_duration_us": covered,
        "actual_delta_time_us_min": min(deltas),
        "actual_delta_time_us_max": max(deltas),
        "path_length_normalized": _round_float(path_length),
        "mean_speed_normalized_per_s": _round_float(mean_speed),
        "max_speed_normalized_per_s": _round_float(
            max((speed for _edge, _overlap, speed in contributions))
        ),
    }


def _aggregate_posture(
    observations: Sequence[tuple[_SmoothedPerson, int]],
    indices: Sequence[int],
    *,
    wrist_index: int | None = None,
) -> tuple[dict[str, Any], dict[int, tuple[tuple[float, float], float]]]:
    coordinate_weighted: dict[int, list[tuple[tuple[float, float], float]]] = defaultdict(list)
    valid_weight = 0.0
    total_weight = 0.0
    for person, overlap in observations:
        for index in indices:
            total_weight += overlap
            point = person.points[index]
            confidence = person.confidences[index]
            clipped_confidence = min(1.0, max(0.0, confidence)) if confidence is not None else 0.0
            if point is None or clipped_confidence <= 0.0:
                continue
            weight = overlap * clipped_confidence
            coordinate_weighted[index].append((point, weight))
            valid_weight += overlap
    aggregated: dict[int, tuple[tuple[float, float], float]] = {}
    for index, values in coordinate_weighted.items():
        weight_sum = math.fsum((weight for _point, weight in values))
        if weight_sum <= 0.0:
            continue
        point = (
            math.fsum((point[0] * weight for point, weight in values)) / weight_sum,
            math.fsum((point[1] * weight for point, weight in values)) / weight_sum,
        )
        confidence = min(
            1.0,
            math.fsum((weight for _point, weight in values))
            / math.fsum((overlap for _person, overlap in observations)),
        )
        aggregated[index] = (point, confidence)
    if aggregated:
        center = (
            math.fsum((item[0][0] for item in aggregated.values())) / len(aggregated),
            math.fsum((item[0][1] for item in aggregated.values())) / len(aggregated),
        )
        center_value: list[float] | None = [_round_float(center[0]), _round_float(center[1])]
    else:
        center_value = None
    posture: dict[str, Any] = {
        "observed_keypoint_fraction": _round_float(
            valid_weight / total_weight if total_weight > 0.0 else 0.0
        ),
        "center_xy_normalized": center_value,
    }
    if wrist_index is not None:
        wrist = aggregated.get(wrist_index)
        posture["wrist_xy_normalized"] = (
            [_round_float(wrist[0][0]), _round_float(wrist[0][1])] if wrist is not None else None
        )
        posture["wrist_confidence"] = _round_float(wrist[1]) if wrist else None
        palm_indices = (91, 96, 100, 104, 108) if wrist_index == 9 else (112, 117, 121, 125, 129)
        palm = [aggregated.get(index) for index in palm_indices]
        posture["palm_confidence"] = (
            _round_float(min((item[1] for item in palm if item is not None)))
            if all((item is not None for item in palm))
            else None
        )
        posture["palm_xy_normalized"] = (
            [
                _round_float(
                    math.fsum((item[0][axis] for item in palm if item is not None)) / len(palm)
                )
                for axis in (0, 1)
            ]
            if all((item is not None for item in palm))
            else None
        )
        posture["mean_keypoint_confidence"] = (
            _round_float(math.fsum((item[1] for item in aggregated.values())) / len(aggregated))
            if aggregated
            else None
        )
    return (posture, aggregated)


def _candidate_record(
    segment_id: str,
    observations: Sequence[tuple[_SmoothedPerson, int]],
    edges: Mapping[str, Sequence[_MotionEdge]],
    sample: AnalysisSample,
) -> tuple[dict[str, Any], dict[int, tuple[tuple[float, float], float]]]:
    ordered = sorted(observations, key=lambda item: item[0].source_frame_index)
    track_id = ordered[0][0].track_id
    intervals = [
        (
            max(person.canonical_time_us, sample.start_us),
            min(person.canonical_end_us, sample.end_us),
        )
        for person, _overlap in ordered
    ]
    coverage = _covered_duration_us(intervals) / (sample.end_us - sample.start_us)
    confidences = [
        (person.detector_confidence, overlap)
        for person, overlap in ordered
        if person.detector_confidence is not None
    ]
    detector_confidence = (
        math.fsum((value * weight for value, weight in confidences))
        / math.fsum((weight for _value, weight in confidences))
        if confidences
        else None
    )
    bbox = (
        min((person.bbox[0] for person, _overlap in ordered)),
        min((person.bbox[1] for person, _overlap in ordered)),
        max((person.bbox[2] for person, _overlap in ordered)),
        max((person.bbox[3] for person, _overlap in ordered)),
    )
    body, body_points = _aggregate_posture(ordered, _BODY_INDICES)
    left_hand, left_points = _aggregate_posture(ordered, _LEFT_HAND_INDICES, wrist_index=9)
    right_hand, right_points = _aggregate_posture(ordered, _RIGHT_HAND_INDICES, wrist_index=10)
    body["motion"] = _motion_summary(edges.get(segment_id, ()), sample, "body")
    left_hand["motion"] = _motion_summary(edges.get(segment_id, ()), sample, "left_hand")
    right_hand["motion"] = _motion_summary(edges.get(segment_id, ()), sample, "right_hand")
    smoothing_counts = Counter(
        (
            status
            for person, _overlap in ordered
            for index, status in enumerate(person.smoothing_statuses)
            if index in _SMOOTHED_KEYPOINT_INDICES and status is not None
        )
    )
    record = {
        "track_id": track_id,
        "continuity_segment_id": segment_id,
        "source_frame_indices": [person.source_frame_index for person, _overlap in ordered],
        "source_frame_count": len(ordered),
        "coverage_fraction": _round_float(min(1.0, coverage)),
        "detector_confidence": _round_float(detector_confidence),
        "bbox_xyxy_normalized": [_round_float(value) for value in bbox],
        "body": body,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "smoothing": dict(sorted(smoothing_counts.items())),
    }
    all_points = dict(body_points)
    all_points.update(left_points)
    all_points.update(right_points)
    return (record, all_points)


def _sample_features(
    rows: Sequence[Mapping[str, Any]],
    people: Sequence[_SmoothedPerson],
    edges: Mapping[str, Sequence[_MotionEdge]],
    samples: Sequence[AnalysisSample],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows_by_sample: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for sample_index in row["analysis_sample_indices"]:
            rows_by_sample[sample_index].append(row)
    people_by_frame: dict[int, list[_SmoothedPerson]] = defaultdict(list)
    for person in people:
        people_by_frame[person.source_frame_index].append(person)

    features: list[dict[str, Any]] = []
    render_inputs: dict[int, dict[str, Any]] = {}
    for sample in samples:
        overlapping_rows = ([] if sample.source_gap_reason == "outside_processing_range"
            else rows_by_sample.get(sample.index, []))
        by_segment: dict[str, list[tuple[_SmoothedPerson, int]]] = defaultdict(list)
        for row in overlapping_rows:
            overlap = _overlap_us(
                row["canonical_time_us"],
                row["canonical_end_us"],
                sample.start_us,
                sample.end_us,
            )
            for person in people_by_frame.get(row["source_frame_index"], []):
                by_segment[person.segment_id].append((person, overlap))
        candidates: list[dict[str, Any]] = []
        point_sets: dict[str, dict[int, tuple[tuple[float, float], float]]] = {}
        for segment_id in sorted(
            by_segment,
            key=lambda value: (
                by_segment[value][0][0].track_id,
                value,
            ),
        ):
            candidate, aggregated_points = _candidate_record(
                segment_id, by_segment[segment_id], edges, sample
            )
            candidates.append(candidate)
            point_sets[segment_id] = aggregated_points
        selected_row = select_source_observation(overlapping_rows, sample.center_us)
        selected_people = people_by_frame.get(selected_row["source_frame_index"], []) if selected_row else []
        simultaneous = any(len(people_by_frame.get(row["source_frame_index"], [])) > 1 for row in overlapping_rows)
        barrier_reasons = []
        if sample.source_gap_reason is not None:
            barrier_reasons.append(sample.source_gap_reason)
        if any(_has_discontinuity(row) for row in overlapping_rows):
            barrier_reasons.append("source_discontinuity")
        if _covered_duration_us([(max(sample.start_us, row["canonical_time_us"]), min(sample.end_us, row["canonical_end_us"])) for row in overlapping_rows]) < sample.end_us - sample.start_us:
            barrier_reasons.append("source_coverage_gap")
        camera_cut_times = [row["canonical_time_us"] for row in overlapping_rows if row.get("camera_cut_before")]
        if not camera_cut_times and not simultaneous and len({candidate["track_id"] for candidate in candidates}) > 1:
            barrier_reasons.append("unresolved_track_switch")
        if sample.source_gap_reason is not None:
            pose_evidence = "source_gap"
            focus_mode = "full_frame_fallback_source_gap"
        elif len(candidates) == 1:
            pose_evidence = "unique_eligible_person"
            # Interval evidence may survive a detection gap in the exact RGB
            # frame. Retain that evidence without borrowing another frame's box.
            focus_mode = "pose_guided_crop" if len(selected_people) == 1 else "full_frame_fallback_no_person"
        elif simultaneous:
            pose_evidence = "multiple_eligible_people"
            focus_mode = "full_frame_fallback_multiple_people"
        elif candidates:
            pose_evidence = "sequential_track_segments"
            focus_mode = "pose_guided_crop" if len(selected_people) == 1 else "full_frame_fallback_no_person"
        else:
            pose_evidence = "no_eligible_person"
            focus_mode = "full_frame_fallback_no_person"
        record = {
            "schema_version": POSE_CONDITIONING_CONTRACT_VERSION,
            "sample_index": sample.index,
            "start_us": sample.start_us,
            "center_us": sample.center_us,
            "end_us": sample.end_us,
            "source_frame_count": len(overlapping_rows),
            "source_frame_ids": [row["source_frame_id"] for row in overlapping_rows],
            "pose_evidence": pose_evidence,
            "focus_mode": focus_mode,
            "contributing_track_ids": sorted(
                {candidate["track_id"] for candidate in candidates}
            ),
            "candidates": candidates,
            "selected_source_frame": {
                "source_frame_id": selected_row["source_frame_id"],
                "source_frame_index": selected_row["source_frame_index"],
                "observation_time_us": selected_row["canonical_time_us"],
                "display_start_us": selected_row["canonical_time_us"],
                "display_end_us": selected_row["canonical_end_us"],
                "source_frame": dict(selected_row.get("source_frame", {})),
                "pose_decoded_frame": dict(selected_row.get("decoded_frame", {})),
                "selection_policy_version": SOURCE_FRAME_SELECTION_POLICY_VERSION,
            } if selected_row is not None else None,
            "target_association_continuous": len(candidates) == 1 and not simultaneous,
            "continuity_barrier": bool(barrier_reasons),
            "continuity_barrier_reasons": barrier_reasons,
            "camera_shot_id": selected_row.get("camera_shot_id") if selected_row else None,
            "camera_cut_times_us": camera_cut_times,
        }
        instantaneous_candidates = []
        for selected_person in selected_people:
            instantaneous, _ = _candidate_record(selected_person.segment_id, [(selected_person, 1)], {}, sample)
            for category in ("body", "left_hand", "right_hand"):
                instantaneous[category].pop("motion", None)
            instantaneous_candidates.append(instantaneous)
        record["instantaneous_candidates"] = instantaneous_candidates
        features.append(record)
        render_inputs[sample.index] = {
            "record": record,
            "points": {index: (point, float(selected_people[0].confidences[index] or 0))
                       for index, point in enumerate(selected_people[0].points)
                       if point is not None and selected_people[0].confidences[index] is not None}
                       if len(selected_people) == 1 and not simultaneous else {},
            "selected_bbox": selected_people[0].bbox if len(selected_people) == 1 else None,
        }
    return features, render_inputs


def _load_image(path: Path) -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise MediaRuntimeError(
            "pose focus rendering requires the reviewed OpenCV pose runtime"
        ) from exc
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise MediaRuntimeError(f"analysis JPEG is not a decodable RGB image: {path.name}")
    return (cv2, np, image)


def _expanded_bbox(
    bbox: Sequence[float] | None, width: int, height: int
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox)
    padding_x = (x2 - x1) * CROP_PADDING_FRACTION
    padding_y = (y2 - y1) * CROP_PADDING_FRACTION
    x1 = max(0.0, x1 - padding_x)
    y1 = max(0.0, y1 - padding_y)
    x2 = min(1.0, x2 + padding_x)
    y2 = min(1.0, y2 + padding_y)
    left = max(0, min(width - 1, math.floor(x1 * width)))
    top = max(0, min(height - 1, math.floor(y1 * height)))
    right = max(1, min(width, math.ceil(x2 * width)))
    bottom = max(1, min(height, math.ceil(y2 * height)))
    if right - left < 2 or bottom - top < 2:
        return None
    return (left, top, right, bottom)


def _draw_skeleton(
    image: Any,
    cv2: Any,
    points: Mapping[int, tuple[tuple[float, float], float]],
    *,
    threshold: float,
) -> None:
    height, width = image.shape[:2]
    coordinates: dict[int, tuple[int, int]] = {}
    for index, (point, confidence) in points.items():
        if confidence < threshold:
            continue
        x = int(round(min(1.0, max(0.0, point[0])) * (width - 1)))
        y = int(round(min(1.0, max(0.0, point[1])) * (height - 1)))
        coordinates[index] = (x, y)
    thickness = max(1, min(width, height) // 360)
    radius = max(1, thickness + 1)
    for left, right in _BODY_CONNECTIONS:
        if left in coordinates and right in coordinates:
            cv2.line(
                image, coordinates[left], coordinates[right], (0, 220, 255), thickness, cv2.LINE_8
            )
    for left, right in _HAND_CONNECTIONS:
        if left in coordinates and right in coordinates:
            color = (255, 170, 0) if left <= 111 else (210, 80, 255)
            cv2.line(image, coordinates[left], coordinates[right], color, thickness, cv2.LINE_8)
    for index in sorted(coordinates):
        color = (0, 220, 255) if index <= 16 else (255, 170, 0) if index <= 111 else (210, 80, 255)
        cv2.circle(image, coordinates[index], radius, color, -1, cv2.LINE_8)


def _encode_jpeg(path: Path, image: Any, cv2: Any) -> None:
    parameters = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        JPEG_QUALITY,
        int(cv2.IMWRITE_JPEG_PROGRESSIVE),
        0,
        int(cv2.IMWRITE_JPEG_OPTIMIZE),
        0,
    ]
    ok, encoded = cv2.imencode(".jpg", image, parameters)
    if not ok:
        raise MediaRuntimeError("OpenCV failed to encode a pose focus JPEG")
    path.write_bytes(encoded.tobytes())


def _render_focus_frames(
    frame_records: Sequence[Mapping[str, Any]],
    render_inputs: Mapping[int, Mapping[str, Any]],
    target_directory: Path,
    raw_manifest: Mapping[str, Any],
    *,
    exact_frames: Mapping[int, Path] | None = None,
) -> list[dict[str, Any]]:
    target_directory.mkdir(parents=True)
    threshold_value = raw_manifest.get("configuration", {}).get(
        "keypoint_confidence_threshold", 0.4
    )
    threshold = float(threshold_value) if _is_real(threshold_value) else 0.4
    records: list[dict[str, Any]] = []
    for frame_record in frame_records:
        sample_index = int(frame_record["sample_index"])
        source = Path(str(frame_record["path"])).resolve()
        target = target_directory / f"sample-{sample_index:08d}.jpg"
        render_input = render_inputs[sample_index]
        feature = render_input["record"]
        if frame_record.get("source_gap_reason") == "outside_processing_range":
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copyfile(source, target)
            feature["rgb_pose_pairing_verified"] = False
            records.append(
                {
                    "sample_index": sample_index,
                    "start_us": frame_record["start_us"],
                    "end_us": frame_record["end_us"],
                    "source_gap_reason": "outside_processing_range",
                    "relative_path": f"pose/focus-frames/{target.name}",
                    "sha256": frame_record["sha256"],
                    "byte_size": target.stat().st_size,
                    "source_analysis_jpeg_sha256": frame_record["sha256"],
                    "conditioning_mode": str(feature["focus_mode"]),
                    "selected_source_frame": None,
                    "pose_rgb_pairing": "unverified_rgb_without_skeleton",
                    "instantaneous_skeleton": False,
                    "transform": {
                        "input_width": 8,
                        "input_height": 8,
                        "source_frame_selection_verified": False,
                    },
                }
            )
            continue
        mode = str(feature["focus_mode"])
        selected = feature.get("selected_source_frame")
        verified = (
            selected is not None
            and exact_frames is not None
            and (selected["source_frame_index"] in exact_frames)
        )
        feature["rgb_pose_pairing_verified"] = verified
        if verified:
            source = exact_frames[selected["source_frame_index"]]
        elif mode == "pose_guided_crop":
            mode = "full_frame_fallback_unverified_source_pairing"
            feature["focus_mode"] = mode
        cv2, np, image = _load_image(source)
        transform: dict[str, Any] = {
            "input_width": int(image.shape[1]),
            "input_height": int(image.shape[0]),
            "source_frame_selection_verified": verified,
            "source_to_full_scene": "FFmpeg default autorotation; scale=trunc(iw*sar/2)*2:trunc(ih/2)*2:flags=lanczos; setsar=1; scale=640:-2:flags=lanczos; setsar=1"
            if verified
            else "analysis JPEG transform retained in preprocessing provenance",
        }
        if mode == "pose_guided_crop":
            crop_box = _expanded_bbox(render_input["selected_bbox"], image.shape[1], image.shape[0])
            if crop_box is None:
                mode = "full_frame_fallback_unusable_crop"
                feature["focus_mode"] = mode
            else:
                _draw_skeleton(image, cv2, render_input["points"], threshold=threshold)
                left, top, right, bottom = crop_box
                crop = image[top:bottom, left:right]
                output_height, output_width = image.shape[:2]
                scale = min(output_width / crop.shape[1], output_height / crop.shape[0])
                resized_width = max(1, min(output_width, int(round(crop.shape[1] * scale))))
                resized_height = max(1, min(output_height, int(round(crop.shape[0] * scale))))
                resized = cv2.resize(
                    crop, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4
                )
                canvas = np.full_like(image, 16)
                offset_x = (output_width - resized_width) // 2
                offset_y = (output_height - resized_height) // 2
                canvas[
                    offset_y : offset_y + resized_height, offset_x : offset_x + resized_width
                ] = resized
                transform.update(
                    {
                        "crop_xyxy_pixels": list(crop_box),
                        "resized_width": resized_width,
                        "resized_height": resized_height,
                        "letterbox_offset_xy": [offset_x, offset_y],
                        "interpolation": "INTER_LANCZOS4",
                    }
                )
                _encode_jpeg(target, canvas, cv2)
        if mode != "pose_guided_crop":
            shutil.copyfile(source, target)
        records.append(
            {
                "sample_index": sample_index,
                "start_us": int(frame_record["start_us"]),
                "end_us": int(frame_record["end_us"]),
                "source_gap_reason": frame_record.get("source_gap_reason"),
                "relative_path": f"pose/focus-frames/{target.name}",
                "sha256": file_sha256(target),
                "byte_size": target.stat().st_size,
                "source_analysis_jpeg_sha256": frame_record["sha256"],
                "conditioning_mode": mode,
                "selected_source_frame": selected if verified else None,
                "pose_rgb_pairing": "verified_same_source_frame"
                if verified
                else "unverified_rgb_without_skeleton",
                "instantaneous_skeleton": bool(verified and mode == "pose_guided_crop"),
                "transform": transform,
            }
        )
    return records


def _balanced_expression_sum(parts: Sequence[str]) -> str:
    """Join Boolean FFmpeg expressions without creating a linear parse tree."""
    if not parts:
        raise ValueError("FFmpeg expression sum requires at least one term")
    level = list(parts)
    while len(level) > 1:
        next_level: list[str] = []
        for position in range(0, len(level), 2):
            if position + 1 == len(level):
                next_level.append(level[position])
            else:
                next_level.append(f"({level[position]}+{level[position + 1]})")
        level = next_level
    return level[0]


def _exact_frame_select_expression(indices: Sequence[int]) -> str:
    """Encode an exact ordered frame-index set as a shallow FFmpeg expression.

    Long equal-stride runs are represented by a bounded ``between``/``mod``
    predicate.  Irregular terms are combined as a balanced tree.  Both forms
    select exactly the supplied decoded-frame ordinals; neither relies on a
    nominal frame rate or reconstructed timestamp.
    """
    if not indices:
        raise ValueError("exact frame selection requires at least one index")
    if any((type(index) is not int or index < 0 for index in indices)):
        raise ValueError("exact frame indices must be non-negative integers")
    if any((right <= left for left, right in zip(indices, indices[1:]))):
        raise ValueError("exact frame indices must be strictly increasing")
    terms: list[str] = []
    position = 0
    while position < len(indices):
        run_end = position + 1
        if position + 2 < len(indices):
            step = indices[position + 1] - indices[position]
            run_end = position + 2
            while run_end < len(indices) and indices[run_end] - indices[run_end - 1] == step:
                run_end += 1
            if run_end - position >= 3:
                start = indices[position]
                end = indices[run_end - 1]
                if step == 1:
                    terms.append(f"between(n,{start},{end})")
                else:
                    terms.append(f"between(n,{start},{end})*not(mod(n-{start},{step}))")
                position = run_end
                continue
        terms.append(f"eq(n,{indices[position]})")
        position += 1
    return _balanced_expression_sum(terms)


def _extract_exact_source_frames(
    source_path: Path,
    rows: Sequence[Mapping[str, Any]],
    samples: Sequence[AnalysisSample],
    directory: Path,
    *,
    ffmpeg: str,
) -> dict[int, Path]:
    """Decode the actual source and select verified decode indices without FPS conversion."""
    indices = sorted(
        {
            selected["source_frame_index"]
            for sample in samples
            if sample.source_gap_reason != "outside_processing_range"
            if (selected := select_source_observation(rows, sample.center_us)) is not None
        }
    )
    if not indices:
        return {}
    directory.mkdir()
    expression_path = directory / "selection.filter"
    select_expression = _exact_frame_select_expression(indices)
    expression_path.write_text(
        "select='"
        + select_expression
        + "',scale=trunc(iw*sar/2)*2:trunc(ih/2)*2:flags=lanczos,setsar=1,scale=640:-2:flags=lanczos,setsar=1",
        encoding="utf-8",
    )
    command = [
        _resolve_executable(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-copyts",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-/filter:v",
        str(expression_path),
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(len(indices)),
        "-start_number",
        "0",
        "-q:v",
        "2",
        "-threads",
        "1",
        str(directory / "selected-%08d.jpg"),
    ]
    try:
        subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = " | ".join(
            (line.strip() for line in (getattr(exc, "stderr", "") or "").splitlines()[-6:])
        )[:2000]
        if (
            isinstance(exc, subprocess.CalledProcessError)
            and "Unrecognized option" in detail
            and ("/filter:v" in detail)
        ):
            command[command.index("-/filter:v")] = "-filter_script:v"
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, subprocess.CalledProcessError) as fallback_exc:
                raise MediaRuntimeError(
                    "exact source-frame extraction failed with the filter-script option"
                ) from fallback_exc
        else:
            raise MediaRuntimeError(f"exact source-frame extraction failed: {detail}") from exc
    files = sorted(directory.glob("selected-*.jpg"))
    if len(files) != len(indices):
        raise MediaRuntimeError(
            "exact source-frame extraction did not preserve selected decode-index coverage"
        )
    return dict(zip(indices, files, strict=True))


def _render_conditioned_overlay(
    focus_directory: Path, target: Path, *, frame_count: int, ffmpeg: str, start_number: int = 0
) -> None:
    executable = _resolve_executable(ffmpeg)
    pattern = focus_directory / "sample-%08d.jpg"
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "image2",
        "-framerate",
        "10",
        "-start_number",
        str(start_number),
        "-i",
        _escape_image2_sequence_pattern(pattern, "%08d"),
        "-map",
        "0:v:0",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black,setsar=1",
        "-frames:v",
        str(frame_count),
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        detail = " | ".join((line.strip() for line in stderr.splitlines()[-6:] if line.strip()))[
            :2000
        ]
        suffix = f": {detail}" if detail else ""
        raise MediaRuntimeError(f"conditioned pose overlay rendering failed{suffix}") from exc
    if not target.is_file() or target.stat().st_size <= 0:
        raise MediaRuntimeError("conditioned pose overlay was not created")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write a stream without materializing it; return the persisted row count."""
    count = 0
    with open_data(path, "w") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with open_data(path) as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("pose feature record must be an object")
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pose features are not valid UTF-8 JSONL") from exc
    return records


def _focus_rights_document(
    rights: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": RIGHTS_CONTRACT_VERSION,
        "rights": validate_rights(rights),
        "artifacts": [
            {
                "filename": Path(str(record["relative_path"])).name,
                "sample_index": record["sample_index"],
                "sha256": record["sha256"],
                "byte_size": record["byte_size"],
                "conditioning_mode": record["conditioning_mode"],
            }
            for record in records
        ],
    }


def _result(
    pose_directory: Path,
    manifest: Mapping[str, Any],
    feature_records: Sequence[Mapping[str, Any]],
    focus_records: Sequence[Mapping[str, Any]],
    *,
    reused: bool,
) -> PoseConditioningResult:
    exact_by_index = {
        record["source_frame_index"]: record
        for record in manifest.get("artifacts", {}).get("exact_source_frames", [])
    }
    exact_records = []
    for record in focus_records:
        selected = record.get("selected_source_frame")
        exact = exact_by_index.get(selected["source_frame_index"]) if selected else None
        if exact is not None:
            exact_records.append(
                {
                    "sample_index": record["sample_index"],
                    "start_us": record["start_us"],
                    "end_us": record["end_us"],
                    "path": str((pose_directory.parent / exact["relative_path"]).resolve()),
                    "sha256": exact["sha256"],
                    "source_gap_reason": record.get("source_gap_reason"),
                    "selected_source_frame": selected,
                    "pose_rgb_pairing": "verified_same_source_frame",
                    "rights": manifest["rights"],
                }
            )
    return PoseConditioningResult(
        manifest=dict(manifest),
        directory=pose_directory,
        manifest_path=pose_directory / "conditioning-manifest.json",
        features_path=pose_directory / "analysis-features.jsonl",
        features_rights_path=rights_sidecar_path(pose_directory / "analysis-features.jsonl"),
        focus_frames_directory=pose_directory / "focus-frames",
        focus_rights_manifest_path=pose_directory / "focus-frames" / "rights-manifest.json",
        conditioned_overlay_path=pose_directory / "conditioned-overlay.mp4",
        conditioned_overlay_rights_path=rights_sidecar_path(
            pose_directory / "conditioned-overlay.mp4"
        ),
        feature_records=tuple((dict(record) for record in feature_records)),
        focus_frame_records=tuple(
            (
                {
                    "sample_index": record["sample_index"],
                    "start_us": record["start_us"],
                    "end_us": record["end_us"],
                    "path": str((pose_directory.parent / str(record["relative_path"])).resolve()),
                    "sha256": record["sha256"],
                    "conditioning_mode": record["conditioning_mode"],
                    "source_gap_reason": record.get("source_gap_reason"),
                    "selected_source_frame": record.get("selected_source_frame"),
                    "pose_rgb_pairing": record.get("pose_rgb_pairing"),
                    "instantaneous_skeleton": record.get("instantaneous_skeleton", False),
                    "transform": record.get("transform"),
                }
                for record in focus_records
            )
        ),
        reused=reused,
        native_feature_records=tuple(
            _read_jsonl(data_path(pose_directory / "native-features.jsonl"))
        ),
        exact_frame_records=tuple(exact_records),
    )


def _validate_committed_conditioning(
    pose_directory: Path,
    raw_manifest: Mapping[str, Any],
    samples: Sequence[AnalysisSample],
    frame_identity: Sequence[Mapping[str, Any]],
    rights: Mapping[str, Any],
    *,
    reused: bool,
    exact_source_rgb_extraction: bool = False,
) -> PoseConditioningResult:
    manifest_path = pose_directory / "conditioning-manifest.json"
    document = read_json(manifest_path)
    if not isinstance(document, dict):
        raise ValueError("pose conditioning manifest must be an object")
    policy = document.get("policy")
    expected_policy = {
        "native_pose_contract_version": NATIVE_POSE_CONTRACT_VERSION,
        "native_packet_policy_version": NATIVE_POSE_PACKET_POLICY_VERSION,
        "source_frame_selection_policy_version": SOURCE_FRAME_SELECTION_POLICY_VERSION,
        "smoothing_policy_version": POSE_SMOOTHING_POLICY_VERSION,
        "aggregation_policy_version": POSE_AGGREGATION_POLICY_VERSION,
        "focus_renderer_version": POSE_FOCUS_RENDERER_VERSION,
        "conditioned_overlay_version": POSE_CONDITIONED_OVERLAY_VERSION,
        "smoothing_half_window_us": SMOOTHING_HALF_WINDOW_US,
        "smoothing_polynomial_degree": SMOOTHING_POLYNOMIAL_DEGREE,
        "smoothing_minimum_support": SMOOTHING_MINIMUM_SUPPORT,
        "smoothing_minimum_side_support": SMOOTHING_MINIMUM_SIDE_SUPPORT,
        "crop_padding_fraction": CROP_PADDING_FRACTION,
        "jpeg_quality": JPEG_QUALITY,
        "conditioned_overlay_frame_rate_hz": 10,
        "conditioned_overlay_timing_semantics": (
            "frame N is analysis sample coverage.conditioned_overlay_start_sample + N; original analysis timestamps remain authoritative"
            if any(s.source_gap_reason == "outside_processing_range" for s in samples) else
            "frame N is analysis sample N at fixed 10 Hz; diagnostic only, not "
            "source-timestamp authority or a model input"
        ),
        "ambiguous_person_policy": "full_frame_no_largest_person_selection",
        "continuity_policy": "split_on_track_gap_or_discontinuity_or_camera_edit",
        "camera_cut_policy": CAMERA_CUT_POLICY,
    }
    input_record = document.get("input")
    # Packet-only additions do not change durable coordinates, source RGB or
    # smoothing. Reuse the validated native bundle; new requests carry the new
    # packet policy through model_input_identity, so old inference is not reused.
    if isinstance(policy, Mapping) and policy.get("native_packet_policy_version") == "native-complete-with-hand-summaries-v0.2":
        expected_policy["native_packet_policy_version"] = policy["native_packet_policy_version"]
    raw_manifest_path = pose_directory / "manifest.json"
    raw_observations_path = data_path(pose_directory / "observations.jsonl")
    raw_manifest_record = (
        input_record.get("raw_pose_manifest")
        if isinstance(input_record, Mapping)
        else None
    )
    raw_observations_record = (
        input_record.get("raw_pose_observations")
        if isinstance(input_record, Mapping)
        else None
    )
    if (
        document.get("schema_version") != POSE_CONDITIONING_CONTRACT_VERSION
        or document.get("status") != "completed"
        or document.get("stage") != "pose_conditioning"
        or policy != expected_policy
        or not isinstance(input_record, Mapping)
        or not isinstance(raw_manifest_record, Mapping)
        or not isinstance(raw_observations_record, Mapping)
        or input_record.get("raw_pose_contract_version") != POSE_CONTRACT_VERSION
        or raw_manifest_record.get("sha256")
        != file_sha256(raw_manifest_path)
        or raw_manifest_record.get("byte_size")
        != raw_manifest_path.stat().st_size
        or raw_observations_record.get("sha256")
        != file_sha256(raw_observations_path)
        or raw_observations_record.get("byte_size")
        != raw_observations_path.stat().st_size
        or input_record.get("analysis_grid_sha256")
        != _canonical_sha256([asdict(sample) for sample in samples])
        or input_record.get("analysis_frame_records_sha256")
        != _canonical_sha256(frame_identity)
        or input_record.get("analysis_sample_count") != len(samples)
        or input_record.get("exact_source_rgb_extraction") != exact_source_rgb_extraction
        or validate_rights(document.get("rights")) != validate_rights(rights)
    ):
        raise ValueError("pose conditioning checkpoint belongs to different inputs")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("pose conditioning artifact provenance is missing")
    features_path = pose_directory / "analysis-features.jsonl"
    features_rights_path = rights_sidecar_path(features_path)
    focus_directory = pose_directory / "focus-frames"
    focus_rights_path = focus_directory / "rights-manifest.json"
    conditioned_overlay_path = pose_directory / "conditioned-overlay.mp4"
    conditioned_overlay_rights_path = rights_sidecar_path(conditioned_overlay_path)
    for name, path in (
        ("exact_source_frames_rights_manifest", pose_directory / "exact-source-frames" / "rights-manifest.json"),
        ("native_features", data_path(pose_directory / "native-features.jsonl")),
        ("native_features_rights", rights_sidecar_path(data_path(pose_directory / "native-features.jsonl"))),
        ("analysis_features", features_path),
        ("analysis_features_rights", features_rights_path),
        ("focus_frames_rights_manifest", focus_rights_path),
        ("conditioned_overlay", conditioned_overlay_path),
        ("conditioned_overlay_rights", conditioned_overlay_rights_path),
    ):
        record = artifacts.get(name)
        if name in {"conditioned_overlay", "conditioned_overlay_rights"} and overlay_omitted(document) and record is None:
            continue
        if (
            not isinstance(record, Mapping)
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or record.get("byte_size") != path.stat().st_size
        ):
            raise ValueError(f"pose conditioning {name} is missing or tampered")
    if read_rights_sidecar(features_path) != validate_rights(rights):
        raise ValueError("pose conditioning feature rights are missing or stale")
    if read_rights_sidecar(data_path(pose_directory / "native-features.jsonl")) != validate_rights(rights):
        raise ValueError("native pose feature rights are missing or stale")
    for record in artifacts.get("exact_source_frames", []):
        path = pose_directory.parent / record["relative_path"]
        if not path.is_file() or file_sha256(path) != record.get("sha256") or path.stat().st_size != record.get("byte_size"):
            raise ValueError("exact source RGB frame is missing or tampered")
    if not overlay_omitted(document) and read_rights_sidecar(conditioned_overlay_path) != validate_rights(rights):
        raise ValueError("conditioned pose overlay rights are missing or stale")
    feature_records = _read_jsonl(features_path)
    if (
        len(feature_records) != len(samples)
        or [record.get("sample_index") for record in feature_records]
        != list(range(len(samples)))
    ):
        raise ValueError("pose features do not have exact analysis-sample coverage")
    focus_records = artifacts.get("focus_frames")
    if (
        not isinstance(focus_records, list)
        or len(focus_records) != len(samples)
        or any(not isinstance(record, Mapping) for record in focus_records)
    ):
        raise ValueError("pose focus frames do not have exact analysis-sample coverage")
    rights_document = read_json(focus_rights_path)
    if rights_document != _focus_rights_document(rights, focus_records):
        raise ValueError("pose focus-frame rights manifest is stale")
    expected_names = {"rights-manifest.json"}
    for sample, feature, record in zip(
        samples, feature_records, focus_records, strict=True
    ):
        expected_relative = f"pose/focus-frames/sample-{sample.index:08d}.jpg"
        path = pose_directory.parent / expected_relative
        if (
            not isinstance(record, Mapping)
            or record.get("sample_index") != sample.index
            or record.get("start_us") != sample.start_us
            or record.get("end_us") != sample.end_us
            or record.get("source_gap_reason") != sample.source_gap_reason
            or record.get("relative_path") != expected_relative
            or record.get("conditioning_mode") != feature.get("focus_mode")
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or record.get("byte_size") != path.stat().st_size
        ):
            raise ValueError("pose focus frame is missing, stale, or misaligned")
        expected_names.add(path.name)
    actual_names = {path.name for path in focus_directory.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("pose focus-frame directory has unexpected artifacts")
    coverage = document.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("analysis_sample_count") != len(samples)
        or coverage.get("feature_record_count") != len(samples)
        or coverage.get("focus_frame_count") != len(samples)
        or coverage.get("conditioned_overlay_frame_count") != (0 if overlay_omitted(document) else
            sum(s.source_gap_reason != "outside_processing_range" for s in samples))
        or (any(s.source_gap_reason == "outside_processing_range" for s in samples)
            and coverage.get("conditioned_overlay_start_sample") != next(s.index for s in samples if s.source_gap_reason != "outside_processing_range"))
    ):
        raise ValueError("pose conditioning coverage metadata is invalid")
    return _result(
        pose_directory,
        document,
        feature_records,
        focus_records,
        reused=reused,
    )


def _commit_outputs(staging: Path, pose_directory: Path) -> None:
    manifest = read_json(staging / "conditioning-manifest.json")
    required = [
        "exact-source-frames",
        "analysis-features.jsonl",
        "analysis-features.jsonl.rights.json",
        "focus-frames",
        data_path(staging / "native-features.jsonl").name,
        rights_sidecar_path(data_path(staging / "native-features.jsonl")).name,
    ]
    if not overlay_omitted(manifest):
        required.extend(("conditioned-overlay.mp4", "conditioned-overlay.mp4.rights.json"))
    if any((not (staging / name).exists() for name in required)):
        raise ValueError("incomplete staged pose conditioning bundle")
    names = (
        "exact-source-frames",
        "native-features.jsonl",
        "native-features.jsonl.rights.json",
        "native-features.jsonl.gz",
        "native-features.jsonl.gz.rights.json",
        "analysis-features.jsonl",
        "analysis-features.jsonl.rights.json",
        "focus-frames",
        "conditioned-overlay.mp4",
        "conditioned-overlay.mp4.rights.json",
        "conditioning-manifest.json",
    )
    token = uuid4().hex
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for name in names:
            target = pose_directory / name
            if not target.exists():
                continue
            backup = pose_directory / f".{name.replace('/', '-')}.{token}.backup"
            target.replace(backup)
            backups.append((target, backup))
        for name in names:
            source = staging / name
            target = pose_directory / name
            if not source.exists():
                continue
            source.replace(target)
            committed.append(target)
    except Exception:
        for path in reversed(committed):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        for target, backup in reversed(backups):
            if backup.exists():
                backup.replace(target)
        raise
    else:
        for _target, backup in backups:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink(missing_ok=True)


def run_pose_conditioning(
    pose_result: PosePreprocessingResult,
    samples: Sequence[AnalysisSample],
    frame_records: Sequence[Mapping[str, Any]],
    run_dir: str | Path,
    *,
    rights: Mapping[str, Any],
    overwrite: bool = False,
    resume: bool = False,
    ffmpeg: str = "ffmpeg",
    source_path: str | Path | None = None,
    progress: Callable[[str, str, Mapping[str, Any]], None] | None = None,
) -> PoseConditioningResult:
    """Build or validate the immutable derived pose-conditioning bundle.

    Source-rate motion always uses differences between exact canonical PTS
    values.  Local smoothing and velocity edges are restricted to one
    uninterrupted short-gap tracker segment.  The focus renderer selects a crop
    only when the exact selected RGB frame has one eligible person and the
    sample has no simultaneous people; it never borrows a neighboring frame's
    box or substitutes a largest-person heuristic for missing identity evidence.
    """

    if type(overwrite) is not bool or type(resume) is not bool:
        raise ValueError("overwrite and resume must be Boolean")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    _validate_samples(samples)
    effective_rights = validate_rights(rights)
    root = Path(run_dir).resolve()
    def step(name: str, work: Callable[[], Any]) -> Any:
        started = perf_counter()
        if progress:
            progress(name, "started", {})
        try:
            result = work()
        except Exception:
            if progress:
                progress(name, "failed", {"wall_seconds": perf_counter()-started})
            raise
        if progress:
            progress(name, "completed", {"wall_seconds": perf_counter()-started,
                     **({"reused": result} if type(result) is bool else {})})
        return result

    raw_manifest, raw_snapshot = step("pose_input_validation", lambda: _validate_raw_pose(
        pose_result, samples, effective_rights, root))
    source = Path(source_path).resolve() if source_path is not None else None
    if source is not None and (not source.is_file() or file_sha256(source) != raw_manifest["input"].get("source_sha256")):
        raise ValueError("exact pose/RGB extraction source does not match the raw pose source hash")
    frame_identity = _frame_identity_payload(frame_records, samples)
    _validate_frame_rights(frame_records, effective_rights)
    pose_directory = pose_result.directory.resolve()
    manifest_path = pose_directory / "conditioning-manifest.json"
    output_paths = (
        pose_directory / "exact-source-frames",
        pose_directory / "native-features.jsonl",
        pose_directory / "native-features.jsonl.rights.json",
        pose_directory / "native-features.jsonl.gz",
        pose_directory / "native-features.jsonl.gz.rights.json",
        manifest_path,
        pose_directory / "analysis-features.jsonl",
        pose_directory / "analysis-features.jsonl.rights.json",
        pose_directory / "focus-frames",
        pose_directory / "conditioned-overlay.mp4",
        pose_directory / "conditioned-overlay.mp4.rights.json",
    )
    existing = [path for path in output_paths if path.exists()]
    if resume and manifest_path.is_file():
        result = _validate_committed_conditioning(
            pose_directory,
            raw_manifest,
            samples,
            frame_identity,
            effective_rights,
            reused=True,
            exact_source_rgb_extraction=source is not None,
        )
        if _snapshot(tuple(raw_snapshot)) != raw_snapshot:
            raise MediaRuntimeError("raw pose artifacts changed during conditioning resume")
        return result
    if resume and existing:
        raise ValueError("pose conditioning resume found an incomplete checkpoint")
    if existing and not overwrite:
        raise FileExistsError(existing[0])

    staging = root / f".pose-conditioning.{uuid4().hex}.tmp"
    staging.mkdir(parents=True)
    try:
        rows, people = step("pose_observation_validation", lambda: _read_observations(
            pose_result, samples, raw_manifest, smooth=False))
        camera_cuts = step("camera_geometry_boundaries", lambda: scan_source(source, rows, ffmpeg=ffmpeg)) if source is not None else []
        if source is not None:
            segment_observations(rows, people, camera_cuts)
        smoothing_identity = {
            "raw_pose_manifest_sha256": file_sha256(pose_result.manifest_path),
            "raw_observations_sha256": raw_manifest["artifacts"]["observations"]["sha256"],
            "smoothing_policy_version": POSE_SMOOTHING_POLICY_VERSION,
            "half_window_us": SMOOTHING_HALF_WINDOW_US, "degree": SMOOTHING_POLYNOMIAL_DEGREE,
            "minimum_support": SMOOTHING_MINIMUM_SUPPORT, "minimum_side_support": SMOOTHING_MINIMUM_SIDE_SUPPORT,
            "keypoint_indices": list(_SMOOTHED_KEYPOINT_INDICES),
            "camera_cut_policy": CAMERA_CUT_POLICY,
            "camera_cuts": camera_cuts,
        }
        cache_directory = pose_directory / "smoothing-cache" / _canonical_sha256(smoothing_identity)

        def smooth() -> bool:
            if load_smoothing(cache_directory, people, smoothing_identity, effective_rights):
                return True
            started = last_report = perf_counter()
            def report(done: int, total: int) -> None:
                nonlocal last_report
                now = perf_counter()
                if progress and (done == total or now-last_report >= 5):
                    progress("pose_smoothing", "progress", {"completed_count": done,
                        "total_count": total, "wall_seconds": now-started})
                    last_report = now
            _smooth_people(people, progress=report)
            step("pose_smoothing_checkpoint", lambda: save_smoothing(cache_directory, people, smoothing_identity, effective_rights))
            return False

        step("pose_smoothing", smooth)
        edges = step("pose_motion_edges", lambda: _motion_edges(people))
        source_frame_count = len(rows)
        native_path = output_data_path(staging / "native-features.jsonl")
        # Native geometry and smoothing support can be much larger than RGB.
        # Serialize each observation as it is built, keeping all source samples.
        native_record_count = step("pose_native_write", lambda: _write_jsonl(
            native_path, iter_native_records(rows, people)))
        write_rights_sidecar(native_path, effective_rights, overwrite=False)
        feature_records, render_inputs = step("pose_sample_aggregation", lambda: _sample_features(
            rows, people, edges, samples))
        people = edges = None
        features_path = staging / "analysis-features.jsonl"
        focus_directory = staging / "focus-frames"
        exact_frames = step("pose_exact_rgb", lambda: _extract_exact_source_frames(source, rows, samples, staging / "exact-source-frames", ffmpeg=ffmpeg)) if source is not None else None
        # Rendering needs only the selected rectangles/points, not complete raw
        # and smoothed tracks. Release them before allocating image buffers.
        rows = None
        gc.collect()
        exact_directory = staging / "exact-source-frames"
        exact_directory.mkdir(exist_ok=True)
        exact_records = [{"source_frame_index": index, **_artifact_record(path, f"pose/exact-source-frames/{path.name}")}
                         for index, path in (exact_frames or {}).items()]
        exact_rights_path = write_json(exact_directory / "rights-manifest.json", {
            "schema_version": RIGHTS_CONTRACT_VERSION, "rights": effective_rights,
            "artifacts": [{"filename": Path(record["relative_path"]).name, "sha256": record["sha256"], "byte_size": record["byte_size"], "source_frame_index": record["source_frame_index"]} for record in exact_records],
        })
        (exact_directory / "selection.filter").unlink(missing_ok=True)
        focus_records = step("pose_focus_render", lambda: _render_focus_frames(
            frame_records, render_inputs, focus_directory, raw_manifest, exact_frames=exact_frames,
        ))
        render_inputs = None
        conditioned_overlay_path = staging / "conditioned-overlay.mp4"
        included_samples = [s for s in samples if s.source_gap_reason != "outside_processing_range"]
        if not compact_enabled():
            step("pose_overlay_render", lambda: _render_conditioned_overlay(
                focus_directory,
                conditioned_overlay_path,
                frame_count=len(included_samples),
                start_number=included_samples[0].index,
                ffmpeg=ffmpeg,
            ))
        conditioned_overlay_rights_path = rights_sidecar_path(conditioned_overlay_path) if compact_enabled() else write_rights_sidecar(
            conditioned_overlay_path, effective_rights, overwrite=False
        )
        _write_jsonl(features_path, feature_records)
        write_rights_sidecar(features_path, effective_rights, overwrite=False)
        focus_rights_path = write_json(
            focus_directory / "rights-manifest.json",
            _focus_rights_document(effective_rights, focus_records),
        )
        policy = {
            "native_pose_contract_version": NATIVE_POSE_CONTRACT_VERSION,
            "native_packet_policy_version": NATIVE_POSE_PACKET_POLICY_VERSION,
            "source_frame_selection_policy_version": SOURCE_FRAME_SELECTION_POLICY_VERSION,
            "smoothing_policy_version": POSE_SMOOTHING_POLICY_VERSION,
            "aggregation_policy_version": POSE_AGGREGATION_POLICY_VERSION,
            "focus_renderer_version": POSE_FOCUS_RENDERER_VERSION,
            "conditioned_overlay_version": POSE_CONDITIONED_OVERLAY_VERSION,
            "smoothing_half_window_us": SMOOTHING_HALF_WINDOW_US,
            "smoothing_polynomial_degree": SMOOTHING_POLYNOMIAL_DEGREE,
            "smoothing_minimum_support": SMOOTHING_MINIMUM_SUPPORT,
            "smoothing_minimum_side_support": SMOOTHING_MINIMUM_SIDE_SUPPORT,
            "crop_padding_fraction": CROP_PADDING_FRACTION,
            "jpeg_quality": JPEG_QUALITY,
            "conditioned_overlay_frame_rate_hz": 10,
            "conditioned_overlay_timing_semantics": (
                "frame N is analysis sample coverage.conditioned_overlay_start_sample + N; original analysis timestamps remain authoritative"
                if len(included_samples) != len(samples) else
                "frame N is analysis sample N at fixed 10 Hz; diagnostic only, not "
                "source-timestamp authority or a model input"
            ),
            "ambiguous_person_policy": "full_frame_no_largest_person_selection",
            "continuity_policy": "split_on_track_gap_or_discontinuity_or_camera_edit",
            "camera_cut_policy": CAMERA_CUT_POLICY,
        }
        manifest = {
            "schema_version": POSE_CONDITIONING_CONTRACT_VERSION,
            **({"diagnostic_overlay": "omitted-compact-storage-v0.1"} if compact_enabled() else {}),
            "status": "completed",
            "stage": "pose_conditioning",
            "implementation": "derived-source-rate-pose-conditioning-v0.4",
            "memory_policy_version": "stream-native-release-intermediates-v0.1",
            "camera_cuts": {"status": "completed" if source is not None else "unavailable_without_source",
                            "policy": CAMERA_CUT_POLICY, "boundaries": camera_cuts},
            "smoothing_checkpoint": {
                "input_sha256": _canonical_sha256(smoothing_identity),
                "manifest_sha256": file_sha256(cache_directory / "manifest.json"),
                "policy_version": "source-rate-smoothing-cache-v0.1",
            },
            "input": {
                "raw_pose_contract_version": POSE_CONTRACT_VERSION,
                "raw_pose_manifest": _artifact_record(
                    pose_result.manifest_path, "pose/manifest.json"
                ),
                "raw_pose_observations": _artifact_record(
                    pose_result.observations_path, "pose/observations.jsonl"
                ),
                "raw_pose_observations_rights": _artifact_record(
                    pose_result.observations_rights_path,
                    "pose/observations.jsonl.rights.json",
                ),
                "analysis_grid_sha256": _canonical_sha256(
                    [asdict(sample) for sample in samples]
                ),
                "analysis_frame_records_sha256": _canonical_sha256(frame_identity),
                "analysis_sample_count": len(samples),
                "source_frame_count": source_frame_count,
                "exact_source_rgb_extraction": source is not None,
            },
            "policy": policy,
            "coverage": {
                "source_frame_count": source_frame_count,
                "native_feature_record_count": native_record_count,
                "analysis_sample_count": len(samples),
                "feature_record_count": len(feature_records),
                "focus_frame_count": len(focus_records),
                "conditioned_overlay_frame_count": 0 if compact_enabled() else len(included_samples),
                "conditioned_overlay_start_sample": included_samples[0].index,
                "unique_person_crop_count": sum(
                    record["conditioning_mode"] == "pose_guided_crop"
                    for record in focus_records
                ),
                "full_frame_fallback_count": sum(
                    record["conditioning_mode"] != "pose_guided_crop"
                    for record in focus_records
                ),
            },
            "artifacts": {
                "exact_source_frames": exact_records,
                "exact_source_frames_rights_manifest": _artifact_record(exact_rights_path, "pose/exact-source-frames/rights-manifest.json"),
                "native_features": _artifact_record(native_path, "pose/native-features.jsonl"),
                "native_features_rights": _artifact_record(rights_sidecar_path(native_path), "pose/native-features.jsonl.rights.json"),
                "analysis_features": _artifact_record(
                    features_path, "pose/analysis-features.jsonl"
                ),
                "analysis_features_rights": _artifact_record(
                    rights_sidecar_path(features_path),
                    "pose/analysis-features.jsonl.rights.json",
                ),
                "focus_frames_rights_manifest": _artifact_record(
                    focus_rights_path, "pose/focus-frames/rights-manifest.json"
                ),
                "focus_frames": focus_records,
                "conditioned_overlay": None if compact_enabled() else _artifact_record(
                    conditioned_overlay_path, "pose/conditioned-overlay.mp4"
                ),
                "conditioned_overlay_rights": None if compact_enabled() else _artifact_record(
                    conditioned_overlay_rights_path,
                    "pose/conditioned-overlay.mp4.rights.json",
                ),
            },
            "rights": effective_rights,
            "raw_pose_artifacts_modified": False,
        }
        write_json(staging / "conditioning-manifest.json", manifest)
        if _snapshot(tuple(raw_snapshot)) != raw_snapshot:
            raise MediaRuntimeError(
                "raw pose artifacts changed while conditioning; staged output was discarded"
            )
        _commit_outputs(staging, pose_directory)
        # Validation reloads the committed bundle. Do not overlap it with the
        # construction-time feature table and manifest.
        del feature_records, focus_records, manifest
        gc.collect()
        return _validate_committed_conditioning(
            pose_directory,
            raw_manifest,
            samples,
            frame_identity,
            effective_rights,
            reused=False,
            exact_source_rgb_extraction=source is not None,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
