# SPDX-License-Identifier: Apache-2.0
"""Child-process runtime for diagnostic RTMW3D preprocessing."""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
import sys
from threading import Thread
from typing import Any, Iterable, Mapping, Sequence
from ..config import PosePreprocessingConfig
from ..export.json_export import read_json, write_json
from ..storage import open_data
from .ffmpeg import file_sha256

_CONTRACT_VERSION = "0.2.0"
_SOURCE_VIDEO_FILTER = "scale=trunc(iw*sar/2)*2:trunc(ih/2)*2:flags=lanczos,setsar=1,showinfo"


@dataclass(slots=True)
class _Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    last_canonical_time_us: int


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


class _IoUTracker:
    """Small deterministic tracker used only to stabilize diagnostic IDs."""

    def __init__(self, *, threshold: float, maximum_gap_us: int) -> None:
        self._threshold = threshold
        self._maximum_gap_us = maximum_gap_us
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    @property
    def track_count(self) -> int:
        return self._next_id

    def assign(
        self, canonical_time_us: int, bboxes: Sequence[tuple[float, float, float, float]]
    ) -> list[int]:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if canonical_time_us - track.last_canonical_time_us <= self._maximum_gap_us
        }
        candidates: list[tuple[float, int, int]] = []
        for detection_index, bbox in enumerate(bboxes):
            for track_id, track in self._tracks.items():
                score = _iou(bbox, track.bbox)
                if score >= self._threshold:
                    candidates.append((-score, track_id, detection_index))
        assignments: list[int | None] = [None] * len(bboxes)
        used_tracks: set[int] = set()
        for _, track_id, detection_index in sorted(candidates):
            if assignments[detection_index] is None and track_id not in used_tracks:
                assignments[detection_index] = track_id
                used_tracks.add(track_id)
        for detection_index, track_id in enumerate(assignments):
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                assignments[detection_index] = track_id
            self._tracks[track_id] = _Track(
                track_id=track_id,
                bbox=bboxes[detection_index],
                last_canonical_time_us=canonical_time_us,
            )
        return [int(value) for value in assignments]


def _finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _points(
    values: Any, *, dimensions: int, divisors: Sequence[float] | None = None
) -> list[list[float | None]]:
    output: list[list[float | None]] = []
    for point in values:
        row: list[float | None] = []
        for axis in range(dimensions):
            numeric = _finite_or_none(point[axis])
            if numeric is not None and divisors is not None:
                numeric /= divisors[axis]
            row.append(numeric)
        output.append(row)
    return output


def _scores(values: Any) -> list[float | None]:
    return [_finite_or_none(value) for value in values]


def _clip_bbox(
    raw: Sequence[Any], width: int, height: int
) -> tuple[float, float, float, float] | None:
    if len(raw) < 4:
        return None
    values = [float(raw[index]) for index in range(4)]
    if not all((math.isfinite(value) for value in values)):
        return None
    x1 = min(max(values[0], 0.0), float(width))
    y1 = min(max(values[1], 0.0), float(height))
    x2 = min(max(values[2], 0.0), float(width))
    y2 = min(max(values[3], 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _candidate_model_paths(tool: Any) -> Iterable[Path]:
    seen: set[str] = set()
    for owner in (tool, getattr(tool, "session", None)):
        if owner is None:
            continue
        for name in ("onnx_model", "model_path", "_model_path"):
            value = getattr(owner, name, None)
            if isinstance(value, (str, Path)):
                path = Path(value)
                key = str(path)
                if key not in seen and path.is_file():
                    seen.add(key)
                    yield path


def _model_identity(configured_location: str, tool: Any) -> dict[str, Any]:
    paths = list(_candidate_model_paths(tool))
    loaded = paths[0] if paths else None
    return {
        "configured_location": configured_location,
        "configured_location_sha256": sha256(configured_location.encode("utf-8")).hexdigest(),
        "loaded_file_sha256": file_sha256(loaded) if loaded is not None else None,
        "loaded_file_byte_size": loaded.stat().st_size if loaded is not None else None,
        "loaded_file_name": loaded.name if loaded is not None else None,
        "absolute_cache_path_recorded": False,
    }


def _start_overlay_encoder(
    ffmpeg: str, target: Path, *, width: int, height: int, crf: int, frame_rate: str
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            frame_rate,
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


@dataclass(slots=True)
class _DecoderState:
    final_detail: str = ""


def _showinfo_reader(
    stderr: Any, events: Queue[dict[str, Any] | None], state: _DecoderState
) -> None:
    current_time_base: tuple[int, int] | None = None
    try:
        for raw_line in stderr:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                state.final_detail = line
            time_base_match = re.search("config in time_base:\\s*(\\d+)/(\\d+)", line)
            if time_base_match:
                current_time_base = (int(time_base_match.group(1)), int(time_base_match.group(2)))
            if " pts:" not in line or " n:" not in line:
                continue
            index_match = re.search("\\bn:\\s*(\\d+)", line)
            pts_match = re.search("\\bpts:\\s*(-?\\d+)", line)
            size_match = re.search("\\bs:(\\d+)x(\\d+)", line)
            sar_match = re.search("\\bsar:(\\d+)/(\\d+)", line)
            if index_match and pts_match and size_match:
                events.put(
                    {
                        "index": int(index_match.group(1)),
                        "pts": int(pts_match.group(1)),
                        "width": int(size_match.group(1)),
                        "height": int(size_match.group(2)),
                        "sample_aspect_ratio": {
                            "numerator": int(sar_match.group(1)),
                            "denominator": int(sar_match.group(2)),
                        }
                        if sar_match
                        else None,
                        "time_base": {
                            "numerator": current_time_base[0],
                            "denominator": current_time_base[1],
                        }
                        if current_time_base is not None
                        else None,
                    }
                )
    finally:
        events.put(None)


def _start_source_decoder(
    ffmpeg: str, source: Path, *, frame_span: tuple[int, int] | None = None
) -> tuple[subprocess.Popen[bytes], Queue[dict[str, Any] | None], _DecoderState, Thread]:
    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostats",
            "-copyts",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            (f"trim=start_frame={frame_span[0]}:end_frame={frame_span[1]}," if frame_span else "")
            + _SOURCE_VIDEO_FILTER,
            "-fps_mode",
            "passthrough",
            *(["-frames:v", str(frame_span[1] - frame_span[0])] if frame_span else []),
            "-c:v",
            "ppm",
            "-f",
            "image2pipe",
            "-",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stderr is not None
    events: Queue[dict[str, Any] | None] = Queue()
    state = _DecoderState()
    thread = Thread(
        target=_showinfo_reader,
        args=(process.stderr, events, state),
        name="pose-source-showinfo",
        daemon=True,
    )
    thread.start()
    return (process, events, state, thread)


def _read_ppm_token(stream: Any) -> bytes | None:
    token = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            return bytes(token) if token else None
        if byte == b"#" and (not token):
            while byte not in {b"\n", b""}:
                byte = stream.read(1)
            continue
        if byte.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(byte)


def _read_ppm_frame(stream: Any, np: Any) -> Any | None:
    magic = _read_ppm_token(stream)
    if magic is None:
        return None
    if magic != b"P6":
        raise RuntimeError("source decoder returned a non-PPM frame")
    width_token = _read_ppm_token(stream)
    height_token = _read_ppm_token(stream)
    maximum_token = _read_ppm_token(stream)
    if width_token is None or height_token is None or maximum_token != b"255":
        raise RuntimeError("source decoder returned a malformed PPM header")
    width = int(width_token)
    height = int(height_token)
    if width <= 0 or height <= 0:
        raise RuntimeError("source decoder returned invalid frame dimensions")
    expected = width * height * 3
    payload = bytearray()
    while len(payload) < expected:
        chunk = stream.read(expected - len(payload))
        if not chunk:
            raise RuntimeError("source decoder truncated a PPM frame")
        payload.extend(chunk)
    rgb = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
    return rgb[:, :, ::-1].copy()


def _overlay_rate(source_frames: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any], str]:
    first_us = int(source_frames[0]["canonical_time_us"])
    last_end_us = int(source_frames[-1]["canonical_end_us"])
    span_us = max(1, last_end_us - first_us)
    rate = Fraction(len(source_frames) * 1000000, span_us).limit_denominator(1000000)
    deltas = [
        int(right["canonical_time_us"]) - int(left["canonical_time_us"])
        for left, right in zip(source_frames, source_frames[1:])
    ]
    approximately_constant = not deltas or max(deltas) - min(deltas) <= 1
    metadata = {
        "numerator": rate.numerator,
        "denominator": rate.denominator,
        "decimal": rate.numerator / rate.denominator,
        "source_timing": "constant_cadence" if approximately_constant else "variable_cadence",
    }
    semantics = "all decoded source frames are retained in presentation order; the overlay uses one constant playback rate spanning the source-video presentation span. Exact per-frame PTS remain authoritative in observations.jsonl"
    return (f"{rate.numerator}/{rate.denominator}", metadata, semantics)


def _close_overlay_encoder(process: subprocess.Popen[bytes]) -> None:
    assert process.stdin is not None
    process.stdin.close()
    process.stdin = None
    _, stderr = process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(f"pose overlay encoding failed{suffix}")


def _write_frame(process: subprocess.Popen[bytes], frame: Any) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write(frame.tobytes())
    except BrokenPipeError as exc:
        raise RuntimeError("pose overlay encoder stopped unexpectedly") from exc


def _draw_observation(
    frame: Any,
    cv2: Any,
    draw_skeleton: Any,
    keypoints_2d: Any,
    scores: Any,
    bboxes: Sequence[tuple[float, float, float, float]],
    track_ids: Sequence[int],
    *,
    source_frame_index: int,
    canonical_time_us: int,
    threshold: float,
) -> Any:
    if len(bboxes):
        frame = draw_skeleton(
            frame,
            keypoints_2d,
            scores,
            openpose_skeleton=False,
            kpt_thr=threshold,
            radius=2,
            line_width=2,
        )
    for bbox, track_id in zip(bboxes, track_ids):
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"person {track_id}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"source frame {source_frame_index} | {canonical_time_us / 1000000:.3f} s",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _validate_request(value: Any) -> tuple[dict[str, Any], PosePreprocessingConfig]:
    if not isinstance(value, dict) or set(value) - {"storage_mode"} != {
        "schema_version",
        "source_path",
        "output_directory",
        "ffmpeg",
        "source_frames",
        "configuration",
    }:
        raise ValueError("pose worker request is malformed")
    if value.get("storage_mode", "debug") not in {"compact", "debug"}:
        raise ValueError("pose worker storage mode is unsupported")
    if value.get("schema_version") != _CONTRACT_VERSION:
        raise ValueError("pose worker request contract is unsupported")
    if not isinstance(value.get("source_frames"), list) or not value["source_frames"]:
        raise ValueError("pose worker request has no source frames")
    if not isinstance(value.get("configuration"), Mapping):
        raise ValueError("pose worker request has no configuration")
    return (value, PosePreprocessingConfig(**value["configuration"]))


def _run(request_path: Path) -> None:
    request, config = _validate_request(read_json(request_path))
    source = Path(request["source_path"])
    source_frames = request["source_frames"]
    output = Path(request["output_directory"])
    compact_storage = request.get("storage_mode", "debug") == "compact"
    observations_path = output / (
        "observations.jsonl.gz" if compact_storage else "observations.jsonl"
    )
    overlay_path = output / "overlay.mp4"
    try:
        import cv2
        import numpy as np
        from rtmlib import Wholebody3d, draw_skeleton
    except ImportError as exc:
        raise RuntimeError(
            "pose dependencies are unavailable; install the project into its virtual environment with `python -m pip install -e .`"
        ) from exc
    engine = Wholebody3d(
        det=config.detector_model,
        det_input_size=(config.detector_input_width, config.detector_input_height),
        pose=config.pose_model,
        pose_input_size=(config.pose_input_width, config.pose_input_height),
        to_openpose=False,
        backend=config.backend,
        device=config.device,
    )
    from .processing_scope import OUTSIDE, excluded_observation

    selected_indices = [
        i for i, frame in enumerate(source_frames) if frame.get("processing_status") != OUTSIDE
    ]
    if not selected_indices:
        raise ValueError("no source frames in processing range")
    scope_start, scope_end = (selected_indices[0], selected_indices[-1] + 1)
    decoder, decoder_events, decoder_state, decoder_thread = _start_source_decoder(
        request["ffmpeg"],
        source,
        frame_span=(scope_start, scope_end)
        if len(selected_indices) != len(source_frames)
        else None,
    )
    try:
        assert decoder.stdout is not None
        first_frame = _read_ppm_frame(decoder.stdout, np)
        if first_frame is None:
            raise RuntimeError("pose worker could not decode source video")
        width = int(first_frame.shape[1])
        height = int(first_frame.shape[0])
        frame_rate, overlay_frame_rate, overlay_timing_semantics = _overlay_rate(source_frames)
        encoder = (
            None
            if compact_storage
            else _start_overlay_encoder(
                request["ffmpeg"],
                overlay_path,
                width=width,
                height=height,
                crf=config.overlay_crf,
                frame_rate=frame_rate,
            )
        )
    except Exception:
        if decoder.poll() is None:
            decoder.kill()
            decoder.wait()
        decoder_thread.join(timeout=5.0)
        raise
    tracker = _IoUTracker(
        threshold=config.tracking_iou_threshold, maximum_gap_us=config.maximum_track_gap_us
    )
    frames_with_people = 0
    person_observation_count = 0
    processed = 0
    decoder_sample_aspect_ratio: Mapping[str, Any] | None = None
    try:
        with open_data(observations_path, "w") as handle:
            for source_frame in source_frames:
                if source_frame.get("processing_status") == OUTSIDE:
                    handle.write(
                        json.dumps(
                            excluded_observation(source_frame, width, height),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    continue
                frame = first_frame if processed == 0 else _read_ppm_frame(decoder.stdout, np)
                if frame is None:
                    raise RuntimeError(
                        "source decoder ended before every probed source frame was decoded"
                    )
                try:
                    event = decoder_events.get(timeout=30.0)
                except Empty as exc:
                    raise RuntimeError(
                        "source decoder did not provide timestamp evidence in time"
                    ) from exc
                if event is None:
                    raise RuntimeError(
                        "source decoder omitted timestamp evidence for a decoded frame"
                    )
                source_frame_index = source_frame.get("source_frame_index")
                if source_frame_index != scope_start + processed:
                    raise RuntimeError("pose worker received an unordered source timeline")
                if (
                    event.get("index") != processed
                    or event.get("pts") != source_frame.get("presentation_timestamp")
                    or event.get("time_base") != source_frame.get("time_base")
                    or (event.get("width") != frame.shape[1])
                    or (event.get("height") != frame.shape[0])
                ):
                    raise RuntimeError(
                        "decoded source frame does not match the probed PTS timeline"
                    )
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise RuntimeError("decoded source dimensions changed within the video")
                if decoder_sample_aspect_ratio is None:
                    decoder_sample_aspect_ratio = event.get("sample_aspect_ratio")
                elif event.get("sample_aspect_ratio") != decoder_sample_aspect_ratio:
                    raise RuntimeError(
                        "decoded source sample aspect ratio changed within the video"
                    )
                people: list[dict[str, Any]] = []
                raw_bboxes = engine.det_model(frame)
                if isinstance(raw_bboxes, tuple):
                    raw_bboxes = raw_bboxes[0]
                selected_raw: list[Any] = []
                bboxes: list[tuple[float, float, float, float]] = []
                for raw_bbox in raw_bboxes:
                    bbox = _clip_bbox(raw_bbox, width, height)
                    if bbox is not None:
                        selected_raw.append(raw_bbox)
                        bboxes.append(bbox)
                order = sorted(range(len(bboxes)), key=lambda index: (*bboxes[index], index))
                bboxes = [bboxes[index] for index in order]
                selected_raw = [selected_raw[index] for index in order]
                if bboxes:
                    keypoints_3d, scores, _keypoints_simcc, keypoints_2d = engine.pose_model(
                        frame, bboxes=bboxes
                    )
                    keypoints_3d = np.asarray(keypoints_3d)
                    scores = np.asarray(scores)
                    keypoints_2d = np.asarray(keypoints_2d)
                    if not len(keypoints_3d) == len(scores) == len(keypoints_2d) == len(bboxes):
                        raise RuntimeError(
                            "pose estimator returned inconsistent person cardinality"
                        )
                    if (
                        keypoints_3d.ndim != 3
                        or keypoints_3d.shape[1:] != (133, 3)
                        or keypoints_2d.ndim != 3
                        or (keypoints_2d.shape[1:] != (133, 2))
                        or (scores.ndim != 2)
                        or (scores.shape[1] != 133)
                    ):
                        raise RuntimeError(
                            "pose estimator did not return COCO-WholeBody-133 arrays"
                        )
                else:
                    keypoints_3d = np.empty((0, 133, 3), dtype=float)
                    scores = np.empty((0, 133), dtype=float)
                    keypoints_2d = np.empty((0, 133, 2), dtype=float)
                canonical_time_us = int(source_frame["canonical_time_us"])
                track_ids = tracker.assign(canonical_time_us, bboxes)
                for person_index, (bbox, track_id) in enumerate(zip(bboxes, track_ids)):
                    raw_bbox = selected_raw[person_index]
                    confidence = _finite_or_none(raw_bbox[4]) if len(raw_bbox) > 4 else None
                    score_values = _scores(scores[person_index])
                    finite_scores = [value for value in score_values if value is not None]
                    people.append(
                        {
                            "detection_index": person_index,
                            "track_id": track_id,
                            "bbox_xyxy_px": list(bbox),
                            "bbox_xyxy_normalized": [
                                bbox[0] / width,
                                bbox[1] / height,
                                bbox[2] / width,
                                bbox[3] / height,
                            ],
                            "detector_confidence": confidence,
                            "mean_keypoint_confidence": sum(finite_scores) / len(finite_scores)
                            if finite_scores
                            else None,
                            "keypoint_confidences": score_values,
                            "keypoints_image_xy_px": _points(
                                keypoints_2d[person_index], dimensions=2
                            ),
                            "keypoints_image_xy_normalized": _points(
                                keypoints_2d[person_index], dimensions=2, divisors=(width, height)
                            ),
                            "keypoints_model_xyz": _points(
                                keypoints_3d[person_index], dimensions=3
                            ),
                            "keypoints_model_xyz_normalized": _points(
                                keypoints_3d[person_index],
                                dimensions=3,
                                divisors=(
                                    config.pose_input_width,
                                    config.pose_input_height,
                                    config.model_depth_range,
                                ),
                            ),
                        }
                    )
                if people:
                    frames_with_people += 1
                    person_observation_count += len(people)
                overlay = (
                    None
                    if compact_storage
                    else _draw_observation(
                        frame.copy(),
                        cv2,
                        draw_skeleton,
                        keypoints_2d,
                        scores,
                        bboxes,
                        track_ids,
                        source_frame_index=source_frame_index,
                        canonical_time_us=canonical_time_us,
                        threshold=config.keypoint_confidence_threshold,
                    )
                )
                observation = {
                    "schema_version": _CONTRACT_VERSION,
                    "source_frame": {
                        key: value
                        for key, value in source_frame.items()
                        if key not in {"source_frame_index", "analysis_sample_indices"}
                    },
                    "source_frame_index": source_frame_index,
                    "source_frame_id": source_frame.get("frame_id"),
                    "decode_index": source_frame.get("decode_index"),
                    "pts": source_frame.get("pts"),
                    "best_effort_timestamp": source_frame.get("best_effort_timestamp"),
                    "presentation_timestamp": source_frame.get("presentation_timestamp"),
                    "timestamp_source": source_frame.get("timestamp_source"),
                    "time_base": source_frame.get("time_base"),
                    "canonical_time_us": canonical_time_us,
                    "canonical_end_us": source_frame.get("canonical_end_us"),
                    "endpoint_basis": source_frame.get("endpoint_basis"),
                    "analysis_sample_indices": source_frame.get("analysis_sample_indices", []),
                    "decoded_frame": {
                        "width": width,
                        "height": height,
                        "sample_aspect_ratio": event.get("sample_aspect_ratio"),
                        "display_transform": "FFmpeg automatic display transform",
                    },
                    "pose_inference_attempted": True,
                    "pose_evidence_available": bool(people),
                    "people": people,
                }
                handle.write(
                    json.dumps(
                        observation,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                if encoder is not None:
                    _write_frame(encoder, overlay)
                processed += 1
                if processed % 100 == 0 or processed == len(source_frames):
                    print(
                        f"PROGRESS source_frame_count={processed} total_source_frame_count={len(source_frames)}",
                        file=sys.stderr,
                        flush=True,
                    )
            if _read_ppm_frame(decoder.stdout, np) is not None:
                raise RuntimeError(
                    "source decoder yielded more frames than the probed source timeline"
                )
        decoder_return_code = decoder.wait()
        decoder_thread.join()
        if decoder_return_code != 0:
            raise RuntimeError("source decoding failed; inspect the local FFmpeg runtime")
        if encoder is not None:
            _close_overlay_encoder(encoder)
    except Exception:
        if decoder.poll() is None:
            decoder.kill()
            decoder.wait()
        decoder_thread.join()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        raise
    models = {
        "detector": _model_identity(config.detector_model, engine.det_model),
        "pose": _model_identity(config.pose_model, engine.pose_model),
    }
    write_json(
        output / ".worker-result.json",
        {
            "schema_version": _CONTRACT_VERSION,
            "frame_count": len(source_frames),
            "processed_frame_count": processed,
            "processed_frame_span": [scope_start, scope_end],
            "frame_width": width,
            "frame_height": height,
            "frames_with_people": frames_with_people,
            "person_observation_count": person_observation_count,
            "track_count": tracker.track_count,
            "overlay_frame_rate": overlay_frame_rate,
            "overlay_timing_semantics": overlay_timing_semantics,
            "runtime": {
                "python_version": ".".join(map(str, sys.version_info[:3])),
                "rtmlib_version": _package_version("rtmlib"),
                "opencv_version": getattr(cv2, "__version__", None),
                "numpy_version": getattr(np, "__version__", None),
                "onnxruntime_version": _package_version("onnxruntime"),
                "source_decoder": "FFmpeg image2pipe PPM with showinfo PTS audit",
                "pose_canvas_pixel_format": "8-bit BGR converted from FFmpeg PPM RGB",
                "source_video_filter": _SOURCE_VIDEO_FILTER,
                "automatic_display_rotation": True,
                "decoder_sample_aspect_ratio": decoder_sample_aspect_ratio,
                "backend": config.backend,
                "device": config.device,
                "process_isolation": "terminating child process",
            },
            "models": models,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        _run(args.request)
        return 0
    except Exception as exc:
        print(f"pose preprocessing failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
