# SPDX-License-Identifier: Apache-2.0
"""Optional FFmpeg adapter for PTS-aware proxy generation."""

from __future__ import annotations
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from bisect import bisect_right
from uuid import uuid4
from ..config import AnalysisConfig
from ..rights import (
    classify_media_rights,
    combine_rights,
    repository_restricted_roots,
    rights_sidecar_path,
    validate_rights,
    write_rights_sidecar,
)
from .timeline import build_analysis_grid, normalize_pts_us, round_fraction_half_even


class MediaRuntimeError(RuntimeError):
    """Raised when the optional media runtime cannot perform a request."""


QUANTIZER_VERSION = "rational-seconds-to-integer-microseconds-half-even-v0.1"


def _format_microseconds_as_seconds(value_us: int) -> str:
    sign = "-" if value_us < 0 else ""
    seconds, micros = divmod(abs(value_us), 1000000)
    return f"{sign}{seconds}.{micros:06d}"


def _quantized_timestamp(
    pts: int, base: Mapping[str, Any], origin: Fraction
) -> tuple[int, dict[str, int]]:
    canonical_us = normalize_pts_us(pts, int(base["numerator"]), int(base["denominator"]), origin)
    exact_us = (Fraction(pts * int(base["numerator"]), int(base["denominator"])) - origin) * 1000000
    error = Fraction(canonical_us) - exact_us
    if abs(error) > Fraction(1, 2):
        raise MediaRuntimeError("canonical timestamp quantization exceeded 0.5 microseconds")
    return (canonical_us, {"numerator": error.numerator, "denominator": error.denominator})


def _resolve_executable(value: str) -> str:
    candidate = Path(value)
    if candidate.parent != Path(".") or candidate.is_absolute():
        if candidate.is_file():
            return str(candidate)
        raise MediaRuntimeError(f"media executable does not exist: {value}")
    resolved = shutil.which(value)
    if resolved is None:
        raise MediaRuntimeError(
            f"{value} was not found; install a reviewed FFmpeg runtime or pass its path"
        )
    return resolved


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _escape_image2_sequence_pattern(path: str | Path, placeholder: str) -> str:
    """Escape literal percents while preserving one image2 sequence marker."""
    pattern = Path(path)
    if not placeholder.startswith("%") or not placeholder.endswith("d"):
        raise ValueError("image2 placeholder must be a decimal sequence marker")
    if pattern.name.count(placeholder) != 1:
        raise ValueError("image2 filename must contain exactly one sequence marker")
    prefix, suffix = pattern.name.split(placeholder)
    escaped_name = prefix.replace("%", "%%") + placeholder + suffix.replace("%", "%%")
    escaped_parent = str(pattern.parent).replace("%", "%%")
    return str(Path(escaped_parent) / escaped_name)


def _commit_preprocessing_pair(
    temporary_proxy: Path,
    proxy: Path,
    temporary_manifest: Path,
    manifest: Path,
    temporary_rights: Path | None = None,
    rights: Path | None = None,
) -> None:
    """Commit proxy, manifest, and optional rights label as one recoverable set."""
    token = uuid4().hex
    proxy_backup = proxy.with_name(f".{proxy.name}.{token}.backup")
    manifest_backup = manifest.with_name(f".{manifest.name}.{token}.backup")
    rights_backup = (
        rights.with_name(f".{rights.name}.{token}.backup") if rights is not None else None
    )
    proxy_committed = False
    manifest_committed = False
    rights_committed = False
    try:
        if proxy.exists():
            proxy.replace(proxy_backup)
        if manifest.exists():
            manifest.replace(manifest_backup)
        if rights is not None and rights.exists() and (rights_backup is not None):
            rights.replace(rights_backup)
        temporary_proxy.replace(proxy)
        proxy_committed = True
        temporary_manifest.replace(manifest)
        manifest_committed = True
        if temporary_rights is not None and rights is not None:
            temporary_rights.replace(rights)
            rights_committed = True
    except OSError:
        if rights_committed and rights is not None:
            rights.unlink(missing_ok=True)
        if manifest_committed:
            manifest.unlink(missing_ok=True)
        if proxy_committed:
            proxy.unlink(missing_ok=True)
        if proxy_backup.exists():
            proxy_backup.replace(proxy)
        if manifest_backup.exists():
            manifest_backup.replace(manifest)
        if rights_backup is not None and rights_backup.exists() and (rights is not None):
            rights_backup.replace(rights)
        raise
    else:
        proxy_backup.unlink(missing_ok=True)
        manifest_backup.unlink(missing_ok=True)
        if rights_backup is not None:
            rights_backup.unlink(missing_ok=True)
    finally:
        temporary_proxy.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        if temporary_rights is not None:
            temporary_rights.unlink(missing_ok=True)


def build_ffmpeg_command(
    input_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate_hz: int = 10,
    maximum_width: int = 960,
    include_audio: bool = True,
    audio_input_path: str | Path | None = None,
    audio_offset_us: int = 0,
    overwrite: bool = False,
) -> list[str]:
    """Build, but do not execute, the deterministic proxy command."""
    if sample_rate_hz != 10:
        raise ValueError("version 0.1 proxy sampling is fixed at 10 Hz")
    if maximum_width < 2:
        raise ValueError("maximum_width must be at least two pixels")
    input_file = Path(input_path)
    output_file = Path(output_path)
    if input_file.resolve() == output_file.resolve():
        raise ValueError("the source video may not be overwritten")
    audio_file = Path(audio_input_path) if audio_input_path is not None else None
    if audio_file is None and audio_offset_us != 0:
        raise ValueError("audio_offset_us requires a sidecar audio input")
    if audio_file is not None:
        if not include_audio:
            raise ValueError("a sidecar audio input cannot be combined with include_audio=False")
        if audio_file.resolve() in {input_file.resolve(), output_file.resolve()}:
            raise ValueError("sidecar audio must be distinct from video and proxy paths")
    if type(audio_offset_us) is not int:
        raise ValueError("audio_offset_us must be an integer")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n"]
    if audio_file is None:
        command.extend(["-copyts", "-start_at_zero"])
    command.extend(["-i", str(input_file)])
    if audio_file is not None:
        command.extend(["-i", str(audio_file)])
    command.extend(["-map", "0:v:0"])
    if include_audio:
        command.extend(["-map", "1:a:0?" if audio_file is not None else "0:a?"])
    video_filter_parts = []
    if audio_file is not None:
        video_shift_us = max(0, -audio_offset_us)
        video_filter_parts.append(
            f"setpts=PTS-STARTPTS+{_format_microseconds_as_seconds(video_shift_us)}/TB"
        )
    video_filter_parts.extend(
        [
            f"fps=fps={sample_rate_hz}:round=near:eof_action=pass",
            f"scale=w='max(2,trunc(min({maximum_width},iw)/2)*2)':h=-2:flags=lanczos",
        ]
    )
    command.extend(
        [
            "-vf",
            ",".join(video_filter_parts),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if include_audio:
        if audio_file is not None:
            audio_shift_us = max(0, audio_offset_us)
            command.extend(
                [
                    "-af",
                    f"asetpts=PTS-STARTPTS+{_format_microseconds_as_seconds(audio_shift_us)}/TB",
                ]
            )
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")
    command.extend(["-movflags", "+faststart", str(output_file)])
    return command


def _run_json(command: Sequence[str]) -> dict[str, Any]:
    try:
        process = subprocess.run(
            list(command), check=True, capture_output=True, text=True, encoding="utf-8"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else str(exc)
        raise MediaRuntimeError(f"media probe failed: {detail}") from exc
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise MediaRuntimeError("media probe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MediaRuntimeError("media probe returned an unexpected document")
    return payload


def probe_media(input_path: str | Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    executable = _resolve_executable(ffprobe)
    return _run_json(
        [
            executable,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(Path(input_path)),
        ]
    )


def probe_video_frames(input_path: str | Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Read raw presentation timestamps for only the selected video stream."""
    executable = _resolve_executable(ffprobe)
    return _run_json(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=pts,best_effort_timestamp,pkt_duration",
            "-of",
            "json",
            str(Path(input_path)),
        ]
    )


def probe_audio_packets(input_path: str | Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Read raw packet timestamps for every selected source audio stream."""
    executable = _resolve_executable(ffprobe)
    return _run_json(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_packets",
            "-show_entries",
            "packet=stream_index,pts,duration",
            "-of",
            "json",
            str(Path(input_path)),
        ]
    )


def _parse_time_base(value: str) -> tuple[int, int]:
    numerator_text, denominator_text = value.split("/", 1)
    numerator, denominator = (int(numerator_text), int(denominator_text))
    if numerator <= 0 or denominator <= 0:
        raise MediaRuntimeError("source stream has an invalid time base")
    return (numerator, denominator)


def _safe_probe_summary(probe: dict[str, Any], *, include_audio: bool = True) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    starts: list[Fraction] = []
    selected_video = False
    for stream in probe.get("streams", []):
        if stream.get("codec_type") not in {"video", "audio"}:
            continue
        if stream.get("codec_type") == "video":
            if selected_video:
                continue
            selected_video = True
        if stream.get("codec_type") == "audio" and (not include_audio):
            continue
        numerator, denominator = _parse_time_base(stream["time_base"])
        start_pts = int(stream.get("start_pts", 0))
        start = Fraction(start_pts * numerator, denominator)
        starts.append(start)
        selected.append(
            {
                "index": int(stream["index"]),
                "codec_type": stream["codec_type"],
                "codec_name": stream.get("codec_name"),
                "time_base": {"numerator": numerator, "denominator": denominator},
                "start_pts": start_pts,
                "duration_ts": int(stream["duration_ts"])
                if stream.get("duration_ts") is not None
                else None,
                "width": stream.get("width"),
                "height": stream.get("height"),
                "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                "source_kind": "embedded",
                "alignment_offset_us": 0,
            }
        )
    if not selected or not any((item["codec_type"] == "video" for item in selected)):
        raise MediaRuntimeError("source contains no supported video stream")
    origin = min(starts)
    stream_ends = []
    for item in selected:
        if item["duration_ts"] is not None:
            base = item["time_base"]
            stream_ends.append(
                Fraction(
                    (item["start_pts"] + item["duration_ts"]) * base["numerator"],
                    base["denominator"],
                )
            )
    if stream_ends:
        duration = max(stream_ends) - origin
        duration_basis = "selected stream start_pts plus duration_ts"
    else:
        duration_text = probe.get("format", {}).get("duration")
        if duration_text is None:
            raise MediaRuntimeError("selected source-stream duration is unavailable")
        duration = Fraction(str(duration_text))
        duration_basis = "container duration fallback; selected stream duration unavailable"
    duration_us = round_fraction_half_even(duration * 1000000)
    if duration_us <= 0:
        raise MediaRuntimeError("source duration must be positive")
    return {
        "streams": selected,
        "global_origin": {"numerator": origin.numerator, "denominator": origin.denominator},
        "duration_us": duration_us,
        "duration_basis": duration_basis,
        "quantization": "exact rational seconds to integer microseconds, nearest ties-to-even",
        "quantizer_version": QUANTIZER_VERSION,
        "origin_basis": "stream metadata fallback before packet/frame inspection",
    }


def _sidecar_audio_stream(probe: Mapping[str, Any], *, alignment_offset_us: int) -> dict[str, Any]:
    """Select one sidecar audio stream in the same shape as probe summaries."""
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        numerator, denominator = _parse_time_base(stream["time_base"])
        return {
            "index": int(stream["index"]),
            "codec_type": "audio",
            "codec_name": stream.get("codec_name"),
            "time_base": {"numerator": numerator, "denominator": denominator},
            "start_pts": int(stream.get("start_pts", 0)),
            "duration_ts": int(stream["duration_ts"])
            if stream.get("duration_ts") is not None
            else None,
            "width": None,
            "height": None,
            "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
            "source_kind": "sidecar",
            "alignment_offset_us": alignment_offset_us,
        }
    raise MediaRuntimeError("sidecar source contains no supported audio stream")


def _inferred_presentation_ends(entries: list[tuple[Fraction, Fraction]]) -> list[Fraction]:
    """Use explicit durations, then adjacent cadence, for presentation ends."""
    ordered = sorted(entries, key=lambda item: item[0])
    positive_deltas = [
        right[0] - left[0] for left, right in zip(ordered, ordered[1:]) if right[0] > left[0]
    ]
    trailing_cadence = positive_deltas[-1] if positive_deltas else Fraction(0)
    result: list[Fraction] = []
    for index, (start, duration) in enumerate(ordered):
        if duration > 0:
            result.append(start + duration)
        elif index + 1 < len(ordered) and ordered[index + 1][0] > start:
            result.append(ordered[index + 1][0])
        elif trailing_cadence > 0:
            result.append(start + trailing_cadence)
        else:
            result.append(start)
    return result


def _apply_actual_timeline(
    summary: Mapping[str, Any], frame_probe: Mapping[str, Any], packet_probe: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace metadata timing with the earliest actual selected presentation time."""
    starts: list[Fraction] = []
    ends: list[Fraction] = []
    video_stream = next((item for item in summary["streams"] if item["codec_type"] == "video"))
    video_base = video_stream["time_base"]
    video_scale = Fraction(int(video_base["numerator"]), int(video_base["denominator"]))
    video_entries: list[tuple[Fraction, Fraction]] = []
    for frame in frame_probe.get("frames", []):
        value = frame.get("pts")
        if value is None:
            value = frame.get("best_effort_timestamp")
        if value is None:
            continue
        start = int(value) * video_scale
        starts.append(start)
        duration = int(frame.get("pkt_duration") or 0) * video_scale
        video_entries.append((start, duration))
    ends.extend(_inferred_presentation_ends(video_entries))
    audio_bases = {
        int(item["index"]): (
            Fraction(int(item["time_base"]["numerator"]), int(item["time_base"]["denominator"])),
            Fraction(int(item.get("alignment_offset_us", 0)), 1000000),
        )
        for item in summary["streams"]
        if item["codec_type"] == "audio"
    }
    audio_entries: dict[int, list[tuple[Fraction, Fraction]]] = {}
    for packet in packet_probe.get("packets", []):
        stream_index = int(packet.get("stream_index", -1))
        value = packet.get("pts")
        if value is None or stream_index not in audio_bases:
            continue
        scale, alignment_offset = audio_bases[stream_index]
        start = int(value) * scale + alignment_offset
        starts.append(start)
        duration = int(packet.get("duration") or 0) * scale
        audio_entries.setdefault(stream_index, []).append((start, duration))
    for entries in audio_entries.values():
        ends.extend(_inferred_presentation_ends(entries))
    if not starts:
        raise MediaRuntimeError("no actual presentation timestamps were available")
    actual_origin = min(starts)
    prior_origin_data = summary["global_origin"]
    prior_origin = Fraction(
        int(prior_origin_data["numerator"]), int(prior_origin_data["denominator"])
    )
    metadata_end = prior_origin + Fraction(int(summary["duration_us"]), 1000000)
    actual_values = [*ends, *starts]
    use_metadata_bound = not str(summary.get("duration_basis", "")).startswith(
        "container duration fallback"
    )
    actual_end = max(
        [*(actual_values or [metadata_end]), *([metadata_end] if use_metadata_bound else [])]
    )
    used_metadata_fallback = actual_end <= actual_origin
    if used_metadata_fallback:
        actual_end = metadata_end
    duration_us = round_fraction_half_even((actual_end - actual_origin) * 1000000)
    if duration_us <= 0:
        raise MediaRuntimeError("actual source presentation span must be positive")
    result = dict(summary)
    result["global_origin"] = {
        "numerator": actual_origin.numerator,
        "denominator": actual_origin.denominator,
    }
    result["duration_us"] = duration_us
    result["origin_basis"] = (
        "earliest actual selected video-frame or audio-packet presentation timestamp"
    )
    if use_metadata_bound:
        result["duration_basis"] = (
            "latest actual selected presentation end with selected-stream metadata safety bound"
        )
    elif used_metadata_fallback:
        result["duration_basis"] = (
            "container duration fallback; no positive selected presentation span was inferable"
        )
    else:
        result["duration_basis"] = (
            "latest actual selected presentation end with missing durations inferred from cadence; container fallback excluded"
        )
    return result


def _apply_sidecar_timeline(
    summary: Mapping[str, Any],
    frame_probe: Mapping[str, Any],
    packet_probe: Mapping[str, Any],
    *,
    alignment_offset_us: int,
) -> dict[str, Any]:
    """Normalize independent input starts, then apply the declared A/V offset."""
    video_stream = next((item for item in summary["streams"] if item["codec_type"] == "video"))
    audio_stream = next(
        (
            item
            for item in summary["streams"]
            if item["codec_type"] == "audio" and item.get("source_kind") == "sidecar"
        )
    )
    video_scale = Fraction(
        int(video_stream["time_base"]["numerator"]), int(video_stream["time_base"]["denominator"])
    )
    audio_scale = Fraction(
        int(audio_stream["time_base"]["numerator"]), int(audio_stream["time_base"]["denominator"])
    )
    video_entries: list[tuple[Fraction, Fraction]] = []
    for frame in frame_probe.get("frames", []):
        value = frame.get("pts")
        if value is None:
            value = frame.get("best_effort_timestamp")
        if value is None:
            continue
        video_entries.append(
            (int(value) * video_scale, int(frame.get("pkt_duration") or 0) * video_scale)
        )
    audio_entries: list[tuple[Fraction, Fraction]] = []
    for packet in packet_probe.get("packets", []):
        if int(packet.get("stream_index", -1)) != int(audio_stream["index"]):
            continue
        value = packet.get("pts")
        if value is None:
            continue
        audio_entries.append(
            (int(value) * audio_scale, int(packet.get("duration") or 0) * audio_scale)
        )
    if not video_entries or not audio_entries:
        raise MediaRuntimeError(
            "sidecar alignment requires actual video-frame and audio-packet timestamps"
        )
    first_video = min((item[0] for item in video_entries))
    first_audio = min((item[0] for item in audio_entries))
    declared_offset = Fraction(alignment_offset_us, 1000000)
    video_shift = max(Fraction(0), -declared_offset)
    audio_shift = max(Fraction(0), declared_offset)
    video_origin = first_video - video_shift
    audio_origin = first_audio - audio_shift
    video_ends = _inferred_presentation_ends(video_entries)
    audio_ends = _inferred_presentation_ends(audio_entries)
    if max(video_ends) <= first_video and video_stream.get("duration_ts"):
        video_ends.append(first_video + int(video_stream["duration_ts"]) * video_scale)
    if max(audio_ends) <= first_audio and audio_stream.get("duration_ts"):
        audio_ends.append(first_audio + int(audio_stream["duration_ts"]) * audio_scale)
    duration = max(max(video_ends) - video_origin, max(audio_ends) - audio_origin)
    duration_us = round_fraction_half_even(duration * 1000000)
    if duration_us <= 0:
        raise MediaRuntimeError("aligned video/sidecar presentation span must be positive")
    streams = [dict(item) for item in summary["streams"]]
    for stream in streams:
        if stream["codec_type"] == "video":
            origin = video_origin
            normalized_from = first_video
            shift = video_shift
        elif stream.get("source_kind") == "sidecar":
            origin = audio_origin
            normalized_from = first_audio
            shift = audio_shift
        else:
            continue
        stream["canonical_origin"] = {
            "numerator": origin.numerator,
            "denominator": origin.denominator,
        }
        stream["input_normalization_origin"] = {
            "numerator": normalized_from.numerator,
            "denominator": normalized_from.denominator,
        }
        stream["post_normalization_shift_us"] = round_fraction_half_even(shift * 1000000)
    result = dict(summary)
    result["streams"] = streams
    result["global_origin"] = {"numerator": 0, "denominator": 1}
    result["duration_us"] = duration_us
    result["origin_basis"] = (
        "independent first actual video/audio presentations normalized to zero, then signed sidecar offset applied and both shifted non-negative"
    )
    result["duration_basis"] = "latest aligned actual video-frame or sidecar-audio-packet end"
    result["multi_input_alignment"] = {
        "policy": "normalize-each-input-start-then-apply-sidecar-offset-v0.1",
        "sidecar_offset_us": alignment_offset_us,
        "video_shift_us": round_fraction_half_even(video_shift * 1000000),
        "audio_shift_us": round_fraction_half_even(audio_shift * 1000000),
    }
    return result


def _map_source_frames(
    frame_probe: Mapping[str, Any],
    summary: Mapping[str, Any],
    duration_us: int,
    sample_rate_hz: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    video_stream = next((item for item in summary["streams"] if item["codec_type"] == "video"))
    base = video_stream["time_base"]
    origin_data = video_stream.get("canonical_origin", summary["global_origin"])
    origin = Fraction(origin_data["numerator"], origin_data["denominator"])
    source_frames: list[dict[str, Any]] = []
    for raw_index, frame in enumerate(frame_probe.get("frames", [])):
        raw_pts = frame.get("pts")
        best_effort = frame.get("best_effort_timestamp")
        presentation_value = raw_pts if raw_pts is not None else best_effort
        if presentation_value is None:
            raise MediaRuntimeError(
                "a decoded source video frame has neither PTS nor a best-effort presentation timestamp"
            )
        presentation_timestamp = int(presentation_value)
        canonical_us, signed_error = _quantized_timestamp(presentation_timestamp, base, origin)
        source_frames.append(
            {
                "frame_id": f"source-frame:{len(source_frames)}",
                "stream_index": int(video_stream["index"]),
                "decode_index": raw_index,
                "pts": int(raw_pts) if raw_pts is not None else None,
                "best_effort_timestamp": int(best_effort) if best_effort is not None else None,
                "presentation_timestamp": presentation_timestamp,
                "timestamp_source": "pts" if raw_pts is not None else "best_effort_timestamp",
                "time_base": dict(base),
                "canonical_time_us": canonical_us,
                "quantizer_version": QUANTIZER_VERSION,
                "signed_quantization_error_us": signed_error,
                "duration_ts": int(frame["pkt_duration"]) if frame.get("pkt_duration") else None,
            }
        )
    if not source_frames:
        raise MediaRuntimeError("no presentation timestamps were available for source video frames")
    if any(
        (
            left["canonical_time_us"] > right["canonical_time_us"]
            for left, right in zip(source_frames, source_frames[1:])
        )
    ):
        raise MediaRuntimeError("video mapping is non-monotonic after quantization")
    source_frames.sort(key=lambda item: (item["canonical_time_us"], item["decode_index"]))
    timestamps = [int(item["canonical_time_us"]) for item in source_frames]
    for index, item in enumerate(source_frames):
        duration_ts = item["duration_ts"]
        end_timestamp: int | None = None
        endpoint_basis: str
        if duration_ts is not None and duration_ts > 0:
            end_timestamp = item["presentation_timestamp"] + duration_ts
            endpoint_basis = "presentation_timestamp_plus_packet_duration"
        elif index + 1 < len(source_frames):
            end_timestamp = source_frames[index + 1]["presentation_timestamp"]
            endpoint_basis = "next_presentation_timestamp"
        elif index > 0:
            cadence = (
                item["presentation_timestamp"] - source_frames[index - 1]["presentation_timestamp"]
            )
            if cadence > 0:
                end_timestamp = item["presentation_timestamp"] + cadence
            endpoint_basis = "preceding_presentation_cadence"
        else:
            endpoint_basis = "minimum_positive_fallback"
        if end_timestamp is not None:
            canonical_end_us, end_error = _quantized_timestamp(end_timestamp, base, origin)
            item["presentation_end_timestamp"] = end_timestamp
            item["canonical_end_us"] = max(item["canonical_time_us"] + 1, canonical_end_us)
            item["signed_end_quantization_error_us"] = end_error
        else:
            item["presentation_end_timestamp"] = None
            item["canonical_end_us"] = item["canonical_time_us"] + 1
            item["signed_end_quantization_error_us"] = None
        item["endpoint_basis"] = endpoint_basis
    serialized_grid: list[dict[str, Any]] = []
    active_start = 0
    for sample in build_analysis_grid(duration_us, sample_rate_hz):
        while (
            active_start < len(source_frames)
            and source_frames[active_start]["canonical_end_us"] <= sample.start_us
        ):
            active_start += 1
        first = active_start
        last = bisect_right(timestamps, sample.end_us - 1)
        references = [
            item["frame_id"]
            for item in source_frames[first:last]
            if item["canonical_time_us"] < sample.end_us
            and sample.start_us < item["canonical_end_us"]
        ]
        serialized_grid.append(
            {
                "index": sample.index,
                "start_us": sample.start_us,
                "end_us": sample.end_us,
                "center_us": sample.center_us,
                "source_references": references,
                "source_gap_reason": None if references else "no_decodable_frame_in_sample",
            }
        )
    return (source_frames, serialized_grid)


def _runtime_banner(executable: str) -> str:
    try:
        process = subprocess.run(
            [executable, "-version"], check=True, capture_output=True, text=True, encoding="utf-8"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MediaRuntimeError("could not record the FFmpeg runtime version") from exc
    return process.stdout.splitlines()[0] if process.stdout else "FFmpeg version unavailable"


def _map_audio_packets(
    packet_probe: Mapping[str, Any], summary: Mapping[str, Any]
) -> list[dict[str, Any]]:
    stream_bases = {
        int(item["index"]): (
            item["time_base"],
            int(item.get("alignment_offset_us", 0)),
            str(item.get("source_kind", "embedded")),
            item.get("canonical_origin"),
        )
        for item in summary["streams"]
        if item["codec_type"] == "audio"
    }
    origin_data = summary["global_origin"]
    origin = Fraction(origin_data["numerator"], origin_data["denominator"])
    packets: list[dict[str, Any]] = []
    for packet in packet_probe.get("packets", []):
        stream_index = int(packet["stream_index"])
        pts_value = packet.get("pts")
        if stream_index not in stream_bases or pts_value is None:
            continue
        base, alignment_offset_us, source_kind, canonical_origin_data = stream_bases[stream_index]
        pts = int(pts_value)
        if isinstance(canonical_origin_data, Mapping):
            stream_origin = Fraction(
                int(canonical_origin_data["numerator"]), int(canonical_origin_data["denominator"])
            )
        else:
            stream_origin = origin - Fraction(alignment_offset_us, 1000000)
        canonical_us, signed_error = _quantized_timestamp(pts, base, stream_origin)
        duration_ts = int(packet["duration"]) if packet.get("duration") else None
        canonical_end_us: int | None = None
        signed_end_error: dict[str, int] | None = None
        if duration_ts is not None and duration_ts > 0:
            canonical_end_us, signed_end_error = _quantized_timestamp(
                pts + duration_ts, base, stream_origin
            )
        packets.append(
            {
                "packet_id": f"audio-packet:{len(packets)}",
                "stream_index": stream_index,
                "source_kind": source_kind,
                "alignment_offset_us": alignment_offset_us,
                "pts": pts,
                "time_base": dict(base),
                "canonical_time_us": canonical_us,
                "quantizer_version": QUANTIZER_VERSION,
                "signed_quantization_error_us": signed_error,
                "duration_ts": duration_ts,
                "canonical_end_us": canonical_end_us,
                "endpoint_basis": "presentation_timestamp_plus_packet_duration"
                if canonical_end_us is not None
                else None,
                "signed_end_quantization_error_us": signed_end_error,
            }
        )
    stream_indices = sorted({int(item["stream_index"]) for item in packets})
    for stream_index in stream_indices:
        input_order = [item for item in packets if int(item["stream_index"]) == stream_index]
        if any(
            (
                int(left["canonical_time_us"]) > int(right["canonical_time_us"])
                for left, right in zip(input_order, input_order[1:])
            )
        ):
            raise MediaRuntimeError("audio mapping is non-monotonic after quantization")
    packets.sort(
        key=lambda item: (item["canonical_time_us"], item["stream_index"], item["packet_id"])
    )
    for stream_index in stream_indices:
        stream_packets = [item for item in packets if int(item["stream_index"]) == stream_index]
        positive_deltas = [
            int(right["canonical_time_us"]) - int(left["canonical_time_us"])
            for left, right in zip(stream_packets, stream_packets[1:])
            if int(right["canonical_time_us"]) > int(left["canonical_time_us"])
        ]
        trailing_cadence = positive_deltas[-1] if positive_deltas else 0
        for index, item in enumerate(stream_packets):
            if item["canonical_end_us"] is not None:
                item["canonical_end_us"] = max(
                    int(item["canonical_time_us"]) + 1, int(item["canonical_end_us"])
                )
                continue
            if index + 1 < len(stream_packets):
                inferred_end = int(stream_packets[index + 1]["canonical_time_us"])
                basis = "next_presentation_timestamp"
            elif trailing_cadence > 0:
                inferred_end = int(item["canonical_time_us"]) + trailing_cadence
                basis = "preceding_presentation_cadence"
            else:
                inferred_end = int(item["canonical_time_us"]) + 1
                basis = "minimum_positive_fallback"
            item["canonical_end_us"] = max(int(item["canonical_time_us"]) + 1, inferred_end)
            item["endpoint_basis"] = basis
    return packets


def _verify_proxy_timeline(
    proxy_summary: Mapping[str, Any],
    proxy_frame_probe: Mapping[str, Any],
    proxy_packet_probe: Mapping[str, Any],
    source_frames: Sequence[Mapping[str, Any]],
    source_audio_packets: Sequence[Mapping[str, Any]],
    *,
    source_duration_us: int,
    sample_rate_hz: int,
) -> dict[str, Any]:
    """Verify the muxed derivative's observed timing against canonical input time."""
    proxy_timeline = _apply_actual_timeline(proxy_summary, proxy_frame_probe, proxy_packet_probe)
    proxy_frames, _ = _map_source_frames(
        proxy_frame_probe, proxy_timeline, proxy_timeline["duration_us"], sample_rate_hz
    )
    period_us = 1000000 // sample_rate_hz
    cadence_errors = [
        abs(right["canonical_time_us"] - left["canonical_time_us"] - period_us)
        for left, right in zip(proxy_frames, proxy_frames[1:])
    ]
    cadence_tolerance_us = 1000
    if cadence_errors and max(cadence_errors) > cadence_tolerance_us:
        raise MediaRuntimeError("generated proxy is not an observed fixed-10-Hz video")
    if not source_frames:
        raise MediaRuntimeError("source video timing is unavailable for proxy verification")
    video_start_error_us = proxy_frames[0]["canonical_time_us"] - int(
        source_frames[0]["canonical_time_us"]
    )
    alignment_tolerance_us = period_us
    if abs(video_start_error_us) > alignment_tolerance_us:
        raise MediaRuntimeError("generated proxy video is misaligned with canonical source time")
    source_video_end_us = max(
        (
            int(item.get("canonical_end_us", int(item["canonical_time_us"]) + period_us))
            for item in source_frames
        )
    )
    proxy_video_end_us = max((int(item["canonical_end_us"]) for item in proxy_frames))
    proxy_video_end_aligned_us = proxy_video_end_us - video_start_error_us
    video_end_error_us = proxy_video_end_aligned_us - source_video_end_us
    if abs(video_end_error_us) > alignment_tolerance_us:
        raise MediaRuntimeError("generated proxy video ends at the wrong canonical time")
    duration_error_us = int(proxy_timeline["duration_us"]) - source_duration_us
    if abs(duration_error_us) > alignment_tolerance_us:
        raise MediaRuntimeError("generated proxy duration is misaligned with canonical source time")
    analysis_grid = build_analysis_grid(source_duration_us, sample_rate_hz)
    expected_frame_count = len(analysis_grid)
    frame_count_delta = len(proxy_frames) - expected_frame_count
    if frame_count_delta == 0:
        frame_count_adjustment = "none"
    elif frame_count_delta == -1:
        final_sample = analysis_grid[-1]
        if (
            source_video_end_us > final_sample.start_us
            or proxy_video_end_aligned_us > final_sample.start_us
        ):
            raise MediaRuntimeError(
                f"generated proxy is missing a visual frame inside the canonical video span (expected {expected_frame_count}, observed {len(proxy_frames)}; final sample [{final_sample.start_us}, {final_sample.end_us}); source video end {source_video_end_us}; aligned proxy video end {proxy_video_end_aligned_us})"
            )
        frame_count_adjustment = "clone_previous_visual_for_nonvisual_terminal_sample"
    elif frame_count_delta == 1:
        surplus_start_us = int(proxy_frames[-1]["canonical_time_us"])
        surplus_start_aligned_us = surplus_start_us - video_start_error_us
        if surplus_start_aligned_us < source_duration_us:
            raise MediaRuntimeError(
                "generated proxy has an unexpected surplus frame inside canonical time"
            )
        frame_count_adjustment = "drop_proxy_frame_starting_after_canonical_end"
    else:
        raise MediaRuntimeError(
            "generated proxy frame count is incompatible with the canonical 10 Hz grid"
        )
    proxy_audio_packets = _map_audio_packets(proxy_packet_probe, proxy_timeline)
    audio_start_error_us: int | None = None
    audio_end_error_us: int | None = None
    if source_audio_packets:
        if not proxy_audio_packets:
            raise MediaRuntimeError("generated proxy audio timestamps could not be verified")
        audio_start_error_us = int(proxy_audio_packets[0]["canonical_time_us"]) - int(
            source_audio_packets[0]["canonical_time_us"]
        )
        if abs(audio_start_error_us) > alignment_tolerance_us:
            raise MediaRuntimeError(
                "generated proxy audio is misaligned with canonical source time"
            )
        source_audio_end_us = max(
            (
                int(item.get("canonical_end_us", int(item["canonical_time_us"]) + period_us))
                for item in source_audio_packets
            )
        )
        proxy_audio_end_us = max((int(item["canonical_end_us"]) for item in proxy_audio_packets))
        audio_end_error_us = proxy_audio_end_us - source_audio_end_us
        if abs(audio_end_error_us) > alignment_tolerance_us:
            raise MediaRuntimeError("generated proxy audio ends at the wrong canonical time")
    return {
        "status": "verified_within_declared_tolerance",
        "sample_rate_hz": sample_rate_hz,
        "proxy_video_frame_count": len(proxy_frames),
        "expected_analysis_sample_count": expected_frame_count,
        "video_frame_count_delta": frame_count_delta,
        "frame_count_adjustment": frame_count_adjustment,
        "proxy_audio_packet_count": len(proxy_audio_packets),
        "maximum_video_cadence_error_us": max(cadence_errors, default=0),
        "cadence_tolerance_us": cadence_tolerance_us,
        "video_start_error_us": video_start_error_us,
        "proxy_video_end_aligned_us": proxy_video_end_aligned_us,
        "video_end_error_us": video_end_error_us,
        "audio_start_error_us": audio_start_error_us,
        "audio_end_error_us": audio_end_error_us,
        "duration_error_us": duration_error_us,
        "alignment_tolerance_us": alignment_tolerance_us,
        "verification_basis": "observed proxy frame and audio-packet PTS after filtering, encoding, and muxing, plus canonical-grid frame cardinality",
    }


def preprocess_video(
    input_path: str | Path,
    run_dir: str | Path,
    *,
    config: AnalysisConfig,
    audio_path: str | Path | None = None,
    audio_offset_us: int = 0,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    include_audio: bool = True,
    overwrite: bool = False,
    resume: bool = False,
    source_rights: Mapping[str, Any] | None = None,
    sidecar_audio_rights: Mapping[str, Any] | None = None,
    start_us: int = 0,
    end_us: int | None = None,
) -> dict[str, Any]:
    """Create a downsampled proxy and timestamp-rich preprocessing manifest."""
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    sidecar_audio = Path(audio_path) if audio_path is not None else None
    effective_source_rights = (
        validate_rights(source_rights)
        if source_rights is not None
        else classify_media_rights(source, restricted_roots=repository_restricted_roots())
    )
    if sidecar_audio is not None:
        if not include_audio:
            raise ValueError("--audio cannot be combined with --no-audio")
        if not sidecar_audio.is_file():
            raise FileNotFoundError(sidecar_audio)
        if sidecar_audio.resolve() == source.resolve():
            raise ValueError("sidecar audio must be a distinct source file")
        effective_audio_rights = (
            validate_rights(sidecar_audio_rights)
            if sidecar_audio_rights is not None
            else classify_media_rights(
                sidecar_audio, restricted_roots=repository_restricted_roots()
            )
        )
    elif sidecar_audio_rights is not None:
        raise ValueError("sidecar_audio_rights requires an explicit sidecar audio input")
    else:
        effective_audio_rights = None
    derived_rights = combine_rights(
        effective_source_rights,
        *([effective_audio_rights] if effective_audio_rights is not None else []),
    )
    if type(audio_offset_us) is not int:
        raise ValueError("audio_offset_us must be an integer")
    if sidecar_audio is None and audio_offset_us != 0:
        raise ValueError("audio_offset_us requires an explicit sidecar audio input")
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy = output_dir / "model-proxy.mp4"
    manifest_path = output_dir / "preprocessing.json"
    proxy_rights_path = rights_sidecar_path(proxy)
    owned_output_names = {
        "model-proxy.mp4",
        "model-proxy.mp4.rights.json",
        "preprocessing.json",
        "source.json",
        "profile.json",
        "frame-evidence.json",
        "candidates.json",
        "prompt-envelopes.json",
        "window-responses.json",
        "candidate-assessments.json",
        "activity-decisions.json",
        "annotations.json",
        "annotations.eaf",
        "run.json",
    }
    source_resolved = source.resolve()
    if any((source_resolved == (output_dir / name).resolve() for name in owned_output_names)):
        raise ValueError("source media may not be a pipeline-owned output path")
    if sidecar_audio is not None and any(
        (sidecar_audio.resolve() == (output_dir / name).resolve() for name in owned_output_names)
    ):
        raise ValueError("sidecar audio may not be a pipeline-owned output path")
    existing_stage_artifacts = [
        path for path in (proxy, manifest_path, proxy_rights_path) if path.exists()
    ]
    if existing_stage_artifacts and (not overwrite):
        raise FileExistsError(existing_stage_artifacts[0])
    if resume and sidecar_audio is None:
        from .resume_media import restore_proxy
        restored = restore_proxy(output_dir, source, config=config,
            rights=effective_source_rights, include_audio=include_audio,
            start_us=start_us, end_us=end_us)
        if restored is not None:
            return restored
    initial_stat = source.stat()
    initial_digest = file_sha256(source)
    after_initial_hash = source.stat()
    if (
        after_initial_hash.st_size != initial_stat.st_size
        or after_initial_hash.st_mtime_ns != initial_stat.st_mtime_ns
    ):
        raise MediaRuntimeError("source media changed while its initial identity was read")
    sidecar_initial_stat = sidecar_audio.stat() if sidecar_audio is not None else None
    sidecar_initial_digest = file_sha256(sidecar_audio) if sidecar_audio is not None else None
    if sidecar_audio is not None:
        sidecar_after_hash = sidecar_audio.stat()
        assert sidecar_initial_stat is not None
        if (
            sidecar_after_hash.st_size != sidecar_initial_stat.st_size
            or sidecar_after_hash.st_mtime_ns != sidecar_initial_stat.st_mtime_ns
        ):
            raise MediaRuntimeError("sidecar audio changed while its initial identity was read")
    probe = probe_media(source, ffprobe=ffprobe)
    summary = _safe_probe_summary(probe, include_audio=include_audio and sidecar_audio is None)
    frame_probe = probe_video_frames(source, ffprobe=ffprobe)
    if sidecar_audio is not None:
        sidecar_probe = probe_media(sidecar_audio, ffprobe=ffprobe)
        summary["streams"].append(
            _sidecar_audio_stream(sidecar_probe, alignment_offset_us=audio_offset_us)
        )
        packet_probe = probe_audio_packets(sidecar_audio, ffprobe=ffprobe)
    else:
        packet_probe = None
    has_audio = any((item["codec_type"] == "audio" for item in summary["streams"]))
    if packet_probe is None:
        packet_probe = (
            probe_audio_packets(source, ffprobe=ffprobe) if has_audio else {"packets": []}
        )
    if sidecar_audio is not None:
        summary = _apply_sidecar_timeline(
            summary, frame_probe, packet_probe, alignment_offset_us=audio_offset_us
        )
    else:
        summary = _apply_actual_timeline(summary, frame_probe, packet_probe)
    from .processing_scope import processing_scope, apply_scope

    scope = (
        processing_scope(
            summary["duration_us"], start_us, end_us, period_us=config.sample_period_us
        )
        if start_us or end_us is not None
        else None
    )
    executable = _resolve_executable(ffmpeg)
    temporary_proxy = output_dir / f".model-proxy.{uuid4().hex}.tmp.mp4"
    temporary_manifest = output_dir / f".preprocessing.{uuid4().hex}.tmp.json"
    command = build_ffmpeg_command(
        source,
        temporary_proxy,
        ffmpeg=executable,
        sample_rate_hz=config.sample_rate_hz,
        maximum_width=config.maximum_proxy_width,
        include_audio=has_audio,
        audio_input_path=sidecar_audio,
        audio_offset_us=audio_offset_us,
        overwrite=False,
    )
    if scope is not None:
        command[-1:-1] = [
            "-ss",
            _format_microseconds_as_seconds(scope["start_us"]),
            "-t",
            _format_microseconds_as_seconds(scope["end_us"] - scope["start_us"]),
        ]
    try:
        subprocess.run(command, check=True)
        if not temporary_proxy.is_file() or temporary_proxy.stat().st_size == 0:
            raise MediaRuntimeError("proxy generation produced no usable file")
        proxy_probe = probe_media(temporary_proxy, ffprobe=ffprobe)
        proxy_summary = _safe_probe_summary(proxy_probe, include_audio=has_audio)
        proxy_frame_probe = probe_video_frames(temporary_proxy, ffprobe=ffprobe)
        proxy_packet_probe = (
            probe_audio_packets(temporary_proxy, ffprobe=ffprobe) if has_audio else {"packets": []}
        )
        if has_audio and (
            not any((item["codec_type"] == "audio" for item in proxy_summary["streams"]))
        ):
            raise MediaRuntimeError("generated proxy unexpectedly omitted source audio")
    except (OSError, subprocess.CalledProcessError) as exc:
        temporary_proxy.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise MediaRuntimeError(f"proxy generation failed: {exc}") from exc
    except Exception:
        temporary_proxy.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        rights_sidecar_path(temporary_proxy).unlink(missing_ok=True)
        raise
    try:
        final_digest = file_sha256(source)
        final_stat = source.stat()
        if (
            final_digest != initial_digest
            or final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
        ):
            raise MediaRuntimeError(
                "source media changed during probing or proxy generation; staged output was discarded"
            )
        if sidecar_audio is not None:
            assert sidecar_initial_stat is not None
            final_sidecar_digest = file_sha256(sidecar_audio)
            final_sidecar_stat = sidecar_audio.stat()
            if (
                final_sidecar_digest != sidecar_initial_digest
                or final_sidecar_stat.st_size != sidecar_initial_stat.st_size
                or final_sidecar_stat.st_mtime_ns != sidecar_initial_stat.st_mtime_ns
            ):
                raise MediaRuntimeError(
                    "sidecar audio changed during probing or proxy generation; staged output was discarded"
                )
        digest = initial_digest
        source_frames, grid = _map_source_frames(
            frame_probe, summary, summary["duration_us"], config.sample_rate_hz
        )
        audio_packets = _map_audio_packets(packet_probe, summary)
        verification_frames, verification_audio = (source_frames, audio_packets)
        verification_duration = summary["duration_us"]
        if scope is not None:
            grid = apply_scope(grid, scope)
            from .processing_scope import OUTSIDE

            source_frames = [
                dict(row, processing_status=OUTSIDE)
                if row["canonical_end_us"] <= scope["start_us"]
                or row["canonical_time_us"] >= scope["end_us"]
                else row
                for row in source_frames
            ]

            def scoped_timing(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
                return [
                    dict(
                        row,
                        canonical_time_us=max(0, row["canonical_time_us"] - scope["start_us"]),
                        canonical_end_us=min(scope["end_us"], row["canonical_end_us"])
                        - scope["start_us"],
                    )
                    for row in rows
                    if row["canonical_time_us"] < scope["end_us"]
                    and row["canonical_end_us"] > scope["start_us"]
                ]

            verification_frames = scoped_timing(source_frames)
            verification_audio = scoped_timing(audio_packets)
            verification_duration = scope["end_us"] - scope["start_us"]
        timeline_verification = _verify_proxy_timeline(
            proxy_summary,
            proxy_frame_probe,
            proxy_packet_probe,
            verification_frames,
            verification_audio,
            source_duration_us=verification_duration,
            sample_rate_hz=config.sample_rate_hz,
        )
        manifest = {
            "schema_version": "0.1.0",
            **({"processing_scope": scope} if scope is not None else {}),
            "source": {
                "source_id": f"sha256:{digest[:16]}",
                "sha256": digest,
                "byte_size": initial_stat.st_size,
                "container_suffix": source.suffix.lower(),
                "rights": effective_source_rights,
                **summary,
                "video_frames": source_frames,
                "audio_packets": audio_packets,
                "sidecar_audio": {
                    "source_id": f"sha256:{sidecar_initial_digest[:16]}",
                    "sha256": sidecar_initial_digest,
                    "byte_size": sidecar_initial_stat.st_size,
                    "container_suffix": sidecar_audio.suffix.lower(),
                    "alignment_offset_us": audio_offset_us,
                    "rights": effective_audio_rights,
                }
                if sidecar_audio is not None
                and sidecar_initial_digest is not None
                and (sidecar_initial_stat is not None)
                else None,
            },
            "proxy": {
                "relative_path": proxy.name,
                "sampling_rate_hz": config.sample_rate_hz,
                "sampling_method": "FFmpeg presentation-timestamp-aware fps filter",
                "maximum_width": config.maximum_proxy_width,
                "width": next(
                    (
                        item["width"]
                        for item in proxy_summary["streams"]
                        if item["codec_type"] == "video"
                    )
                ),
                "height": next(
                    (
                        item["height"]
                        for item in proxy_summary["streams"]
                        if item["codec_type"] == "video"
                    )
                ),
                "resize_policy": "preserve aspect ratio; never upscale",
                "video_codec": "H.264",
                "sha256": file_sha256(temporary_proxy),
                "byte_size": temporary_proxy.stat().st_size,
                "rights": derived_rights,
                "audio_requested": include_audio,
                "audio_supplied": has_audio,
                "audio_source": "sidecar"
                if sidecar_audio is not None
                else "embedded"
                if has_audio
                else None,
                "audio_alignment_offset_us": audio_offset_us if sidecar_audio is not None else 0,
                "audio_codec": "AAC" if has_audio else None,
                "frame_mapping": grid,
                "frame_mapping_semantics": "decoded source-frame coverage overlapping each canonical 10 Hz sample; not an assertion of the FFmpeg fps filter's chosen source frame",
                "timeline_verification": timeline_verification,
                **(
                    {"processing_scope": scope, "proxy_time_origin_us": scope["start_us"]}
                    if scope is not None
                    else {}
                ),
            },
            "runtime": {
                "adapter": "FFmpeg command-line runtime",
                "version_banner": _runtime_banner(executable),
                "command_paths_recorded": False,
                "video_filter": command[command.index("-vf") + 1],
                "audio_filter": command[command.index("-af") + 1] if "-af" in command else None,
                "av_offset_policy": "normalize independent video/sidecar starts, apply configured offset, and shift non-negative"
                if sidecar_audio is not None
                else "copy embedded source timestamps and shift selected streams by one global origin",
            },
        }
        from ..export.json_export import write_json

        write_json(temporary_manifest, manifest)
        temporary_rights = write_rights_sidecar(temporary_proxy, derived_rights, overwrite=False)
        _commit_preprocessing_pair(
            temporary_proxy,
            proxy,
            temporary_manifest,
            manifest_path,
            temporary_rights,
            proxy_rights_path,
        )
        return manifest
    except Exception:
        temporary_proxy.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
