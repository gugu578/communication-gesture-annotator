# SPDX-License-Identifier: Apache-2.0
"""Validated, byte-preserving reuse of a completed preprocessing bundle."""

from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping, Sequence
from uuid import uuid4
from ..config import AnalysisConfig, InferenceConfig, PosePreprocessingConfig
from ..domain import AnalysisSample
from ..export.json_export import write_json
from ..storage import data_path, overlay_omitted
from ..rights import (
    RIGHTS_CONTRACT_VERSION,
    combine_rights,
    read_rights_sidecar,
    rights_sidecar_path,
    validate_rights,
)
from .ffmpeg import file_sha256
from .pose import PosePreprocessingResult, run_pose_preprocessing

PREPROCESSING_REUSE_CONTRACT_VERSION = "0.1.0"
_PREPROCESSING_SCHEMA_VERSION = "0.1.0"
_FRAME_MAPPING_SEMANTICS = "decoded source-frame coverage overlapping each canonical 10 Hz sample; not an assertion of the FFmpeg fps filter's chosen source frame"


@dataclass(frozen=True, slots=True)
class ReusedPreprocessingResult:
    """A locally validated preprocessing bundle ready for a fresh model run."""

    manifest: Mapping[str, Any]
    samples: tuple[AnalysisSample, ...]
    frame_records: tuple[Mapping[str, Any], ...]
    preprocessing_path: Path
    proxy_path: Path
    proxy_rights_path: Path
    pose_result: PosePreprocessingResult
    reuse_manifest_path: Path
    source_bundle_sha256: str


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("preprocessing reuse JSON contains duplicate object keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"preprocessing reuse JSON contains non-finite number {value!r}")


def _read_strict_json(path: Path) -> tuple[Any, str]:
    try:
        payload = path.read_bytes()
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("preprocessing reuse artifact is not strict UTF-8 JSON") from exc
    return (document, sha256(payload).hexdigest())


def _regular_file(path: Path, label: str) -> Path:
    is_junction = getattr(path, "is_junction", lambda: False)()
    if not path.is_file() or path.is_symlink() or is_junction:
        raise ValueError(f"preprocessing reuse {label} must be a regular file")
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse_flag and attributes & reparse_flag:
        raise ValueError(f"preprocessing reuse {label} may not be a reparse point")
    return path


def _validate_run_root(path: Path, label: str) -> Path:
    resolved = path.resolve()
    _regular_directory(path, label)
    return resolved


def _regular_directory(path: Path, label: str) -> Path:
    is_junction = getattr(path, "is_junction", lambda: False)()
    if not path.is_dir() or path.is_symlink() or is_junction:
        raise ValueError(f"preprocessing reuse {label} must be a regular directory")
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse_flag and attributes & reparse_flag:
        raise ValueError(f"preprocessing reuse {label} may not be a reparse point")
    return path


def _artifact_record(path: Path, relative_path: str) -> dict[str, Any]:
    _regular_file(path, relative_path)
    return {
        "relative_path": relative_path,
        "byte_size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _contained_file(root: Path, relative_path: str, label: str) -> Path:
    path = root / relative_path
    _regular_file(path, label)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"preprocessing reuse {label} escapes its run directory")
    return path


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _validate_digest_and_size(path: Path, record: Mapping[str, Any], label: str) -> dict[str, Any]:
    _regular_file(path, label)
    expected_digest = record.get("sha256")
    expected_size = record.get("byte_size")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or type(expected_size) is not int
        or (expected_size <= 0)
        or (path.stat().st_size != expected_size)
        or (file_sha256(path) != expected_digest)
    ):
        raise ValueError(f"preprocessing reuse {label} does not match its manifest")
    return {"relative_path": label, "byte_size": expected_size, "sha256": expected_digest}


def _samples_from_manifest(proxy: Mapping[str, Any]) -> tuple[AnalysisSample, ...]:
    mapping = proxy.get("frame_mapping")
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("preprocessing reuse frame mapping is missing")
    samples: list[AnalysisSample] = []
    for position, item in enumerate(mapping):
        if not isinstance(item, Mapping) or set(item) != {
            "index",
            "start_us",
            "center_us",
            "end_us",
            "source_references",
            "source_gap_reason",
        }:
            raise ValueError("preprocessing reuse frame mapping has an invalid record")
        references = item.get("source_references")
        if not isinstance(references, list) or any(
            (not isinstance(value, str) for value in references)
        ):
            raise ValueError("preprocessing reuse source references are invalid")
        sample = AnalysisSample(
            index=item.get("index"),
            start_us=item.get("start_us"),
            center_us=item.get("center_us"),
            end_us=item.get("end_us"),
            source_references=tuple(references),
            source_gap_reason=item.get("source_gap_reason"),
        )
        if sample.index != position or sample.start_us != position * 100000:
            raise ValueError("preprocessing reuse requires the fixed ordered 10 Hz grid")
        if position < len(mapping) - 1 and sample.end_us != (position + 1) * 100000:
            raise ValueError("preprocessing reuse requires contiguous 100 ms samples")
        if position and sample.start_us != samples[-1].end_us:
            raise ValueError("preprocessing reuse sample grid is not contiguous")
        samples.append(sample)
    return tuple(samples)


def _validate_proxy_verification(
    document: Mapping[str, Any], proxy: Mapping[str, Any], *, sample_count: int, sample_rate_hz: int
) -> None:
    """Require the timing evidence produced by the current proxy contract."""
    runtime = document.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "adapter",
            "version_banner",
            "command_paths_recorded",
            "video_filter",
            "audio_filter",
            "av_offset_policy",
        }
        or runtime.get("adapter") != "FFmpeg command-line runtime"
        or (runtime.get("command_paths_recorded") is not False)
        or (not isinstance(runtime.get("version_banner"), str))
        or (not runtime["version_banner"])
        or (not isinstance(runtime.get("video_filter"), str))
        or (not runtime["video_filter"])
        or (not isinstance(runtime.get("av_offset_policy"), str))
        or (not runtime["av_offset_policy"])
        or (
            runtime.get("audio_filter") is not None
            and (not isinstance(runtime.get("audio_filter"), str))
        )
    ):
        raise ValueError("reused preprocessing lacks its verified FFmpeg runtime record")
    verification = proxy.get("timeline_verification")
    period_us = 1000000 // sample_rate_hz
    if (
        proxy.get("frame_mapping_semantics") != _FRAME_MAPPING_SEMANTICS
        or not isinstance(verification, Mapping)
        or verification.get("status") != "verified_within_declared_tolerance"
        or (verification.get("sample_rate_hz") != sample_rate_hz)
        or (verification.get("expected_analysis_sample_count") != sample_count)
        or (type(verification.get("proxy_video_frame_count")) is not int)
        or (type(verification.get("video_frame_count_delta")) is not int)
        or (verification["video_frame_count_delta"] not in {-1, 0, 1})
        or (
            verification["proxy_video_frame_count"] - sample_count
            != verification["video_frame_count_delta"]
        )
        or (
            verification.get("frame_count_adjustment")
            not in {
                "none",
                "clone_previous_visual_for_nonvisual_terminal_sample",
                "drop_proxy_frame_starting_after_canonical_end",
            }
        )
        or (type(verification.get("cadence_tolerance_us")) is not int)
        or (verification["cadence_tolerance_us"] < 0)
        or (type(verification.get("maximum_video_cadence_error_us")) is not int)
        or (verification["maximum_video_cadence_error_us"] < 0)
        or (verification["maximum_video_cadence_error_us"] > verification["cadence_tolerance_us"])
        or (verification.get("alignment_tolerance_us") != period_us)
        or any(
            (
                type(verification.get(name)) is not int or abs(verification[name]) > period_us
                for name in ("video_start_error_us", "video_end_error_us", "duration_error_us")
            )
        )
        or (not isinstance(verification.get("verification_basis"), str))
        or (not verification["verification_basis"])
    ):
        raise ValueError("reused proxy lacks valid recorded timeline verification")


def _validate_audio_identity(
    source: Mapping[str, Any],
    proxy: Mapping[str, Any],
    *,
    audio_path: Path | None,
    audio_offset_us: int,
    include_audio: bool,
    sidecar_audio_rights: Mapping[str, Any] | None,
) -> None:
    if type(audio_offset_us) is not int:
        raise ValueError("audio_offset_us must be an integer")
    if proxy.get("audio_requested") is not include_audio:
        raise ValueError("reused preprocessing used a different audio mode")
    sidecar = source.get("sidecar_audio")
    if audio_path is None:
        if (
            sidecar is not None
            or audio_offset_us != 0
            or sidecar_audio_rights is not None
            or (proxy.get("audio_source") == "sidecar")
            or (proxy.get("audio_alignment_offset_us") != 0)
        ):
            raise ValueError("reused preprocessing used a different sidecar-audio input")
        return
    if not include_audio:
        raise ValueError("--audio cannot be combined with --no-audio")
    if not isinstance(sidecar, Mapping):
        raise ValueError("reused preprocessing did not use the requested sidecar audio")
    audio = _regular_file(audio_path.resolve(), "sidecar audio")
    expected_rights = validate_rights(sidecar_audio_rights)
    if (
        sidecar.get("sha256") != file_sha256(audio)
        or sidecar.get("byte_size") != audio.stat().st_size
        or sidecar.get("alignment_offset_us") != audio_offset_us
        or (validate_rights(sidecar.get("rights")) != expected_rights)
        or (proxy.get("audio_source") != "sidecar")
        or (proxy.get("audio_alignment_offset_us") != audio_offset_us)
    ):
        raise ValueError("sidecar audio does not match reused preprocessing provenance")


def _validate_analysis_frames(
    root: Path, samples: Sequence[AnalysisSample], proxy_rights: Mapping[str, Any]
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any], list[dict[str, Any]]]:
    directory = root / "analysis-frames"
    _regular_directory(directory, "analysis-frames directory")
    manifest_path = _contained_file(
        root, "analysis-frames/rights-manifest.json", "analysis-frame rights manifest"
    )
    rights_document, rights_digest = _read_strict_json(manifest_path)
    if not isinstance(rights_document, Mapping):
        raise ValueError("analysis-frame rights manifest must be an object")
    artifacts = rights_document.get("artifacts")
    if (
        rights_document.get("schema_version") != RIGHTS_CONTRACT_VERSION
        or validate_rights(rights_document.get("rights")) != validate_rights(proxy_rights)
        or (not isinstance(artifacts, list))
    ):
        raise ValueError("analysis-frame rights manifest has incompatible rights")
    expected_names = [f"sample-{sample.index:08d}.jpg" for sample in samples]
    records: dict[str, str] = {}
    for item in artifacts:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"filename", "sha256"}
            or (not isinstance(item.get("filename"), str))
            or (not isinstance(item.get("sha256"), str))
            or (item["filename"] in records)
        ):
            raise ValueError("analysis-frame rights manifest has invalid records")
        records[item["filename"]] = item["sha256"]
    if list(records) != expected_names:
        raise ValueError("analysis-frame rights manifest does not match the sample grid")
    actual_names = sorted((path.name for path in directory.glob("sample-*.jpg")))
    if actual_names != expected_names:
        raise ValueError("analysis-frame files do not exactly match the sample grid")
    frame_records: list[Mapping[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for sample, name in zip(samples, expected_names, strict=True):
        path = _contained_file(root, f"analysis-frames/{name}", "analysis frame")
        digest = file_sha256(path)
        if records[name] != digest or len(digest) != 64 or path.stat().st_size <= 0:
            raise ValueError("analysis frame does not match its rights manifest")
        frame_records.append(
            {
                "sample_index": sample.index,
                "start_us": sample.start_us,
                "end_us": sample.end_us,
                "path": str(path.resolve()),
                "sha256": digest,
                **(
                    {"source_gap_reason": sample.source_gap_reason}
                    if sample.source_gap_reason is not None
                    else {}
                ),
                "rights": validate_rights(proxy_rights),
            }
        )
        identities.append({"filename": name, "byte_size": path.stat().st_size, "sha256": digest})
    return (
        tuple(frame_records),
        {
            "relative_path": "analysis-frames/rights-manifest.json",
            "byte_size": manifest_path.stat().st_size,
            "sha256": rights_digest,
        },
        [
            {
                "relative_path": f"analysis-frames/{item['filename']}",
                "byte_size": item["byte_size"],
                "sha256": item["sha256"],
            }
            for item in identities
        ],
    )


def _validate_bundle(
    root: Path,
    *,
    source_path: Path,
    analysis_config: AnalysisConfig,
    inference_config: InferenceConfig,
    pose_config: PosePreprocessingConfig,
    source_rights: Mapping[str, Any],
    audio_path: Path | None,
    audio_offset_us: int,
    include_audio: bool,
    sidecar_audio_rights: Mapping[str, Any] | None,
    ffmpeg: str,
) -> tuple[
    Mapping[str, Any],
    tuple[AnalysisSample, ...],
    tuple[Mapping[str, Any], ...],
    PosePreprocessingResult,
    list[dict[str, Any]],
    str,
]:
    preprocessing_path = _contained_file(root, "preprocessing.json", "preprocessing.json")
    document, preprocessing_digest = _read_strict_json(preprocessing_path)
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != _PREPROCESSING_SCHEMA_VERSION
    ):
        raise ValueError("reused preprocessing.json has an unsupported schema_version")
    source = document.get("source")
    proxy = document.get("proxy")
    if not isinstance(source, Mapping) or not isinstance(proxy, Mapping):
        raise ValueError("reused preprocessing.json is incomplete")
    video = _regular_file(source_path.resolve(), "source video")
    expected_source_rights = validate_rights(source_rights)
    if (
        source.get("sha256") != file_sha256(video)
        or source.get("byte_size") != video.stat().st_size
        or validate_rights(source.get("rights")) != expected_source_rights
    ):
        raise ValueError("source video does not match reused preprocessing provenance")
    _validate_audio_identity(
        source,
        proxy,
        audio_path=audio_path,
        audio_offset_us=audio_offset_us,
        include_audio=include_audio,
        sidecar_audio_rights=sidecar_audio_rights,
    )
    combined_rights = combine_rights(
        expected_source_rights,
        *([validate_rights(sidecar_audio_rights)] if sidecar_audio_rights is not None else []),
    )
    if (
        proxy.get("relative_path") != "model-proxy.mp4"
        or type(proxy.get("audio_requested")) is not bool
        or type(proxy.get("audio_supplied")) is not bool
        or (proxy.get("sampling_rate_hz") != analysis_config.sample_rate_hz)
        or (proxy.get("maximum_width") != analysis_config.maximum_proxy_width)
        or (validate_rights(proxy.get("rights")) != combined_rights)
    ):
        raise ValueError("reused proxy is incompatible with the current analysis configuration")
    proxy_path = _contained_file(root, "model-proxy.mp4", "model-proxy.mp4")
    proxy_record = _validate_digest_and_size(proxy_path, proxy, "model-proxy.mp4")
    proxy_rights_path = _contained_file(
        root, "model-proxy.mp4.rights.json", "model-proxy.mp4.rights.json"
    )
    if read_rights_sidecar(proxy_path) != combined_rights:
        raise ValueError("reused proxy rights sidecar is missing or stale")
    proxy_rights_record = _artifact_record(proxy_rights_path, "model-proxy.mp4.rights.json")
    samples = _samples_from_manifest(proxy)
    _validate_proxy_verification(
        document, proxy, sample_count=len(samples), sample_rate_hz=analysis_config.sample_rate_hz
    )
    if samples[-1].end_us != source.get("duration_us"):
        raise ValueError("reused analysis grid does not cover the source duration")
    frames, frame_rights_record, analysis_frame_records = _validate_analysis_frames(
        root, samples, combined_rights
    )
    source_frames = source.get("video_frames")
    if not isinstance(source_frames, list):
        raise ValueError("reused preprocessing lacks the source-frame timeline")
    _regular_directory(root / "pose", "pose directory")
    pose_result = run_pose_preprocessing(
        video,
        source_frames,
        samples,
        root,
        config=pose_config,
        rights=expected_source_rights,
        expected_source_sha256=source["sha256"],
        expected_source_byte_size=source["byte_size"],
        preprocessing_sha256=preprocessing_digest,
        ffmpeg=ffmpeg,
        overwrite=False,
        resume=True,
    )
    pose_artifacts = pose_result.manifest.get("artifacts")
    expected_pose_paths = {
        "observations": "pose/" + pose_result.observations_path.name,
        "observations_rights": "pose/" + pose_result.observations_rights_path.name,
        "overlay": "pose/overlay.mp4",
        "overlay_rights": "pose/overlay.mp4.rights.json",
    }
    if not isinstance(pose_artifacts, Mapping) or set(pose_artifacts) != set(expected_pose_paths):
        raise ValueError("raw pose manifest has an unsupported artifact set")
    for name, relative_path in expected_pose_paths.items():
        record = pose_artifacts.get(name)
        if (
            name in {"overlay", "overlay_rights"}
            and overlay_omitted(pose_result.manifest)
            and (record is None)
        ):
            continue
        if not isinstance(record, Mapping) or record.get("relative_path") != relative_path:
            raise ValueError("raw pose manifest contains a noncanonical artifact path")
    raw_records = [
        _artifact_record(_contained_file(root, relative, relative), relative)
        for relative in [
            "pose/manifest.json",
            *(
                relative
                for name, relative in expected_pose_paths.items()
                if pose_artifacts[name] is not None
            ),
        ]
    ]
    records = [
        _artifact_record(preprocessing_path, "preprocessing.json"),
        proxy_record,
        proxy_rights_record,
        frame_rights_record,
        *analysis_frame_records,
        *raw_records,
    ]
    frame_set_digest = _canonical_sha256(
        [
            {
                "filename": Path(str(record["relative_path"])).name,
                "byte_size": record["byte_size"],
                "sha256": record["sha256"],
            }
            for record in analysis_frame_records
        ]
    )
    bundle_digest = _canonical_sha256(
        {
            "records": records,
            "analysis_frame_count": len(frames),
            "analysis_frame_set_sha256": frame_set_digest,
        }
    )
    if inference_config.frame_width < 2 or not 1 <= inference_config.jpeg_quality <= 100:
        raise ValueError("current analysis-frame configuration is invalid")
    return (document, samples, frames, pose_result, records, bundle_digest)


def _copy_regular(source: Path, target: Path) -> None:
    _regular_file(source, str(source.name))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)
    if target.is_symlink() or not target.is_file():
        raise ValueError("preprocessing reuse copy did not create a regular file")
    if source.stat().st_size != target.stat().st_size or file_sha256(source) != file_sha256(target):
        raise ValueError("preprocessing reuse copy changed artifact bytes")


def _validate_fresh_target(path: Path, *, allow_preflight_artifacts: bool = False) -> bool:
    """Allow only a fresh directory plus this invocation's synthetic preflight trace."""
    if not path.exists():
        return False
    is_junction = getattr(path, "is_junction", lambda: False)()
    if not path.is_dir() or path.is_symlink() or is_junction:
        raise FileExistsError(path)
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse_flag and attributes & reparse_flag:
        raise ValueError("preprocessing reuse destination may not be a reparse point")
    allowed = {"run-events.jsonl", "model-traces"}
    children = list(path.iterdir())
    if children and (not allow_preflight_artifacts):
        raise FileExistsError(path)
    if any((child.name not in allowed for child in children)):
        raise FileExistsError(path)
    event_log = path / "run-events.jsonl"
    if event_log.exists():
        _regular_file(event_log, "destination preflight event log")
    trace_root = path / "model-traces"
    if trace_root.exists():
        _regular_directory(trace_root, "destination preflight model-traces")
    return True


def _commit_staged_bundle(
    staging: Path, target: Path, *, merge: bool, allow_preflight_artifacts: bool
) -> None:
    if not merge:
        staging.replace(target)
        return
    _validate_fresh_target(target, allow_preflight_artifacts=allow_preflight_artifacts)
    committed: list[tuple[Path, Path]] = []
    try:
        for source in sorted(staging.iterdir(), key=lambda item: item.name):
            destination = target / source.name
            if destination.exists():
                raise FileExistsError(destination)
            source.replace(destination)
            committed.append((source, destination))
    except Exception:
        for source, destination in reversed(committed):
            if destination.exists():
                destination.replace(source)
        raise


def _remap_result(
    root: Path,
    manifest: Mapping[str, Any],
    samples: tuple[AnalysisSample, ...],
    frames: tuple[Mapping[str, Any], ...],
    pose_manifest: Mapping[str, Any],
    source_bundle_sha256: str,
) -> ReusedPreprocessingResult:
    frame_records = tuple(
        (
            {
                **dict(frame),
                "path": str(
                    (root / "analysis-frames" / f"sample-{sample.index:08d}.jpg").resolve()
                ),
            }
            for sample, frame in zip(samples, frames, strict=True)
        )
    )
    pose_directory = root / "pose"
    pose_result = PosePreprocessingResult(
        manifest=dict(pose_manifest),
        directory=pose_directory,
        manifest_path=pose_directory / "manifest.json",
        observations_path=data_path(pose_directory / "observations.jsonl"),
        observations_rights_path=rights_sidecar_path(
            data_path(pose_directory / "observations.jsonl")
        ),
        overlay_path=pose_directory / "overlay.mp4",
        reused=True,
    )
    return ReusedPreprocessingResult(
        manifest=dict(manifest),
        samples=samples,
        frame_records=frame_records,
        preprocessing_path=root / "preprocessing.json",
        proxy_path=root / "model-proxy.mp4",
        proxy_rights_path=root / "model-proxy.mp4.rights.json",
        pose_result=pose_result,
        reuse_manifest_path=root / "preprocessing-reuse.json",
        source_bundle_sha256=source_bundle_sha256,
    )


def reuse_preprocessing(
    source_run_dir: str | Path,
    target_run_dir: str | Path,
    *,
    source_path: str | Path,
    analysis_config: AnalysisConfig,
    inference_config: InferenceConfig,
    pose_config: PosePreprocessingConfig,
    source_rights: Mapping[str, Any],
    audio_path: str | Path | None = None,
    audio_offset_us: int = 0,
    include_audio: bool = True,
    sidecar_audio_rights: Mapping[str, Any] | None = None,
    ffmpeg: str = "ffmpeg",
    allow_preflight_artifacts: bool = False,
) -> ReusedPreprocessingResult:
    """Copy only validated non-model preprocessing into a fresh run directory."""
    source_root = _validate_run_root(Path(source_run_dir), "source run directory")
    target = Path(target_run_dir).resolve()
    if (
        source_root == target
        or source_root.is_relative_to(target)
        or target.is_relative_to(source_root)
    ):
        raise ValueError("preprocessing source and destination runs must be disjoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    if type(allow_preflight_artifacts) is not bool:
        raise ValueError("allow_preflight_artifacts must be Boolean")
    merge_into_preflight = _validate_fresh_target(
        target, allow_preflight_artifacts=allow_preflight_artifacts
    )
    manifest, samples, frames, pose_result, records, bundle_digest = _validate_bundle(
        source_root,
        source_path=Path(source_path),
        analysis_config=analysis_config,
        inference_config=inference_config,
        pose_config=pose_config,
        source_rights=source_rights,
        audio_path=Path(audio_path) if audio_path is not None else None,
        audio_offset_us=audio_offset_us,
        include_audio=include_audio,
        sidecar_audio_rights=sidecar_audio_rights,
        ffmpeg=ffmpeg,
    )
    staging = target.parent / f".{target.name}.preprocessing-reuse.{uuid4().hex}.tmp"
    staging.mkdir()
    try:
        for record in records:
            relative = record["relative_path"]
            _copy_regular(source_root / relative, staging / relative)
        from .smoothing_cache import copy_smoothing_caches

        smoothing_caches = copy_smoothing_caches(
            source_root / "pose",
            staging / "pose",
            file_sha256(source_root / "pose" / "manifest.json"),
            pose_result.manifest.get("rights", source_rights),
        )
        source_status = source_root / "pipeline-status.json"
        source_status_record = (
            _artifact_record(source_status, "pipeline-status.json")
            if source_status.is_file() and (not source_status.is_symlink())
            else None
        )
        write_json(
            staging / "preprocessing-reuse.json",
            {
                "schema_version": PREPROCESSING_REUSE_CONTRACT_VERSION,
                "status": "completed",
                "copy_policy": "validated-preprocessing-only-v0.1",
                "source_bundle_sha256": bundle_digest,
                "copied_artifacts": records,
                "smoothing_caches": smoothing_caches,
                "analysis_frames": {
                    "count": len(frames),
                    "set_sha256": _canonical_sha256(
                        [
                            {
                                "filename": Path(str(record["relative_path"])).name,
                                "byte_size": record["byte_size"],
                                "sha256": record["sha256"],
                            }
                            for record in records
                            if str(record["relative_path"]).startswith("analysis-frames/sample-")
                        ]
                    ),
                    "generation_parameters_recorded": False,
                    "requested_frame_width_not_asserted": inference_config.frame_width,
                    "requested_jpeg_quality_not_asserted": inference_config.jpeg_quality,
                },
                "source_pipeline_status": source_status_record,
                "copy_method": "independent_regular_file_byte_copy",
                "model_artifacts_copied": False,
                "excluded_artifact_classes": [
                    "pose_conditioning",
                    "model_checkpoints",
                    "model_inputs",
                    "model_traces",
                    "phase0_and_later_outputs",
                    "annotations",
                    "diagnostics",
                ],
            },
        )
        for record in records:
            staged = staging / str(record["relative_path"])
            if (
                not staged.is_file()
                or staged.stat().st_size != record["byte_size"]
                or file_sha256(staged) != record["sha256"]
            ):
                raise ValueError("staged preprocessing reuse bundle failed validation")
        source_document = manifest["source"]
        current_source = _regular_file(Path(source_path).resolve(), "source video")
        if (
            current_source.stat().st_size != source_document["byte_size"]
            or file_sha256(current_source) != source_document["sha256"]
        ):
            raise ValueError("source video changed during preprocessing reuse")
        _validate_audio_identity(
            source_document,
            manifest["proxy"],
            audio_path=Path(audio_path).resolve() if audio_path is not None else None,
            audio_offset_us=audio_offset_us,
            include_audio=include_audio,
            sidecar_audio_rights=sidecar_audio_rights,
        )
        _commit_staged_bundle(
            staging,
            target,
            merge=merge_into_preflight,
            allow_preflight_artifacts=allow_preflight_artifacts,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _remap_result(target, manifest, samples, frames, pose_result.manifest, bundle_digest)


def load_reused_preprocessing(
    run_dir: str | Path,
    *,
    source_path: str | Path,
    analysis_config: AnalysisConfig,
    inference_config: InferenceConfig,
    pose_config: PosePreprocessingConfig,
    source_rights: Mapping[str, Any],
    audio_path: str | Path | None = None,
    audio_offset_us: int = 0,
    include_audio: bool = True,
    sidecar_audio_rights: Mapping[str, Any] | None = None,
    ffmpeg: str = "ffmpeg",
) -> ReusedPreprocessingResult:
    """Revalidate a target-local copied bundle during that run's own resume."""
    root = _validate_run_root(Path(run_dir), "run directory")
    reuse_document, _ = _read_strict_json(
        _regular_file(root / "preprocessing-reuse.json", "preprocessing-reuse.json")
    )
    if (
        not isinstance(reuse_document, Mapping)
        or reuse_document.get("schema_version") != PREPROCESSING_REUSE_CONTRACT_VERSION
        or reuse_document.get("status") != "completed"
        or (reuse_document.get("model_artifacts_copied") is not False)
        or (reuse_document.get("copy_method") != "independent_regular_file_byte_copy")
    ):
        raise ValueError("preprocessing-reuse.json is incomplete or unsupported")
    manifest, samples, frames, pose_result, records, bundle_digest = _validate_bundle(
        root,
        source_path=Path(source_path),
        analysis_config=analysis_config,
        inference_config=inference_config,
        pose_config=pose_config,
        source_rights=source_rights,
        audio_path=Path(audio_path) if audio_path is not None else None,
        audio_offset_us=audio_offset_us,
        include_audio=include_audio,
        sidecar_audio_rights=sidecar_audio_rights,
        ffmpeg=ffmpeg,
    )
    analysis_frames = reuse_document.get("analysis_frames")
    frame_records = [
        record
        for record in records
        if str(record.get("relative_path", "")).startswith("analysis-frames/sample-")
    ]
    expected_frame_set_digest = _canonical_sha256(
        [
            {
                "filename": Path(str(record["relative_path"])).name,
                "byte_size": record["byte_size"],
                "sha256": record["sha256"],
            }
            for record in frame_records
        ]
    )
    if (
        reuse_document.get("copy_policy") != "validated-preprocessing-only-v0.1"
        or reuse_document.get("source_bundle_sha256") != bundle_digest
        or reuse_document.get("copied_artifacts") != records
        or (not isinstance(analysis_frames, Mapping))
        or (analysis_frames.get("count") != len(frames))
        or (analysis_frames.get("set_sha256") != expected_frame_set_digest)
    ):
        raise ValueError("preprocessing reuse bundle no longer matches its provenance")
    return _remap_result(root, manifest, samples, frames, pose_result.manifest, bundle_digest)
