# SPDX-License-Identifier: Apache-2.0
"""Staged orchestration for Research Stage 2 vision-model experiments."""

from __future__ import annotations
from ..storage import data_path, phase_result_path, release_phase1_images
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Mapping, Sequence
from ..audit.progress import StepProgress
from ..config import ResearchConfig
from ..domain import AnalysisResult, AnalysisSample, to_json_value
from ..export.json_export import read_json, read_json_snapshot, write_json
from ..media.image_packaging import (
    ContactSheetLayout,
    FrameImage,
    plan_image_packages,
    render_image_packages,
)
from ..media.pose_conditioning import PoseConditioningResult
from ..rights import is_restricted_rights, restricted_transport_mode, validate_rights
from .front_end_contracts import (
    OVERVIEW_PROMPT_VERSION,
    PHASE0A_EXEMPLAR_CONTRACT_VERSION,
    PHASE0A_EXEMPLAR_PROMPT_VERSION,
    VISIBILITY_GATE_VERSION,
    final_overview_schema,
    overview_system_prompt,
    phase0a_exemplar_schema_for_count,
    phase0a_exemplar_verification_system_prompt,
    validate_overview_response,
    validate_phase0a_exemplar_response,
)
from .llm_contracts import (
    PROFILE_EVIDENCE_SCOPE_VERSION,
    PROFILE_MAX_ROOT_EVIDENCE_INDICES,
    PROFILE_NORMALIZATION_VERSION,
    PROFILE_SEGMENT_SCHEMA,
    ModelCallResult,
    Phase2Model,
    validate_profile_evidence_scope,
)
from .request_windows import RequestWindow, fixed_period_indices, plan_index_windows
from .representatives import PositiveRepresentative, export_positive_representatives

_PROFILE_SUBJECT_ID_CANONICALIZATION_VERSION = "profile-subject-id-canonicalization-v0.1"
_PHASE0A_FINAL_STORYBOARD_POLICY_VERSION = "uniform-recording-coverage-v0.2"
_PHASE0A_EXEMPLAR_SELECTION_POLICY_VERSION = "verified-selected-live-target-temporal-strata-v0.1"
_PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION = (
    "single-image-selected-live-target-confirmation-v0.1"
)
_PHASE0A_EXEMPLAR_MINIMUM_POSITIVE_PROBABILITY = 0.8


@dataclass(frozen=True, slots=True)
class Phase2RunResult:
    """Root outcome for Milestone 1's single selected participant."""

    run_id: str | None
    target_mode: str | None
    analyses: tuple[AnalysisResult, ...]
    analysis_paths: tuple[Path, ...]
    index_path: Path | None
    completed_stage: str = "phase3"
    status_path: Path | None = None
    stage_artifact_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _SubjectRunResult:
    """One subject outcome at either the Phase 2 or Phase 3 boundary."""

    analysis: AnalysisResult | None
    analysis_path: Path | None
    phase2_evidence_path: Path


_MODEL_CHECKPOINT_SCHEMA_VERSION = "1.0"
_MODEL_CHECKPOINT_POLICY_VERSION = "validated-model-call-checkpoint-v0.1"
_MODEL_CAPABILITY_IDENTITY_POLICY_VERSION = "stable-model-capability-identity-v0.1"
_PIPELINE_PROVENANCE_VERSION = "research-stage2-pose-conditioned-v0.13"
PIPELINE_STAGE_ORDER = ("preprocessing", "phase0a", "phase0b", "phase1", "phase2", "phase3")
MODEL_PIPELINE_STOP_STAGES = PIPELINE_STAGE_ORDER[1:]
_PIPELINE_STATUS_SCHEMA_VERSION = "0.1.0"
_PIPELINE_STATUS_ARTIFACT_NAME = "pipeline-status.json"
_MODEL_RUN_INTERMEDIATE_ARTIFACT_NAMES = (
    "recording-overview.json",
    "scene-usability.json",
    "recording-profile.json",
    "model-capabilities.json",
    "phase2-evidence.json",
)
_MODEL_RUN_FINAL_ARTIFACT_NAMES = (
    "source.json",
    "profile.json",
    "frame-evidence.json",
    "candidates.json",
    "prompt-envelopes.json",
    "window-responses.json",
    "candidate-assessments.json",
    "activity-decisions.json",
    "decision-trace.json",
    "annotations.json",
    "annotations.eaf",
    "run.json",
    "inference-plan.json",
    "analysis-index.json",
)


def validate_pipeline_stage(value: str, *, include_preprocessing: bool = True) -> str:
    """Return one canonical requested stopping stage or fail closed."""
    allowed = PIPELINE_STAGE_ORDER if include_preprocessing else MODEL_PIPELINE_STOP_STAGES
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"pipeline stage must be one of: {choices}")
    return value


def _pipeline_stage_rank(value: str) -> int:
    return PIPELINE_STAGE_ORDER.index(validate_pipeline_stage(value))


def write_pipeline_status(
    run_dir: str | Path,
    *,
    requested_stop_after: str,
    completed_stage: str,
    artifacts: Sequence[str | Path] = (),
    source_sha256: str | None = None,
    preprocessing_sha256: str | None = None,
) -> Path:
    """Atomically record the highest fully completed pipeline stage.

    This is a lifecycle marker, not a cache.  Model stages always reconstruct
    their values from input-bound validated checkpoints on resume.
    """
    requested = validate_pipeline_stage(requested_stop_after)
    completed = validate_pipeline_stage(completed_stage)
    if _pipeline_stage_rank(completed) > _pipeline_stage_rank(requested):
        raise ValueError("completed pipeline stage cannot follow the requested stop stage")
    for label, digest in (
        ("source_sha256", source_sha256),
        ("preprocessing_sha256", preprocessing_sha256),
    ):
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
            raise ValueError(f"{label} must be a SHA-256 digest or null")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    status_artifacts = list(artifacts)
    artifact_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for artifact in status_artifacts:
        path = Path(artifact)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            raise ValueError("pipeline status artifacts must be files inside the run directory")
        relative_path = str(resolved.relative_to(resolved_root)).replace("\\", "/")
        if relative_path == _PIPELINE_STATUS_ARTIFACT_NAME:
            raise ValueError("pipeline status may not cite itself")
        if relative_path in seen_paths:
            raise ValueError("pipeline status artifact paths must be unique")
        seen_paths.add(relative_path)
        artifact_records.append(
            {
                "path": relative_path,
                "byte_size": resolved.stat().st_size,
                "sha256": _file_sha256(resolved),
            }
        )
    status = (
        "completed"
        if completed == "phase3"
        else "stopped_as_requested"
        if completed == requested
        else "running"
    )
    return write_json(
        root / _PIPELINE_STATUS_ARTIFACT_NAME,
        {
            "schema_version": _PIPELINE_STATUS_SCHEMA_VERSION,
            "status": status,
            "requested_stop_after": requested,
            "completed_stage": completed,
            "stage_order": list(PIPELINE_STAGE_ORDER),
            "source_sha256": source_sha256,
            "preprocessing_sha256": preprocessing_sha256,
            "artifacts": artifact_records,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _validated_pipeline_status(root: Path) -> Mapping[str, Any] | None:
    path = root / _PIPELINE_STATUS_ARTIFACT_NAME
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"pipeline lifecycle artifact must be a file: {path}")
    payload, _ = read_json_snapshot(path)
    if not isinstance(payload, Mapping):
        raise ValueError("pipeline status must be a JSON object")
    if payload.get("schema_version") != _PIPELINE_STATUS_SCHEMA_VERSION:
        raise ValueError("pipeline status has an unsupported schema_version")
    requested = validate_pipeline_stage(payload.get("requested_stop_after"))
    completed = validate_pipeline_stage(payload.get("completed_stage"))
    if _pipeline_stage_rank(completed) > _pipeline_stage_rank(requested):
        raise ValueError("pipeline status completed stage follows its requested stop")
    expected_status = (
        "completed"
        if completed == "phase3"
        else "stopped_as_requested"
        if completed == requested
        else "running"
    )
    if payload.get("status") != expected_status:
        raise ValueError("pipeline status lifecycle state is inconsistent")
    if payload.get("stage_order") != list(PIPELINE_STAGE_ORDER):
        raise ValueError("pipeline status stage order is invalid")
    artifact_records = payload.get("artifacts")
    if not isinstance(artifact_records, list):
        raise ValueError("pipeline status artifacts must be an array")
    resolved_root = root.resolve()
    seen: set[str] = set()
    for record in artifact_records:
        if not isinstance(record, Mapping):
            raise ValueError("pipeline status artifact record is invalid")
        relative_path = record.get("path")
        expected_size = record.get("byte_size")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path in seen
            or (type(expected_size) is not int)
            or (expected_size < 0)
            or (not isinstance(expected_sha256, str))
            or (len(expected_sha256) != 64)
        ):
            raise ValueError("pipeline status artifact identity is invalid")
        seen.add(relative_path)
        artifact_path = (root / relative_path).resolve()
        if (
            not artifact_path.is_relative_to(resolved_root)
            or not artifact_path.is_file()
            or artifact_path.stat().st_size != expected_size
            or (_file_sha256(artifact_path) != expected_sha256)
        ):
            raise ValueError("pipeline status artifact identity does not match")
    return payload


def _completed_intermediate_stage(present: Sequence[str]) -> str:
    """Identify lifecycle stages by markers, independent of legacy file count."""
    for marker, stage in (
        ("phase2-evidence.json", "phase2"),
        ("model-capabilities.json", "phase1"),
        ("scene-usability.json", "phase0b"),
        ("recording-overview.json", "phase0a"),
    ):
        if marker in present:
            return stage
    return "preprocessing"


def validate_model_run_resume_state(run_dir: str | Path) -> tuple[str, ...]:
    """Validate a resumable, incomplete model-run lifecycle prefix.

    Intermediate JSON files are lifecycle markers only.  The resumed pipeline
    deterministically reconstructs and overwrites them from input-bound,
    locally revalidated model-call checkpoints; it never treats their content
    as a cache. An incomplete lifecycle with durable Phase2 evidence may have
    partially serialized final outputs. Rebuild those from validated model
    checkpoints; only a completed Phase3 lifecycle is protected from resume.
    Legacy runs without a lifecycle marker retain the final-output guard.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError("--resume requires an existing run directory")
    pipeline_status = _validated_pipeline_status(root)
    present_intermediates: list[str] = []
    missing_seen = False
    for name in _MODEL_RUN_INTERMEDIATE_ARTIFACT_NAMES:
        path = data_path(root / name)
        if path.exists() and (not path.is_file()):
            raise ValueError(f"resume lifecycle artifact must be a file: {path}")
        if path.is_file():
            if missing_seen:
                raise ValueError(
                    f"--resume requires an ordered intermediate artifact prefix; found {name} after a missing earlier stage"
                )
            present_intermediates.append(name)
        else:
            missing_seen = True
    if pipeline_status is not None and pipeline_status["completed_stage"] == "phase3":
        raise ValueError("a completed Phase 3 run cannot be resumed")
    interrupted_finalization = (
        pipeline_status is not None and "phase2-evidence.json" in present_intermediates
    )
    for name in _MODEL_RUN_FINAL_ARTIFACT_NAMES:
        path = data_path(root / name)
        if path.exists() and not path.is_file():
            raise ValueError(f"resume final artifact must be a file: {path}")
        if path.exists() and not interrupted_finalization:
            raise FileExistsError(
                f"final output exists without an incomplete Phase2 lifecycle: {path}"
            )
    if pipeline_status is not None:
        completed_stage = str(pipeline_status["completed_stage"])
        artifact_completed_stage = _completed_intermediate_stage(present_intermediates)
        if _pipeline_stage_rank(completed_stage) > _pipeline_stage_rank(artifact_completed_stage):
            raise ValueError("pipeline status claims a stage beyond the durable artifact prefix")
    if not present_intermediates:
        if pipeline_status is None or pipeline_status.get("completed_stage") != "preprocessing":
            raise ValueError(
                "--resume requires preprocessing lifecycle status or an incomplete run beginning with recording-overview.json"
            )
        status_artifacts = pipeline_status.get("artifacts")
        assert isinstance(status_artifacts, list)
        if not any(
            (
                isinstance(record, Mapping) and record.get("path") == "preprocessing.json"
                for record in status_artifacts
            )
        ):
            raise ValueError("preprocessing-only resume status must cite preprocessing.json")
    checkpoint_root = root / "model-checkpoints"
    if checkpoint_root.exists() and (not checkpoint_root.is_dir()):
        raise ValueError("model-checkpoints artifact path must be a directory")
    checkpoint_available = checkpoint_root.is_dir() and any(
        (path.is_file() for path in checkpoint_root.glob("checkpoint-*.json"))
    )
    skipped_phase1 = (
        pipeline_status is not None
        and "recording-overview.json" in present_intermediates
        and (
            read_json(root / "recording-overview.json").get("policy_version")
            == "operator-skipped-phase1-v0.1"
        )
    )
    if present_intermediates and (not checkpoint_available) and (not skipped_phase1):
        raise ValueError(
            "--resume requires at least one validated model checkpoint in model-checkpoints"
        )
    return tuple(present_intermediates)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_model_capabilities(
    capabilities: Mapping[str, Any], *, semantic: bool
) -> dict[str, Any]:
    """Return capability fields used by checkpoint identity.

    The synthetic probe has never been part of checkpoint identity.  Open
    WebUI's top-level model-list entry contains mutable registry metadata, so
    its opaque digest is retained in provenance but omitted from the stable
    semantic identity.  The model-details digest remains semantic because it
    may bind model weights, configuration, or templates.
    """
    result = {
        str(key): deepcopy(value)
        for key, value in capabilities.items()
        if key != "synthetic_request_probe"
    }
    if (
        semantic
        and str(result.get("adapter", "")).startswith("open-webui-research-")
        and _is_sha256(result.get("model_entry_sha256"))
    ):
        result.pop("model_entry_sha256", None)
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all((character in "0123456789abcdef" for character in value))
    )


class _CheckpointingPhase2Model:
    """Replay successful calls only when complete input identities match."""

    def __init__(
        self,
        delegate: Phase2Model,
        *,
        run_root: Path,
        scope: Mapping[str, Any],
        resume: bool,
        trace: Any | None,
    ) -> None:
        self._delegate = delegate
        self._run_root = run_root
        self._checkpoint_root = run_root / "model-checkpoints"
        self._scope_sha256 = _digest(scope)
        self._resume = resume
        self._trace = trace
        self.reused_count = 0
        self.persisted_count = 0

    def check_capabilities(self) -> Mapping[str, Any]:
        return self._delegate.check_capabilities()

    def complete_json(
        self,
        *,
        request_id: str,
        task: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[Path],
        response_schema: Mapping[str, Any],
        trace_context: Mapping[str, Any],
        response_validator: Any = None,
    ) -> ModelCallResult:
        identity = _digest(
            {
                "scope": self._scope_sha256,
                "request_id": request_id,
                "task": task,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_schema": response_schema,
                "trace_context": trace_context,
                "images": [_file_sha256(Path(p)) for p in images],
            }
        )
        path = self._checkpoint_root / (
            "checkpoint-" + sha256(request_id.encode()).hexdigest()[:24] + ".json"
        )
        if self._resume and path.is_file():
            saved = read_json_snapshot(path)[0]
            if saved.get("input_sha256") != identity:
                raise ValueError(f"Checkpoint inputs changed for {request_id}; start a fresh run")
            call = saved.get("call")
            if not isinstance(call, dict) or _digest(call) != saved.get("call_sha256"):
                raise ValueError("Checkpoint result hash does not match")
            if call.get("request_id") != request_id:
                raise ValueError("Checkpoint request identity does not match")
            if response_validator:
                response_validator(call["payload"])
            self.reused_count += 1
            _record_event(self._trace, "model", "checkpoint", "reused", request_id=request_id)
            return ModelCallResult(**call)
        result = self._delegate.complete_json(
            request_id=request_id,
            task=task,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            response_schema=response_schema,
            trace_context=trace_context,
            response_validator=response_validator,
        )
        if response_validator:
            response_validator(result.payload)
        # Stable key ordering makes downstream prompts identical after resume.
        call = json.loads(json.dumps(to_json_value(result), sort_keys=True, allow_nan=False))
        write_json(
            path,
            {
                "schema_version": "1.0",
                "request_id": request_id,
                "task": task,
                "input_sha256": identity,
                "call": call,
                "call_sha256": _digest(call),
            },
        )
        from ..adapters.image_recovery import release_checkpointed_images
        release_checkpointed_images(self._run_root, request_id)
        self.persisted_count += 1
        _record_event(self._trace, "model", "checkpoint", "persisted", request_id=request_id)
        return ModelCallResult(**call)


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _uniform_indices(count: int, selected_count: int) -> list[int]:
    if count <= 0:
        return []
    if count <= selected_count:
        return list(range(count))
    if selected_count == 1:
        return [count // 2]
    values = {
        round(position * (count - 1) / (selected_count - 1)) for position in range(selected_count)
    }
    return sorted(values)


def _frame_by_index(
    frames: Sequence[Mapping[str, Any]], samples: Sequence[AnalysisSample]
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for frame in frames:
        index = frame.get("sample_index")
        path = frame.get("path")
        digest = frame.get("sha256")
        if (
            type(index) is not int
            or index in result
            or (not isinstance(path, (str, Path)))
            or (not isinstance(digest, str))
            or (len(digest) != 64)
        ):
            raise ValueError("extracted frame manifest is malformed")
        result[index] = frame
    if set(result) != {sample.index for sample in samples}:
        raise ValueError("one extracted analysis image is required for every sample")
    return result


def _merge_exact_frames(
    exact_frames: Sequence[Mapping[str, Any]],
    samples: Sequence[AnalysisSample],
    fallback: Mapping[int, Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    """Use exact RGB for processed samples; keep excluded timeline placeholders."""
    # Range limits intentionally omit images outside the processed interval.
    processed = [s for s in samples if s.source_gap_reason != "outside_processing_range"]
    return {**fallback, **_frame_by_index(exact_frames, processed)}


def _pose_conditioning_inputs(
    conditioning: PoseConditioningResult | None, samples: Sequence[AnalysisSample]
) -> tuple[dict[int, Mapping[str, Any]], dict[int, Mapping[str, Any]], Mapping[str, Any] | None]:
    """Validate the optional derived pose stream before any model call.

    Programmatic callers may omit the optional bundle.
    The ``run-llm`` command supplies it unconditionally.  Valid absence inside
    a bundle is represented by an explicit full-frame fallback record, never by
    silently dropping a sample.
    """
    if conditioning is None:
        return ({}, {}, None)
    focus = _frame_by_index(conditioning.focus_frame_records, samples)
    features: dict[int, Mapping[str, Any]] = {}
    for expected, record in enumerate(conditioning.feature_records):
        if not isinstance(record, Mapping) or record.get("sample_index") != expected:
            raise ValueError("pose conditioning must contain one ordered feature record per sample")
        sample = samples[expected]
        if record.get("start_us") != sample.start_us or record.get("end_us") != sample.end_us:
            raise ValueError("pose-conditioning feature timestamps do not match the grid")
        features[expected] = record
    if set(features) != {sample.index for sample in samples}:
        raise ValueError("pose conditioning does not cover the complete analysis grid")
    identity = conditioning.model_input_identity()
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("pose conditioning omitted its model-input identity")
    return (focus, features, deepcopy(dict(identity)))


def _frame_export_source_root(frames: Mapping[int, Mapping[str, Any]]) -> Path:
    """Return the narrow common directory containing extracted analysis JPEGs."""
    if not frames:
        raise ValueError("analysis-frame source root requires at least one frame")
    resolved: list[Path] = []
    for frame in frames.values():
        value = frame.get("path")
        if not isinstance(value, (str, Path)):
            raise ValueError("analysis-frame path is invalid")
        resolved.append(Path(value).resolve())
    candidate = resolved[0].parent
    if not all((path.parent == candidate for path in resolved)):
        raise ValueError(
            "analysis frames must share one source directory for representative export"
        )
    return candidate


def _enforce_restricted_transport(
    *,
    source: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    config: ResearchConfig,
    capabilities: Mapping[str, Any],
) -> str | None:
    """Fail closed before any restricted frame reaches a model request."""
    declared: list[dict[str, Any]] = []
    source_rights = source.get("rights")
    if source_rights is not None:
        declared.append(validate_rights(source_rights))
    proxy = preprocessing.get("proxy")
    proxy = proxy if isinstance(proxy, Mapping) else {}
    proxy_rights = proxy.get("rights")
    if proxy_rights is not None:
        declared.append(validate_rights(proxy_rights))
    for frame in frames:
        if frame.get("rights") is not None:
            declared.append(validate_rights(frame["rights"]))
    restricted = any((is_restricted_rights(value) for value in declared))
    if not restricted:
        return None
    if proxy_rights is None or not is_restricted_rights(proxy_rights):
        raise ValueError("restricted source rights were not propagated to the model proxy")
    transport_mode = restricted_transport_mode(
        restricted=True,
        base_url=config.open_webui.base_url,
        destination_scope=str(capabilities.get("destination_scope")),
        verify_tls=config.open_webui.verify_tls,
        allow_non_private_network=config.open_webui.allow_non_private_network,
        operator_asserts_local_provider=config.open_webui.operator_asserts_local_provider,
        operator_authorizes_restricted_remote_processing=config.open_webui.operator_authorizes_restricted_remote_processing,
    )
    for frame in frames:
        frame_rights = frame.get("rights")
        if frame_rights is None or not is_restricted_rights(frame_rights):
            raise ValueError(
                "restricted proxy rights were not propagated to every derived model frame"
            )
    return transport_mode


def _paths(indices: Sequence[int], frame_by_index: Mapping[int, Mapping[str, Any]]) -> list[Path]:
    return [Path(frame_by_index[index]["path"]) for index in indices]


def _frame_refs(
    indices: Sequence[int], frame_by_index: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": index,
            "start_us": frame_by_index[index].get("start_us"),
            "end_us": frame_by_index[index].get("end_us"),
            "sha256": frame_by_index[index]["sha256"],
            **(
                {"source_gap_reason": frame_by_index[index]["source_gap_reason"]}
                if frame_by_index[index].get("source_gap_reason") is not None
                else {}
            ),
        }
        for index in indices
    ]


def _visual_mapping_text(references: Sequence[Mapping[str, Any]]) -> str:
    """Return a compact model-readable map from supplied images to samples."""
    compact: list[dict[str, Any]] = []
    for reference in references:
        if reference.get("kind") == "individual_frame":
            compact.append(
                {
                    "image_ordinal": reference["image_ordinal"],
                    "sample_index": reference["sample_index"],
                }
            )
            continue
        compact.append(
            {
                "image_ordinal": reference["image_ordinal"],
                "package_id": reference.get("package_id"),
                "cells": [
                    {
                        "ordinal": cell["ordinal"],
                        "row": cell["row"],
                        "column": cell["column"],
                        "sample_index": cell["sample_index"],
                    }
                    for cell in reference.get("cells", [])
                ],
            }
        )
    return json.dumps(compact, separators=(",", ":"))


def _contiguous_sample_span(indices: Sequence[int], *, label: str) -> list[int]:
    """Return a validated global half-open span for contiguous sample indices."""
    ordered = list(indices)
    if not ordered:
        raise ValueError(f"{label} must contain at least one sample")
    if any((type(index) is not int or index < 0 for index in ordered)):
        raise ValueError(f"{label} must contain non-negative integer sample indices")
    expected = list(range(ordered[0], ordered[-1] + 1))
    if ordered != expected:
        raise ValueError(f"{label} must be ordered, unique, and contiguous")
    return [ordered[0], ordered[-1] + 1]


def _contiguous_runs(indices: Sequence[int], *, label: str) -> list[list[int]]:
    """Partition sorted unique sample indices into maximal contiguous runs."""
    ordered = list(indices)
    if any((type(index) is not int or index < 0 for index in ordered)):
        raise ValueError(f"{label} must contain non-negative integer sample indices")
    if ordered != sorted(set(ordered)):
        raise ValueError(f"{label} must be ordered and unique")
    runs: list[list[int]] = []
    for index in ordered:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _without_mapping_key(value: Any, omitted_key: str) -> Any:
    """Copy a JSON-like value while removing one private model-facing key."""
    if isinstance(value, Mapping):
        return {
            str(key): _without_mapping_key(item, omitted_key)
            for key, item in value.items()
            if key != omitted_key
        }
    if isinstance(value, (list, tuple)):
        return [_without_mapping_key(item, omitted_key) for item in value]
    return value


def _dense_frame_context(
    indices: Sequence[int], frames: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Describe a dense request without repeating one record per 10 Hz image."""
    sample_span = _contiguous_sample_span(indices, label="dense frame context")
    gaps: list[list[Any]] = []
    previous_end: int | None = None
    for index in indices:
        if index not in frames:
            raise ValueError(f"dense frame context is missing sample {index}")
        frame = frames[index]
        start_us = frame.get("start_us")
        end_us = frame.get("end_us")
        if (
            type(start_us) is not int
            or type(end_us) is not int
            or start_us < 0
            or (end_us <= start_us)
        ):
            raise ValueError("dense frame context has invalid canonical timestamps")
        if previous_end is not None and start_us != previous_end:
            raise ValueError("dense frame context timestamps must be contiguous")
        previous_end = end_us
        reason = frame.get("source_gap_reason")
        if reason is not None:
            gaps.append([index, str(reason)])
    first = frames[sample_span[0]]
    last = frames[sample_span[1] - 1]
    return {
        "analysis_rate_hz": 10,
        "sample_span": sample_span,
        "canonical_time_span_us": [first["start_us"], last["end_us"]],
        "source_gap_samples": gaps,
    }


def _dense_visual_mapping(
    references: Sequence[Mapping[str, Any]], expected_indices: Sequence[int]
) -> dict[str, Any]:
    """Compress and validate dense image-to-global-sample provenance for the model."""
    expected = list(expected_indices)
    _contiguous_sample_span(expected, label="dense visual mapping samples")
    if not references:
        raise ValueError("dense visual mapping has no image references")
    modes = {reference.get("kind") for reference in references}
    if len(modes) != 1 or not modes <= {"individual_frame", "contact_sheet"}:
        raise ValueError("dense visual mapping must use one supported image mode")
    if [reference.get("image_ordinal") for reference in references] != list(range(len(references))):
        raise ValueError("dense visual mapping image ordinals must be contiguous from zero")
    mode = next(iter(modes))
    if mode == "individual_frame":
        samples: list[int] = []
        images: list[list[int]] = []
        for reference in references:
            image_ordinal = reference["image_ordinal"]
            sample_index = reference.get("sample_index")
            if type(sample_index) is not int:
                raise ValueError("individual image reference has no integer sample index")
            samples.append(sample_index)
            images.append([image_ordinal, sample_index])
        if samples != expected:
            raise ValueError("individual image mapping does not cover the expected samples")
        return {
            "mode": "individual",
            "fields": ["image_ordinal", "global_sample_index"],
            "images": images,
        }
    samples = []
    images = []
    maximum_row = 0
    maximum_column = 0
    label_kinds: set[str] = set()
    for reference in references:
        cells = reference.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("contact-sheet image reference has no cells")
        positions: list[tuple[int, int]] = []
        sheet_samples: list[int] = []
        for local_ordinal, cell in enumerate(cells):
            if not isinstance(cell, Mapping):
                raise ValueError("contact-sheet cell reference is malformed")
            if cell.get("ordinal") != local_ordinal:
                raise ValueError("contact-sheet cell ordinals must reset from zero")
            if cell.get("visible_label") == f"CURRENT f{cell.get('sample_index')}":
                label_kinds.add("global")
            elif cell.get("visible_label") == str(local_ordinal):
                label_kinds.add("local")
            else:
                raise ValueError(
                    "contact-sheet visible label does not match its global frame or local ordinal"
                )
            row = cell.get("row")
            column = cell.get("column")
            sample_index = cell.get("sample_index")
            if (
                type(row) is not int
                or type(column) is not int
                or row < 0
                or (column < 0)
                or (type(sample_index) is not int)
            ):
                raise ValueError("contact-sheet cell mapping is malformed")
            positions.append((row, column))
            maximum_row = max(maximum_row, row)
            maximum_column = max(maximum_column, column)
            sheet_samples.append(sample_index)
        if positions != sorted(positions) or positions[0] != (0, 0):
            raise ValueError("contact-sheet cells must be in row-major order")
        if sheet_samples != list(range(sheet_samples[0], sheet_samples[0] + len(sheet_samples))):
            raise ValueError("contact-sheet samples must be contiguous")
        samples.extend(sheet_samples)
        images.append([reference["image_ordinal"], sheet_samples[0], sheet_samples[-1] + 1])
    if len(label_kinds) != 1:
        raise ValueError("contact-sheet labels must use one consistent numbering policy")
    if samples != expected:
        raise ValueError("contact-sheet mapping does not cover the expected samples")
    return {
        "mode": "contact_sheet",
        "fields": ["image_ordinal", "start_sample", "end_sample_exclusive"],
        "images": images,
        "cell_order": "row-major",
        "observed_grid_shape": [maximum_row + 1, maximum_column + 1],
        "visible_label_rule": "CURRENT fN labels give the global analysis sample index N directly; each cell is one frame"
        if label_kinds == {"global"}
        else "labels are image-local ordinals starting at 0 on every image; global_sample_index = image start_sample + visible label",
    }


def _role_visual_references(
    references: Sequence[Mapping[str, Any]], *, image_ordinal_offset: int, role: str
) -> list[dict[str, Any]]:
    """Copy visual references into their actual mixed-request image positions."""
    if type(image_ordinal_offset) is not int or image_ordinal_offset < 0:
        raise ValueError("visual-reference image offset must be non-negative")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("visual-reference role must be nonempty")
    if [reference.get("image_ordinal") for reference in references] != list(range(len(references))):
        raise ValueError("source visual-reference ordinals must start at zero")
    result: list[dict[str, Any]] = []
    for reference in references:
        copied = deepcopy(dict(reference))
        copied["image_ordinal"] = int(reference["image_ordinal"]) + image_ordinal_offset
        copied["role"] = role
        result.append(copied)
    return result


def _positive_exemplar_payload(
    *,
    indices: Sequence[int],
    frames: Mapping[int, Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    image_ordinal_offset: int,
    role: str,
    interpretation: str,
) -> dict[str, Any]:
    """Describe positive, non-temporal reference images separately from a sequence."""
    if not indices:
        return {
            "role": role,
            "sample_metadata": [],
            "visual_mapping": [],
            "interpretation": interpretation,
        }
    mapped = _role_visual_references(
        references, image_ordinal_offset=image_ordinal_offset, role=role
    )
    return {
        "role": role,
        "sample_metadata": _frame_refs(indices, frames),
        "visual_mapping": mapped,
        "interpretation": interpretation,
    }


def _positive_exemplar_sample_capacity(
    config: ResearchConfig,
    *,
    chronological_stage: str,
    chronological_sample_count: int,
    desired_count: int = 8,
) -> int:
    """Return a safe positive-reference sample count within a mixed image budget."""
    chronological_images = config.inference.packaged_image_count(
        chronological_stage, chronological_sample_count
    )
    available_images = config.inference.maximum_images_per_request - chronological_images
    if available_images <= 0:
        return 0
    if config.inference.profile_image_mode == "individual":
        return min(desired_count, available_images)
    per_image = config.inference.profile_sheet_columns * config.inference.profile_sheet_rows
    return min(desired_count, available_images * per_image)


def _bounded_profile_contact_sheet_layout(
    config: ResearchConfig, *, maximum_cells: int
) -> ContactSheetLayout:
    """Return a profile-tile sheet bounded to one role's occupied capacity."""
    configured_capacity = (
        config.inference.profile_sheet_columns * config.inference.profile_sheet_rows
    )
    if maximum_cells <= 0 or maximum_cells > configured_capacity:
        raise ValueError(
            "bounded profile contact-sheet capacity must be positive and may not exceed the configured profile sheet"
        )
    columns = min(config.inference.profile_sheet_columns, maximum_cells)
    rows = (maximum_cells + columns - 1) // columns
    return ContactSheetLayout(
        columns=columns,
        rows=rows,
        tile_width_px=config.inference.profile_tile_width,
        tile_height_px=config.inference.profile_tile_height,
        gutter_px=config.inference.contact_sheet_gutter,
        label_mode="ordinal",
        jpeg_quality=config.inference.profile_jpeg_quality,
    )


def _request_visual_inputs(
    *,
    root: Path,
    request_id: str,
    stage: str,
    indices: Sequence[int],
    frame_by_index: Mapping[int, Mapping[str, Any]],
    mode: str,
    layout: ContactSheetLayout,
    maximum_images: int,
    ffmpeg: str,
    overwrite: bool,
    records: list[dict[str, Any]],
    trace: Any | None,
    visible_labels: Mapping[int, str] | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Prepare individual images or deterministic contact sheets for one call."""
    if not indices:
        raise ValueError(f"{request_id} has no visual inputs")
    if any(
        (
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in request_id
        )
    ):
        raise ValueError("model request identifier is not safe for an artifact path")
    _record_event(
        trace,
        stage,
        "visual_packaging",
        "started",
        request_id=request_id,
        mode=mode,
        source_sample_count=len(indices),
    )
    if mode == "individual" and visible_labels is None:
        if len(indices) > maximum_images:
            raise ValueError(
                f"{request_id} requires {len(indices)} images; model limit is {maximum_images}"
            )
        paths = _paths(indices, frame_by_index)
        references = [
            {"kind": "individual_frame", "image_ordinal": ordinal, **reference}
            for ordinal, reference in enumerate(_frame_refs(indices, frame_by_index))
        ]
        records.append(
            {
                "request_id": request_id,
                "stage": stage,
                "mode": mode,
                "source_sample_count": len(indices),
                "image_count": len(paths),
                "image_refs": references,
            }
        )
        _record_event(
            trace,
            stage,
            "visual_packaging",
            "completed",
            request_id=request_id,
            mode=mode,
            image_count=len(paths),
        )
        return (paths, references)
    if visible_labels is not None:
        layout = replace(
            layout,
            label_mode="global_sample"
            if all((v.startswith(("CURRENT f", "REFERENCE f")) for v in visible_labels.values()))
            else "target_time",
            **{"columns": 1, "rows": 1} if mode == "individual" else {},
        )
    if mode not in {"contact_sheet", "individual"}:
        raise ValueError("unsupported visual-input mode")
    frames = [FrameImage.from_mapping(frame_by_index[index]) for index in indices]
    plans = plan_image_packages(
        frames, layout, maximum_images=maximum_images, visible_labels=visible_labels
    )
    rendered = render_image_packages(
        plans, root / "model-inputs" / request_id, ffmpeg=ffmpeg, overwrite=overwrite, reuse_existing=overwrite
    )
    references = [
        {"kind": "contact_sheet", "image_ordinal": ordinal, **reference}
        for ordinal, reference in enumerate(rendered.trace_refs())
    ]
    records.append(
        {
            "request_id": request_id,
            "stage": stage,
            "mode": mode,
            "package_set_id": rendered.package_set_id,
            "manifest": str(rendered.manifest_path.relative_to(root)).replace("\\", "/"),
            "source_sample_count": len(indices),
            "image_count": len(rendered.images),
            "layout": layout.to_dict(),
            "image_refs": references,
        }
    )
    _record_event(
        trace,
        stage,
        "visual_packaging",
        "completed",
        request_id=request_id,
        mode=mode,
        image_count=len(rendered.images),
        package_set_id=rendered.package_set_id,
    )
    return (rendered.image_paths(), references)


def _record_event(trace: Any | None, stage: str, event: str, status: str, **details: Any) -> None:
    if trace is not None:
        trace.log_event(stage=stage, event=event, status=status, details=details)


def _model_call(model: Phase2Model, trace: Any | None, stage: str, **arguments: Any) -> Any:
    """Record safe failure metadata while preserving the original exception."""
    try:
        return model.complete_json(**arguments)
    except Exception as exc:
        _record_event(
            trace,
            stage,
            "model_request",
            "failed",
            request_id=arguments.get("request_id"),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def _normalize_profile(raw: Mapping[str, Any]) -> dict[str, Any]:
    subjects: list[dict[str, Any]] = []
    subject_id_mappings: list[dict[str, Any]] = []
    for index, value in enumerate(raw["selected_subjects"], 1):
        item = dict(value)
        item["model_subject_id"] = item.pop("subject_id")
        item["subject_id"] = f"subject-{index:04d}"
        subject_id_mappings.append(
            {
                "subject_ordinal": index,
                "model_subject_id": item["model_subject_id"],
                "canonical_subject_id": item["subject_id"],
            }
        )
        subjects.append(item)
    result = dict(raw)
    result["selected_subjects"] = subjects
    result["profile_version"] = OVERVIEW_PROMPT_VERSION
    result["identity_scope"] = "within-recording visual continuity only"
    result["subject_id_canonicalization"] = {
        "policy_version": _PROFILE_SUBJECT_ID_CANONICALIZATION_VERSION,
        "scope": "application-facing selected-subject identifiers only",
        "model_payload_modified": False,
        "mappings": subject_id_mappings,
    }
    return result


def _profile_call_normalization_record(call: Any) -> dict[str, Any]:
    """Persist raw/canonical identity and deterministic profile corrections."""
    events = [deepcopy(dict(event)) for event in getattr(call, "contract_normalization_events", ())]
    model_payload = getattr(call, "model_payload", None)
    if not isinstance(model_payload, Mapping):
        raise ValueError("profile call did not preserve its raw model payload")
    computed_raw_sha256 = _digest(model_payload)
    raw_sha256 = getattr(call, "model_payload_sha256", None)
    if raw_sha256 is None:
        raw_sha256 = computed_raw_sha256
    elif raw_sha256 != computed_raw_sha256:
        raise ValueError("profile call raw model payload digest is inconsistent")
    computed_canonical_sha256 = _digest(call.payload)
    canonical_sha256 = getattr(call, "payload_sha256", None)
    if canonical_sha256 is None:
        canonical_sha256 = computed_canonical_sha256
    elif canonical_sha256 != computed_canonical_sha256:
        raise ValueError("profile call canonical payload digest is inconsistent")
    policy_version = getattr(call, "contract_normalization_policy_version", None)
    if policy_version is None:
        policy_version = PROFILE_NORMALIZATION_VERSION
    if policy_version != PROFILE_NORMALIZATION_VERSION:
        raise ValueError("profile call normalization policy is incompatible")
    if not isinstance(call.trace_reference, Mapping):
        raise ValueError("profile call sensitive exchange trace is missing")
    if bool(events) != (raw_sha256 != canonical_sha256):
        raise ValueError("profile call normalization ledger is incomplete")
    return {
        "policy_version": policy_version,
        "normalization_applied": bool(events),
        "event_count": len(events),
        "events": events,
        "raw_model_payload_preserved": True,
        "raw_model_payload_sha256": raw_sha256,
        "canonical_payload_sha256": canonical_sha256,
        "sensitive_model_exchange_trace": call.trace_reference,
    }


def _profile_prompt(
    indices: Sequence[int],
    frames: Mapping[int, Mapping[str, Any]],
    visual_inputs: Sequence[Mapping[str, Any]],
) -> str:
    return (
        "Consolidate the supplied chronological Phase 0a overview summaries and global storyboard. Earlier requests covered the complete recording with overlapping 0.5 Hz storyboard windows. Resolve recording-local identity continuity across segments and camera angles. Merge repeated sightings of the same person into one selected subject rather than enumerating samples, images, views, or segment-local observations as subjects. The image order and canonical sample metadata are:\n"
        + json.dumps(_frame_refs(indices, frames), separators=(",", ":"))
        + "\nVisual image/contact-sheet mapping:\n"
        + _visual_mapping_text(visual_inputs)
        + "\nSelect exactly one contextually central recurring participant and describe stable within-recording identity cues plus the family of legitimate live camera views in which that target recurs. Slides, static pictures, embedded clips, audience shots, and unrelated people are not target views. The sample_index values in this list are direct final-storyboard evidence. Use opaque within-recording IDs. Return empty typical_rest_postures arrays and an empty annotation_relevance_spans array; those are owned by later stages. Return JSON."
    )


def _profile_consolidation_prompt(
    indices: Sequence[int],
    frames: Mapping[int, Mapping[str, Any]],
    profile_inputs: Sequence[Mapping[str, Any]],
    visual_inputs: Sequence[Mapping[str, Any]],
    evidence_scope: Mapping[str, Any],
) -> str:
    return (
        _profile_prompt(indices, frames, visual_inputs)
        + "\nBounded chronological profile summaries, in chronological order:\n"
        + json.dumps(list(profile_inputs), ensure_ascii=False, separators=(",", ":"))
        + "\nAuthoritative final evidence scope:\n"
        + json.dumps(dict(evidence_scope), separators=(",", ":"))
        + "\nDirect storyboard indices have images in this request. Inherited profile indices are useful only for consolidating identity and recurring-view descriptions. For this final response, evidence_sample_indices may contain up to eight ordinary visual citations from direct_storyboard_sample_indices that support the target/profile decision. They are evidence, not certified positive exemplars. A separate exact per-image request will verify positive main-scene examples. Never pad the evidence array merely to reach a count. "
        + "\nPeople occurring in only brief or contextually unrelated segments must be ignored in the final selection. Preserve the family of legitimate recurring target views. Do not infer rest postures or exact scene boundaries."
    )


def _phase0a_exemplar_verification_prompt(
    *,
    profile: Mapping[str, Any],
    indices: Sequence[int],
    frames: Mapping[int, Mapping[str, Any]],
    visual_inputs: Sequence[Mapping[str, Any]],
) -> str:
    """Build the independent exact classifier for final storyboard candidates."""
    target_profile = {
        "target_mode": profile["target_mode"],
        "selected_subjects": _without_mapping_key(profile["selected_subjects"], "model_subject_id"),
        "scene_summary": profile["scene_summary"],
    }
    payload = {
        "phase0a_target_profile": target_profile,
        "candidate_sample_metadata": _frame_refs(indices, frames),
        "visual_mapping": json.loads(_visual_mapping_text(visual_inputs)),
        "classification_policy": {
            "classify_each_image_independently": True,
            "exact_ordered_sample_coverage": list(indices),
            "positive_requires_selected_live_target_physically_visible": True,
            "speaker_plus_slide_is_positive_only_when_selected_live_target_visible": True,
            "full_screen_slide_is_not_positive": True,
            "person_depicted_inside_slide_picture_or_clip_is_not_live_target": True,
            "some_other_speaker_is_not_the_selected_target": True,
            "uncertain_when_live_target_presence_cannot_be_established": True,
        },
    }
    return (
        "Verify every supplied Phase 0a storyboard candidate. First use all images as a set to understand the fixed selected target's recording-local appearance and legitimate view family. Then return one independent ordered classification for every candidate; do not skip an image and do not use temporal proximity as evidence. Input:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _profile_segment_prompt(
    indices: Sequence[int],
    owned_indices: Sequence[int],
    frames: Mapping[int, Mapping[str, Any]],
    segment_id: str,
    visual_inputs: Sequence[Mapping[str, Any]],
) -> str:
    return (
        "Analyze every ordered 0.5 Hz storyboard sample in this chronological recording segment. This segment is one of the configured overlapping windows that together cover the complete recording. Describe the recurring people visible in this segment as candidates for later recording-level consolidation; merge repeated sightings of the same person into one candidate and never create one subject per sample, image, or view. Do not name or identify anyone. If no person is visible, return target_mode=none and an empty selected_subjects array. Segment: "
        + segment_id
        + ". Ordered canonical sample metadata:\n"
        + json.dumps(_frame_refs(indices, frames), separators=(",", ":"))
        + "\nVisual image/contact-sheet mapping:\n"
        + _visual_mapping_text(visual_inputs)
        + "\nThe overlap is visual context, but prevalence counting for this segment belongs only to these owned storyboard samples: "
        + json.dumps(list(owned_indices), separators=(",", ":"))
        + ". Describe only recurring target candidates, stable identity cues, and legitimate camera views. Slides, static pictures, embedded clips, audience shots, and unrelated people are not target views. Return empty typical_rest_postures arrays and an empty annotation_relevance_spans array. Evidence indices may come from any supplied input sample. Return only JSON."
    )


def _bounded_profile_input(item: Mapping[str, Any], maximum_characters: int) -> dict[str, Any]:
    """Create a bounded, digest-backed semantic summary for the next level."""
    result = item["result"]
    if not isinstance(result, Mapping):
        raise ValueError("profile consolidation result must be an object")
    subjects = result.get("selected_subjects")
    ignored = result.get("ignored_transient_people")
    if not isinstance(subjects, list) or any(
        (not isinstance(subject, Mapping) for subject in subjects)
    ):
        raise ValueError("profile selected_subjects must be an object array")
    if not isinstance(ignored, list):
        raise ValueError("profile ignored_transient_people must be an array")
    subject_fields = ["id", "prev", "desc", "cues", "views", "why"]
    compact: dict[str, Any] = {
        "target_mode": result.get("target_mode"),
        "subject_fields": subject_fields,
        "selected_subjects": [
            [str(subject.get("subject_id", "")), subject.get("prevalence"), "", "", "", ""]
            for subject in subjects
        ],
        "ignored_count": len(ignored),
        "scene": "",
        "confidence": result.get("confidence"),
    }

    def compact_json() -> str:
        return json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    excerpt = compact_json()
    if len(excerpt) > maximum_characters:
        raise ValueError("profile_summary_maximum_characters is too small for the semantic summary")
    slots: list[tuple[list[Any] | dict[str, Any], int | str, str]] = []
    for row, subject in zip(compact["selected_subjects"], subjects, strict=True):
        cues = subject.get("identity_cues", [])
        views = subject.get("recurring_views", [])
        slots.extend(
            [
                (row, 2, str(subject.get("visual_description", ""))),
                (
                    row,
                    3,
                    " | ".join((str(value) for value in cues)) if isinstance(cues, list) else "",
                ),
                (
                    row,
                    4,
                    " | ".join((str(value) for value in views)) if isinstance(views, list) else "",
                ),
                (row, 5, str(subject.get("selection_reason", ""))),
            ]
        )
    slots.append((compact, "scene", str(result.get("scene_summary", ""))))
    for ordinal, (container, key, text) in enumerate(slots):
        text = text.strip()
        if not text:
            continue
        remaining_slots = len(slots) - ordinal
        available = maximum_characters - len(excerpt)
        character_budget = max(0, available // max(1, remaining_slots))
        low = 0
        high = min(len(text), character_budget)
        accepted = ""
        while low <= high:
            midpoint = (low + high) // 2
            candidate = text[:midpoint]
            container[key] = candidate
            candidate_json = compact_json()
            if len(candidate_json) <= maximum_characters:
                accepted = candidate
                excerpt = candidate_json
                low = midpoint + 1
            else:
                high = midpoint - 1
        container[key] = accepted
        excerpt = compact_json()
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "profile_id": item["profile_id"],
        "sample_range": list(item["sample_range"]),
        "owned_sample_span": list(item.get("owned_sample_span", item["sample_range"])),
        "canonical_range_us": list(item["canonical_range_us"]),
        "leaf_profile_count": item["leaf_profile_count"],
        "leaf_profile_span": list(item["leaf_profile_span"]),
        "leaf_lineage_sha256": item["leaf_lineage_sha256"],
        "result_sha256": _digest(item["result"]),
        "authoritative_evidence_sample_indices": _profile_result_evidence_indices(item["result"]),
        "result_excerpt": excerpt,
        "result_excerpt_truncated": excerpt != serialized,
    }


def _profile_reduction_prompt(
    *,
    level: int,
    batch_number: int,
    inputs: Sequence[Mapping[str, Any]],
    evidence_scope: Mapping[str, Any],
) -> str:
    return (
        f"Consolidate this bounded chronological batch of recording profile summaries. This is an intermediate reduction, not the final target selection. Preserve recurring within-recording identity and camera-angle cues; merge repeated sightings of the same person and ignore transient people. Do not infer rest postures or scene boundaries. Use owned_sample_span to avoid double-counting overlapping storyboard context. Reduction level {level}, batch {batch_number}. Inputs:\n"
        + json.dumps(list(inputs), ensure_ascii=False, separators=(",", ":"))
        + "\nAuthoritative reduction evidence scope: "
        + json.dumps(dict(evidence_scope), separators=(",", ":"))
        + ". All evidence in this image-free reduction is inherited from validated upstream profiles. Every root evidence index must belong to allowed_evidence_sample_indices. The allowed scope may exceed the response limit; select at most 128 root evidence items and respect every nested array limit. Return empty typical_rest_postures arrays and annotation_relevance_spans. If the batch contains no person evidence, return target_mode=none. Return only JSON."
    )


def _profile_result_evidence_indices(result: Mapping[str, Any]) -> list[int]:
    """Collect every canonical citation exposed by one validated profile."""
    if not isinstance(result, Mapping):
        raise ValueError("profile consolidation result must be an object")
    collected: set[int] = set()

    def add_indices(value: Any, *, label: str) -> None:
        if not isinstance(value, list) or any(
            (type(index) is not int or index < 0 for index in value)
        ):
            raise ValueError(f"{label} must be a non-negative integer array")
        collected.update(value)

    add_indices(result.get("evidence_sample_indices"), label="profile root evidence")
    subjects = result.get("selected_subjects")
    if not isinstance(subjects, list):
        raise ValueError("profile selected_subjects must be an array")
    for subject in subjects:
        if not isinstance(subject, Mapping):
            raise ValueError("profile selected subject must be an object")
        postures = subject.get("typical_rest_postures")
        if not isinstance(postures, list):
            raise ValueError("profile typical_rest_postures must be an array")
        for posture in postures:
            if not isinstance(posture, Mapping):
                raise ValueError("profile typical rest posture must be an object")
            add_indices(posture.get("evidence_sample_indices"), label="profile posture evidence")
    spans = result.get("annotation_relevance_spans")
    if not isinstance(spans, list):
        raise ValueError("profile annotation_relevance_spans must be an array")
    for span in spans:
        if not isinstance(span, Mapping):
            raise ValueError("profile annotation relevance span must be an object")
        start = span.get("start_sample")
        end = span.get("end_sample_inclusive")
        if type(start) is not int or start < 0 or type(end) is not int or (end < 0):
            raise ValueError("profile annotation relevance bounds are invalid")
        collected.update((start, end))
        add_indices(span.get("evidence_sample_indices"), label="profile relevance evidence")
    return sorted(collected)


def _profile_evidence_scope(
    *, direct_storyboard_indices: Sequence[int], inherited_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the authoritative direct/inherited consolidation evidence union."""
    if any((type(index) is not int or index < 0 for index in direct_storyboard_indices)):
        raise ValueError("direct profile storyboard indices are invalid")
    direct = sorted(set(direct_storyboard_indices))
    upstream: set[int] = set()
    for record in inherited_records:
        if not isinstance(record, Mapping) or "result" not in record:
            raise ValueError("profile consolidation record is invalid")
        upstream.update(_profile_result_evidence_indices(record["result"]))
    inherited = sorted(upstream - set(direct))
    return validate_profile_evidence_scope(
        {
            "policy_version": PROFILE_EVIDENCE_SCOPE_VERSION,
            "direct_storyboard_sample_indices": direct,
            "inherited_profile_sample_indices": inherited,
            "allowed_evidence_sample_indices": sorted({*direct, *inherited}),
            "maximum_output_evidence_items": PROFILE_MAX_ROOT_EVIDENCE_INDICES,
            "irrelevant_gate_evidence_policy": "direct-final-storyboard-only",
        }
    )


def _phase0a_final_storyboard_indices(
    *,
    available_indices: Sequence[int],
    inherited_records: Sequence[Mapping[str, Any]],
    maximum_count: int,
) -> tuple[list[int], dict[int, list[dict[str, Any]]]]:
    """Select uniform recording-wide candidates and retain citation provenance.

    Generic upstream overview evidence must not control the final visual sample:
    it can be concentrated in an early window and is not a positive-exemplar
    decision. The dedicated verifier therefore receives a uniform sample across
    the full recording. Upstream citations remain attached only as provenance.
    """
    available = list(available_indices)
    if (
        not available
        or any((type(index) is not int or index < 0 for index in available))
        or available != sorted(set(available))
    ):
        raise ValueError(
            "Phase 0a final-storyboard availability must be non-empty, sorted, and unique"
        )
    if type(maximum_count) is not int or maximum_count <= 0:
        raise ValueError("Phase 0a final-storyboard maximum_count must be positive")
    capacity = min(maximum_count, len(available))
    available_set = set(available)
    citations: dict[int, list[dict[str, Any]]] = {}
    seen_sources_by_index: dict[int, set[str]] = {}
    for record in inherited_records:
        if not isinstance(record, Mapping):
            raise ValueError("Phase 0a final-storyboard source record is invalid")
        result = record.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("Phase 0a final-storyboard source result is invalid")
        subjects = result.get("selected_subjects")
        if result.get("target_mode") == "none" or not isinstance(subjects, list) or (not subjects):
            continue
        evidence = result.get("evidence_sample_indices")
        if (
            not isinstance(evidence, list)
            or any((type(index) is not int or index < 0 for index in evidence))
            or evidence != sorted(set(evidence))
        ):
            raise ValueError("Phase 0a final-storyboard source evidence is invalid")
        source_id = str(record.get("request_id") or record.get("profile_id") or "")
        if not source_id:
            raise ValueError("Phase 0a final-storyboard source has no request identity")
        source = {
            "request_id": source_id,
            "profile_id": record.get("profile_id"),
            "response_sha256": record.get("response_sha256"),
            "source_kind": "reduction"
            if "reduction_level" in record
            else "segment"
            if "segment_id" in record
            else "consolidation_input",
        }
        for index in evidence:
            if index not in available_set:
                continue
            seen = seen_sources_by_index.setdefault(index, set())
            if source_id in seen:
                continue
            seen.add(source_id)
            citations.setdefault(index, []).append(deepcopy(source))
    positions = _uniform_indices(len(available), capacity)
    selected = [available[position] for position in positions]
    return (selected, {index: deepcopy(citations.get(index, [])) for index in sorted(selected)})


def _select_phase0a_verified_exemplars(
    samples: Sequence[Mapping[str, Any]],
    *,
    segment_profiles: Sequence[Mapping[str, Any]],
    maximum_count: int,
    minimum_positive_probability: float,
) -> list[dict[str, Any]]:
    """Select verified positives across owned Phase 0a temporal strata."""
    if type(maximum_count) is not int or maximum_count < 0:
        raise ValueError("Phase 0a exemplar maximum_count must be non-negative")
    if (
        not isinstance(minimum_positive_probability, (int, float))
        or isinstance(minimum_positive_probability, bool)
        or (not 0.0 <= float(minimum_positive_probability) <= 1.0)
    ):
        raise ValueError("Phase 0a exemplar minimum_positive_probability must be in [0, 1]")
    ordered_samples = list(samples)
    sample_indices = [
        sample.get("sample_index") if isinstance(sample, Mapping) else None
        for sample in ordered_samples
    ]
    if any(
        (type(index) is not int or index < 0 for index in sample_indices)
    ) or sample_indices != sorted(set(sample_indices)):
        raise ValueError("Phase 0a exemplar verification samples must be ordered and unique")
    spans: list[tuple[int, int, str]] = []
    for ordinal, profile in enumerate(segment_profiles, start=1):
        if not isinstance(profile, Mapping):
            raise ValueError("Phase 0a segment profile must be an object")
        span = profile.get("owned_sample_span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any((type(bound) is not int for bound in span))
            or (span[0] < 0)
            or (span[1] <= span[0])
        ):
            raise ValueError("Phase 0a segment owned span is invalid")
        segment_id = str(profile.get("segment_id") or f"overview-segment-{ordinal:04d}")
        spans.append((span[0], span[1], segment_id))
    spans.sort()
    if any((previous[1] > following[0] for previous, following in zip(spans, spans[1:]))):
        raise ValueError("Phase 0a segment owned spans must not overlap")
    strata: dict[str, list[dict[str, Any]]] = {segment_id: [] for _, _, segment_id in spans}
    fallback_stratum = "recording-wide-unowned-fallback"
    for raw_sample in ordered_samples:
        assert isinstance(raw_sample, Mapping)
        if raw_sample.get("classification") != "positive_main_scene":
            continue
        probabilities = raw_sample.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise ValueError("Phase 0a exemplar probabilities are missing")
        positive_probability = probabilities.get("positive_main_scene")
        if not isinstance(positive_probability, (int, float)) or isinstance(
            positive_probability, bool
        ):
            raise ValueError("Phase 0a positive exemplar probability is invalid")
        if float(positive_probability) < float(minimum_positive_probability):
            continue
        sample_index = int(raw_sample["sample_index"])
        matching = [segment_id for start, end, segment_id in spans if start <= sample_index < end]
        if len(matching) > 1:
            raise ValueError("Phase 0a exemplar belongs to overlapping temporal strata")
        stratum = matching[0] if matching else fallback_stratum
        strata.setdefault(stratum, []).append(deepcopy(dict(raw_sample)))
    nonempty = [(stratum, candidates) for stratum, candidates in strata.items() if candidates]
    if maximum_count == 0 or not nonempty:
        return []
    capacity = min(maximum_count, sum((len(candidates) for _, candidates in nonempty)))
    if len(nonempty) > capacity:
        positions = _uniform_indices(len(nonempty), capacity)
        nonempty = [nonempty[position] for position in positions]
    quotas = {stratum: 0 for stratum, _ in nonempty}
    remaining = capacity
    while remaining:
        progressed = False
        for stratum, candidates in nonempty:
            if remaining == 0:
                break
            if quotas[stratum] >= len(candidates):
                continue
            quotas[stratum] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    selected: list[dict[str, Any]] = []
    for stratum, candidates in nonempty:
        count = quotas[stratum]
        positions = _uniform_indices(len(candidates), count)
        for rank, position in enumerate(positions, start=1):
            selected.append(
                {
                    **deepcopy(candidates[position]),
                    "selection_temporal_stratum": stratum,
                    "rank_within_temporal_stratum": rank,
                }
            )
    return sorted(selected, key=lambda sample: int(sample["sample_index"]))


def _visibility_prompt(
    *,
    profile: Mapping[str, Any],
    context_indices: Sequence[int],
    core_indices: Sequence[int],
    frames: Mapping[int, Mapping[str, Any]],
    visual_inputs: Sequence[Mapping[str, Any]],
    positive_exemplar_indices: Sequence[int] = (),
    positive_exemplar_visual_inputs: Sequence[Mapping[str, Any]] = (),
    main_scene_only: bool = False,
) -> str:
    """Build one dense Phase 0b request with exact output ownership."""
    core_span = _contiguous_sample_span(core_indices, label="visibility core samples")
    target_profile = {
        "target_mode": profile["target_mode"],
        "selected_subjects": _without_mapping_key(profile["selected_subjects"], "model_subject_id"),
        "ignored_transient_people": profile["ignored_transient_people"],
        "scene_summary": profile["scene_summary"],
    }
    payload = {
        "phase0a_target_profile": target_profile,
        "input_context": _dense_frame_context(context_indices, frames),
        "required_core_sample_span": core_span,
        "required_core_sample_indices": list(core_indices),
        "visual_mapping": _dense_visual_mapping(visual_inputs, context_indices),
        "positive_main_scene_exemplars": _positive_exemplar_payload(
            indices=positive_exemplar_indices,
            frames=frames,
            references=positive_exemplar_visual_inputs,
            image_ordinal_offset=len(visual_inputs),
            role="positive_main_scene_exemplar",
            interpretation="Positive, non-temporal references to the selected target/view family. They are not part of the chronological sequence and are not exhaustive; a legitimate unseen target angle may still be main_scene. This list may be empty when Phase 0a found no directly supplied image it could honestly certify as positive; in that case use the recording-wide target profile and chronological evidence without inventing a positive exemplar.",
        ),
        "transition_detection_policy": {
            "multiple_transitions_allowed": True,
            "compare_every_adjacent_context_sample": True,
            "partition_full_context_into_all_maximal_scene_runs": True,
            "boundary_semantics": "first_sample_depicting_the_new_scene",
            "confirmation_delay_allowed": False,
            "majority_labeling_allowed": False,
            "detect_on_full_context_emit_core_only": True,
        },
        "output_contract": {
            "one_record_per_required_core_sample": True,
            "context_samples_must_not_be_returned": True,
            "sample_order": "strictly increasing global sample index",
            "boundary_resolution_us": 100000,
        },
    }
    payload.pop("transition_detection_policy")
    payload["output_contract"] = {
        "samples": [{"sample_index": i, "main_scene": "Boolean"} for i in core_indices],
        "reference_cells_are_not_targets": True,
    }
    return "Decide main-scene membership for each CURRENT frame. Input:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    return (
        "Classify the selected target's scene membership and hand usability for every required core sample. When positive exemplar images are supplied, use them to help establish the recording-wide target/view family; otherwise use the target profile and chronological context. Then inspect the complete chronological context, find every maximal scene run and every boundaryâ€”including multiple switches inside one requestâ€”and finally emit only the owned core samples. Input:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _visibility_relevance_spans(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Run-length encode exact dense Phase 0b labels for its own artifact."""
    if not records:
        raise ValueError("visibility records cannot be empty")
    indices = [record.get("sample_index") for record in records]
    if indices != list(range(len(records))):
        raise ValueError("visibility records must cover the canonical grid")

    def projected(record: Mapping[str, Any]) -> tuple[str, float, str]:
        if record.get("assessment_status") == "not_assessed":
            return ("uncertain", 0.0, "not assessed: outside requested visibility context")
        raw = str(record["relevance"])
        return (
            raw,
            float(record["probabilities"][raw]),
            "deterministic run-length encoding of exact Phase 0b sample labels",
        )

    result: list[dict[str, Any]] = []
    start = 0
    current, first_confidence, current_reason = projected(records[0])
    probabilities: list[float] = [first_confidence]
    for ordinal in range(1, len(records) + 1):
        next_projection = projected(records[ordinal]) if ordinal < len(records) else None
        if (
            next_projection is not None
            and next_projection[0] == current
            and (next_projection[2] == current_reason)
        ):
            probabilities.append(next_projection[1])
            continue
        end = ordinal - 1
        result.append(
            {
                "start_sample": start,
                "end_sample_inclusive": end,
                "relevance": current,
                "confidence": min(probabilities),
                "reason": current_reason,
                "evidence_sample_indices": [start] if start == end else [start, end],
            }
        )
        if ordinal < len(records):
            start = ordinal
            assert next_projection is not None
            current, confidence, current_reason = next_projection
            probabilities = [confidence]
    return result


def _visibility_transition_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive first-new-sample categorical boundaries from exact Phase 0b labels."""
    if not records:
        return []
    indices = [record.get("sample_index") for record in records]
    if indices != list(range(len(records))):
        raise ValueError("visibility transition input must cover the canonical grid")
    fields = (
        "scene_membership",
        "scene_type",
        "target_visibility",
        "left_hand_usability",
        "right_hand_usability",
        "view",
        "relevance",
    )

    def state(record: Mapping[str, Any]) -> dict[str, Any]:
        return {field: record[field] for field in fields}

    result: list[dict[str, Any]] = []
    previous = records[0]
    previous_state = state(previous)
    for current in records[1:]:
        current_state = state(current)
        if (
            current_state != previous_state
            and current.get("assessment_status") != "not_assessed"
            and (previous.get("assessment_status") != "not_assessed")
        ):
            sample_index = int(current["sample_index"])
            result.append(
                {
                    "first_new_state_sample": sample_index,
                    "last_previous_state_sample": sample_index - 1,
                    "from": previous_state,
                    "to": current_state,
                    "previous_request_id": previous.get("request_id"),
                    "current_request_id": current.get("request_id"),
                    "crosses_request_ownership_seam": previous.get("request_id")
                    != current.get("request_id"),
                }
            )
        previous = current
        previous_state = current_state
    return result


def _validated_preprocessing_reuse_identity(
    root: Path, value: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {
        "contract_version",
        "copy_policy",
        "manifest",
        "manifest_sha256",
        "source_bundle_sha256",
        "model_artifacts_reused",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("preprocessing reuse identity is incomplete")
    if (
        value.get("contract_version") != "0.1.0"
        or value.get("copy_policy") != "validated-preprocessing-only-v0.1"
        or value.get("manifest") != "preprocessing-reuse.json"
        or (value.get("model_artifacts_reused") is not False)
    ):
        raise ValueError("preprocessing reuse identity is unsupported")
    for name in ("manifest_sha256", "source_bundle_sha256"):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any((character not in "0123456789abcdef" for character in digest))
        ):
            raise ValueError("preprocessing reuse identity has an invalid digest")
    manifest_path = root / "preprocessing-reuse.json"
    if not manifest_path.is_file() or _file_sha256(manifest_path) != value["manifest_sha256"]:
        raise ValueError("preprocessing reuse manifest identity does not match")
    document = read_json(manifest_path)
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != value["contract_version"]
        or document.get("copy_policy") != value["copy_policy"]
        or (document.get("source_bundle_sha256") != value["source_bundle_sha256"])
        or (document.get("model_artifacts_copied") is not False)
    ):
        raise ValueError("preprocessing reuse manifest contradicts its identity")
    return deepcopy(dict(value))


# Scene selection owns exclusions; Phase2 sees the admitted same-speaker sequence.
def run_phase2_pipeline(
    *,
    run_dir: str | Path,
    source: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    analysis_samples: Sequence[AnalysisSample],
    extracted_frames: Sequence[Mapping[str, Any]],
    config: ResearchConfig,
    model: Phase2Model,
    audio_supplied: bool,
    ffmpeg: str = "ffmpeg",
    export_eaf: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    stop_after: str = "phase3",
    trace: Any | None = None,
    preflight_capabilities: Mapping[str, Any] | None = None,
    pose_conditioning: PoseConditioningResult | None = None,
    preprocessing_reuse: Mapping[str, Any] | None = None,
) -> Phase2RunResult:
    """Run the Phase 0 split, optional Phase 1, and Phases 2/3."""
    started = perf_counter()
    if type(overwrite) is not bool or type(resume) is not bool:
        raise ValueError("overwrite and resume must be Boolean")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    stop_after = validate_pipeline_stage(stop_after, include_preprocessing=False)
    if not config.inference.phase1_enabled and stop_after in {"phase0a", "phase0b"}:
        raise ValueError("Phase1 is disabled; phase0a and phase0b are skipped")
    config.validate_export(export_eaf, detection_run=True)
    from .probability_pass import selected_span

    selected_span(config.detection, len(analysis_samples))
    if export_eaf and stop_after != "phase3":
        raise ValueError("EAF export requires --stop-after phase3")
    from . import main_scene_visibility as scene_visibility

    main_scene_only = True
    visibility_prompt_version = scene_visibility.PROMPT_VERSION
    visibility_contract_version = scene_visibility.PROMPT_VERSION
    visibility_items = config.inference.visibility_max_frames
    visibility_overlap_items = 0
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    preprocessing_reuse_identity = _validated_preprocessing_reuse_identity(
        root, preprocessing_reuse
    )
    reuse_manifest_path = root / "preprocessing-reuse.json"
    if preprocessing_reuse_identity is None and reuse_manifest_path.exists():
        raise ValueError("preprocessing-reuse.json is present without a validated reuse identity")
    reuse_status_artifacts = (
        (reuse_manifest_path,) if preprocessing_reuse_identity is not None else ()
    )
    root_owned = tuple(
        (
            data_path(root / name)
            for name in (*_MODEL_RUN_INTERMEDIATE_ARTIFACT_NAMES, *_MODEL_RUN_FINAL_ARTIFACT_NAMES)
        )
    )
    existing_root = [path for path in root_owned if path.exists()]
    checkpoint_root = root / "model-checkpoints"
    if resume:
        resume_prefix = validate_model_run_resume_state(root)
        durable_completed_stage = _completed_intermediate_stage(resume_prefix)
        if _pipeline_stage_rank(stop_after) < _pipeline_stage_rank(durable_completed_stage):
            raise ValueError(
                f"--stop-after {stop_after} precedes already completed {durable_completed_stage}; use --force or a different run directory"
            )
    elif existing_root and (not overwrite):
        raise FileExistsError(existing_root[0])
    if checkpoint_root.exists() and (not checkpoint_root.is_dir()):
        raise ValueError("model-checkpoints artifact path must be a directory")
    if checkpoint_root.is_dir() and (not overwrite) and (not resume):
        raise FileExistsError(checkpoint_root)
    if overwrite and checkpoint_root.is_dir():
        resolved_root = root.resolve()
        resolved_checkpoint_root = checkpoint_root.resolve()
        if (
            resolved_checkpoint_root.parent != resolved_root
            or resolved_checkpoint_root.name != "model-checkpoints"
        ):
            raise ValueError("refusing to clear an unexpected checkpoint path")
        shutil.rmtree(resolved_checkpoint_root)
    artifact_overwrite = overwrite or resume
    visual_root = root / "model-inputs"
    if visual_root.exists() and (not visual_root.is_dir()):
        raise ValueError("model-inputs artifact path must be a directory")
    if overwrite and visual_root.is_dir():
        resolved_root = root.resolve()
        resolved_visual_root = visual_root.resolve()
        if (
            resolved_visual_root.parent != resolved_root
            or resolved_visual_root.name != "model-inputs"
        ):
            raise ValueError("refusing to clear an unexpected model-input artifact path")
        shutil.rmtree(resolved_visual_root)
    diagnostics_root = root / "diagnostics"
    if overwrite and diagnostics_root.is_dir():
        resolved_root = root.resolve()
        resolved_diagnostics = diagnostics_root.resolve()
        if (
            resolved_diagnostics.parent != resolved_root
            or resolved_diagnostics.name != "diagnostics"
        ):
            raise ValueError("refusing to clear an unexpected diagnostics path")
        shutil.rmtree(resolved_diagnostics)
    if not analysis_samples:
        raise ValueError("Research Stage 2 requires at least one analysis sample")
    _record_event(trace, "pipeline", "analysis", "started", sample_count=len(analysis_samples))
    frame_by_index = _frame_by_index(extracted_frames, analysis_samples)
    pose_frame_by_index, pose_feature_by_index, pose_input_identity = _pose_conditioning_inputs(
        pose_conditioning, analysis_samples
    )
    frame_export_root = _frame_export_source_root(frame_by_index)
    _record_event(trace, "preflight", "capability_check", "started")
    capabilities = dict(
        preflight_capabilities if preflight_capabilities is not None else model.check_capabilities()
    )
    _record_event(trace, "preflight", "capability_check", "completed", capability=capabilities)
    checkpoint_model_capabilities = _checkpoint_model_capabilities(capabilities, semantic=False)
    checkpoint_scope_common = {
        "checkpoint_policy_version": _MODEL_CHECKPOINT_POLICY_VERSION,
        "pipeline_provenance_version": _PIPELINE_PROVENANCE_VERSION,
        "source": to_json_value(source),
        "preprocessing_sha256": _digest(to_json_value(preprocessing)),
        **(
            {"preprocessing_reuse": deepcopy(preprocessing_reuse_identity)}
            if preprocessing_reuse_identity is not None
            else {}
        ),
        "analysis_samples_sha256": _digest([to_json_value(sample) for sample in analysis_samples]),
        "analysis": config.analysis.to_dict(),
        "inference": config.inference.to_dict(),
        **{"detection": config.detection.model_input_dict()},
        "model_endpoint": {
            "contract_version": config.open_webui.contract_version,
            "base_url": config.open_webui.base_url,
            "model": config.open_webui.model,
        },
    }
    semantic_model_capabilities = _checkpoint_model_capabilities(
        checkpoint_model_capabilities, semantic=True
    )
    stable_open_webui_capability_identity = (
        semantic_model_capabilities != checkpoint_model_capabilities
    )
    checkpoint_scope = {
        **deepcopy(checkpoint_scope_common),
        **(
            {"model_capability_identity_policy_version": _MODEL_CAPABILITY_IDENTITY_POLICY_VERSION}
            if stable_open_webui_capability_identity
            else {}
        ),
        "model_capabilities": semantic_model_capabilities,
    }
    checkpointing_model = _CheckpointingPhase2Model(
        model, run_root=root, scope=checkpoint_scope, resume=resume, trace=trace
    )
    model = checkpointing_model
    restricted_transport = _enforce_restricted_transport(
        source=source,
        preprocessing=preprocessing,
        frames=extracted_frames,
        config=config,
        capabilities=capabilities,
    )
    _record_event(
        trace,
        "preflight",
        "restricted_transport_policy",
        "completed",
        restricted_input=restricted_transport is not None,
        transport_mode=restricted_transport,
        destination_scope=capabilities.get("destination_scope"),
        allow_non_private_network=config.open_webui.allow_non_private_network,
        verify_tls=config.open_webui.verify_tls,
        operator_asserts_local_provider=config.open_webui.operator_asserts_local_provider,
        operator_authorizes_restricted_remote_processing=config.open_webui.operator_authorizes_restricted_remote_processing,
    )
    if not config.inference.phase1_enabled:
        from .phase1_bypass import run_without_phase1

        return run_without_phase1(
            root=root,
            samples=analysis_samples,
            source=source,
            preprocessing=preprocessing,
            config=config,
            preprocessing_reuse=preprocessing_reuse_identity,
            frame_by_index=frame_by_index,
            pose_frame_by_index=pose_frame_by_index,
            pose_feature_by_index=pose_feature_by_index,
            pose_input_identity=pose_input_identity,
            pose_conditioning=pose_conditioning,
            model=model,
            audio_supplied=audio_supplied,
            export_eaf=export_eaf,
            overwrite=artifact_overwrite,
            resume=resume,
            trace=trace,
            capability_info=capabilities,
            restricted_transport=restricted_transport,
            ffmpeg=ffmpeg,
            stop_after=stop_after,
        )
    visual_input_records: list[dict[str, Any]] = []
    profile_layout = ContactSheetLayout(
        columns=config.inference.profile_sheet_columns,
        rows=config.inference.profile_sheet_rows,
        tile_width_px=config.inference.profile_tile_width,
        tile_height_px=config.inference.profile_tile_height,
        gutter_px=config.inference.contact_sheet_gutter,
        label_mode="ordinal",
        jpeg_quality=config.inference.profile_jpeg_quality,
    )
    exemplar_layout = _bounded_profile_contact_sheet_layout(
        config,
        maximum_cells=min(
            8, config.inference.profile_sheet_columns * config.inference.profile_sheet_rows
        ),
    )
    profile_grid_selected = fixed_period_indices(
        len(analysis_samples),
        sample_period_us=config.analysis.sample_period_us,
        selection_period_us=config.inference.profile_sample_period_us,
    )
    processing_scope = preprocessing.get("processing_scope") or preprocessing.get("proxy", {}).get(
        "processing_scope"
    )
    if processing_scope:
        first, stop = processing_scope["sample_span"]
        profile_grid_selected = tuple(
            range(
                first,
                stop,
                max(
                    1, config.inference.profile_sample_period_us // config.analysis.sample_period_us
                ),
            )
        )
    planned_profile_windows = plan_index_windows(
        profile_grid_selected,
        items_per_window=config.inference.profile_window_us
        // config.inference.profile_sample_period_us,
        overlap_items=config.inference.profile_overlap_us
        // config.inference.profile_sample_period_us,
    )
    profile_selected = tuple(
        (
            index
            for index in profile_grid_selected
            if analysis_samples[index].source_gap_reason is None
        )
    )
    if not profile_selected:
        raise ValueError("Phase 0a sparse storyboard contains no decodable source samples")
    profile_windows_list: list[RequestWindow] = []
    for planned_window in planned_profile_windows:
        input_indices = tuple(
            (
                index
                for index in planned_window.input_indices
                if analysis_samples[index].source_gap_reason is None
            )
        )
        output_indices = tuple(
            (
                index
                for index in planned_window.output_indices
                if analysis_samples[index].source_gap_reason is None
            )
        )
        if not output_indices:
            continue
        profile_windows_list.append(
            RequestWindow(
                sequence_index=len(profile_windows_list),
                input_indices=input_indices,
                output_indices=output_indices,
            )
        )
    profile_windows = tuple(profile_windows_list)
    _record_event(
        trace,
        "overview",
        "window_plan",
        "completed",
        sample_count=len(profile_selected),
        source_gap_excluded_sample_count=len(profile_grid_selected) - len(profile_selected),
        segment_count=len(profile_windows),
    )
    segment_profiles: list[dict[str, Any]] = []
    segment_progress = StepProgress(
        trace,
        "overview_segments",
        len(profile_windows),
        model=model,
        sample_count=len(profile_selected),
    )
    for segment_number, profile_window in enumerate(profile_windows, start=1):
        indices = list(profile_window.input_indices)
        start = indices[0]
        end = indices[-1] + 1
        segment_id = f"overview-segment-{segment_number:04d}"
        canonical_range = [
            analysis_samples[start].start_us,
            min(
                int(processing_scope["end_us"] if processing_scope else source["duration_us"]),
                analysis_samples[start].start_us + config.inference.profile_window_us,
            ),
        ]
        images, image_refs = _request_visual_inputs(
            root=root,
            request_id=segment_id,
            stage="overview_segment",
            indices=indices,
            frame_by_index=frame_by_index,
            mode=config.inference.profile_image_mode,
            layout=profile_layout,
            maximum_images=config.inference.maximum_images_per_request,
            ffmpeg=ffmpeg,
            overwrite=artifact_overwrite,
            records=visual_input_records,
            trace=trace,
        )
        _record_event(
            trace,
            "overview",
            "segment_request",
            "started",
            request_id=segment_id,
            sample_range=[start, end],
            canonical_range_us=canonical_range,
        )
        segment_call = _model_call(
            model,
            trace,
            "overview",
            request_id=segment_id,
            task="overview_segment",
            system_prompt=overview_system_prompt()
            + " This request is one chronological segment. Its people are candidates for a later recording-level consolidation, not the final target selection.",
            user_prompt=_profile_segment_prompt(
                indices, profile_window.output_indices, frame_by_index, segment_id, image_refs
            ),
            images=images,
            response_schema=PROFILE_SEGMENT_SCHEMA,
            trace_context={
                "prompt_version": OVERVIEW_PROMPT_VERSION,
                "profile_stage": "overlapping-half-hertz-target-overview-window",
                "segment_id": segment_id,
                "canonical_range_us": canonical_range,
                "input_sample_indices": indices,
                "owned_sample_indices": list(profile_window.output_indices),
                "image_refs": image_refs,
            },
            response_validator=lambda value, allowed=set(indices): validate_overview_response(
                value, allowed, allow_empty=True, require_nested_evidence_in_root=False
            ),
        )
        segment_result = dict(segment_call.payload)
        segment_normalization = _profile_call_normalization_record(segment_call)
        segment_profiles.append(
            {
                "profile_id": segment_id,
                "segment_id": segment_id,
                "sample_range": [start, end],
                "canonical_range_us": canonical_range,
                "storyboard_sample_indices": indices,
                "owned_storyboard_sample_indices": list(profile_window.output_indices),
                "owned_sample_span": [
                    profile_window.output_indices[0],
                    profile_window.output_indices[-1] + 1,
                ],
                "request_id": segment_call.request_id,
                "response_sha256": segment_call.response_sha256,
                "model_exchange_trace": segment_call.trace_reference,
                "leaf_profile_count": 1,
                "leaf_profile_span": [segment_id, segment_id],
                "leaf_lineage_sha256": _digest([segment_id]),
                "contract_normalization": segment_normalization,
                "result": segment_result,
            }
        )
        _record_event(
            trace,
            "overview",
            "segment_request",
            "completed",
            request_id=segment_id,
            sample_range=[start, end],
            latency_seconds=segment_call.latency_seconds,
            repaired=segment_call.repaired,
            contract_normalized=bool(segment_normalization["event_count"]),
            normalization_event_count=segment_normalization["event_count"],
        )
        segment_progress.advance()
    segment_progress.finish()
    consolidation_records: list[dict[str, Any]] = list(segment_profiles)
    profile_reductions: list[dict[str, Any]] = []
    reduction_level = 1
    batch_size = config.inference.profile_consolidation_batch_size
    while len(consolidation_records) > batch_size:
        reduction_progress = StepProgress(
            trace,
            "overview_reduction",
            (len(consolidation_records) + batch_size - 1) // batch_size,
            model=model,
            reduction_level=reduction_level,
        )
        next_level: list[dict[str, Any]] = []
        for batch_number, offset in enumerate(
            range(0, len(consolidation_records), batch_size), start=1
        ):
            batch = consolidation_records[offset : offset + batch_size]
            bounded_inputs = [
                _bounded_profile_input(item, config.inference.profile_summary_maximum_characters)
                for item in batch
            ]
            reduction_evidence_scope = _profile_evidence_scope(
                direct_storyboard_indices=(), inherited_records=batch
            )
            allowed_evidence = reduction_evidence_scope["allowed_evidence_sample_indices"]
            reduction_id = f"overview-reduction-{reduction_level:02d}-{batch_number:04d}"
            _record_event(
                trace,
                "overview",
                "reduction_request",
                "started",
                request_id=reduction_id,
                reduction_level=reduction_level,
                input_profile_count=len(batch),
                evidence_scope_policy_version=reduction_evidence_scope["policy_version"],
                inherited_evidence_sample_count=len(
                    reduction_evidence_scope["inherited_profile_sample_indices"]
                ),
                allowed_evidence_sample_count=len(allowed_evidence),
            )
            reduction_call = _model_call(
                model,
                trace,
                "overview",
                request_id=reduction_id,
                task="overview_reduce",
                system_prompt=overview_system_prompt()
                + " This is a summary-only intermediate reduction with no images. Use only supplied summaries and do not invent visual evidence.",
                user_prompt=_profile_reduction_prompt(
                    level=reduction_level,
                    batch_number=batch_number,
                    inputs=bounded_inputs,
                    evidence_scope=reduction_evidence_scope,
                ),
                images=[],
                response_schema=PROFILE_SEGMENT_SCHEMA,
                trace_context={
                    "prompt_version": OVERVIEW_PROMPT_VERSION,
                    "profile_stage": "hierarchical-overview-summary-reduction",
                    "reduction_level": reduction_level,
                    "input_profile_count": len(batch),
                    "input_summary_sha256": _digest(bounded_inputs),
                    "allowed_evidence_sample_indices": allowed_evidence,
                    "profile_evidence_scope": reduction_evidence_scope,
                    "image_refs": [],
                },
                response_validator=lambda value, allowed=set(allowed_evidence): (
                    validate_overview_response(
                        value, allowed, allow_empty=True, require_nested_evidence_in_root=False
                    )
                ),
            )
            reduction_result = dict(reduction_call.payload)
            reduction_normalization = _profile_call_normalization_record(reduction_call)
            first = batch[0]
            last = batch[-1]
            record = {
                "profile_id": reduction_id,
                "sample_range": [first["sample_range"][0], last["sample_range"][1]],
                "owned_sample_span": [first["owned_sample_span"][0], last["owned_sample_span"][1]],
                "canonical_range_us": [
                    first["canonical_range_us"][0],
                    last["canonical_range_us"][1],
                ],
                "request_id": reduction_call.request_id,
                "response_sha256": reduction_call.response_sha256,
                "model_exchange_trace": reduction_call.trace_reference,
                "leaf_profile_count": sum((int(item["leaf_profile_count"]) for item in batch)),
                "leaf_profile_span": [first["leaf_profile_span"][0], last["leaf_profile_span"][1]],
                "leaf_lineage_sha256": _digest([item["leaf_lineage_sha256"] for item in batch]),
                "contract_normalization": reduction_normalization,
                "evidence_scope": reduction_evidence_scope,
                "result": reduction_result,
            }
            profile_reductions.append(
                {
                    **record,
                    "reduction_level": reduction_level,
                    "input_profile_ids": [item["profile_id"] for item in batch],
                    "allowed_evidence_sample_indices": allowed_evidence,
                }
            )
            next_level.append(record)
            _record_event(
                trace,
                "overview",
                "reduction_request",
                "completed",
                request_id=reduction_id,
                reduction_level=reduction_level,
                input_profile_count=len(batch),
                latency_seconds=reduction_call.latency_seconds,
                repaired=reduction_call.repaired,
                contract_normalized=bool(reduction_normalization["event_count"]),
                normalization_event_count=reduction_normalization["event_count"],
            )
            reduction_progress.advance()
        reduction_progress.finish()
        consolidation_records = next_level
        reduction_level += 1
    consolidation_progress = StepProgress(trace, "overview_consolidation", 1, model=model)
    final_profile_inputs = [
        _bounded_profile_input(item, config.inference.profile_summary_maximum_characters)
        for item in consolidation_records
    ]
    profile_indices, final_storyboard_citation_sources = _phase0a_final_storyboard_indices(
        available_indices=profile_selected,
        inherited_records=[
            *consolidation_records,
            *reversed(profile_reductions),
            *segment_profiles,
        ],
        maximum_count=config.inference.profile_samples,
    )
    cited_final_storyboard_indices = sorted(
        (index for index, sources in final_storyboard_citation_sources.items() if sources)
    )
    final_evidence_scope = _profile_evidence_scope(
        direct_storyboard_indices=profile_indices, inherited_records=consolidation_records
    )
    final_allowed_evidence = final_evidence_scope["allowed_evidence_sample_indices"]
    profile_images, profile_image_refs = _request_visual_inputs(
        root=root,
        request_id="overview-recording",
        stage="overview_consolidation",
        indices=profile_indices,
        frame_by_index=frame_by_index,
        mode=config.inference.profile_image_mode,
        layout=profile_layout,
        maximum_images=config.inference.maximum_images_per_request,
        ffmpeg=ffmpeg,
        overwrite=artifact_overwrite,
        records=visual_input_records,
        trace=trace,
    )
    _record_event(
        trace,
        "overview",
        "model_request",
        "started",
        request_id="overview-recording",
        storyboard_sample_indices=profile_indices,
        canonical_range_us=[0, source["duration_us"]],
        evidence_scope_policy_version=final_evidence_scope["policy_version"],
        direct_evidence_sample_count=len(final_evidence_scope["direct_storyboard_sample_indices"]),
        inherited_evidence_sample_count=len(
            final_evidence_scope["inherited_profile_sample_indices"]
        ),
        allowed_evidence_sample_count=len(final_allowed_evidence),
        final_storyboard_selection_policy_version=_PHASE0A_FINAL_STORYBOARD_POLICY_VERSION,
        generic_upstream_evidence_sample_count=len(cited_final_storyboard_indices),
        uniform_recording_coverage_sample_count=len(profile_indices),
    )
    profile_call = _model_call(
        model,
        trace,
        "overview",
        request_id="overview-recording",
        task="overview",
        system_prompt=overview_system_prompt(),
        user_prompt=_profile_consolidation_prompt(
            profile_indices,
            frame_by_index,
            final_profile_inputs,
            profile_image_refs,
            final_evidence_scope,
        ),
        images=profile_images,
        response_schema=final_overview_schema(profile_indices),
        trace_context={
            "prompt_version": OVERVIEW_PROMPT_VERSION,
            "canonical_range_us": [0, source["duration_us"]],
            "profile_input_count": len(final_profile_inputs),
            "profile_input_sha256": _digest(final_profile_inputs),
            "allowed_evidence_sample_indices": final_allowed_evidence,
            "profile_evidence_scope": final_evidence_scope,
            "final_storyboard_selection_policy_version": _PHASE0A_FINAL_STORYBOARD_POLICY_VERSION,
            "generic_upstream_evidence_sample_indices": cited_final_storyboard_indices,
            "final_storyboard_citation_sources": final_storyboard_citation_sources,
            "image_refs": profile_image_refs,
        },
        response_validator=lambda value: validate_overview_response(
            value,
            set(final_allowed_evidence),
            require_single_subject=True,
            require_nested_evidence_in_root=False,
            require_direct_evidence=True,
            allowed_direct_indices=set(profile_indices),
        ),
    )
    contract_profile = dict(profile_call.payload)
    recording_profile = _normalize_profile(contract_profile)
    final_profile_normalization = _profile_call_normalization_record(profile_call)
    consolidation_progress.advance()
    consolidation_progress.finish(
        selected_subject_count=len(recording_profile["selected_subjects"])
    )
    _record_event(
        trace,
        "overview",
        "model_request",
        "completed",
        request_id=profile_call.request_id,
        latency_seconds=profile_call.latency_seconds,
        repaired=profile_call.repaired,
        contract_normalized=bool(final_profile_normalization["event_count"]),
        normalization_event_count=final_profile_normalization["event_count"],
    )
    if not isinstance(profile_call.model_payload, Mapping):
        raise ValueError("final profile call did not preserve its raw model payload")
    recording_profile["model_output"] = deepcopy(dict(profile_call.model_payload))
    recording_profile["contract_normalized_output"] = deepcopy(dict(profile_call.payload))
    recording_profile["response_sha256"] = profile_call.response_sha256
    recording_profile["request_id"] = profile_call.request_id
    recording_profile["model_exchange_trace"] = profile_call.trace_reference
    recording_profile["storyboard_sample_indices"] = profile_indices
    recording_profile["evidence_scope"] = final_evidence_scope
    recording_profile["segment_profiles"] = segment_profiles
    recording_profile["profile_reductions"] = profile_reductions
    segment_normalization_event_count = sum(
        (int(item["contract_normalization"]["event_count"]) for item in segment_profiles)
    )
    reduction_normalization_event_count = sum(
        (int(item["contract_normalization"]["event_count"]) for item in profile_reductions)
    )
    final_normalization_event_count = int(final_profile_normalization["event_count"])
    all_profile_call_normalizations = [
        *(item["contract_normalization"] for item in segment_profiles),
        *(item["contract_normalization"] for item in profile_reductions),
        final_profile_normalization,
    ]
    recording_profile["contract_normalization"] = {
        "policy_version": PROFILE_NORMALIZATION_VERSION,
        "final_profile": final_profile_normalization,
        "segment_event_count": segment_normalization_event_count,
        "reduction_event_count": reduction_normalization_event_count,
        "final_profile_event_count": final_normalization_event_count,
        "total_event_count": segment_normalization_event_count
        + reduction_normalization_event_count
        + final_normalization_event_count,
        "normalized_irrelevant_gate_policy": "downgrade-to-uncertain",
        "raw_model_payloads_preserved": all(
            (bool(item["raw_model_payload_preserved"]) for item in all_profile_call_normalizations)
        ),
    }
    recording_profile["storyboard_coverage"] = {
        "policy": "complete recording in overlapping 0.5-Hz storyboard windows, bounded hierarchical consolidation, then global storyboard consolidation",
        "canonical_analysis_rate_hz": config.analysis.sample_rate_hz,
        "profile_sample_period_us": config.inference.profile_sample_period_us,
        "profile_window_us": config.inference.profile_window_us,
        "profile_overlap_us": config.inference.profile_overlap_us,
        "planned_storyboard_sample_count": len(profile_grid_selected),
        "covered_storyboard_sample_count": len(profile_selected),
        "source_gap_excluded_sample_count": len(profile_grid_selected) - len(profile_selected),
        "segment_count": len(segment_profiles),
        "reduction_request_count": len(profile_reductions),
        "reduction_level_count": reduction_level - 1,
        "maximum_inputs_per_reduction": batch_size,
        "maximum_result_excerpt_characters": config.inference.profile_summary_maximum_characters,
        "final_profile_input_count": len(final_profile_inputs),
        "consolidation_storyboard_sample_indices": profile_indices,
        "consolidation_storyboard_selection": {
            "policy_version": _PHASE0A_FINAL_STORYBOARD_POLICY_VERSION,
            "uniform_recording_coverage_sample_indices": profile_indices,
            "generic_upstream_evidence_sample_indices": cited_final_storyboard_indices,
            "upstream_citation_sources_by_sample": {
                str(index): sources
                for index, sources in final_storyboard_citation_sources.items()
                if sources
            },
            "semantics": "generic upstream citations are provenance only and do not control the uniform final storyboard or positive-exemplar verification",
        },
        "evidence_scope_policy_version": final_evidence_scope["policy_version"],
        "direct_evidence_sample_count": len(
            final_evidence_scope["direct_storyboard_sample_indices"]
        ),
        "inherited_evidence_sample_count": len(
            final_evidence_scope["inherited_profile_sample_indices"]
        ),
        "allowed_evidence_sample_count": len(final_allowed_evidence),
        "canonical_range_us": [0, source["duration_us"]],
    }
    phase0a_exemplar_capacity = _positive_exemplar_sample_capacity(
        config,
        chronological_stage="visibility",
        chronological_sample_count=min(len(analysis_samples), visibility_items),
    )
    exemplar_verification_request_id = "overview-exemplar-verification"
    exemplar_progress = StepProgress(
        trace, "scene_exemplars", 1, model=model, sample_count=len(profile_indices)
    )
    target_subject_ids = [
        str(subject["subject_id"]) for subject in recording_profile["selected_subjects"]
    ]
    _record_event(
        trace,
        "overview",
        "exemplar_verification",
        "started",
        request_id=exemplar_verification_request_id,
        candidate_sample_count=len(profile_indices),
        prompt_version=PHASE0A_EXEMPLAR_PROMPT_VERSION,
        contract_version=PHASE0A_EXEMPLAR_CONTRACT_VERSION,
    )
    exemplar_verification_call = _model_call(
        model,
        trace,
        "overview",
        request_id=exemplar_verification_request_id,
        task="overview_exemplar_verification",
        system_prompt=phase0a_exemplar_verification_system_prompt(),
        user_prompt=_phase0a_exemplar_verification_prompt(
            profile=recording_profile,
            indices=profile_indices,
            frames=frame_by_index,
            visual_inputs=profile_image_refs,
        ),
        images=profile_images,
        response_schema=phase0a_exemplar_schema_for_count(len(profile_indices)),
        trace_context={
            "prompt_version": PHASE0A_EXEMPLAR_PROMPT_VERSION,
            "contract_version": PHASE0A_EXEMPLAR_CONTRACT_VERSION,
            "candidate_sample_indices": profile_indices,
            "target_subject_ids": target_subject_ids,
            "image_refs": profile_image_refs,
        },
        response_validator=lambda value: validate_phase0a_exemplar_response(
            value, expected_indices=profile_indices, expected_subject_ids=target_subject_ids
        ),
    )
    exemplar_verification = dict(exemplar_verification_call.payload)
    exemplar_progress.advance()
    exemplar_progress.finish()
    if not isinstance(exemplar_verification_call.model_payload, Mapping):
        raise ValueError("Phase 0a exemplar verification did not preserve its raw model payload")
    selected_exemplar_records = _select_phase0a_verified_exemplars(
        exemplar_verification["samples"],
        segment_profiles=segment_profiles,
        maximum_count=min(8, phase0a_exemplar_capacity),
        minimum_positive_probability=_PHASE0A_EXEMPLAR_MINIMUM_POSITIVE_PROBABILITY,
    )
    proposed_exemplar_records = list(selected_exemplar_records)
    proposed_exemplar_indices = [
        int(sample["sample_index"]) for sample in proposed_exemplar_records
    ]
    confirmation_layout = _bounded_profile_contact_sheet_layout(config, maximum_cells=1)
    exemplar_confirmation_requests: list[dict[str, Any]] = []
    confirmation_by_index: dict[int, dict[str, Any]] = {}
    confirmed_exemplar_records: list[dict[str, Any]] = []
    confirmation_progress = StepProgress(
        trace, "scene_exemplar_confirmation", len(proposed_exemplar_records), model=model
    )
    for proposed_sample in proposed_exemplar_records:
        sample_index = int(proposed_sample["sample_index"])
        confirmation_request_id = f"overview-exemplar-confirmation-{sample_index:08d}"
        confirmation_images, confirmation_image_refs = _request_visual_inputs(
            root=root,
            request_id=confirmation_request_id,
            stage="overview_exemplar_confirmation",
            indices=[sample_index],
            frame_by_index=frame_by_index,
            mode=config.inference.profile_image_mode,
            layout=confirmation_layout,
            maximum_images=config.inference.maximum_images_per_request,
            ffmpeg=ffmpeg,
            overwrite=artifact_overwrite,
            records=visual_input_records,
            trace=trace,
        )
        if len(confirmation_images) != 1:
            raise RuntimeError("Phase 0a exemplar confirmation must use exactly one image")
        _record_event(
            trace,
            "overview",
            "exemplar_confirmation",
            "started",
            request_id=confirmation_request_id,
            sample_index=sample_index,
            policy_version=_PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
        )
        confirmation_call = _model_call(
            model,
            trace,
            "overview",
            request_id=confirmation_request_id,
            task="overview_exemplar_verification",
            system_prompt=phase0a_exemplar_verification_system_prompt(),
            user_prompt=_phase0a_exemplar_verification_prompt(
                profile=recording_profile,
                indices=[sample_index],
                frames=frame_by_index,
                visual_inputs=confirmation_image_refs,
            ),
            images=confirmation_images,
            response_schema=phase0a_exemplar_schema_for_count(1),
            trace_context={
                "prompt_version": PHASE0A_EXEMPLAR_PROMPT_VERSION,
                "contract_version": PHASE0A_EXEMPLAR_CONTRACT_VERSION,
                "confirmation_policy_version": _PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
                "verification_pass": "individual_confirmation",
                "candidate_sample_indices": [sample_index],
                "target_subject_ids": target_subject_ids,
                "image_refs": confirmation_image_refs,
            },
            response_validator=lambda value, expected=(sample_index,): (
                validate_phase0a_exemplar_response(
                    value, expected_indices=expected, expected_subject_ids=target_subject_ids
                )
            ),
        )
        if not isinstance(confirmation_call.model_payload, Mapping):
            raise ValueError(
                "Phase 0a exemplar confirmation did not preserve its raw model payload"
            )
        confirmation_payload = dict(confirmation_call.payload)
        confirmation_sample = deepcopy(dict(confirmation_payload["samples"][0]))
        confirmed = (
            confirmation_sample["classification"] == "positive_main_scene"
            and float(confirmation_sample["probabilities"]["positive_main_scene"])
            >= _PHASE0A_EXEMPLAR_MINIMUM_POSITIVE_PROBABILITY
        )
        confirmation_record = {
            "policy_version": _PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
            "sample_index": sample_index,
            "request_id": confirmation_call.request_id,
            "response_sha256": confirmation_call.response_sha256,
            "model_exchange_trace": confirmation_call.trace_reference,
            "model_output": deepcopy(dict(confirmation_call.model_payload)),
            "contract_normalized_output": confirmation_payload,
            "image_refs": confirmation_image_refs,
            "confirmed_positive": confirmed,
            "repaired": confirmation_call.repaired,
            "latency_seconds": confirmation_call.latency_seconds,
        }
        exemplar_confirmation_requests.append(confirmation_record)
        confirmation_by_index[sample_index] = confirmation_record
        if confirmed:
            confirmed_exemplar_records.append(proposed_sample)
        _record_event(
            trace,
            "overview",
            "exemplar_confirmation",
            "completed",
            request_id=confirmation_call.request_id,
            sample_index=sample_index,
            policy_version=_PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
            confirmed_positive=confirmed,
            latency_seconds=confirmation_call.latency_seconds,
            repaired=confirmation_call.repaired,
        )
        confirmation_progress.advance()
    confirmation_progress.finish(sample_count=len(confirmed_exemplar_records))
    selected_exemplar_records = confirmed_exemplar_records
    phase0a_exemplar_indices = [int(sample["sample_index"]) for sample in selected_exemplar_records]
    selected_record_by_index = {
        int(sample["sample_index"]): sample for sample in selected_exemplar_records
    }
    verified_positive_indices = [
        int(sample["sample_index"])
        for sample in exemplar_verification["samples"]
        if sample["classification"] == "positive_main_scene"
    ]
    phase0a_representatives = tuple(
        (
            PositiveRepresentative(
                sample_index=index,
                positive_label="positive_main_scene",
                probability=float(
                    selected_record_by_index[index]["probabilities"]["positive_main_scene"]
                ),
                view=None,
                category=str(selected_record_by_index[index]["scene_type"]),
                rank_within_stratum=int(
                    selected_record_by_index[index]["rank_within_temporal_stratum"]
                ),
                rationale="dedicated Phase 0a verification and isolated single-image confirmation both classified this image as a high-confidence selected-live-target main-scene exemplar",
                metadata={
                    "source_stage": "phase0a_exemplar_verification",
                    "verification_request_id": exemplar_verification_call.request_id,
                    "verification_response_sha256": exemplar_verification_call.response_sha256,
                    "verification_contract_version": PHASE0A_EXEMPLAR_CONTRACT_VERSION,
                    "verification_prompt_version": PHASE0A_EXEMPLAR_PROMPT_VERSION,
                    "confirmation_policy_version": _PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
                    "confirmation_request_id": confirmation_by_index[index]["request_id"],
                    "confirmation_response_sha256": confirmation_by_index[index]["response_sha256"],
                    "confirmation_model_exchange_trace": confirmation_by_index[index][
                        "model_exchange_trace"
                    ],
                    "selection_policy_version": _PHASE0A_EXEMPLAR_SELECTION_POLICY_VERSION,
                    "minimum_positive_probability": _PHASE0A_EXEMPLAR_MINIMUM_POSITIVE_PROBABILITY,
                    "temporal_stratum": selected_record_by_index[index][
                        "selection_temporal_stratum"
                    ],
                    "verification_decision": {
                        key: deepcopy(selected_record_by_index[index][key])
                        for key in (
                            "scene_type",
                            "selected_target_visibility",
                            "classification",
                            "probabilities",
                            "reason",
                        )
                    },
                    "confirmation_decision": {
                        key: deepcopy(
                            confirmation_by_index[index]["contract_normalized_output"]["samples"][
                                0
                            ][key]
                        )
                        for key in (
                            "scene_type",
                            "selected_target_visibility",
                            "classification",
                            "probabilities",
                            "reason",
                        )
                    },
                    "candidate_selection_basis": "uniform_recording_coverage",
                    "upstream_model_citations": final_storyboard_citation_sources.get(index, []),
                    "supplied_to_phase0b": True,
                },
            )
            for index in phase0a_exemplar_indices
        )
    )
    phase0a_export_dir = root / "diagnostics" / "phase0a-main-scene-exemplars"
    phase0a_manifest_path = export_positive_representatives(
        phase0a_representatives,
        frame_by_index=frame_by_index,
        destination=phase0a_export_dir,
        source_root=frame_export_root,
        export_metadata={
            "stage": "phase0a",
            "prompt_version": PHASE0A_EXEMPLAR_PROMPT_VERSION,
            "contract_version": PHASE0A_EXEMPLAR_CONTRACT_VERSION,
            "request_id": exemplar_verification_call.request_id,
            "response_sha256": exemplar_verification_call.response_sha256,
            "selection_policy_version": _PHASE0A_EXEMPLAR_SELECTION_POLICY_VERSION,
            "confirmation_policy_version": _PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
            "selection_semantics": "high-confidence independently verified selected-live-target main-scene examples across temporal strata that also passed an isolated single-image confirmation",
            "minimum_positive_probability": _PHASE0A_EXEMPLAR_MINIMUM_POSITIVE_PROBABILITY,
            "per_image_probability_available": True,
            "per_image_view_available": False,
            "candidate_sample_indices": profile_indices,
            "verified_positive_sample_indices": verified_positive_indices,
            "proposed_positive_sample_indices": proposed_exemplar_indices,
            "confirmed_positive_sample_indices": phase0a_exemplar_indices,
            "confirmation_request_ids": [
                record["request_id"] for record in exemplar_confirmation_requests
            ],
            "confirmation_response_sha256_by_sample": {
                str(record["sample_index"]): record["response_sha256"]
                for record in exemplar_confirmation_requests
            },
            "phase0b_image_capacity": phase0a_exemplar_capacity,
            "phase0b_supplied_sample_indices": phase0a_exemplar_indices,
        },
        overwrite=artifact_overwrite,
    )
    if phase0a_exemplar_indices:
        phase0a_exemplar_images, phase0a_exemplar_image_refs = _request_visual_inputs(
            root=root,
            request_id="phase0a-main-scene-exemplars-global-v1",
            stage="overview_exemplars",
            visible_labels={i: f"REFERENCE f{i}" for i in phase0a_exemplar_indices},
            indices=phase0a_exemplar_indices,
            frame_by_index=frame_by_index,
            mode=config.inference.profile_image_mode,
            layout=exemplar_layout,
            maximum_images=config.inference.maximum_images_per_request,
            ffmpeg=ffmpeg,
            overwrite=artifact_overwrite,
            records=visual_input_records,
            trace=trace,
        )
    else:
        phase0a_exemplar_images = []
        phase0a_exemplar_image_refs = []
    phase0a_manifest_sha256 = sha256(phase0a_manifest_path.read_bytes()).hexdigest()
    recording_profile["positive_main_scene_exemplars"] = {
        "selection_policy_version": _PHASE0A_EXEMPLAR_SELECTION_POLICY_VERSION,
        "verification_contract_version": PHASE0A_EXEMPLAR_CONTRACT_VERSION,
        "verification_prompt_version": PHASE0A_EXEMPLAR_PROMPT_VERSION,
        "verification_request_id": exemplar_verification_call.request_id,
        "verification_response_sha256": exemplar_verification_call.response_sha256,
        "verification_model_exchange_trace": exemplar_verification_call.trace_reference,
        "verification_model_output": deepcopy(dict(exemplar_verification_call.model_payload)),
        "verification_contract_output": exemplar_verification,
        "candidate_sample_indices": profile_indices,
        "verified_positive_sample_indices": verified_positive_indices,
        "proposed_positive_sample_indices": proposed_exemplar_indices,
        "minimum_positive_probability": _PHASE0A_EXEMPLAR_MINIMUM_POSITIVE_PROBABILITY,
        "confirmation": {
            "policy_version": _PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
            "prompt_version": PHASE0A_EXEMPLAR_PROMPT_VERSION,
            "contract_version": PHASE0A_EXEMPLAR_CONTRACT_VERSION,
            "minimum_positive_probability": _PHASE0A_EXEMPLAR_MINIMUM_POSITIVE_PROBABILITY,
            "input_cardinality": "exactly_one_proposed_positive_per_request",
            "fixed_target_profile": True,
            "other_exemplars_supplied": False,
            "backfill_allowed": False,
            "proposed_sample_indices": proposed_exemplar_indices,
            "confirmed_sample_indices": phase0a_exemplar_indices,
            "request_ids": [record["request_id"] for record in exemplar_confirmation_requests],
            "requests": exemplar_confirmation_requests,
        },
        "sample_indices": phase0a_exemplar_indices,
        "phase0b_supplied_sample_indices": phase0a_exemplar_indices,
        "positive_evidence_cardinality": "zero_to_eight verified and individually confirmed selected-live-target positives; never padded or backfilled",
        "export_directory": str(phase0a_export_dir.relative_to(root)).replace("\\", "/"),
        "manifest": str(phase0a_manifest_path.relative_to(root)).replace("\\", "/"),
        "manifest_sha256": phase0a_manifest_sha256,
        "model_input_request_id": "phase0a-main-scene-exemplars",
    }
    _record_event(
        trace,
        "overview",
        "exemplar_verification",
        "completed",
        request_id=exemplar_verification_call.request_id,
        latency_seconds=exemplar_verification_call.latency_seconds,
        repaired=exemplar_verification_call.repaired,
        verified_positive_count=len(verified_positive_indices),
        selected_positive_count=len(proposed_exemplar_indices),
        selected_positive_sample_indices=proposed_exemplar_indices,
    )
    _record_event(
        trace,
        "overview",
        "exemplar_confirmation",
        "completed",
        policy_version=_PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
        proposed_positive_count=len(proposed_exemplar_indices),
        proposed_positive_sample_indices=proposed_exemplar_indices,
        confirmed_positive_count=len(phase0a_exemplar_indices),
        confirmed_positive_sample_indices=phase0a_exemplar_indices,
        rejected_positive_sample_indices=sorted(
            set(proposed_exemplar_indices) - set(phase0a_exemplar_indices)
        ),
        request_ids=[record["request_id"] for record in exemplar_confirmation_requests],
    )
    _record_event(
        trace,
        "overview",
        "representative_export",
        "completed",
        directory=str(phase0a_export_dir.relative_to(root)).replace("\\", "/"),
        representative_count=len(phase0a_exemplar_indices),
        representative_sample_indices=phase0a_exemplar_indices,
        phase0b_supplied_sample_indices=phase0a_exemplar_indices,
        manifest_sha256=phase0a_manifest_sha256,
    )
    recording_profile["overview_version"] = OVERVIEW_PROMPT_VERSION
    overview_path = write_json(root / "recording-overview.json", recording_profile)
    overview_artifact_sha256 = sha256(overview_path.read_bytes()).hexdigest()
    status_path = write_pipeline_status(
        root,
        requested_stop_after=stop_after,
        completed_stage="phase0a",
        artifacts=(overview_path, *reuse_status_artifacts),
        source_sha256=str(source["sha256"])
        if isinstance(source.get("sha256"), str) and len(str(source["sha256"])) == 64
        else None,
        preprocessing_sha256=_digest(to_json_value(preprocessing)),
    )
    if stop_after == "phase0a":
        _record_event(trace, "pipeline", "requested_stop", "completed", completed_stage="phase0a")
        return Phase2RunResult(
            run_id=None,
            target_mode=str(recording_profile["target_mode"]),
            analyses=(),
            analysis_paths=(),
            index_path=None,
            completed_stage="phase0a",
            status_path=status_path,
            stage_artifact_paths=(overview_path,),
        )
    _record_event(
        trace,
        "overview",
        "target_selection",
        "completed",
        selected_subject_count=len(recording_profile["selected_subjects"]),
    )
    visibility_layout = ContactSheetLayout(
        columns=config.inference.visibility_sheet_columns,
        rows=config.inference.visibility_sheet_rows,
        tile_width_px=config.inference.visibility_tile_width,
        tile_height_px=config.inference.visibility_tile_height,
        gutter_px=config.inference.contact_sheet_gutter,
        label_mode="ordinal",
        jpeg_quality=config.inference.visibility_jpeg_quality,
    )
    from .visibility_scope import visibility_scope, unassessed_visibility_sample

    visibility_selection = visibility_scope(
        config, len(analysis_samples), processing=processing_scope
    )
    visibility_start, visibility_end = visibility_selection["assessed_sample_span"]
    visibility_windows = plan_index_windows(
        range(visibility_start, visibility_end),
        items_per_window=visibility_items,
        overlap_items=visibility_overlap_items,
    )
    target_subject_ids = [
        str(subject["subject_id"]) for subject in recording_profile["selected_subjects"]
    ]
    recurring_views = {
        str(view)
        for subject in recording_profile["selected_subjects"]
        for view in subject.get("recurring_views", [])
    }
    _record_event(
        trace,
        "visibility",
        "window_plan",
        "completed",
        sample_count=visibility_end - visibility_start,
        recording_sample_count=len(analysis_samples),
        time_selection=visibility_selection,
        window_count=len(visibility_windows),
        window_us=visibility_items * config.analysis.sample_period_us,
        overlap_us=visibility_overlap_items * config.analysis.sample_period_us,
    )
    visibility_records: list[dict[str, Any]] = []
    visibility_requests: list[dict[str, Any]] = []
    visibility_contract_normalizations: list[dict[str, Any]] = []
    visibility_source_gap_normalizations: list[dict[str, Any]] = []
    visibility_progress = StepProgress(
        trace,
        "visibility",
        len(visibility_windows),
        model=model,
        sample_count=visibility_end - visibility_start,
        target_frames=visibility_items,
    )
    for visibility_window in visibility_windows:
        context = list(visibility_window.input_indices)
        core = list(visibility_window.output_indices)
        start = core[0]
        end = core[-1] + 1
        prefix = "visibility-main-scene-v1"
        request_id = f"{prefix}-{start:08d}-{end:08d}"
        images, image_refs = _request_visual_inputs(
            root=root,
            request_id=request_id,
            stage="visibility",
            visible_labels={i: f"CURRENT f{i}" for i in context},
            indices=context,
            frame_by_index=frame_by_index,
            mode=config.inference.visibility_image_mode,
            layout=visibility_layout,
            maximum_images=config.inference.maximum_images_per_request,
            ffmpeg=ffmpeg,
            overwrite=artifact_overwrite,
            records=visual_input_records,
            trace=trace,
        )
        combined_visibility_images = [*images, *phase0a_exemplar_images]
        if len(combined_visibility_images) > config.inference.maximum_images_per_request:
            raise ValueError(
                "Phase 0b full-scene and positive exemplar images exceed maximum_images_per_request"
            )
        combined_visibility_refs = [
            *_role_visual_references(
                image_refs, image_ordinal_offset=0, role="chronological_context"
            ),
            *_role_visual_references(
                phase0a_exemplar_image_refs,
                image_ordinal_offset=len(image_refs),
                role="positive_main_scene_exemplar",
            ),
        ]
        _record_event(
            trace,
            "visibility",
            "model_request",
            "started",
            request_id=request_id,
            core_sample_range=[start, end],
            context_sample_range=[context[0], context[-1] + 1],
        )
        visibility_call = _model_call(
            model,
            trace,
            "visibility",
            request_id=request_id,
            task=scene_visibility.TASK,
            system_prompt=scene_visibility.SYSTEM_PROMPT,
            user_prompt=_visibility_prompt(
                profile=recording_profile,
                context_indices=context,
                core_indices=core,
                frames=frame_by_index,
                visual_inputs=image_refs,
                positive_exemplar_indices=phase0a_exemplar_indices,
                positive_exemplar_visual_inputs=phase0a_exemplar_image_refs,
                main_scene_only=main_scene_only,
            ),
            images=combined_visibility_images,
            response_schema=scene_visibility.response_schema(len(core)),
            trace_context={
                "prompt_version": visibility_prompt_version,
                "target_subject_ids": target_subject_ids,
                "allowed_recurring_views": sorted(recurring_views),
                "core_sample_indices": core,
                "context_sample_indices": context,
                "image_refs": combined_visibility_refs,
                "positive_main_scene_exemplar_sample_indices": phase0a_exemplar_indices,
            },
            response_validator=lambda value, expected=tuple(core): (
                scene_visibility.validate_response(value, expected)
            ),
        )
        visibility_normalization = {
            "policy_version": scene_visibility.PROJECTION_VERSION,
            "request_id": request_id,
            "normalization_applied": False,
            "event_count": 0,
            "events": [],
            "projection": "binary_scene_to_gate_fields_without_hand_claims",
        }
        visibility_contract_normalizations.append(visibility_normalization)
        validated_visibility = scene_visibility.project_response(visibility_call.payload)
        for record in validated_visibility["samples"]:
            canonical_record = deepcopy(dict(record))
            sample_index = int(canonical_record["sample_index"])
            source_gap_reason = analysis_samples[sample_index].source_gap_reason
            if source_gap_reason is not None:
                canonical_record.update(
                    {
                        "scene_membership": "uncertain",
                        "scene_type": "uncertain",
                        "target_visibility": "uncertain",
                        "left_hand_usability": "unusable",
                        "right_hand_usability": "unusable",
                        "view": None,
                        "relevance": "irrelevant",
                        "probabilities": {"relevant": 0.0, "irrelevant": 1.0, "uncertain": 0.0},
                        "reason": "decoder source gap; visual evidence unavailable",
                    }
                )
                canonical_record.update(main_scene=False, assessment_status="source_gap")
                visibility_source_gap_normalizations.append(
                    {
                        "sample_index": sample_index,
                        "source_gap_reason": source_gap_reason,
                        "request_id": request_id,
                        "model_sample_sha256": _digest(record),
                        "canonical_sample_sha256": _digest(canonical_record),
                    }
                )
            visibility_records.append(
                {
                    **canonical_record,
                    "request_id": request_id,
                    "response_sha256": visibility_call.response_sha256,
                }
            )
        visibility_requests.append(
            {
                "request_id": request_id,
                "core_sample_indices": core,
                "context_sample_indices": context,
                "positive_main_scene_exemplar_sample_indices": phase0a_exemplar_indices,
                "image_refs": combined_visibility_refs,
                "response_sha256": visibility_call.response_sha256,
                "model_exchange_trace": visibility_call.trace_reference,
                "model_output": deepcopy(dict(visibility_call.model_payload))
                if isinstance(visibility_call.model_payload, Mapping)
                else None,
                "contract_normalized_output": deepcopy(dict(visibility_call.payload)),
                "contract_normalization": deepcopy(visibility_normalization),
            }
        )
        _record_event(
            trace,
            "visibility",
            "model_request",
            "completed",
            request_id=request_id,
            core_sample_range=[start, end],
            latency_seconds=visibility_call.latency_seconds,
            repaired=visibility_call.repaired,
            contract_normalized=visibility_normalization["normalization_applied"],
            normalization_event_count=visibility_normalization["event_count"],
        )
        visibility_progress.advance()
    if [record["sample_index"] for record in visibility_records] != list(
        range(visibility_start, visibility_end)
    ):
        raise RuntimeError("Phase 0b did not produce exact requested visibility coverage")
    visibility_progress.finish()
    visibility_records = [
        *(unassessed_visibility_sample(i) for i in range(visibility_start)),
        *visibility_records,
        *(unassessed_visibility_sample(i) for i in range(visibility_end, len(analysis_samples))),
    ]
    visibility_spans = _visibility_relevance_spans(visibility_records)
    visibility_transitions = _visibility_transition_records(visibility_records)
    _record_event(
        trace,
        "visibility",
        "transition_diagnostics",
        "completed",
        transition_count=len(visibility_transitions),
        transition_sample_indices=[
            item["first_new_state_sample"] for item in visibility_transitions
        ],
        request_seam_transition_count=sum(
            (bool(item["crosses_request_ownership_seam"]) for item in visibility_transitions)
        ),
        boundary_semantics="first_sample_depicting_the_new_state",
    )
    visibility_analysis = {
        "schema_version": "0.1.0",
        "contract_version": visibility_contract_version,
        "prompt_version": visibility_prompt_version,
        "analysis_rate_hz": config.analysis.sample_rate_hz,
        "time_selection": visibility_selection,
        "sample_period_us": config.analysis.sample_period_us,
        "window_us": visibility_items * config.analysis.sample_period_us,
        "overlap_us": visibility_overlap_items * config.analysis.sample_period_us,
        "target_subject_ids": target_subject_ids,
        "overview_artifact": "recording-overview.json",
        "overview_artifact_sha256": overview_artifact_sha256,
        "pose_conditioning": {
            "consumed": False,
            "available": pose_input_identity is not None,
            "reserved_for_stages": ["phase1", "phase2"],
        },
        "samples": visibility_records,
        "relevance_spans": visibility_spans,
        "transitions": visibility_transitions,
        "transition_detection": {
            "policy_version": "phase0b-explicit-transitions-v0.1",
            "multiple_transitions_per_window_allowed": True,
            "boundary_semantics": "first_sample_depicting_the_new_state",
            "derived_from_exact_sample_labels": True,
            "transition_count": len(visibility_transitions),
        },
        "requests": visibility_requests,
        "contract_normalization": {
            "policy_version": scene_visibility.PROJECTION_VERSION,
            "request_count": len(visibility_contract_normalizations),
            "normalized_request_count": sum(
                (
                    bool(record["normalization_applied"])
                    for record in visibility_contract_normalizations
                )
            ),
            "event_count": sum(
                (int(record["event_count"]) for record in visibility_contract_normalizations)
            ),
            "requests": visibility_contract_normalizations,
        },
        "source_gap_normalization": {
            "policy_version": "visibility-source-gap-v0.1",
            "event_count": len(visibility_source_gap_normalizations),
            "events": visibility_source_gap_normalizations,
        },
        "gate_policy": {
            "version": VISIBILITY_GATE_VERSION,
            "enabled": True,
            "minimum_irrelevant_probability": 0.9,
            "other_scene_or_target_absent": "gate_both_hands",
            "either_hand_unusable": "gate_both_hands_for_joint_phase2a",
            "uncertain_or_low_confidence": "analyze",
        },
    }
    visibility_analysis["gate_policy"] = {
        "version": "property-evidence-resolution-v0.1",
        "enabled": True,
        "minimum_irrelevant_probability": 0.9,
        "other_scene_or_target_absent": "retain_report_and_recheck_pose_conflict",
        "either_hand_unusable": "resolve_each_property_across_pose_and_rgb",
        "uncertain_or_low_confidence": "analyze_available_evidence",
    }
    visibility_analysis["gate_policy"] = {
        "version": scene_visibility.PROMPT_VERSION,
        "enabled": True,
        "main_scene_true": "admit_selected_live_speaker; hand usability not assessed",
        "main_scene_false": "exclude_from_phase2",
        "probabilities": "local binary encoding, not model confidence",
    }
    visibility_path = write_json(root / "scene-usability.json", visibility_analysis)
    visibility_artifact_sha256 = sha256(visibility_path.read_bytes()).hexdigest()
    status_path = write_pipeline_status(
        root,
        requested_stop_after=stop_after,
        completed_stage="phase0b",
        artifacts=(overview_path, visibility_path, *reuse_status_artifacts),
        source_sha256=str(source["sha256"])
        if isinstance(source.get("sha256"), str) and len(str(source["sha256"])) == 64
        else None,
        preprocessing_sha256=_digest(to_json_value(preprocessing)),
    )
    if stop_after == "phase0b":
        _record_event(trace, "pipeline", "requested_stop", "completed", completed_stage="phase0b")
        return Phase2RunResult(
            run_id=None,
            target_mode=str(recording_profile["target_mode"]),
            analyses=(),
            analysis_paths=(),
            index_path=None,
            completed_stage="phase0b",
            status_path=status_path,
            stage_artifact_paths=(overview_path, visibility_path),
        )
    recording_profile["annotation_relevance_spans"] = visibility_spans
    recording_profile["profile_version"] = "recording-context-v0.2"
    recording_profile["overview"] = {
        "artifact": "recording-overview.json",
        "artifact_sha256": overview_artifact_sha256,
        "prompt_version": OVERVIEW_PROMPT_VERSION,
    }
    recording_profile["visibility"] = {
        "artifact": "scene-usability.json",
        "artifact_sha256": visibility_artifact_sha256,
        "prompt_version": visibility_prompt_version,
        "contract_version": visibility_contract_version,
        "sample_count": len(visibility_records),
        "relevance_span_count": len(visibility_spans),
    }
    phase1_admission_records = None
    phase1_admission_paths: list[Path] = []
    from .phase1_admission import VERSION as admission_version, build_admission

    if pose_conditioning is not None and getattr(pose_conditioning, "focus_frame_records", ()):
        from .selected_pose import SelectedPoseView

        pose_conditioning = SelectedPoseView(
            pose_conditioning, analysis_samples, visibility_analysis
        )
        pose_feature_by_index = {f["sample_index"]: f for f in pose_conditioning.feature_records}
        pose_input_identity = pose_conditioning.model_input_identity()
        selection_path = write_json(
            root / "phase1-selected-pose.json",
            {
                **pose_conditioning.selection_document,
                "projection_sha256": pose_conditioning.projection_sha256,
                "source_rights": source.get("rights"),
            },
        )
        phase1_admission_paths.append(selection_path)
    admission_frames = (
        _merge_exact_frames(pose_conditioning.exact_frame_records, analysis_samples, frame_by_index)
        if pose_conditioning is not None and pose_conditioning.exact_frame_records
        else frame_by_index
    )
    phase1_admission_records = build_admission(
        samples=analysis_samples,
        visibility=visibility_analysis,
        pose_features=pose_feature_by_index,
        config=config,
        frames=admission_frames,
        visibility_hash=visibility_artifact_sha256,
    )
    admission_path = write_json(
        phase_result_path(root / "phase1-admission.json"),
        {
            "schema_version": "0.1.0",
            "policy_version": admission_version,
            "stage": "phase1",
            "source_sha256": source.get("sha256"),
            "source_rights": source.get("rights"),
            "visibility_artifact_sha256": visibility_artifact_sha256,
            "records": [phase1_admission_records[i] for i in range(len(analysis_samples))],
        },
    )
    phase1_admission_paths.append(admission_path)
    recording_profile["phase1_admission"] = {
        "artifact": admission_path.name,
        "sha256": _file_sha256(admission_path),
        "policy_version": admission_version,
    }
    _record_event(
        trace,
        "phase1",
        "speaker_scene_admission",
        "completed",
        admitted_samples=sum(
            (not r["exclude_from_detection"] for r in phase1_admission_records.values())
        ),
        excluded_samples=sum(
            (r["exclude_from_detection"] for r in phase1_admission_records.values())
        ),
        additional_model_calls=0,
    )
    recording_profile["pose_conditioning"] = (
        {
            "consumed_by_stages": ["phase1", "phase2"],
            "not_consumed_by_stages": ["phase0a", "phase0b", "phase3"],
            "input_identity": deepcopy(dict(pose_input_identity)),
        }
        if pose_input_identity is not None
        else {"consumed_by_stages": [], "not_available": True}
    )
    recording_profile_path = write_json(root / "recording-profile.json", recording_profile)
    recording_profile_artifact_sha256 = sha256(recording_profile_path.read_bytes()).hexdigest()
    model_capabilities_path = write_json(root / "model-capabilities.json", capabilities)
    root_material = {
        "source_sha256": source.get("sha256"),
        "proxy_sha256": preprocessing.get("proxy", {}).get("sha256")
        if isinstance(preprocessing.get("proxy"), Mapping)
        else None,
        "model": config.open_webui.model,
        "config": config.to_dict(),
        "overview_sha256": overview_artifact_sha256,
        "visibility_sha256": visibility_artifact_sha256,
        "pose_conditioning_sha256": _digest(pose_input_identity)
        if pose_input_identity is not None
        else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    root_run_id = f"llm-{_digest(root_material)[:16]}"
    subjects = recording_profile["selected_subjects"]
    if recording_profile["target_mode"] != "individual" or len(subjects) != 1:
        raise RuntimeError("Phase 0a must select exactly one participant")
    status_path = write_pipeline_status(
        root,
        requested_stop_after=stop_after,
        completed_stage="phase1",
        artifacts=(
            overview_path,
            visibility_path,
            recording_profile_path,
            model_capabilities_path,
            *reuse_status_artifacts,
            *phase1_admission_paths,
        ),
        source_sha256=str(source["sha256"])
        if isinstance(source.get("sha256"), str) and len(str(source["sha256"])) == 64
        else None,
        preprocessing_sha256=_digest(to_json_value(preprocessing)),
    )
    if stop_after == "phase1":
        _record_event(trace, "pipeline", "requested_stop", "completed", completed_stage="phase1")
        return Phase2RunResult(
            run_id=root_run_id,
            target_mode=str(recording_profile["target_mode"]),
            analyses=(),
            analysis_paths=(),
            index_path=None,
            completed_stage="phase1",
            status_path=status_path,
            stage_artifact_paths=(
                overview_path,
                visibility_path,
                recording_profile_path,
                model_capabilities_path,
                *phase1_admission_paths,
            ),
        )
    analyses: list[AnalysisResult] = []
    paths: list[Path] = []
    phase2_evidence_paths: list[Path] = []
    if overwrite:
        subjects_root = root / "subjects"
        if subjects_root.is_dir():
            for child in subjects_root.iterdir():
                if child.is_dir() and child.name.startswith("subject-"):
                    shutil.rmtree(child)
    for subject in subjects:
        joint_options: dict[str, Any] = {}
        from .joint_runner import run_joint_subject

        subject_runner = run_joint_subject
        joint_options["pose_conditioning"] = pose_conditioning
        joint_options["resume"] = resume
        joint_options["phase1_admission"] = phase1_admission_records
        subject_result = subject_runner(
            root=root,
            root_run_id=root_run_id,
            output_dir=root,
            subject=subject,
            recording_profile=recording_profile,
            recording_profile_artifact_sha256=recording_profile_artifact_sha256,
            visibility_analysis=visibility_analysis,
            visibility_artifact_sha256=visibility_artifact_sha256,
            source=source,
            preprocessing=preprocessing,
            preprocessing_reuse=preprocessing_reuse_identity,
            samples=analysis_samples,
            frame_by_index=frame_by_index,
            pose_frame_by_index=pose_frame_by_index,
            pose_feature_by_index=pose_feature_by_index,
            pose_input_identity=pose_input_identity,
            config=config,
            model=model,
            audio_supplied=audio_supplied,
            export_eaf=export_eaf,
            overwrite=artifact_overwrite,
            trace=trace,
            capability_info=capabilities,
            restricted_transport=restricted_transport,
            ffmpeg=ffmpeg,
            visual_input_records=visual_input_records,
            stop_after=stop_after,
            **joint_options,
        )
        phase2_evidence_paths.append(subject_result.phase2_evidence_path)
        if subject_result.analysis is not None:
            assert subject_result.analysis_path is not None
            analyses.append(subject_result.analysis)
            paths.append(subject_result.analysis_path)
    if stop_after == "phase2":
        if len(phase2_evidence_paths) != 1:
            raise RuntimeError("Phase 2 requires exactly one subject evidence artifact")
        phase2_evidence_path = phase2_evidence_paths[0]
        status_path = write_pipeline_status(
            root,
            requested_stop_after=stop_after,
            completed_stage="phase2",
            artifacts=(
                overview_path,
                visibility_path,
                recording_profile_path,
                model_capabilities_path,
                phase2_evidence_path,
                *reuse_status_artifacts,
                *phase1_admission_paths,
            ),
            source_sha256=str(source["sha256"])
            if isinstance(source.get("sha256"), str) and len(str(source["sha256"])) == 64
            else None,
            preprocessing_sha256=_digest(to_json_value(preprocessing)),
        )
        _record_event(trace, "pipeline", "requested_stop", "completed", completed_stage="phase2")
        return Phase2RunResult(
            run_id=root_run_id,
            target_mode=str(recording_profile["target_mode"]),
            analyses=(),
            analysis_paths=(),
            index_path=None,
            completed_stage="phase2",
            status_path=status_path,
            stage_artifact_paths=(phase2_evidence_path,),
        )
    inference_plan = {
        "schema_version": "0.3.0",
        "canonical_analysis_rate_hz": config.analysis.sample_rate_hz,
        "annotation_duration_cap_us": None,
        **(
            {"preprocessing_reuse": deepcopy(preprocessing_reuse_identity)}
            if preprocessing_reuse_identity is not None
            else {}
        ),
        "pose_conditioning": {
            "input_identity": deepcopy(dict(pose_input_identity)),
            "source_rate_processing": True,
            "projection_rate_hz": config.analysis.sample_rate_hz,
            "phase0a_consumes_pose": False,
            "phase0b_consumes_full_scene_plus_auxiliary_pose": False,
            "phase1_consumes_pose_guided_focus": False,
            "phase2_consumes_pose_guided_focus_and_motion": True,
            "phase3_consumes_pose": False,
            "identity_policy": "crop only when one pose candidate is unambiguous; otherwise retain the original full frame",
        }
        if pose_input_identity is not None
        else {"available": False},
        "long_candidate_policy": "unknown runs longer than maximum_bridged_missing_samples interrupt continuity and split adjacent candidates with incomplete boundary flags; remaining long candidates use overlapping request windows, and artificial request edges never become G-unit boundaries",
        "overview": {
            "phase": "0a",
            "sample_period_us": config.inference.profile_sample_period_us,
            "window_us": config.inference.profile_window_us,
            "overlap_us": config.inference.profile_overlap_us,
            "prompt_version": OVERVIEW_PROMPT_VERSION,
            "artifact": "recording-overview.json",
            "artifact_sha256": overview_artifact_sha256,
        },
        "exemplar_verification": {
            "phase": "0a",
            "prompt_version": PHASE0A_EXEMPLAR_PROMPT_VERSION,
            "contract_version": PHASE0A_EXEMPLAR_CONTRACT_VERSION,
            "request_id": recording_profile["positive_main_scene_exemplars"][
                "verification_request_id"
            ],
            "candidate_sample_indices": recording_profile["positive_main_scene_exemplars"][
                "candidate_sample_indices"
            ],
            "proposed_positive_sample_indices": proposed_exemplar_indices,
            "selected_positive_sample_indices": phase0a_exemplar_indices,
            "minimum_positive_probability": recording_profile["positive_main_scene_exemplars"][
                "minimum_positive_probability"
            ],
            "selection_policy_version": recording_profile["positive_main_scene_exemplars"][
                "selection_policy_version"
            ],
            "confirmation": {
                "policy_version": _PHASE0A_EXEMPLAR_CONFIRMATION_POLICY_VERSION,
                "request_ids": [record["request_id"] for record in exemplar_confirmation_requests],
                "confirmed_positive_sample_indices": phase0a_exemplar_indices,
                "backfill_allowed": False,
            },
            "artifact": "recording-overview.json",
            "artifact_sha256": overview_artifact_sha256,
        },
        "visibility": {
            "phase": "0b",
            "time_selection": visibility_selection,
            "sample_period_us": config.analysis.sample_period_us,
            "window_us": visibility_items * config.analysis.sample_period_us,
            "overlap_us": visibility_overlap_items * config.analysis.sample_period_us,
            "prompt_version": visibility_prompt_version,
            "artifact": "scene-usability.json",
            "artifact_sha256": visibility_artifact_sha256,
            "output_ownership": "exactly one ordered label per assessed sample; explicit not-assessed placeholders outside visibility context",
            "gate": {**visibility_analysis["gate_policy"]},
        },
        "activity": {},
        "interpretation": {"model_requests": 0, "method": "selected_probability"},
        "model_limits": config.inference.to_dict(),
        "checkpointing": {
            "schema_version": _MODEL_CHECKPOINT_SCHEMA_VERSION,
            "policy_version": _MODEL_CHECKPOINT_POLICY_VERSION,
            "model_capability_identity_policy_version": _MODEL_CAPABILITY_IDENTITY_POLICY_VERSION,
            "stable_model_capability_identity_enabled": stable_open_webui_capability_identity,
            "resume_requested": resume,
            "reused_model_call_count": checkpointing_model.reused_count,
            "persisted_model_call_count": checkpointing_model.persisted_count,
            "directory": "model-checkpoints",
            "initial_model_capabilities_snapshot": "model-checkpoints/initial-model-capabilities.json"
            if (root / "model-checkpoints" / "initial-model-capabilities.json").is_file()
            else None,
        },
        "requests": visual_input_records,
    }
    inference_plan["detection"] = config.detection.to_dict()
    from .probability_pass import (
        VERSION as probability_version,
        PROMPT_VERSION as probability_prompt_version,
        HISTORY_VERSION,
        MOTION_QUALITY_VERSION,
        MOTION_INPUT_POLICY,
        MOTION_SCORE_VERSION,
        schedule_metadata,
        selected_span,
    )

    inference_plan["activity"].update(
        prompt_version=probability_prompt_version,
        contract_version=probability_version,
        activity_assessment="single_pass_probability",
        assessment_pass_count=1,
        filtering_stage="phase1",
        uses_positive_static_exemplars=False,
        factual_rechecks=False,
        boundary_recovery=False,
        candidate_computation="local_selected_probability_and_duration_rules",
        probability_score_source=config.detection.probability_score_source,
        window_us=schedule_metadata(config.detection)["maximum_context_us"],
        overlap_us=schedule_metadata(config.detection)["history_us"]
        + schedule_metadata(config.detection)["lookahead_us"],
        target_schedule=schedule_metadata(config.detection),
        previous_estimates_policy_version=HISTORY_VERSION,
        probability_history_enabled=config.detection.probability_history_enabled,
        motion_quality_policy_version=MOTION_QUALITY_VERSION,
        motion_input_policy=MOTION_INPUT_POLICY,
        motion_measurement_policy_version=MOTION_SCORE_VERSION,
        probability_motion_trace_enabled=config.detection.probability_motion_trace_enabled,
        rgb_policy_version="clean-compact-phase2-rgb-v0.1",
        skeleton_overlay=False,
        context_admission_policy="phase1-speaker-scene-admission-v0.2",
        selected_sample_span=list(selected_span(config.detection, len(analysis_samples))),
        probability_formula=("q_llm" if config.detection.probability_score_source == "llm" else
            "sigmoid(logit(clamp(q_motion,.05,.95))+context_strength*(2*q_llm-1)); q_llm when motion unavailable"),
    )
    from .probability_finalization import finalization_metadata

    inference_plan["interpretation"] = {
        "phase": "3",
        **finalization_metadata(merge_gap_ms=config.detection.effective_finalization_merge_gap_ms),
    }
    inference_plan["activity"]["hand_attribution_required"] = False
    inference_plan["activity"]["requests_may_subdivide_for_pose_capacity"] = False
    if pose_input_identity is not None:
        inference_plan["pose_conditioning"]["phase3_consumes_pose"] = False
        inference_plan["pose_conditioning"]["native_packet_phase2"] = True
        inference_plan["pose_conditioning"]["native_packet_phase3"] = False
    inference_plan["canonical_schema_version"] = "0.2.0"
    write_json(root / "inference-plan.json", inference_plan)
    index = {
        "schema_version": "0.3.0",
        "run_id": root_run_id,
        "status": "completed",
        "target_mode": recording_profile["target_mode"],
        "selected_subject_count": len(subjects),
        "recording_profile": "recording-profile.json",
        "recording_profile_sha256": recording_profile_artifact_sha256,
        "recording_overview": "recording-overview.json",
        "recording_overview_sha256": overview_artifact_sha256,
        "scene_usability": "scene-usability.json",
        "scene_usability_sha256": visibility_artifact_sha256,
        "model_capabilities": "model-capabilities.json",
        "phase2_evidence": phase2_evidence_paths[0].relative_to(root).as_posix(),
        "phase2_evidence_sha256": _file_sha256(phase2_evidence_paths[0]),
        "inference_plan": "inference-plan.json",
        **(
            {"preprocessing_reuse": deepcopy(preprocessing_reuse_identity)}
            if preprocessing_reuse_identity is not None
            else {}
        ),
        "pose_conditioning": "pose/conditioning-manifest.json"
        if pose_input_identity is not None
        else None,
        "inference_plan_sha256": sha256((root / "inference-plan.json").read_bytes()).hexdigest(),
        "restricted_transport_mode": restricted_transport,
        "operator_authorization_is_legal_clearance": False,
        "analyses": [
            {
                "subject_id": result.participant_profiles[0].participant_id,
                "run_id": result.run_id,
                "annotations": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for result, path in zip(analyses, paths, strict=True)
        ],
        "configuration": config.to_dict(),
        "timing": {"wall_seconds": perf_counter() - started},
    }
    index_path = root / "analysis-index.json"
    write_json(index_path, index)
    release_phase1_images(root)
    status_path = write_pipeline_status(
        root,
        requested_stop_after=stop_after,
        completed_stage="phase3",
        artifacts=(
            overview_path,
            visibility_path,
            recording_profile_path,
            model_capabilities_path,
            *phase2_evidence_paths,
            root / "inference-plan.json",
            index_path,
            *paths,
            *reuse_status_artifacts,
        ),
        source_sha256=str(source["sha256"])
        if isinstance(source.get("sha256"), str) and len(str(source["sha256"])) == 64
        else None,
        preprocessing_sha256=_digest(to_json_value(preprocessing)),
    )
    _record_event(
        trace,
        "pipeline",
        "analysis",
        "completed",
        selected_subject_count=len(subjects),
        wall_seconds=perf_counter() - started,
    )
    return Phase2RunResult(
        run_id=root_run_id,
        target_mode=recording_profile["target_mode"],
        analyses=tuple(analyses),
        analysis_paths=tuple(paths),
        index_path=index_path,
        completed_stage="phase3",
        status_path=status_path,
        stage_artifact_paths=tuple(phase2_evidence_paths),
    )
