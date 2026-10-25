# SPDX-License-Identifier: Apache-2.0
"""Joint probability orchestration with source-rate motion evidence."""

from __future__ import annotations
from ..storage import data_path, phase_result_path, minimal_enabled
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from .. import __version__
from ..audit.progress import StepProgress
from ..config import ResearchConfig
from ..domain.models import (
    AnalysisResult,
    AnalysisSample,
    JointActivityEvidence,
    JointActivityState,
    ParticipantProfile,
    to_json_value,
)
from ..export.json_export import write_analysis, write_json
from ..media.image_packaging import ContactSheetLayout
from ..media.pose_conditioning import PoseConditioningResult
from . import llm_runner as runtime
from . import probability_pass
from .grouping import propose_joint_activity_candidates
from .probability_finalization import (
    POLICY_VERSION as PROBABILITY_FINALIZATION_VERSION,
    finalize_probability_candidates,
    finalization_metadata,
)
from .evidence_resolution import reported_uncertainty
from .llm_contracts import Phase2Model

JOINT_PIPELINE_VERSION = "research-joint-detection-v0.3"


def _layout(config: ResearchConfig, stage: str) -> ContactSheetLayout:
    inf = config.inference
    return ContactSheetLayout(
        columns=getattr(inf, f"{stage}_sheet_columns"),
        rows=getattr(inf, f"{stage}_sheet_rows"),
        tile_width_px=getattr(inf, f"{stage}_tile_width"),
        tile_height_px=getattr(inf, f"{stage}_tile_height"),
        gutter_px=inf.contact_sheet_gutter,
        label_mode="ordinal",
        jpeg_quality=getattr(inf, f"{stage}_jpeg_quality"),
    )


def _joint_evidence(
    sample: AnalysisSample,
    value: Mapping[str, Any],
    request_id: str,
    response_hash: str,
    resolution: Mapping[str, Any],
    *,
    previous: JointActivityEvidence | None = None,
    motion_record: Mapping[str, Any] | None = None,
) -> JointActivityEvidence:
    state = JointActivityState(value["state"])
    reason = value.get("reason")
    references = (f"model-response:{request_id}:{response_hash}", *sample.source_references)
    records = [dict(resolution)]
    if motion_record is not None:
        records.append(
            {"policy": motion_record["policy_version"], "motion_grounding": dict(motion_record)}
        )
    if previous is not None:
        records.append(
            {
                "policy": "bounded-joint-reestimation-v0.1",
                "previous_evidence": to_json_value(previous),
            }
        )
    source_support = []
    for interval in resolution.get("source_support_intervals", ()):
        if (
            not isinstance(interval, Mapping)
            or type(interval.get("start_us")) is not int
            or type(interval.get("end_us")) is not int
            or (not 0 <= interval["start_us"] < interval["end_us"])
        ):
            raise ValueError(
                "joint evidence source support must have actual positive source intervals"
            )
        source_support.append(deepcopy(dict(interval)))
    return JointActivityEvidence(
        state=state,
        state_probabilities=dict(value["probabilities"]),
        evidence_quality=float(value["quality"]),
        evidence_references=references,
        reason=str(reason or "insufficient_model_evidence")
        if state == JointActivityState.UNKNOWN
        else reason,
        uncertainty_causes=tuple(
            reported_uncertainty(
                value.get("uncertainty_causes", ()),
                reason=reason
                if state == JointActivityState.UNKNOWN or value.get("uncertainty_causes")
                else None,
                evidence_refs=references,
            )
        ),
        source_support_intervals=tuple(source_support),
        resolution_records=tuple(records),
        continuity_barrier=bool(resolution.get("continuity_barrier", False)),
    )


def run_joint_subject(
    *,
    root: Path,
    root_run_id: str,
    output_dir: Path,
    subject: Mapping[str, Any],
    recording_profile: Mapping[str, Any],
    recording_profile_artifact_sha256: str,
    visibility_analysis: Mapping[str, Any],
    visibility_artifact_sha256: str,
    source: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    preprocessing_reuse: Mapping[str, Any] | None,
    samples: Sequence[AnalysisSample],
    frame_by_index: Mapping[int, Mapping[str, Any]],
    pose_frame_by_index: Mapping[int, Mapping[str, Any]],
    pose_feature_by_index: Mapping[int, Mapping[str, Any]],
    pose_input_identity: Mapping[str, Any] | None,
    config: ResearchConfig,
    model: Phase2Model,
    audio_supplied: bool,
    export_eaf: bool,
    overwrite: bool,
    trace: Any | None,
    capability_info: Mapping[str, Any],
    restricted_transport: str | None,
    ffmpeg: str,
    visual_input_records: list[dict[str, Any]],
    stop_after: str,
    pose_conditioning: PoseConditioningResult | None = None,
    resume: bool = False,
    phase1_admission: Mapping[int, Mapping[str, Any]] | None = None,
) -> Any:
    """Run the probability pass and finalize its candidates locally."""
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_id = str(subject["subject_id"])
    activity_prompt_version = probability_pass.PROMPT_VERSION
    policy = config.analysis
    profile_summary = {
        "active_subject": runtime._without_mapping_key(subject, "model_subject_id"),
        "scene_summary": recording_profile["scene_summary"],
    }
    profile_summary["active_subject"] = runtime._without_mapping_key(
        profile_summary["active_subject"], "typical_rest_postures"
    )
    profile = ParticipantProfile(
        participant_id=participant_id,
        selection_policy=recording_profile.get(
            "selection_policy", "sparse-overview-plus-dense-usability-v0.2"
        ),
        modalities_supplied=("video", "audio") if audio_supplied else ("video",),
        modalities_used=("video",),
        summary=profile_summary,
        evidence_references=(f"artifact:recording-profile:{recording_profile_artifact_sha256}",),
        model_output=deepcopy(recording_profile["model_output"]),
        contract_normalized_output=deepcopy(recording_profile["contract_normalized_output"]),
        contract_normalization=deepcopy(
            recording_profile["contract_normalization"]["final_profile"]
        ),
        subject_id_canonicalization=deepcopy(recording_profile["subject_id_canonicalization"]),
    )
    if pose_conditioning is not None and pose_conditioning.exact_frame_records:
        frame_by_index = runtime._merge_exact_frames(
            pose_conditioning.exact_frame_records, samples, frame_by_index
        )
    activity_frames = pose_frame_by_index or frame_by_index
    clean_rgb = None
    from ..media.phase2_rgb import CleanPhase2Frames

    clean_rgb = CleanPhase2Frames(
        frame_by_index,
        getattr(pose_conditioning, "focus_frame_records", ()),
        output_dir / "phase2-rgb",
        source["rights"],
    )
    activity_frames = clean_rgb.frames

    def visuals(
        request_id: str,
        indices: Sequence[int],
        stage: str,
        *,
        visible_labels: Mapping[int, str] | None = None,
    ) -> tuple[list[Path], list[dict[str, Any]]]:
        if stage == "activity" and clean_rgb is not None:
            from ..media.phase2_rgb import compact_sheets, VERSION as rgb_version

            if visible_labels is None:
                raise ValueError("Phase2 clean RGB requires explicit frame labels")
            runtime._record_event(
                trace,
                stage,
                "visual_packaging",
                "started",
                request_id=request_id,
                mode="compact_clean_rgb",
                source_sample_count=len(indices),
            )
            clean_rgb.prepare(indices)
            layout = _layout(config, stage)
            if config.inference.activity_image_mode == "individual":
                layout = replace(layout, columns=1, rows=1)
            images, references = compact_sheets(
                activity_frames,
                indices,
                visible_labels,
                layout,
                output_dir / "phase2-rgb" / "sheets",
                source["rights"],
                config.inference.maximum_images_per_request
                - int(config.detection.probability_motion_trace_enabled),
            )
            visual_input_records.append(
                {
                    "request_id": request_id,
                    "stage": stage,
                    "mode": "compact_clean_rgb",
                    "renderer_version": rgb_version,
                    "source_sample_count": len(indices),
                    "image_count": len(images),
                    "image_refs": references,
                }
            )
            runtime._record_event(
                trace,
                stage,
                "visual_packaging",
                "completed",
                request_id=request_id,
                mode="compact_clean_rgb",
                source_sample_count=len(indices),
                image_count=len(images),
            )
            return (images, references)
        images, references = runtime._request_visual_inputs(
            root=root,
            request_id=request_id,
            stage=stage,
            indices=indices,
            frame_by_index=activity_frames if stage == "activity" else frame_by_index,
            mode=getattr(config.inference, f"{stage}_image_mode"),
            layout=_layout(config, stage),
            maximum_images=config.inference.maximum_images_per_request
            - int(stage == "activity" and config.detection.probability_motion_trace_enabled),
            ffmpeg=ffmpeg,
            overwrite=overwrite,
            records=visual_input_records,
            trace=trace,
            visible_labels=visible_labels,
        )
        return (images, references)

    from .phase1_admission import VERSION as admission_version

    if (
        phase1_admission is None
        or set(phase1_admission) != {s.index for s in samples}
        or any((r.get("admission_version") != admission_version for r in phase1_admission.values()))
    ):
        raise ValueError("single-pass Phase2 requires completed Phase1 speaker/scene admission")
    resolutions = deepcopy(dict(phase1_admission))
    resolution_trace = [resolutions[i] for i in sorted(resolutions)]
    slots: list[JointActivityEvidence | None] = [None] * len(samples)
    selected_span = probability_pass.selected_span(config.detection, len(samples))
    for sample in samples:
        resolution = resolutions[sample.index]
        if not selected_span[0] <= sample.index < selected_span[1]:
            slots[sample.index] = JointActivityEvidence(
                state=JointActivityState.UNKNOWN,
                state_probabilities={"passive": 0.0, "active": 0.0, "unknown": 1.0},
                evidence_quality=0.0,
                reason="outside_requested_time_range",
                continuity_barrier=True,
                resolution_records=(
                    {
                        "policy_version": "phase2-time-selection-v0.1",
                        "exclude_from_detection": True,
                        "reason": "outside_requested_time_range",
                        "selected_sample_span": list(selected_span),
                    },
                    dict(resolution),
                ),
            )
            continue
        if resolution.get("exclude_from_detection"):
            slots[sample.index] = JointActivityEvidence(
                state=JointActivityState.UNKNOWN,
                state_probabilities={"passive": 0.0, "active": 0.0, "unknown": 1.0},
                evidence_quality=0.0,
                reason="decoder_gap" if sample.source_gap_reason else "unresolved_scene_evidence",
                evidence_references=(
                    f"artifact:scene-usability:{visibility_artifact_sha256}",
                    *sample.source_references,
                ),
                continuity_barrier=bool(resolution.get("continuity_barrier")),
                resolution_records=(dict(resolution),),
            )
    activity_traces: list[dict[str, Any]] = []
    activity_traces = probability_pass.run_pass(
        samples=samples,
        subject=subject,
        resolutions=resolutions,
        frames=activity_frames,
        config=config,
        model=model,
        conditioning=pose_conditioning,
        visuals=visuals,
        slots=slots,
        trace=trace,
        artifact_dir=output_dir,
        source_rights=source["rights"],
        overwrite=overwrite and (not resume),
    )
    if any((value is None for value in slots)):
        raise ValueError("joint detection did not cover the analysis grid")
    evidence = [value for value in slots if value is not None]
    slots = list(evidence)
    resolution_trace = [resolutions[i] for i in sorted(resolutions)]
    candidate_progress = StepProgress(trace, "candidates", 0, subject_id=participant_id)
    candidates, bridged = propose_joint_activity_candidates(
        samples, evidence, policy, participant_id=participant_id
    )
    score_diagnostics = probability_pass.candidate_diagnostics(
        samples, evidence, policy, participant_id=participant_id
    )
    candidate_progress.finish(candidate_count=len(candidates))
    windows = []
    phase2_document = {
        "schema_version": "0.2.0",
        "stage": "phase2",
        "status": "completed",
        "subject_id": participant_id,
        "analysis_rate_hz": policy.sample_rate_hz,
        "source_sha256": source.get("sha256"),
        "source_rights": source.get("rights"),
        "preprocessing_sha256": runtime._digest(preprocessing),
        "frame_evidence": {
            "samples": [
                {"sample": to_json_value(s), "joint": to_json_value(e)}
                for s, e in zip(samples, evidence, strict=True)
            ],
            "bridged_sample_indices": sorted(bridged),
        },
        "activity_candidates": to_json_value(candidates),
        "phase2_request_ids": [item["request_id"] for item in activity_traces],
        "phase2_model_requests": activity_traces,
        "evidence_resolution": resolution_trace,
        "activity_assessment": "single_pass_probability",
        "prompt_version": activity_prompt_version,
        **{
            "assessment_pass_count": 1,
            "filtering_stage": "phase1",
            "target_schedule": probability_pass.schedule_metadata(config.detection),
            "previous_estimates_policy_version": probability_pass.HISTORY_VERSION,
            "probability_history_enabled": config.detection.probability_history_enabled,
            "motion_quality_artifact": "phase2-motion-quality.json",
            "motion_quality_policy_version": probability_pass.MOTION_QUALITY_VERSION,
            "motion_input_policy": probability_pass.MOTION_INPUT_POLICY,
            "motion_measurement_policy_version": probability_pass.MOTION_SCORE_VERSION,
            "probability_motion_trace_enabled": config.detection.probability_motion_trace_enabled,
            "rgb_policy_version": "clean-compact-phase2-rgb-v0.1",
            "skeleton_overlay": False,
            "selected_sample_span": list(selected_span),
            "selected_time_range_us": [
                samples[selected_span[0]].start_us,
                samples[selected_span[1] - 1].end_us,
            ],
            "phase1_admission": recording_profile.get("phase1_admission"),
            "candidate_computation": "local_combined_probability_and_duration_rules",
            "phase3_finalization": finalization_metadata(
                merge_gap_ms=config.detection.effective_finalization_merge_gap_ms
            ),
        },
        "score_candidate_diagnostics": to_json_value(score_diagnostics)
        if score_diagnostics is not None
        else None,
    }
    if minimal_enabled() and clean_rgb is not None:
        clean_rgb.release_except(set())
    phase2_path = write_json(phase_result_path(output_dir / "phase2-evidence.json"), phase2_document)
    write_json(
        phase_result_path(output_dir / "evidence-resolution.json"),
        {
            "schema_version": "0.1.0",
            "source_rights": source.get("rights"),
            "records": resolution_trace,
        },
    )
    if stop_after == "phase2":
        return runtime._SubjectRunResult(
            analysis=None, analysis_path=None, phase2_evidence_path=phase2_path
        )
    envelopes = []
    responses = []
    assessments = []
    resolved_windows = []
    windows = resolved_windows
    final_progress = StepProgress(trace, "finalization", 0, candidate_count=len(candidates))
    runtime._record_event(trace, "finalization", "combined_probability", "started")
    decisions, units = finalize_probability_candidates(
        samples,
        evidence,
        candidates,
        policy,
        participant_id=participant_id,
        merge_gap_ms=config.detection.effective_finalization_merge_gap_ms,
    )
    runtime._record_event(
        trace,
        "finalization",
        "combined_probability",
        "completed",
        candidate_count=len(candidates),
        g_unit_count=len(units),
        model_request_count=0,
    )
    final_progress.finish(g_unit_count=len(units))
    snapshot = config.detection_dict()
    result = AnalysisResult(
        schema_version="0.2.0",
        run_id=f"{root_run_id}-{participant_id}",
        created_at=datetime.now(timezone.utc).isoformat(),
        capability={
            "name": "open-webui-g-unit-analysis",
            "version": "0.10.0",
            "model_identifier": str(capability_info.get("resolved_model", config.open_webui.model)),
            "audio_supplied": audio_supplied,
            "audio_used": False,
        },
        source=source,
        preprocessing=preprocessing,
        config=policy,
        participant_profiles=[profile],
        prompt_envelopes=envelopes,
        activity_candidates=candidates,
        processing_windows=windows,
        window_responses=responses,
        candidate_assessments=assessments,
        samples=decisions,
        g_units=units,
        warnings=["Experimental machine predictions; audio was not used by the image adapter."],
        provenance={
            "software": "communication-gesture-annotator",
            "software_version": __version__,
            "policy_version": PROBABILITY_FINALIZATION_VERSION,
            "phase": JOINT_PIPELINE_VERSION,
            "deterministic": False,
            "human_action_required": False,
            "detection_scope": {"contract_version": "0.2.0", "attribution_status": "not_included"},
            "research_configuration": snapshot,
            "research_configuration_sha256": runtime._digest(snapshot),
            "model_capabilities": dict(capability_info),
            "model_capabilities_sha256": runtime._digest(capability_info),
            "restricted_transport_mode": restricted_transport,
            "operator_authorization_is_legal_clearance": False,
            "pose_conditioning": {
                "input_identity": pose_input_identity,
                "consumed_by_stages": ["phase2"] + [],
            },
            "prompt_versions": [activity_prompt_version] + [],
            **{
                "phase3_finalization": finalization_metadata(
                    merge_gap_ms=config.detection.effective_finalization_merge_gap_ms
                )
            },
            "evidence_resolution_artifact_sha256": runtime._file_sha256(
                data_path(output_dir / "evidence-resolution.json")
            ),
            **({"preprocessing_reuse": dict(preprocessing_reuse)} if preprocessing_reuse else {}),
        },
    )
    result.validate()
    from ..storage import compact_enabled

    diagnostic_duplicates = {
        "frame-evidence",
        "prompt-envelopes",
        "window-responses",
        "candidate-assessments",
        "activity-decisions",
        "decision-trace",
    }
    for name, value in {
        "source": source,
        "preprocessing": preprocessing,
        "profile": profile,
        "frame-evidence": phase2_document["frame_evidence"],
        "candidates": candidates,
        "prompt-envelopes": envelopes,
        "window-responses": responses,
        "candidate-assessments": assessments,
        "activity-decisions": decisions,
        "decision-trace": {
            "schema_version": "0.5.0",
            "joint_detection": True,
            "step2_model_requests": activity_traces,
            "candidates": to_json_value(candidates),
        },
    }.items():
        if compact_enabled() and name in diagnostic_duplicates:
            continue
        write_json(output_dir / f"{name}.json", value)
    annotations = write_analysis(output_dir / "annotations.json", result)
    if export_eaf:
        from ..export.eaf import export_eaf_document as write_eaf
        from ..export.json_export import read_json

        write_eaf(
            read_json(annotations),
            output_dir / "annotations.eaf",
            tier_id=config.analysis.eaf_tier_id,
            overwrite=overwrite,
        )
    write_json(
        output_dir / "completion.json",
        {
            "schema_version": "0.1.0",
            "detection": {
                "status": "completed",
                "artifact": annotations.name,
                "sha256": runtime._file_sha256(annotations),
            },
        },
    )
    write_json(
        output_dir / "run.json",
        {
            "schema_version": "0.4.0",
            "run_id": result.run_id,
            "status": "completed",
            "scope": "joint_detection",
            "annotations": annotations.name,
            "annotations_sha256": runtime._file_sha256(annotations),
            "configuration": config.detection_dict(),
            "optional_completion_artifact": "completion.json",
            "warnings": result.warnings,
            "failures": [],
        },
    )
    return runtime._SubjectRunResult(
        analysis=result, analysis_path=annotations, phase2_evidence_path=phase2_path
    )
