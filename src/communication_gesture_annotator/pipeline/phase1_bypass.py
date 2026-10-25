# SPDX-License-Identifier: Apache-2.0
"""Explicit operator-selected scene assumption; no Phase1 model requests."""

from __future__ import annotations
from ..storage import phase_result_path
from pathlib import Path
from typing import Any, Mapping, Sequence
from ..config import ResearchConfig
from ..domain import AnalysisSample
from ..export.json_export import write_json
from .main_scene_visibility import project_response
from .phase1_admission import VERSION as ADMISSION_VERSION, build_admission
from .visibility_scope import unassessed_visibility_sample

VERSION = "operator-skipped-phase1-v0.1"


def run_without_phase1(
    *,
    root: Path,
    samples: Sequence[AnalysisSample],
    source: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    config: ResearchConfig,
    **options: Any,
) -> Any:
    from . import llm_runner as runtime
    from .joint_runner import run_joint_subject

    write_pipeline_status = runtime.write_pipeline_status
    from .probability_pass import schedule_metadata, selected_span, PROMPT_VERSION
    from .probability_finalization import finalization_metadata

    runtime._record_event(options["trace"], "phase1", "step", "skipped", planned_calls=0)
    if options["pose_conditioning"] is None:
        raise ValueError("skipping Phase1 does not skip mandatory pose preprocessing")
    first, stop = selected_span(config.detection, len(samples))
    assumption = "Phase1 disabled by operator: selected visible participant and scene assumed, not model verified."
    subject = {"subject_id": "subject-0001", "description": assumption, "typical_rest_postures": []}
    rows = project_response(
        {"samples": [{"sample_index": s.index, "main_scene": True} for s in samples]}
    )["samples"]
    for i, sample in enumerate(samples):
        if i < first or i >= stop or sample.source_gap_reason:
            rows[i] = unassessed_visibility_sample(i)
        else:
            rows[i].update(
                assessment_status="operator_assumed",
                decision_source=VERSION,
                probabilities_source="operator_assumption_not_model_confidence",
                reason=assumption,
            )
    visibility = {
        "schema_version": "0.1.0",
        "status": "skipped",
        "policy_version": VERSION,
        "samples": rows,
        "requests": [],
        "model_request_count": 0,
    }
    visibility_path = write_json(root / "scene-usability.json", visibility)
    visibility_hash = runtime._file_sha256(visibility_path)
    frames = options["frame_by_index"]
    conditioning = options["pose_conditioning"]
    selection_paths = []
    if getattr(conditioning, "focus_frame_records", ()):
        from .selected_pose import SelectedPoseView

        conditioning = SelectedPoseView(conditioning, samples, visibility)
        options["pose_conditioning"] = conditioning
        options["pose_feature_by_index"] = {
            f["sample_index"]: f for f in conditioning.feature_records
        }
        options["pose_input_identity"] = conditioning.model_input_identity()
        selection_paths.append(
            write_json(
                root / "phase1-selected-pose.json",
                {
                    **conditioning.selection_document,
                    "projection_sha256": conditioning.projection_sha256,
                    "source_rights": source.get("rights"),
                    "scene_admission_source": VERSION,
                },
            )
        )
    admission_frames = (
        runtime._merge_exact_frames(conditioning.exact_frame_records, samples, frames)
        if conditioning.exact_frame_records
        else frames
    )
    admission = build_admission(
        samples=samples,
        visibility=visibility,
        pose_features=options["pose_feature_by_index"],
        config=config,
        frames=admission_frames,
        visibility_hash=visibility_hash,
    )
    admission_path = write_json(
        phase_result_path(root / "phase1-admission.json"),
        {
            "schema_version": "0.1.0",
            "policy_version": ADMISSION_VERSION,
            "stage": "phase1",
            "decision_source": VERSION,
            "source_sha256": source.get("sha256"),
            "source_rights": source.get("rights"),
            "visibility_artifact_sha256": visibility_hash,
            "records": [admission[i] for i in range(len(samples))],
        },
    )
    profile = {
        "target_mode": "individual",
        "selected_subjects": [subject],
        "scene_summary": assumption,
        "selection_policy": VERSION,
        "model_output": None,
        "contract_normalized_output": None,
        "contract_normalization": {"final_profile": None},
        "subject_id_canonicalization": None,
    }
    profile_path = write_json(root / "recording-profile.json", profile)
    profile_hash = runtime._file_sha256(profile_path)
    overview_path = write_json(
        root / "recording-overview.json",
        {
            "status": "skipped",
            "policy_version": VERSION,
            "model_request_count": 0,
            "operator_assumption": assumption,
        },
    )
    capabilities_path = write_json(root / "model-capabilities.json", options["capability_info"])
    records: list[dict[str, Any]] = []
    plan = {
        "schema_version": "0.1.0",
        "phase1": {"enabled": False, "policy_version": VERSION, "model_request_count": 0},
        "activity": {
            "prompt_version": PROMPT_VERSION,
            "target_schedule": schedule_metadata(config.detection),
            "selected_sample_span": [first, stop],
            "assessment_pass_count": 1,
        },
        "interpretation": finalization_metadata(
            merge_gap_ms=config.detection.effective_finalization_merge_gap_ms
        ),
        "model_limits": config.inference.to_dict(),
        "requests": records,
    }
    artifacts = [
        overview_path,
        visibility_path,
        admission_path,
        profile_path,
        capabilities_path,
        *selection_paths,
    ]
    run_id = f"llm-{runtime._digest({'profile': profile, 'source': source.get('sha256'), 'config': config.to_dict()})[:16]}"
    stop_after = options["stop_after"]
    status = write_pipeline_status(
        root,
        requested_stop_after=stop_after,
        completed_stage="phase1",
        artifacts=artifacts,
        source_sha256=source.get("sha256"),
        preprocessing_sha256=runtime._digest(preprocessing),
    )
    runtime._record_event(
        options["trace"], "phase1", "operator_skip", "completed", model_request_count=0
    )
    result = None
    if stop_after != "phase1":
        result = run_joint_subject(
            root=root,
            root_run_id=run_id,
            output_dir=root,
            subject=subject,
            recording_profile=profile,
            recording_profile_artifact_sha256=profile_hash,
            visibility_analysis=visibility,
            visibility_artifact_sha256=visibility_hash,
            source=source,
            preprocessing=preprocessing,
            samples=samples,
            config=config,
            visual_input_records=records,
            phase1_admission=admission,
            **options,
        )
        artifacts.append(result.phase2_evidence_path)
        if result.analysis_path is not None:
            artifacts.append(result.analysis_path)
        if stop_after == "phase3":
            artifacts.append(write_json(root / "inference-plan.json", plan))
        status = write_pipeline_status(
            root,
            requested_stop_after=stop_after,
            completed_stage=stop_after,
            artifacts=artifacts,
            source_sha256=source.get("sha256"),
            preprocessing_sha256=runtime._digest(preprocessing),
        )
    paths = (result.analysis_path,) if result and result.analysis_path else ()
    index = (
        write_json(
            root / "analysis-index.json",
            {
                "schema_version": "0.3.0",
                "run_id": run_id,
                "status": "completed" if stop_after == "phase3" else "stopped",
                "target_mode": "individual",
                "selected_subject_count": 1,
                "configuration": config.to_dict(),
                "phase1": {"enabled": False},
                "analyses": [
                    {
                        "subject_id": subject["subject_id"],
                        "annotations": p.name,
                        "sha256": runtime._file_sha256(p),
                    }
                    for p in paths
                ],
            },
        )
        if stop_after == "phase3"
        else None
    )
    return runtime.Phase2RunResult(
        run_id=run_id,
        target_mode="individual",
        analyses=(result.analysis,) if result and result.analysis else (),
        analysis_paths=paths,
        index_path=index,
        completed_stage=stop_after,
        status_path=status,
        stage_artifact_paths=tuple(artifacts),
    )
