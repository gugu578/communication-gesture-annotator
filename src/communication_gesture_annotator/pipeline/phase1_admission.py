# SPDX-License-Identifier: Apache-2.0
"""Freeze speaker/scene eligibility before the single-pass Phase2 starts."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
from ..config import ResearchConfig
from ..domain import AnalysisSample
from .joint_resolution import resolve_preprocessed_samples
from .main_scene_visibility import PROMPT_VERSION as MAIN_SCENE_VERSION

VERSION = "phase1-speaker-scene-admission-v0.3"


def build_admission(*, samples: Sequence[AnalysisSample], visibility: Mapping[str, Any],
                    pose_features: Mapping[int, Mapping[str, Any]], config: ResearchConfig,
                    frames: Mapping[int, Mapping[str, Any]], visibility_hash: str) -> dict[int, dict[str, Any]]:
    """Use dense Phase1 scene facts; a plausible skeleton never selects a speaker."""
    result = resolve_preprocessed_samples(samples=samples, visibility=visibility,
        pose_features=pose_features, config=config, frames=frames, visibility_hash=visibility_hash)
    for index, record in result.items():
        scene = record["raw_scene_assessment"]
        scene_only = scene.get("assessment_kind") == MAIN_SCENE_VERSION
        # A reconstruction switch prevents derivatives, not recognition of the
        # same visible speaker in earlier RGB. Keep context eligibility separate
        # from sample-level detection/measurement eligibility.
        context_reasons = []
        if scene.get("assessment_status") == "not_assessed":
            context_reasons.append("outside_requested_visibility_context")
        if scene.get("scene_membership") != "main_scene":
            context_reasons.append("main_scene_not_established")
        if scene.get("target_visibility") != "visible":
            context_reasons.append("selected_speaker_not_visible")
        if samples[index].source_gap_reason is not None:
            context_reasons.append("source_gap")
        pose = pose_features.get(index, {})
        if "selected_pose_association" in pose:
            record["selected_pose_association"] = dict(pose["selected_pose_association"])
        record.update(phase2_context_usable=not context_reasons,
            phase2_context_break_reasons=context_reasons,
            pose_motion_discontinuity=bool(pose.get("continuity_barrier") or pose.get("camera_cut_times_us")),
            pose_motion_discontinuity_reasons=list(pose.get("continuity_barrier_reasons", []))
                + (["camera_edit"] if pose.get("camera_cut_times_us") else []),
            camera_shot_id=pose.get("camera_shot_id"),
            camera_cut_times_us=list(pose.get("camera_cut_times_us", [])))
        reasons = []
        if scene.get("assessment_status") == "not_assessed":
            reasons.append("outside_requested_visibility_context")
        if scene.get("scene_membership") != "main_scene":
            reasons.append("main_scene_not_established")
        if scene.get("target_visibility") != "visible":
            reasons.append("selected_speaker_not_visible")
        if record.get("continuity_barrier"):
            reasons.append("source_or_pose_continuity_barrier")
        # RGB hand detail may be missing while qualified reconstructed motion is
        # available. Neither RGB-only relevance nor symmetry overrides that pose.
        # v0.8 does not assess hand usability. An unknown field is not evidence
        # of missing hands; the admitted RGB must remain available to Phase2.
        if not scene_only and not any(p.get("usable") for p in record.get("properties", ())):
            reasons.append("manual_evidence_unavailable")
        record.update(admission_version=VERSION, admission_stage="phase1",
            exclusion_reasons=reasons, exclude_from_detection=bool(reasons),
            recheck_required=False, phase2_may_recheck=False)
        if reasons:
            record["continuity_barrier"] = True
            record["barrier_inside"] = True
    return result
