# SPDX-License-Identifier: Apache-2.0
"""Bind generic evidence-resolution rules to recorded media and model calls."""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Sequence
from ..config import ResearchConfig
from ..domain.models import AnalysisSample
from .evidence_resolution import MeasurementEvidence, resolve_sample_evidence


def _point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all((isinstance(v, (int, float)) and -1 <= v <= 2 for v in value))
    )


def resolve_preprocessed_samples(
    *,
    samples: Sequence[AnalysisSample],
    visibility: Mapping[str, Any],
    pose_features: Mapping[int, Mapping[str, Any]],
    config: ResearchConfig,
    frames: Mapping[int, Mapping[str, Any]],
    visibility_hash: str,
) -> dict[int, dict[str, Any]]:
    """Resolve existing scene and modality evidence without model requests."""
    result: dict[int, dict[str, Any]] = {}
    visibility_rows = {int(v["sample_index"]): v for v in visibility["samples"]}
    for sample in samples:
        index = sample.index
        rgb = visibility_rows[index]
        not_assessed = rgb.get("assessment_status") == "not_assessed"
        rgb_probabilities = (
            {"relevant": 0.0, "irrelevant": 0.0, "uncertain": 0.0}
            if not_assessed
            else rgb["probabilities"]
        )
        feature = pose_features.get(index, {})
        candidates = feature.get("candidates", [])
        single = len(candidates) == 1 and feature.get("pose_evidence") == "unique_eligible_person"
        candidate = candidates[0] if single else {}
        associated = (
            single
            and rgb.get("target_visibility") == "visible"
            and (rgb.get("scene_membership") != "other_scene")
        )
        excluded = (
            rgb.get("relevance") == "irrelevant"
            and float(rgb["probabilities"].get("irrelevant", 0)) >= 0.9
            and (
                rgb.get("scene_membership") == "other_scene"
                or rgb.get("target_visibility") == "not_visible"
            )
        )
        native_barrier = bool(feature.get("continuity_barrier", False))
        barrier = sample.source_gap_reason is not None or native_barrier
        pose: list[MeasurementEvidence] = []
        visual: list[MeasurementEvidence] = []
        refs = (f"artifact:scene-usability:{visibility_hash}:sample:{index}",)
        pose_refs = tuple(feature.get("source_frame_ids", []))
        for side in ("left", "right"):
            hand = candidate.get(f"{side}_hand", {})
            for landmark, key in (
                ("wrist", "wrist_xy_normalized"),
                ("palm", "center_xy_normalized"),
            ):
                position = hand.get(key)
                confidence = hand.get(
                    f"{landmark}_confidence", hand.get("mean_keypoint_confidence", 0.0)
                )
                name = f"{side}_{landmark}_position"
                pose.append(
                    MeasurementEvidence(
                        name,
                        "pose",
                        _point(position),
                        associated,
                        float(confidence or 0),
                        _point(position),
                        float(candidate.get("coverage_fraction", 0)) > 0,
                        not barrier,
                        pose_refs,
                        "position",
                    )
                )
                usable_rgb = (
                    rgb.get(f"{side}_hand_usability") == "usable"
                    and rgb.get("target_visibility") != "not_visible"
                )
                visual.append(
                    MeasurementEvidence(
                        name,
                        "rgb",
                        usable_rgb,
                        not excluded,
                        max((float(v) for v in rgb_probabilities.values())),
                        True,
                        True,
                        not barrier,
                        refs,
                        "position",
                    )
                )
            motion = hand.get("motion", {})
            available_motion = isinstance(motion, Mapping) and bool(
                motion.get(
                    "interval_count",
                    motion.get("source_interval_count", motion.get("motion_interval_count", 0)),
                )
            )
            pose.append(
                MeasurementEvidence(
                    f"{side}_motion",
                    "pose",
                    available_motion,
                    associated,
                    float(hand.get("wrist_confidence", 0) or 0),
                    available_motion,
                    float(candidate.get("coverage_fraction", 0)) > 0,
                    not barrier,
                    pose_refs,
                    "motion",
                )
            )
        resolution = resolve_sample_evidence(
            index,
            pose,
            visual,
            scene_excluded=excluded,
            source_barrier=barrier,
            minimum_confidence=config.pose_preprocessing.keypoint_confidence_threshold,
        )
        plausible_pose = single and any(
            (
                _point(candidate.get(f"{side}_hand", {}).get("wrist_xy_normalized"))
                for side in ("left", "right")
            )
        )
        resolution.update(
            {
                "raw_scene_assessment": deepcopy(rgb),
                "plausible_pose_conflict": bool(excluded and plausible_pose),
                "recheck_required": bool(excluded and plausible_pose and (not barrier)),
                "continuity_barrier": barrier,
                "barrier_inside": barrier,
                "exclude_from_detection": bool(barrier or excluded),
            }
        )
        selected = frames.get(index, {}).get("selected_source_frame")
        if isinstance(selected, Mapping):
            start, end = (selected.get("display_start_us"), selected.get("display_end_us"))
            if type(start) is int and type(end) is int and (0 <= start < end):
                resolution["source_support_intervals"] = [
                    {
                        "start_us": start,
                        "end_us": end,
                        "support_kind": "selected_rgb_frame_display_coverage",
                        "observation_time_us": selected.get("observation_time_us"),
                        "source_frame_index": selected.get("source_frame_index"),
                    }
                ]
        result[index] = resolution
    return result
