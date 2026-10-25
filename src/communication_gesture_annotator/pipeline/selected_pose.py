# SPDX-License-Identifier: Apache-2.0
"""Bind existing pose tracks to Phase1's visible speaker, without model calls.

A sole detected track in an admitted main-scene interval is the existing target
hypothesis. Extend that hypothesis only along an uninterrupted identical segment;
never select the largest/moving person or bridge a source/track/view break.
"""

from __future__ import annotations

from copy import deepcopy
from bisect import bisect_left
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from ..domain import AnalysisSample

VERSION = "phase1-selected-pose-continuity-v0.2"


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_segments(samples: Sequence[AnalysisSample], visibility: Mapping[str, Any],
                    features: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Resolve competing detections only with a continuous unique-track anchor."""
    scenes = {r["sample_index"]: r for r in visibility["samples"]}
    candidates: dict[int, set[str]] = {}
    anchors: set[tuple[int, str]] = set()
    for sample, feature in zip(samples, features, strict=True):
        scene = scenes[sample.index]
        admitted = (scene.get("scene_membership") == "main_scene" and scene.get("target_visibility") == "visible"
                    and scene.get("assessment_status") != "not_assessed" and sample.source_gap_reason is None
                    and not feature.get("continuity_barrier"))
        entries = feature.get("candidates", ()) if admitted else ()
        candidates[sample.index] = {c["continuity_segment_id"] for c in entries
                                    if c.get("coverage_fraction", 0) > 0 and c.get("continuity_segment_id") is not None}
        if len(entries) == 1 and len(candidates[sample.index]) == 1:
            anchors.add((sample.index, next(iter(candidates[sample.index]))))
    supported: dict[int, dict[str, int]] = {s.index: {} for s in samples}
    active: dict[str, list[int]] = {}

    def finish(segment: str) -> None:
        indices = active.pop(segment)
        seeds = [i for i in indices if (i, segment) in anchors]
        if seeds:
            for i in indices:
                offset = bisect_left(seeds, i)
                neighbours = seeds[max(0, offset-1):offset+1]
                supported[i][segment] = min(neighbours, key=lambda n: (abs(n-i), n))

    prior_end = None
    for sample in samples:
        present = candidates[sample.index]
        for segment in list(active):
            if segment not in present or prior_end != sample.start_us:
                finish(segment)
        for segment in present:
            active.setdefault(segment, []).append(sample.index)
        prior_end = sample.end_us
    for segment in list(active):
        finish(segment)
    result = {}
    for sample in samples:
        choices = supported[sample.index]
        segment = next(iter(choices)) if len(choices) == 1 else None
        feature = features[sample.index]
        exact = feature.get("instantaneous_candidates", ())
        if feature.get("camera_cut_times_us") and len(exact) == 1:
            # One canonical bin may straddle an edit. Use the exact delivered
            # RGB's supported segment for its crop; do not join the derivatives.
            exact_segment = exact[0].get("continuity_segment_id")
            if exact_segment in choices:
                segment = exact_segment
        result[sample.index] = {"policy_version": VERSION, "sample_index": sample.index,
            "continuity_segment_id": segment,
            "anchor_sample_index": choices.get(segment),
            "status": "supported_track_hypothesis" if segment is not None else "unresolved",
            "reason": "unique_track_in_admitted_scene_and_same_segment" if segment is not None else
                      "no_unique_continuous_anchor_or_conflicting_anchors"}
    return result


class SelectedPoseView:
    """An auditable projection over immutable preprocessing, not a second copy."""

    def __init__(
        self, original: Any, samples: Sequence[AnalysisSample], visibility: Mapping[str, Any]
    ) -> None:
        from ..media.pose_conditioning import _expanded_bbox

        self.original = original
        selections = select_segments(samples, visibility, original.feature_records)
        self.selection_document = {
            "policy_version": VERSION,
            "native_artifact_sha256": original.manifest["artifacts"]["native_features"]["sha256"],
            "records": list(selections.values()),
        }
        self.projection_sha256 = _digest(self.selection_document)
        self.native_feature_records = []
        by_source = {}
        for row in original.native_feature_records:
            indices = row.get("analysis_sample_indices", ())
            segments = {selections[i]["continuity_segment_id"] for i in indices if i in selections}
            chosen = next(iter(segments)) if len(segments) == 1 and None not in segments else None
            people = [p for p in row["people"] if p["continuity_segment_id"] == chosen]
            people = people if len(people) == 1 else []
            projected = {
                **row,
                "people": people,
                "target_association": "unique_track_hypothesis"
                if people
                else "unresolved_selected_speaker",
            }
            self.native_feature_records.append(projected)
            by_source[row["source_frame_index"]] = projected
        self.feature_records = []
        self.focus_frame_records = []
        focus = {f["sample_index"]: f for f in original.focus_frame_records}
        for old in original.feature_records:
            record = dict(old)
            selected = selections[old["sample_index"]]
            segment = selected["continuity_segment_id"]
            record["selected_pose_association"] = selected
            record["candidates"] = [
                c for c in record.get("candidates", ()) if c["continuity_segment_id"] == segment
            ]
            record["instantaneous_candidates"] = [
                c
                for c in record.get("instantaneous_candidates", ())
                if c["continuity_segment_id"] == segment
            ]
            record["pose_evidence"] = (
                "unique_eligible_person" if record["candidates"] else "unresolved_selected_speaker"
            )
            record["target_association_continuous"] = bool(record["candidates"])
            self.feature_records.append(record)
            frame = deepcopy(dict(focus[old["sample_index"]]))
            source = frame.get("selected_source_frame") or {}
            native = by_source.get(source.get("source_frame_index"), {})
            people = native.get("people", [])
            transform = frame.setdefault("transform", {})
            width, height = transform.get("input_width"), transform.get("input_height")
            crop = (
                _expanded_bbox(people[0]["bbox_xyxy_normalized"], width, height)
                if segment is not None
                and len(people) == 1
                and people[0]["continuity_segment_id"] == segment
                and frame.get("pose_rgb_pairing") == "verified_same_source_frame"
                and type(width) is int
                and type(height) is int
                else None
            )
            transform["crop_xyxy_pixels"] = list(crop) if crop else None
            frame["conditioning_mode"] = (
                "pose_guided_crop" if crop else "full_frame_fallback_unresolved_selected_speaker"
            )
            frame["selected_pose_association"] = selected
            self.focus_frame_records.append(frame)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def model_input_identity(self) -> dict[str, Any]:
        return {
            **self.original.model_input_identity(),
            "selected_pose_policy_version": VERSION,
            "selected_pose_projection_sha256": self.projection_sha256,
        }

    def temporal_packet(
        self, start_us: int, end_us: int, *, maximum_chars: int | None = None
    ) -> dict[str, Any]:
        from ..media.native_pose import native_pose_packet

        packet = native_pose_packet(
            self.native_feature_records,
            self.feature_records,
            start_us=start_us,
            end_us=end_us,
            artifact_sha256=self.selection_document["native_artifact_sha256"],
            maximum_chars=maximum_chars,
        )
        packet["selected_pose_projection_sha256"] = self.projection_sha256
        return packet
