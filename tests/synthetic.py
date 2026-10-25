# SPDX-License-Identifier: Apache-2.0
"""Synthetic source-rate observations and scripted model responses; no corpus data."""

from types import SimpleNamespace
import math
from typing import Any
from communication_gesture_annotator.domain import AnalysisSample
from communication_gesture_annotator.media.native_pose import (
    iter_native_records,
    native_pose_packet,
    PosePacketCapacityError,
)
from communication_gesture_annotator.pipeline import probability_pass as p
from communication_gesture_annotator.pipeline.phase1_admission import build_admission
from communication_gesture_annotator.pipeline.llm_contracts import ModelCallResult
from test_runtime import ScriptedModel


def _samples(count):
    return [
        AnalysisSample(i, i * 100000, (i + 1) * 100000, i * 100000 + 50000) for i in range(count)
    ]


class SyntheticNativeConditioning:
    """Real packet construction over independently timed synthetic pose observations."""

    exact_frame_records: tuple = ()

    def __init__(self, sample_count: int) -> None:
        self.feature_records = [
            {"sample_index": sample.index, "start_us": sample.start_us, "end_us": sample.end_us}
            for sample in _samples(sample_count)
        ]
        rows, people = ([], [])
        for index in range(sample_count * 3):
            timestamp = round(index * 1000000 / 30)
            end = round((index + 1) * 1000000 / 30)
            points = [[0.4, 0.4] for _ in range(133)]
            points[5], points[6], points[11], points[12] = (
                [0.3, 0.2],
                [0.7, 0.2],
                [0.3, 0.7],
                [0.7, 0.7],
            )
            points[9] = [round(0.5 + 0.2 * math.sin(index / 6), 9), 0.5]
            confidences = [0.9] * 133
            raw_person = {
                "track_id": 0,
                "keypoints_image_xy_normalized": points,
                "keypoint_confidences": confidences,
            }
            rows.append(
                {
                    "source_frame_id": f"source-frame:{index}",
                    "source_frame_index": index,
                    "source_frame": {
                        "raw_pts": 1000 + index,
                        "time_base": {"numerator": 1, "denominator": 30},
                    },
                    "canonical_time_us": timestamp,
                    "canonical_end_us": end,
                    "analysis_sample_indices": [index // 3],
                    "decoded_frame": {"width": 640, "height": 400},
                    "people": [raw_person],
                }
            )
            people.append(
                SimpleNamespace(
                    source_frame_index=index,
                    track_id=0,
                    segment_id="track-0-segment-0",
                    bbox=(0.1, 0.1, 0.9, 0.9),
                    confidences=confidences,
                    points=[tuple(point) for point in points],
                    smoothing_support=[None] * 133,
                )
            )
        self.native_feature_records = list(iter_native_records(rows, people))
        self.attempted_intervals: list[tuple[int, int, bool]] = []

    def temporal_packet(
        self, start_us: int, end_us: int, *, maximum_chars: int | None = None
    ) -> dict[str, Any]:
        try:
            packet = native_pose_packet(
                self.native_feature_records,
                self.feature_records,
                start_us=start_us,
                end_us=end_us,
                artifact_sha256="a" * 64,
                maximum_chars=maximum_chars,
            )
        except PosePacketCapacityError:
            self.attempted_intervals.append((start_us, end_us, False))
            raise
        self.attempted_intervals.append((start_us, end_us, True))
        return packet


def scene(index: int, **changes: object) -> dict[str, object]:
    return {
        "sample_index": index,
        "target_visibility": "visible",
        "scene_membership": "main_scene",
        "relevance": "irrelevant",
        "probabilities": {"relevant": 0.01, "irrelevant": 0.98, "uncertain": 0.01},
        "left_hand_usability": "unusable",
        "right_hand_usability": "usable",
        **changes,
    }


def feature() -> dict[str, object]:
    hand = {
        "wrist_xy_normalized": [0.4, 0.6],
        "center_xy_normalized": [0.5, 0.6],
        "wrist_confidence": 0.95,
        "palm_confidence": 0.8,
        "motion": {"interval_count": 2, "mean_speed_normalized_per_s": 0.2},
    }
    return {
        "pose_evidence": "unique_eligible_person",
        "source_frame_ids": ["source-frame-42"],
        "candidates": [{"coverage_fraction": 1.0, "left_hand": hand, "right_hand": hand}],
    }


class ProbabilityModel(ScriptedModel):
    def complete_json(self, **kwargs):
        if kwargs["task"] != p.TASK:
            return super().complete_json(**kwargs)
        self.calls.append(kwargs)
        core = kwargs["trace_context"]["core_sample_indices"]
        start, end = getattr(self, "active_range", (3, 10))
        payload = {"non_g_unit_probability": [0.05 if start <= i < end else 0.95 for i in core]}
        parsed = kwargs["response_validator"](payload)
        return ModelCallResult(
            payload=parsed,
            model_payload=payload,
            request_id=kwargs["request_id"],
            response_sha256="b" * 64,
            latency_seconds=0.01,
            endpoint_mode="test",
        )


def admission(count, cfg, *, excluded=()):
    rows = [
        scene(i, relevance="relevant")
        if i not in excluded
        else scene(i, target_visibility="not_visible", scene_membership="other_scene")
        for i in range(count)
    ]
    return build_admission(
        samples=_samples(count),
        visibility={"samples": rows},
        pose_features={i: feature() for i in range(count)},
        config=cfg,
        frames={},
        visibility_hash="a" * 64,
    )
