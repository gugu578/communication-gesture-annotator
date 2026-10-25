# SPDX-License-Identifier: Apache-2.0
"""Current probability, grouping and transport contracts with synthetic inputs."""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from typing import Any, Sequence
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from communication_gesture_annotator.adapters.open_webui import OpenWebUIClient, OpenWebUIError
from communication_gesture_annotator.config import (
    AnalysisConfig,
    DetectionConfig,
    InferenceConfig,
    OpenWebUIConfig,
)
from communication_gesture_annotator.domain import (
    AnalysisSample,
    JointActivityEvidence,
    JointActivityState,
)
from communication_gesture_annotator.pipeline.grouping import propose_joint_activity_candidates
from communication_gesture_annotator.pipeline.motion_quality import score_measurement
from communication_gesture_annotator.pipeline.probability_pass import (
    SCHEMA,
    VERSION,
    candidate_policy,
    candidate_diagnostics,
    validate_response,
)
from communication_gesture_annotator.pipeline.probability_finalization import (
    finalize_probability_candidates,
)


def observations(
    scores: Sequence[float],
) -> tuple[list[AnalysisSample], list[JointActivityEvidence]]:
    samples = [
        AnalysisSample(i, i * 100000, (i + 1) * 100000, i * 100000 + 50000)
        for i in range(len(scores))
    ]
    evidence = []
    for score in scores:
        state = JointActivityState.ACTIVE if score < 0.5 else JointActivityState.PASSIVE
        evidence.append(
            JointActivityEvidence(
                state,
                {s.value: float(s == state) for s in JointActivityState},
                1.0,
                resolution_records=(
                    {
                        "policy": VERSION,
                        "motion_grounding": {
                            "policy_version": VERSION,
                            "non_g_unit_probability": score,
                            "candidate_policy": candidate_policy(DetectionConfig()),
                        },
                    },
                ),
            )
        )
    return samples, evidence


class ProbabilityTests(unittest.TestCase):
    def test_motion_path_local_fingers_and_missingness(self) -> None:
        hand = {
            "values": {
                "palm_path": 0.1,
                "palm_supported_us": 100000,
                "palm_speed": 9.0,
                "orientation_speed": 0.0,
                "extension_speed": 0.0,
                "articulation_speed": 0.0,
            },
            "flags": [],
        }
        entry = {
            "start_us": 0,
            "end_us": 100000,
            "hands": {s: deepcopy(hand) for s in ("left", "right")},
        }
        score = score_measurement(entry, DetectionConfig(), None)
        self.assertEqual(score["score_components"]["left"]["transport"], 1.0)
        self.assertEqual(score["motion_non_g_probability"], 0.5)
        entry["hands"]["left"]["values"]["articulation_speed"] = 10.0
        fingers = score_measurement(entry, DetectionConfig(), None)
        self.assertLess(fingers["motion_non_g_probability"], score["motion_non_g_probability"])
        self.assertEqual(fingers["score_components"]["left"]["fingers"], 2.0)
        entry["hands"] = {}
        self.assertIsNone(
            score_measurement(entry, DetectionConfig(), None)["motion_non_g_probability"]
        )
        with self.assertRaises(ValueError):
            DetectionConfig(motion_articulation_scale=0)

    def test_short_departure_suppression_and_visible_edge_fragment(self) -> None:
        samples, evidence = observations([0.9] * 4 + [0.1] * 2 + [0.9] * 4)
        diagnostics = candidate_diagnostics(samples, evidence, AnalysisConfig())
        self.assertEqual(diagnostics.discarded_departure_indices, (4, 5))
        self.assertEqual(diagnostics.candidates, ())
        samples, evidence = observations([0.1] * 4)
        candidates, _ = propose_joint_activity_candidates(samples, evidence, AnalysisConfig())
        decisions, units = finalize_probability_candidates(
            samples, evidence, candidates, AnalysisConfig()
        )
        self.assertEqual(len(units), 1)
        self.assertTrue(all(d.is_g_unit for d in decisions))
        self.assertFalse(candidates[0].start_observed)
        self.assertFalse(candidates[0].end_observed)

    def test_inclusive_gap_merge_retains_raw_decisions(self) -> None:
        samples, evidence = observations([0.1] * 4 + [0.9] * 3 + [0.1] * 4)
        candidates, _ = propose_joint_activity_candidates(samples, evidence, AnalysisConfig())
        self.assertEqual(len(candidates), 2)
        for limit, expected in ((0, 2), (299, 2), (300, 1)):
            decisions, units = finalize_probability_candidates(
                samples, evidence, candidates, AnalysisConfig(), merge_gap_ms=limit
            )
            self.assertEqual(len(units), expected)
            self.assertEqual(decisions[5].raw_decision, "not_g_unit")
            self.assertEqual(decisions[5].is_g_unit, limit == 300)
            if limit == 300:
                self.assertIn("finalization_merged_short_gap", decisions[5].policy_overrides)

    def test_repair_keeps_all_images_context_and_configured_attempt_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "synthetic.png"
            self.assertTrue(cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8)))
            payloads = []
            responses = iter(
                ['{"non_g_unit_probability":[0.2]}', '{"non_g_unit_probability":[0.2,0.7]}']
            )

            def request(method: str, path: str, payload: dict[str, Any], **kwargs: Any) -> tuple:
                payloads.append(deepcopy(payload))
                body = next(responses)
                return (
                    200,
                    {"choices": [{"message": {"content": body}, "finish_reason": "stop"}]},
                    hashlib.sha256(body.encode()).hexdigest(),
                    (),
                )

            client = OpenWebUIClient(
                OpenWebUIConfig(model="synthetic", inter_request_cooldown_seconds=0),
                InferenceConfig(maximum_repair_attempts=1),
                environ={},
            )
            client._capability_snapshot = {"vision": True}
            args = dict(
                request_id="synthetic-request",
                task="non_g_probability",
                system_prompt="system",
                user_prompt="all original history and motion",
                images=[image],
                response_schema=SCHEMA,
                trace_context={"core_sample_indices": [7, 8]},
                response_validator=lambda v: validate_response(v, [7, 8]),
            )
            with patch.object(client, "_request_json", side_effect=request):
                result = client.complete_json(**args)
            self.assertTrue(result.repaired)
            self.assertEqual(len(payloads), 2)
            before, after = [p["messages"][1]["content"] for p in payloads]
            self.assertEqual(before[1:], after[1:])
            self.assertTrue(after[0]["text"].startswith(before[0]["text"]))
            self.assertEqual(result.payload["non_g_unit_probability"], [0.2, 0.7])
            client.inference = replace(client.inference, maximum_repair_attempts=0)
            responses = iter(['{"non_g_unit_probability":[0.2]}'])
            payloads.clear()
            with patch.object(client, "_request_json", side_effect=request):
                with self.assertRaises(OpenWebUIError):
                    client.complete_json(**args)
            self.assertEqual(len(payloads), 1)


if __name__ == "__main__":
    unittest.main()
