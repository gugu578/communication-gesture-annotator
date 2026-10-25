# SPDX-License-Identifier: Apache-2.0
"""Checkpoint replay preserves chronological inputs without rebuilding images."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import Mock, patch

from communication_gesture_annotator.config import ResearchConfig, OpenWebUIConfig
from communication_gesture_annotator.domain import AnalysisSample
from communication_gesture_annotator.export.json_export import read_json, write_json
from communication_gesture_annotator.pipeline import probability_pass as probability
from communication_gesture_annotator.pipeline.llm_contracts import ModelCallResult
from communication_gesture_annotator.pipeline.llm_runner import _CheckpointingPhase2Model
from communication_gesture_annotator.pipeline.phase1_admission import build_admission


class OfflineModel:
    def __init__(self, stop_after: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stop_after = stop_after

    def complete_json(self, **kwargs: Any) -> ModelCallResult:
        if self.stop_after is not None and len(self.calls) == self.stop_after:
            raise RuntimeError("synthetic interruption")
        self.calls.append(kwargs)
        payload = {"non_g_unit_probability": [0.12 if 3 <= i < 17 else 0.93
            for i in kwargs["trace_context"]["core_sample_indices"]]}
        return ModelCallResult(payload=kwargs["response_validator"](payload), model_payload=payload,
            request_id=kwargs["request_id"], response_sha256="b" * 64,
            latency_seconds=0.01, endpoint_mode="test")


class FastResumeTest(unittest.TestCase):
    def inputs(self) -> dict[str, Any]:
        cfg = ResearchConfig(open_webui=OpenWebUIConfig(model="synthetic"))
        cfg = replace(cfg, detection=replace(cfg.detection, probability_target_frames=3,
            probability_history_frames=4, probability_future_frames=3))
        samples = [AnalysisSample(i, i*100000, (i+1)*100000, i*100000+50000) for i in range(24)]
        hand = {"wrist_xy_normalized": [0.4, 0.6], "center_xy_normalized": [0.5, 0.6],
                "wrist_confidence": 0.95, "palm_confidence": 0.8,
                "motion": {"interval_count": 2, "mean_speed_normalized_per_s": 0.2}}
        features = {i: {"pose_evidence": "unique_eligible_person", "source_frame_ids": [f"f:{i}"],
            "candidates": [{"coverage_fraction": 1.0, "left_hand": hand, "right_hand": hand}]}
            for i in range(24)}
        visibility = {"samples": [{"sample_index": i, "target_visibility": "visible",
            "scene_membership": "main_scene", "relevance": "relevant",
            "probabilities": {"relevant": 0.98, "irrelevant": 0.01, "uncertain": 0.01},
            "left_hand_usability": "usable", "right_hand_usability": "usable"} for i in range(24)]}
        resolutions = build_admission(samples=samples, visibility=visibility, pose_features=features,
            config=cfg, frames={}, visibility_hash="a"*64)
        # Include a cut and an excluded interval to exercise prior-output quarantine.
        for i, row in resolutions.items():
            row["camera_shot_id"] = int(i >= 11)
        resolutions[18]["exclude_from_detection"] = True
        resolutions[18]["phase2_context_usable"] = False
        return dict(samples=samples, subject={"subject_id": "s"}, resolutions=resolutions,
            frames={s.index: {"start_us": s.start_us, "end_us": s.end_us} for s in samples},
            config=cfg, conditioning=None, trace=None)

    @staticmethod
    def visuals(request_id: str, indices: list[int], stage: str, **kwargs: Any) -> Any:
        return [], [{"kind": "individual_frame", "sample_index": i, "image_ordinal": n}
                    for n, i in enumerate(indices)]

    def run_case(self, root: Path, inputs: dict[str, Any], model: OfflineModel,
                 *, resume: bool = False, visuals: Any = None) -> tuple[Any, Any]:
        wrapper = _CheckpointingPhase2Model(model, run_root=root, scope={"test": "resume"},
                                           resume=resume, trace=None)
        slots = [None]*len(inputs["samples"])
        traces = probability.run_pass(**inputs, model=wrapper, slots=slots,
            visuals=visuals or self.visuals, artifact_dir=root)
        return traces, slots

    def test_interrupted_replay_preserves_next_prompt_and_all_decisions(self) -> None:
        with TemporaryDirectory() as directory:
            root, inputs = Path(directory), self.inputs()
            uninterrupted = OfflineModel()
            expected_traces, expected_slots = self.run_case(root/"complete", inputs, uninterrupted)
            first = OfflineModel(stop_after=4)
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                self.run_case(root/"partial", inputs, first)
            remaining = OfflineModel()
            visuals = Mock(side_effect=self.visuals)
            traces, slots = self.run_case(root/"partial", inputs, remaining, resume=True, visuals=visuals)
            self.assertEqual(traces, expected_traces)
            self.assertEqual(slots, expected_slots)
            self.assertEqual(visuals.call_count, len(remaining.calls))
            for actual, expected in zip(remaining.calls, uninterrupted.calls[4:], strict=True):
                for key in ("request_id", "user_prompt", "system_prompt", "trace_context", "response_schema"):
                    self.assertEqual(actual[key], expected[key])
            with patch("communication_gesture_annotator.media.motion_trace.write_motion_trace",
                       side_effect=AssertionError("cached chart rendered")):
                cached = OfflineModel(stop_after=0)
                self.assertEqual(self.run_case(root/"partial", inputs, cached, resume=True,
                    visuals=Mock(side_effect=AssertionError("cached images rebuilt"))), (traces, slots))

    def test_legacy_compact_prefix_only_rebuilds_boundary_requests(self) -> None:
        with TemporaryDirectory() as directory:
            root, inputs = Path(directory), self.inputs()
            first = OfflineModel(stop_after=5)
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                self.run_case(root, inputs, first)
            for sidecar in (root/"model-checkpoints").glob("replay-*.json"):
                sidecar.unlink()
            visuals, remaining = Mock(side_effect=self.visuals), OfflineModel()
            traces, slots = self.run_case(root, inputs, remaining, resume=True, visuals=visuals)
            rebuilt = [c.args[0] for c in visuals.call_args_list]
            self.assertEqual(rebuilt[:2], [first.calls[0]["request_id"], first.calls[-1]["request_id"]])
            self.assertEqual(len(rebuilt), 2+len(remaining.calls))
            expected_model = OfflineModel()
            _, expected_slots = self.run_case(root/"comparison", inputs, expected_model)
            self.assertEqual(slots, expected_slots)
            self.assertEqual(remaining.calls[0]["user_prompt"], expected_model.calls[5]["user_prompt"])
            self.assertIn("checkpoint_restore", traces[1])
            self.assertIsNone(traces[1]["model_view_sha256"])

    def test_changed_evidence_or_checkpoint_is_rejected_before_new_requests(self) -> None:
        with TemporaryDirectory() as directory:
            root, inputs = Path(directory), self.inputs()
            self.run_case(root, inputs, OfflineModel())
            changed = {**inputs, "frames": deepcopy(inputs["frames"])}
            changed["frames"][0]["end_us"] += 1
            with self.assertRaisesRegex(ValueError, "resume evidence"):
                self.run_case(root, changed, OfflineModel(stop_after=0), resume=True)
            path = next((root/"model-checkpoints").glob("replay-*.json"))
            checkpoint = path.with_name(path.name.replace("replay-", "checkpoint-", 1))
            saved = read_json(checkpoint)
            saved["input_sha256"] = "c"*64
            write_json(checkpoint, saved)
            with self.assertRaisesRegex(ValueError, "resume evidence"):
                self.run_case(root, inputs, OfflineModel(stop_after=0), resume=True)

    def test_legacy_last_boundary_must_pass_before_unfinished_work(self) -> None:
        with TemporaryDirectory() as directory:
            root, inputs = Path(directory), self.inputs()
            first = OfflineModel(stop_after=5)
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                self.run_case(root, inputs, first)
            from hashlib import sha256
            last = root/"model-checkpoints"/("checkpoint-"+
                sha256(first.calls[-1]["request_id"].encode()).hexdigest()[:24]+".json")
            document = read_json(last)
            document["input_sha256"] = "d"*64
            write_json(last, document)
            for sidecar in (root/"model-checkpoints").glob("replay-*.json"):
                sidecar.unlink()
            remaining = OfflineModel(stop_after=0)
            with self.assertRaises(ValueError):
                self.run_case(root, inputs, remaining, resume=True)
            self.assertEqual(remaining.calls, [])


if __name__ == "__main__":
    unittest.main()
