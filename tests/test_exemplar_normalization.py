# SPDX-License-Identifier: Apache-2.0
"""Synthetic checks for conservative exemplar probability repair and resume."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from communication_gesture_annotator.adapters.open_webui import OpenWebUIClient
from communication_gesture_annotator.audit import TraceWriter
from communication_gesture_annotator.config import InferenceConfig, OpenWebUIConfig
from communication_gesture_annotator.domain import to_json_value
from communication_gesture_annotator.pipeline.exemplar_normalization import (
    EXEMPLAR_NORMALIZATION_VERSION,
    ExemplarValidationResult,
    normalize_exemplar_probabilities,
    validate_exemplar_normalization_events,
)
from communication_gesture_annotator.pipeline.front_end_contracts import (
    phase0a_exemplar_schema_for_count,
    validate_phase0a_exemplar_response,
)
from communication_gesture_annotator.pipeline.llm_contracts import ModelCallResult, TaskContractViolation
from communication_gesture_annotator.pipeline.llm_runner import _CheckpointingPhase2Model


def _payload(positive: float = 0.99, negative: float = 0.0, uncertain: float = 0.0) -> dict[str, Any]:
    return {
        "target_subject_ids": ["synthetic-target"],
        "samples": [{
            "sample_index": 7,
            "scene_type": "live_target",
            "selected_target_visibility": "visible",
            "classification": "positive_main_scene",
            "probabilities": {
                "positive_main_scene": positive,
                "not_positive_main_scene": negative,
                "uncertain": uncertain,
            },
            "reason": "Synthetic visible selected participant.",
        }],
    }


def _validate(value: Any) -> ExemplarValidationResult:
    return validate_phase0a_exemplar_response(
        value, expected_indices=[7], expected_subject_ids=["synthetic-target"]
    )


def _request(validator: Any = _validate) -> dict[str, Any]:
    return {
        "request_id": "synthetic-exemplar-confirmation-00000007",
        "task": "overview_exemplar_verification",
        "system_prompt": "Classify the synthetic exemplar.",
        "user_prompt": "Return a decision for global sample 7.",
        "images": [],
        "response_schema": phase0a_exemplar_schema_for_count(1),
        "trace_context": {
            "candidate_sample_indices": [7],
            "target_subject_ids": ["synthetic-target"],
            "image_refs": [],
        },
        "response_validator": validator,
    }


def _client(root: Path) -> OpenWebUIClient:
    client = OpenWebUIClient(
        OpenWebUIConfig(
            base_url="http://127.0.0.1:1", model="synthetic-vision-model",
            inter_request_cooldown_seconds=0,
        ),
        InferenceConfig(maximum_repair_attempts=1),
        environ={"GESTURE_ANNOTATOR_OPEN_WEBUI_API_KEY": "synthetic-test-key"},
        trace_writer=TraceWriter(root),
    )
    client._capability_snapshot = {"vision": True}
    return client


def _response(payload: dict[str, Any]) -> tuple[int, dict[str, Any], str, tuple[Any, ...]]:
    # Mock the HTTP boundary: these tests never contact a model service.
    return (200, {"choices": [{"message": {
        "role": "assistant", "content": json.dumps(payload),
    }, "finish_reason": "stop"}]}, "a" * 64, ())


class ExemplarNormalizationTests(unittest.TestCase):
    def test_one_percent_shortfall_goes_only_to_uncertainty_and_retains_raw(self) -> None:
        raw = _payload()
        saved = deepcopy(raw)
        result = _validate(raw)
        probabilities = result["samples"][0]["probabilities"]
        self.assertEqual(raw, saved)
        self.assertEqual(probabilities["positive_main_scene"], 0.99)
        self.assertEqual(probabilities["not_positive_main_scene"], 0.0)
        self.assertAlmostEqual(probabilities["uncertain"], 0.01)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertEqual(result.normalization_policy_version, EXEMPLAR_NORMALIZATION_VERSION)
        self.assertEqual(len(result.normalization_events), 1)
        self.assertEqual(validate_exemplar_normalization_events(result.normalization_events),
                         result.normalization_events)

    def test_normalization_cannot_promote_exemplar_over_acceptance_threshold(self) -> None:
        # Proportional normalization would increase .795 to .803, crossing .8.
        result = _validate(_payload(0.795, 0.195, 0.0))
        probabilities = result["samples"][0]["probabilities"]
        self.assertEqual(probabilities["positive_main_scene"], 0.795)
        self.assertLess(probabilities["positive_main_scene"], 0.8)
        self.assertEqual(probabilities["not_positive_main_scene"], 0.195)
        self.assertAlmostEqual(probabilities["uncertain"], 0.01)

    def test_valid_and_existing_numeric_tolerance_are_unchanged(self) -> None:
        for raw in (_payload(0.9, 0.05, 0.05), _payload(0.9999995)):
            with self.subTest(raw=raw):
                result = _validate(raw)
                self.assertEqual(result, raw)
                self.assertFalse(getattr(result, "normalization_events", ()))
                # Historical checkpoints have no normalization policy.
                self.assertIsNone(getattr(result, "normalization_policy_version", None))

    def test_canonical_result_is_idempotent(self) -> None:
        first = _validate(_payload())
        second = _validate(first)
        self.assertEqual(first, second)
        self.assertFalse(getattr(second, "normalization_events", ()))
        self.assertEqual(len(first.normalization_events), 1)

    def test_model_call_serializes_canonical_payload_and_separate_events(self) -> None:
        raw = _payload()
        validated = _validate(raw)
        call = ModelCallResult(
            payload=validated,
            model_payload=raw,
            request_id="synthetic-exemplar-serialization",
            response_sha256="a" * 64,
            latency_seconds=0.001,
            endpoint_mode="synthetic",
            contract_normalization_events=tuple(validated.normalization_events),
            contract_normalization_policy_version=validated.normalization_policy_version,
        )
        for name, convert in (("dataclasses.asdict", asdict), ("to_json_value", to_json_value)):
            with self.subTest(serializer=name):
                # asdict rebuilds dict subclasses from an iterator of key/value pairs.
                serialized = json.loads(json.dumps(convert(call), allow_nan=False))
                self.assertEqual(serialized["payload"], dict(validated))
                self.assertEqual(serialized["model_payload"], raw)
                self.assertEqual(serialized["contract_normalization_events"],
                                 validated.normalization_events)
                self.assertEqual(serialized["contract_normalization_policy_version"],
                                 EXEMPLAR_NORMALIZATION_VERSION)
        self.assertEqual(len(validated.normalization_events), 1)

    def test_helper_preserves_both_decided_classes(self) -> None:
        probabilities = _payload(0.0, 0.99)["samples"][0]["probabilities"]
        saved = deepcopy(probabilities)
        result, event = normalize_exemplar_probabilities(probabilities, sample_index=7)
        self.assertEqual(probabilities, saved)
        self.assertEqual(result["positive_main_scene"], 0.0)
        self.assertEqual(result["not_positive_main_scene"], 0.99)
        self.assertAlmostEqual(result["uncertain"], 0.01)
        self.assertIsNotNone(event)

    def test_invalid_distributions_still_fail_with_specific_code(self) -> None:
        valid = _payload()["samples"][0]["probabilities"]
        cases = [
            {**valid, "positive_main_scene": 0.98},
            {**valid, "positive_main_scene": 0.989999},
            {**valid, "positive_main_scene": 1.0, "uncertain": 0.01},
            {**valid, "positive_main_scene": True},
            {**valid, "positive_main_scene": float("nan")},
            {**valid, "positive_main_scene": float("inf")},
            {**valid, "positive_main_scene": -0.01},
            {**valid, "positive_main_scene": 1.01},
            {**valid, "extra": 0.0},
            {key: value for key, value in valid.items() if key != "uncertain"},
        ]
        for probabilities in cases:
            with self.subTest(probabilities=probabilities):
                raw = _payload()
                raw["samples"][0]["probabilities"] = probabilities
                with self.assertRaises(TaskContractViolation) as raised:
                    _validate(raw)
                self.assertEqual(raised.exception.code, "overview_exemplar.probabilities_invalid")

    def test_maximum_class_remains_required_before_and_after_correction(self) -> None:
        for raw in (_payload(0.4, 0.59, 0), _payload(0.33, 0.33, 0.33)):
            with self.subTest(raw=raw):
                with self.assertRaises(TaskContractViolation) as raised:
                    _validate(raw)
                self.assertEqual(raised.exception.code,
                                 "overview_exemplar.classification_not_maximum")

    def test_semantics_and_exact_target_coverage_are_not_relaxed(self) -> None:
        cases = []
        wrong_scene = _payload()
        wrong_scene["samples"][0]["scene_type"] = "slide"
        cases.append(wrong_scene)
        absent = _payload()
        absent["samples"][0]["selected_target_visibility"] = "not_visible"
        cases.append(absent)
        wrong_subject = _payload()
        wrong_subject["target_subject_ids"] = ["different-synthetic-target"]
        cases.append(wrong_subject)
        wrong_index = _payload()
        wrong_index["samples"][0]["sample_index"] = 8
        cases.append(wrong_index)
        duplicate = _payload()
        duplicate["samples"] *= 2
        cases.append(duplicate)
        missing = _payload()
        missing["samples"] = []
        cases.append(missing)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _validate(raw)

    def test_normalization_ledger_rejects_tampering(self) -> None:
        events = _validate(_payload()).normalization_events
        wrong_policy = deepcopy(events)
        wrong_policy[0]["policy_version"] = "unsupported-policy"
        extra = deepcopy(events)
        extra[0]["model_authored_prose"] = "Not part of the numeric event contract."
        promoted = deepcopy(events)
        promoted[0]["canonical_probabilities"]["positive_main_scene"] = 1.0
        promoted[0]["canonical_probabilities"]["uncertain"] = 0.0
        wrong_index = deepcopy(events)
        wrong_index[0]["sample_index"] = True
        for value in (wrong_policy, extra, promoted, wrong_index, "not an event list", [{}]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_exemplar_normalization_events(value)

    def test_adapter_accepts_shortfall_once_and_checkpoint_retains_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = _client(root)
            first = _CheckpointingPhase2Model(client, run_root=root,
                scope={"synthetic": True}, resume=False, trace=None)
            with patch.object(client, "_request_json", return_value=_response(_payload())) as http:
                result = first.complete_json(**_request())
                resumed = _CheckpointingPhase2Model(client, run_root=root,
                    scope={"synthetic": True}, resume=True, trace=None)
                reused = resumed.complete_json(**_request())
            self.assertEqual(http.call_count, 1)
            self.assertEqual(resumed.reused_count, 1)
            self.assertFalse(result.repaired)
            self.assertEqual(result.model_payload, _payload())
            self.assertNotEqual(result.model_payload_sha256, result.payload_sha256)
            self.assertEqual(result.contract_normalization_policy_version,
                             EXEMPLAR_NORMALIZATION_VERSION)
            self.assertEqual(len(result.contract_normalization_events), 1)
            self.assertEqual(reused.payload, result.payload)
            self.assertEqual(reused.model_payload, result.model_payload)
            self.assertEqual(list(reused.contract_normalization_events),
                             list(result.contract_normalization_events))
            parsed_path = next(root.glob("model-traces/exchange-*/parsed-response.json"))
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["model_output"], _payload())
            self.assertEqual(parsed["contract_normalized_output"], result.payload)
            self.assertEqual(parsed["contract_normalization_policy_version"],
                             EXEMPLAR_NORMALIZATION_VERSION)

    def test_unchanged_historical_checkpoint_reuses_with_new_validator(self) -> None:
        def old_validator(value: Any) -> dict[str, Any]:
            # The old contract returned a plain mapping without event metadata.
            return dict(_validate(value))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = _client(root)
            first = _CheckpointingPhase2Model(client, run_root=root,
                scope={"synthetic": True}, resume=False, trace=None)
            raw = _payload(0.9, 0.05, 0.05)
            with patch.object(client, "_request_json", return_value=_response(raw)) as http:
                result = first.complete_json(**_request(old_validator))
                resumed = _CheckpointingPhase2Model(client, run_root=root,
                    scope={"synthetic": True}, resume=True, trace=None)
                reused = resumed.complete_json(**_request())
            self.assertEqual(http.call_count, 1)
            self.assertEqual(resumed.reused_count, 1)
            self.assertEqual(reused.payload, result.payload)
            self.assertFalse(reused.contract_normalization_events)
            self.assertIsNone(reused.contract_normalization_policy_version)


if __name__ == "__main__":
    unittest.main()
