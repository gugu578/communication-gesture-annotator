# SPDX-License-Identifier: Apache-2.0
"""Bounded, conservative arithmetic repair for Phase1 exemplar probabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .llm_contracts import TaskContractViolation


EXEMPLAR_NORMALIZATION_VERSION = "exemplar-probability-normalization-v0.1"
_LABELS = {"positive_main_scene", "not_positive_main_scene", "uncertain"}
_SUM_TOLERANCE = 1e-6
_MAXIMUM_DEFICIT = 0.01


def _validated_probabilities(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _LABELS:
        raise TaskContractViolation(
            "overview_exemplar.probabilities_invalid",
            "Phase 0a exemplar probabilities must contain exactly the three classes",
        )
    result: dict[str, float] = {}
    for label in sorted(_LABELS):
        probability = value[label]
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not isfinite(float(probability))
            or not 0.0 <= probability <= 1.0
        ):
            raise TaskContractViolation(
                "overview_exemplar.probabilities_invalid",
                "Phase 0a exemplar probabilities must be finite values in [0, 1]",
            )
        result[label] = float(probability)
    return result


def normalize_exemplar_probabilities(
    value: Any, *, sample_index: int
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Fill at most one missing percentage point with uncertainty, never confidence.

    Do not rescale a deficient vector: that could promote an otherwise ineligible
    exemplar across the positive-confidence threshold. Existing valid vectors and
    their historical sum tolerance are preserved. Overfull vectors remain invalid.
    """
    raw = _validated_probabilities(value)
    deficit = 1.0 - sum(raw.values())
    if abs(deficit) <= _SUM_TOLERANCE:
        return raw, None
    if not _SUM_TOLERANCE < deficit <= _MAXIMUM_DEFICIT + 1e-12:
        raise TaskContractViolation(
            "overview_exemplar.probabilities_invalid",
            "Phase 0a exemplar probabilities must sum to 1; only a shortfall of at most 0.01 is repairable",
        )
    canonical = {**raw, "uncertain": raw["uncertain"] + deficit}
    return canonical, {
        "policy_version": EXEMPLAR_NORMALIZATION_VERSION,
        "event": "missing_mass_to_uncertain",
        "sample_index": sample_index,
        "raw_probabilities": raw,
        "canonical_probabilities": canonical.copy(),
    }


def validate_exemplar_normalization_events(value: Any) -> list[dict[str, Any]]:
    """Keep the audit ledger content-free and verify its exact arithmetic."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("exemplar normalization events must be an array")
    events: list[dict[str, Any]] = []
    for event in value:
        if not isinstance(event, Mapping) or set(event) != {
            "policy_version", "event", "sample_index",
            "raw_probabilities", "canonical_probabilities",
        }:
            raise ValueError("exemplar normalization event has an invalid field set")
        index = event["sample_index"]
        if type(index) is not int or index < 0:
            raise ValueError("exemplar normalization event sample_index is invalid")
        _validated_probabilities(event["canonical_probabilities"])
        _, expected = normalize_exemplar_probabilities(
            event["raw_probabilities"], sample_index=index
        )
        if expected is None or dict(event) != expected:
            raise ValueError("exemplar normalization event is inconsistent")
        events.append(expected)
    return events


class ExemplarValidationResult(dict[str, Any]):
    """Canonical response with an audit ledger outside the model response schema."""

    def __init__(
        self, values: Mapping[str, Any], *, normalization_events: Sequence[Mapping[str, Any]] = ()
    ) -> None:
        super().__init__(values)
        self.normalization_events = validate_exemplar_normalization_events(normalization_events)
        # Existing successful checkpoints recorded no policy; keep them reusable.
        self.normalization_policy_version = (
            EXEMPLAR_NORMALIZATION_VERSION if self.normalization_events else None
        )
