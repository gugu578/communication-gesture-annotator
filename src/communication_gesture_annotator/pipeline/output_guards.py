# SPDX-License-Identifier: Apache-2.0
"""Output-only guards. These rules never prepare or modify a model request."""
from __future__ import annotations

from typing import Sequence

from ..domain import AnalysisSample, HandEvidenceState, JointActivityEvidence

CANDIDATE_VERSION = "contextual-score-candidates-v0.3"
LEGACY_CANDIDATE_VERSION = "contextual-score-candidates-v0.2"


def qualify_continuations(
    samples: Sequence[AnalysisSample],
    scores: Sequence[float | None],
    states: Sequence[HandEvidenceState],
    reasons: Sequence[str],
    *,
    rest_threshold: float,
    minimum_departure_us: int,
    maximum_bridged_missing_samples: int,
    bridgeable: Sequence[bool],
) -> tuple[list[HandEvidenceState], list[str]]:
    """Weak rest may continue supported activity, never bootstrap a phrase.

    Keep raw positive edge fragments and the existing brief-departure rule.
    Only promotions caused by a missing strong rest seed are restricted.
    """
    result, explanations = list(states), list(reasons)
    supported = False
    positive_duration = 0
    missing_count = 0
    for i, state in enumerate(states):
        if state is HandEvidenceState.UNKNOWN:
            positive_duration = 0
            missing_count += 1
            if not bridgeable[i] or missing_count > maximum_bridged_missing_samples:
                supported = False
            continue
        missing_count = 0
        if state is HandEvidenceState.NON_REST and scores[i] is not None and scores[i] < rest_threshold:
            positive_duration += samples[i].end_us - samples[i].start_us
            supported |= positive_duration >= minimum_departure_us
        else:
            positive_duration = 0
            if reasons[i] == "no_sustained_rest_seed":
                if not supported:
                    result[i] = HandEvidenceState.REST
                    explanations[i] = "uncertain_without_activity_seed"
            elif state is HandEvidenceState.REST:
                supported = False
    return result, explanations


def excluded_from_output(sample: AnalysisSample, evidence: JointActivityEvidence) -> bool:
    """A hard evidence barrier remains a barrier after temporal grouping."""
    return bool(
        sample.source_gap_reason
        or evidence.is_barrier
        or any(r.get("exclude_from_detection", False) for r in evidence.resolution_records)
    )
