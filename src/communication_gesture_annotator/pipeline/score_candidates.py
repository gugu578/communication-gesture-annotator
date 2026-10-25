# SPDX-License-Identifier: Apache-2.0
"""Derive candidates from combined probabilities and sustained rest on the analysis grid."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
from ..config import AnalysisConfig
from ..domain import ActivityCandidate, AnalysisSample, HandEvidenceState
from .grouping import _propose_from_states

from .output_guards import CANDIDATE_VERSION, LEGACY_CANDIDATE_VERSION, qualify_continuations

POLICY_VERSION = LEGACY_CANDIDATE_VERSION


@dataclass(frozen=True, slots=True)
class ScoreEvidence:
    non_g_score: float | None
    excluded: bool = False
    barrier: bool = False
    bridgeable: bool = True
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.non_g_score is not None and (
            not math.isfinite(self.non_g_score) or not 0 <= self.non_g_score <= 1
        ):
            raise ValueError("score must be finite in [0,1] or absent")


@dataclass(frozen=True, slots=True)
class ScoreCandidatePolicy:
    rest_threshold: float = 0.6
    rest_seed_threshold: float | None = None
    minimum_departure_us: int = 300000

    def __post_init__(self) -> None:
        if not math.isfinite(self.rest_threshold) or not 0 < self.rest_threshold < 1:
            raise ValueError("rest threshold must be in (0,1)")
        if self.rest_seed_threshold is not None and (
            not self.rest_threshold <= self.rest_seed_threshold < 1
        ):
            raise ValueError("rest seed threshold must be at least the rest threshold and below 1")
        if type(self.minimum_departure_us) is not int or self.minimum_departure_us < 0:
            raise ValueError("minimum departure duration must be nonnegative integer microseconds")


@dataclass(frozen=True, slots=True)
class ScoreCandidateResult:
    candidates: tuple[ActivityCandidate, ...]
    raw_scores: tuple[float | None, ...]
    decision_scores: tuple[float | None, ...]
    states: tuple[HandEvidenceState, ...]
    reasons: tuple[str, ...]
    bridged_sample_indices: tuple[int, ...]
    discarded_departure_indices: tuple[int, ...]


# Duration and seed rules suppress brief fragments while preserving longer phrases.
def propose_score_candidates(
    samples: Sequence[AnalysisSample],
    evidence: Sequence[ScoreEvidence],
    analysis: AnalysisConfig,
    policy: ScoreCandidatePolicy = ScoreCandidatePolicy(),
    *,
    participant_id: str = "participant-1",
    policy_version: str = LEGACY_CANDIDATE_VERSION,
) -> ScoreCandidateResult:
    """Apply rest seeds, brief-departure suppression and gap rules without additional model calls."""
    if policy_version not in {LEGACY_CANDIDATE_VERSION, CANDIDATE_VERSION}:
        raise ValueError("unsupported score candidate policy")
    if len(samples) != len(evidence):
        raise ValueError("every analysis sample needs score evidence")
    if any(
        (s.index != i or (i and s.start_us != samples[i - 1].end_us) for i, s in enumerate(samples))
    ):
        raise ValueError("score replay requires a contiguous, indexed canonical grid")
    barriers = {
        i
        for i, (s, e) in enumerate(zip(samples, evidence, strict=True))
        if s.source_gap_reason or e.barrier or e.excluded
    }
    raw = [None if i in barriers else e.non_g_score for i, e in enumerate(evidence)]
    scores = list(raw)
    states, reasons = ([], [])
    for i, (score, e) in enumerate(zip(scores, evidence, strict=True)):
        if score is None:
            states.append(HandEvidenceState.UNKNOWN)
            reasons.append("excluded_or_barrier" if i in barriers else "score_unavailable")
            continue
        rest = score >= policy.rest_threshold
        reason = "score_rest" if rest else "score_activity"
        states.append(HandEvidenceState.REST if rest else HandEvidenceState.NON_REST)
        reasons.append(reason)
    if policy.rest_seed_threshold is not None:
        i = 0
        while i < len(states):
            if states[i] is not HandEvidenceState.REST:
                i += 1
                continue
            end = i + 1
            while end < len(states) and states[end] is HandEvidenceState.REST:
                end += 1
            seed_duration = 0
            qualified = False
            for j in range(i, end):
                seed_duration = (
                    seed_duration + samples[j].end_us - samples[j].start_us
                    if scores[j] >= policy.rest_seed_threshold
                    else 0
                )
                qualified |= seed_duration >= analysis.minimum_separating_rest_us
            if not qualified:
                for j in range(i, end):
                    states[j], reasons[j] = (HandEvidenceState.NON_REST, "no_sustained_rest_seed")
            i = end
    if policy_version == CANDIDATE_VERSION:
        states, reasons = qualify_continuations(
            samples, scores, states, reasons, rest_threshold=policy.rest_threshold,
            minimum_departure_us=policy.minimum_departure_us,
            maximum_bridged_missing_samples=analysis.maximum_bridged_missing_samples,
            bridgeable=[i not in barriers and e.bridgeable for i, e in enumerate(evidence)],
        )
    discarded = []
    original_states = list(states)
    i = 0
    while i < len(states):
        if original_states[i] is not HandEvidenceState.NON_REST:
            i += 1
            continue
        end = i + 1
        while end < len(states) and original_states[end] is HandEvidenceState.NON_REST:
            end += 1
        before, after = (i, end)
        while before > 0 and original_states[before - 1] is HandEvidenceState.REST:
            before -= 1
        while after < len(states) and original_states[after] is HandEvidenceState.REST:
            after += 1
        preceding_duration = samples[i].start_us - samples[before].start_us
        following_duration = samples[after - 1].end_us - samples[end - 1].end_us
        if (
            samples[end - 1].end_us - samples[i].start_us < policy.minimum_departure_us
            and preceding_duration >= analysis.minimum_separating_rest_us
            and (following_duration >= analysis.minimum_separating_rest_us)
        ):
            for j in range(i, end):
                states[j], reasons[j] = (
                    HandEvidenceState.REST,
                    "discarded_short_departure_between_supported_rest",
                )
                discarded.append(j)
        i = end
    candidates, bridged = _propose_from_states(
        samples,
        states,
        [i not in barriers and e.bridgeable for i, e in enumerate(evidence)],
        [e.references for e in evidence],
        analysis,
        participant_id=participant_id,
        proposal_basis=policy_version,
        barriers=barriers,
    )
    return ScoreCandidateResult(
        tuple(candidates),
        tuple(raw),
        tuple(scores),
        tuple(states),
        tuple(reasons),
        tuple(sorted(bridged)),
        tuple(discarded),
    )
