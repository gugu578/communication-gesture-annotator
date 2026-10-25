# SPDX-License-Identifier: Apache-2.0
"""Deterministic Step 2 evidence grouping."""

from __future__ import annotations
from collections.abc import Sequence
from ..config import AnalysisConfig
from ..domain import (
    ActivityCandidate,
    AnalysisSample,
    HandEvidenceState,
    JointActivityEvidence,
    JointActivityState,
)


def bridge_short_unknown_runs(
    states: Sequence[HandEvidenceState],
    maximum_length: int,
    bridgeable: Sequence[bool] | None = None,
) -> tuple[list[HandEvidenceState], set[int]]:
    """Bridge bounded one/two-sample missing runs for continuity only.

    The caller retains the raw evidence. Returned indices identify values that
    were inherited from compatible surrounding contextual states.
    """
    if bridgeable is not None and len(bridgeable) != len(states):
        raise ValueError("bridgeability flags must align with evidence states")
    effective = list(states)
    bridged: set[int] = set()
    index = 0
    while index < len(states):
        if states[index] is not HandEvidenceState.UNKNOWN:
            index += 1
            continue
        end = index
        while end < len(states) and states[end] is HandEvidenceState.UNKNOWN:
            end += 1
        run_length = end - index
        before = states[index - 1] if index > 0 else None
        after = states[end] if end < len(states) else None
        reasons_allow_bridge = bridgeable is None or all(bridgeable[index:end])
        if (
            run_length <= maximum_length
            and before is not None
            and (before is after)
            and reasons_allow_bridge
        ):
            for bridged_index in range(index, end):
                effective[bridged_index] = before
                bridged.add(bridged_index)
        index = end
    return (effective, bridged)


def _supported_rest_closes(
    samples: Sequence[AnalysisSample],
    rest_start: int,
    rest_end_inclusive: int,
    last_active: int,
    config: AnalysisConfig,
) -> bool:
    separation_us = samples[rest_end_inclusive].end_us - samples[rest_start].start_us
    return separation_us >= config.minimum_separating_rest_us


def joint_unknown_is_bridgeable(evidence: JointActivityEvidence) -> bool:
    """A factual discontinuity always prevents bridging, however short."""
    if evidence.is_barrier:
        return False
    if evidence.effective_state is not JointActivityState.UNKNOWN:
        return True
    return not any(
        (
            cause["status"] == "confirmed"
            and cause["cause"]
            not in {
                "occluded",
                "out_of_frame",
                "insufficient_model_evidence",
                "pose_unavailable",
                "rgb_unusable",
                "missing_evidence",
            }
            for cause in evidence.uncertainty_causes
        )
    )


def propose_joint_activity_candidates(
    samples: Sequence[AnalysisSample],
    evidence: Sequence[JointActivityEvidence],
    config: AnalysisConfig,
    *,
    participant_id: str = "participant-1",
) -> tuple[list[ActivityCandidate], set[int]]:
    from .probability_pass import candidate_diagnostics

    result = candidate_diagnostics(samples, evidence, config, participant_id=participant_id)
    if result is None:
        return ([], set())
    return (list(result.candidates), set(result.bridged_sample_indices))


def _propose_from_states(
    samples: Sequence[AnalysisSample],
    raw_states: Sequence[HandEvidenceState],
    bridgeable: Sequence[bool],
    evidence_references: Sequence[tuple[str, ...]],
    config: AnalysisConfig,
    *,
    participant_id: str,
    proposal_basis: str,
    barriers: frozenset[int] | set[int] = frozenset(),
) -> tuple[list[ActivityCandidate], set[int]]:
    states, bridged = bridge_short_unknown_runs(
        raw_states, config.maximum_bridged_missing_samples, bridgeable
    )
    interrupting_unknown_starts: set[int] = set(barriers)
    index = 0
    while index < len(states):
        if states[index] is not HandEvidenceState.UNKNOWN:
            index += 1
            continue
        end = index + 1
        while end < len(states) and states[end] is HandEvidenceState.UNKNOWN:
            end += 1
        if end - index > config.maximum_bridged_missing_samples:
            interrupting_unknown_starts.add(index)
        index = end
    candidates: list[ActivityCandidate] = []
    open_start: int | None = None
    open_start_observed = False
    unresolved_gap = False
    rest_start: int | None = None
    last_active: int | None = None
    preceding_rest_count = 0

    def append_candidate(end_index: int, end_observed: bool) -> None:
        nonlocal open_start, open_start_observed, unresolved_gap
        assert open_start is not None
        if end_index <= open_start:
            return
        references = sorted(
            {
                reference
                for references in evidence_references[open_start:end_index]
                for reference in references
            }
        )
        number = len(candidates) + 1
        candidates.append(
            ActivityCandidate(
                candidate_id=f"candidate-{number:04d}",
                participant_id=participant_id,
                start_sample=open_start,
                end_sample_exclusive=end_index,
                start_us=samples[open_start].start_us,
                end_us=samples[end_index - 1].end_us,
                start_observed=open_start_observed,
                end_observed=end_observed,
                proposal_basis=proposal_basis,
                evidence_references=tuple(references),
                has_unresolved_gap=unresolved_gap,
            )
        )
        open_start = None
        open_start_observed = False
        unresolved_gap = False

    for index, state in enumerate(states):
        if state is HandEvidenceState.NON_REST:
            if open_start is None:
                open_start = index
                open_start_observed = (
                    preceding_rest_count > 0
                    and samples[index].start_us - samples[index - preceding_rest_count].start_us
                    >= config.minimum_separating_rest_us
                )
            last_active = index
            rest_start = None
            preceding_rest_count = 0
            continue
        if state is HandEvidenceState.REST:
            if open_start is None:
                preceding_rest_count += 1
                continue
            if rest_start is None:
                rest_start = index
            assert last_active is not None
            if _supported_rest_closes(samples, rest_start, index, last_active, config):
                append_candidate(rest_start, True)
                preceding_rest_count = index - rest_start + 1
                rest_start = None
                last_active = None
            continue
        preceding_rest_count = 0
        rest_start = None
        if index in interrupting_unknown_starts:
            if open_start is not None:
                append_candidate(index, False)
            last_active = None
            continue
        if open_start is not None:
            unresolved_gap = True
    if open_start is not None:
        append_candidate(len(samples), False)
    return (candidates, bridged)
