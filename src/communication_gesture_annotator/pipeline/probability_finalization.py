# SPDX-License-Identifier: Apache-2.0
"""Local Phase3: finalize combined-score candidates without model interpretation."""

from __future__ import annotations
from dataclasses import replace
from typing import Any, Sequence
from ..config import AnalysisConfig
from ..domain import (
    ActivityCandidate,
    AnalysisSample,
    GUnit,
    JointActivityEvidence,
    JointSampleDecision,
    to_json_value,
)
from .motion_grounding import evidence_motion_record
from .probability_pass import candidate_diagnostics
from .output_guards import CANDIDATE_VERSION, LEGACY_CANDIDATE_VERSION, excluded_from_output

MERGE_POLICY_VERSION = "combined-probability-finalization-v0.3"
POLICY_VERSION = "probability-finalization-v0.4"
POLICY_VERSIONS = {MERGE_POLICY_VERSION, POLICY_VERSION}


def finalization_metadata(
    policy_version: str = POLICY_VERSION, *, merge_gap_ms: int = 300
) -> dict[str, Any]:
    if policy_version not in POLICY_VERSIONS:
        raise ValueError("unsupported probability finalization policy")
    if type(merge_gap_ms) is not int or merge_gap_ms < 0:
        raise ValueError("merge_gap_ms must be a nonnegative integer")
    return {
        "policy_version": policy_version,
        "mode": "local_selected_probability" if policy_version == POLICY_VERSION else "local_combined_probability",
        "model_request_count": 0,
        "deterministic_given_phase2": True,
        "requires_supported_boundaries": False,
        "automatic_hand_attribution": False,
        **(
            {
                "merge_gap_ms": merge_gap_ms,
                "merge_comparison": "gap_us <= merge_gap_ms * 1000",
                "merge_disabled": merge_gap_ms == 0,
            }
            if policy_version in {MERGE_POLICY_VERSION, POLICY_VERSION}
            else {}
        ),
    }


# Final membership comes from combined-score candidates, without another model call.
def finalize_probability_candidates(
    samples: Sequence[AnalysisSample],
    evidence: Sequence[JointActivityEvidence],
    candidates: Sequence[ActivityCandidate],
    config: AnalysisConfig,
    *,
    participant_id: str = "participant-1",
    policy_version: str = POLICY_VERSION,
    merge_gap_ms: int = 300,
) -> tuple[list[JointSampleDecision], list[GUnit]]:
    """Use only preserved combined scores, admission and the existing gap rules.

    Raw membership is the saved score below its saved rest threshold. Grouping
    applies sustained-rest/departure rules. Missing observed rest at a cut or
    selection edge does not invalidate the visible phrase fragment. Candidate
    boundary flags stay unchanged. v0.3 joins final positive intervals across
    gaps at most merge_gap_ms, retaining raw probabilities and gap overrides.
    v0.4 requires activity-supported continuations and preserves hard barriers.

    """
    finalization_metadata(policy_version, merge_gap_ms=merge_gap_ms)
    if len(samples) != len(evidence):
        raise ValueError("every sample requires probability evidence")
    diagnostics = candidate_diagnostics(samples, evidence, config, participant_id=participant_id,
        candidate_policy_version=CANDIDATE_VERSION if policy_version == POLICY_VERSION else LEGACY_CANDIDATE_VERSION)
    expected = diagnostics.candidates if diagnostics is not None else ()
    if to_json_value(candidates) != to_json_value(expected):
        raise ValueError("final candidates must reconstruct from the combined probability")
    bridged = set(diagnostics.bridged_sample_indices) if diagnostics else set()
    discarded = set(diagnostics.discarded_departure_indices) if diagnostics else set()
    candidate_by_index: dict[int, ActivityCandidate] = {}
    outcomes: dict[str, str | None] = {}
    for candidate in candidates:
        reason = None
        if candidate.has_unresolved_gap:
            reason = "missing_evidence"
        outcomes[candidate.candidate_id] = reason
        for index in range(candidate.start_sample, candidate.end_sample_exclusive):
            candidate_by_index[index] = candidate
    decisions: list[JointSampleDecision] = []
    for index, (sample, joint) in enumerate(zip(samples, evidence, strict=True)):
        record = evidence_motion_record(joint)
        q = record["non_g_unit_probability"] if record else None
        raw_positive = q is not None and q < record["candidate_policy"]["rest_threshold"]
        probabilities = (
            {"g_unit": 1 - q, "not_g_unit": q, "other": 0.0}
            if q is not None
            else {"g_unit": 0.0, "not_g_unit": 0.0, "other": 1.0}
        )
        candidate = candidate_by_index.get(index)
        positive = candidate is not None and outcomes[candidate.candidate_id] is None
        reason = (
            None
            if positive
            else outcomes[candidate.candidate_id]
            if candidate
            else "discarded_short_departure_between_supported_rest"
            if index in discarded
            else "excluded_or_barrier"
            if sample.source_gap_reason
            or joint.is_barrier
            or any((r.get("exclude_from_detection", False) for r in joint.resolution_records))
            else "probability_unavailable"
            if q is None
            else "outside_activity"
        )
        overrides: list[str] = []
        if positive and (not raw_positive):
            if index in bridged:
                overrides.append("bridged_short_missing_evidence")
            elif diagnostics is not None and diagnostics.reasons[index] == "no_sustained_rest_seed":
                overrides.append("no_sustained_rest_seed")
            else:
                overrides.append("filled_short_internal_rest")
        elif raw_positive and (not positive):
            overrides.append(str(reason))
        decisions.append(
            JointSampleDecision(
                sample=sample,
                participant_id=participant_id,
                joint_evidence=joint,
                is_g_unit=positive,
                articulator=None,
                class_probabilities=probabilities,
                evidence_quality=joint.evidence_quality,
                evidence_references=joint.evidence_references
                + joint.pose_evidence_references
                + joint.rgb_evidence_references,
                candidate_id=candidate.candidate_id if candidate else None,
                other_reason=reason,
                decision_basis=policy_version,
                raw_decision="g_unit" if raw_positive else "not_g_unit",
                policy_overrides=tuple(overrides),
            )
        )
    units = [
        GUnit(
            g_unit_id=f"g-unit-{number:04d}",
            candidate_id=c.candidate_id,
            participant_id=c.participant_id,
            start_us=c.start_us,
            end_us=c.end_us,
            start_sample=c.start_sample,
            end_sample_exclusive=c.end_sample_exclusive,
            articulator=None,
            decision_basis=policy_version,
        )
        for number, c in enumerate((c for c in candidates if outcomes[c.candidate_id] is None), 1)
    ]
    if policy_version in {MERGE_POLICY_VERSION, POLICY_VERSION} and merge_gap_ms:
        merged: list[GUnit] = []
        for unit in units:
            if (
                merged
                and unit.participant_id == merged[-1].participant_id
                and (0 <= unit.start_us - merged[-1].end_us <= merge_gap_ms * 1000)
                and (policy_version != POLICY_VERSION or all(
                    not excluded_from_output(samples[i], evidence[i])
                    and evidence_motion_record(evidence[i]) is not None
                    for i in range(merged[-1].end_sample_exclusive, unit.start_sample)
                ))
            ):
                previous = merged[-1]
                for index in range(previous.end_sample_exclusive, unit.start_sample):
                    decision = decisions[index]
                    decisions[index] = replace(
                        decision,
                        is_g_unit=True,
                        other_reason=None,
                        candidate_id=previous.candidate_id,
                        policy_overrides=decision.policy_overrides
                        + ("finalization_merged_short_gap",),
                    )
                merged[-1] = replace(
                    previous, end_us=unit.end_us, end_sample_exclusive=unit.end_sample_exclusive
                )
            else:
                merged.append(replace(unit, g_unit_id=f"g-unit-{len(merged) + 1:04d}"))
        units = merged
    return (decisions, units)
