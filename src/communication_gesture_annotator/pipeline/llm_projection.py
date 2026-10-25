# SPDX-License-Identifier: Apache-2.0
"""Shared LLM-only output projection for evaluation and EAF; no inference."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any

from ..config import AnalysisConfig
from ..domain import AnalysisSample, JointActivityEvidence, JointActivityState, to_json_value
from .output_guards import CANDIDATE_VERSION
from .probability_finalization import POLICY_VERSION, finalize_probability_candidates
from .probability_pass import VERSION, candidate_diagnostics

PROJECTION_VERSION = "llm-only-output-projection-v2"


@dataclass(frozen=True)
class LLMProjection:
    membership: tuple[bool, ...]
    units: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def project_llm_output(document: Mapping[str, Any]) -> LLMProjection | None:
    """Group saved LLM estimates with current rules, without changing the source.

    Non-probability legacy formats return None and retain their existing semantics.
    A probability run with missing/invalid raw estimates fails rather than using
    a fused score as if it were an LLM answer. Reference annotations never enter
    this calculation; masking for assessment happens afterwards.
    """
    rows = document["samples"]
    probability_run = document.get("provenance", {}).get("policy_version") in {
        "combined-probability-finalization-v0.3", POLICY_VERSION,
    } or any(
        record.get("policy") == VERSION
        for row in rows for record in (row.get("joint_evidence") or {}).get("resolution_records", ())
    )
    if not probability_run:
        return None
    samples = [AnalysisSample(**row["sample"]) for row in rows]
    config = AnalysisConfig(**document["config"])
    participant_ids = {row["participant_id"] for row in rows}
    if len(participant_ids) != 1:
        raise ValueError("LLM-only output requires one selected participant")
    participant = participant_ids.pop()
    evidence: list[JointActivityEvidence] = []
    policies: list[Mapping[str, Any]] = []
    for row in rows:
        value = dict(row["joint_evidence"])
        records = []
        seen = 0
        for record in value.get("resolution_records", ()):
            if record.get("policy") == VERSION and "motion_grounding" in record:
                seen += 1
                mg = record["motion_grounding"]
                raw = mg.get("llm_non_g_unit_probability")
                if type(raw) not in (float, int) or not isfinite(raw) or not 0 <= raw <= 1:
                    raise ValueError("LLM-only output requires a valid saved raw LLM estimate for every scored sample")
                policies.append(mg["candidate_policy"])
                records.append(dict(record, motion_grounding=dict(mg,
                    non_g_unit_probability=raw, score_source="llm")))
            else:
                records.append(record)
        if seen > 1:
            raise ValueError("LLM-only output found ambiguous saved probability records")
        # Check each resolved sample before candidate_diagnostics can return
        # early for an empty policy list. Unknown/excluded samples need no score.
        if seen == 0 and value["state"] != JointActivityState.UNKNOWN.value:
            raise ValueError("LLM-only output requires a saved raw LLM estimate for every scored sample")
        value["resolution_records"] = tuple(records)
        value["state"] = JointActivityState(value["state"])
        if value.get("resolved_state") is not None:
            value["resolved_state"] = JointActivityState(value["resolved_state"])
        for key in ("evidence_references", "pose_evidence_references", "rgb_evidence_references"):
            if key in value:
                value[key] = tuple(value[key])
        evidence.append(JointActivityEvidence(**value))
    diagnostics = candidate_diagnostics(samples, evidence, config, participant_id=participant,
        candidate_policy_version=CANDIDATE_VERSION)
    gap = document.get("provenance", {}).get("phase3_finalization", {}).get("merge_gap_ms", 300)
    decisions, units = finalize_probability_candidates(samples, evidence,
        diagnostics.candidates if diagnostics else (), config, participant_id=participant,
        policy_version=POLICY_VERSION, merge_gap_ms=gap)
    membership = tuple(d.is_g_unit for d in decisions)
    unit_rows = tuple(to_json_value(unit) for unit in units)
    identity = {"samples": [(s.start_us, s.end_us, p) for s, p in zip(samples, membership, strict=True)],
                "units": unit_rows}
    metadata = {
        "policy_version": PROJECTION_VERSION, "score_source": "llm", "model_calls": 0,
        "candidate_policy_version": CANDIDATE_VERSION, "finalization_policy_version": POLICY_VERSION,
        "candidate_policy": dict(policies[0]) if policies else None,
        "minimum_separating_rest_us": config.minimum_separating_rest_us,
        "maximum_bridged_missing_samples": config.maximum_bridged_missing_samples,
        "merge_gap_ms": gap,
        "changed_sample_count": sum(p != row["is_g_unit"] for p, row in zip(membership, rows, strict=True)),
        "output_sha256": sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return LLMProjection(membership, unit_rows, metadata)


def decision_basis(projection: LLMProjection | None) -> dict[str, Any]:
    """Never label a historical non-probability format as LLM-only."""
    return projection.metadata if projection else {"score_source": "saved_legacy_decisions", "model_calls": 0}
