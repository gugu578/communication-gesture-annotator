# SPDX-License-Identifier: Apache-2.0
"""Core data structures for the fixed-rate research pipeline."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Mapping
from ..config import AnalysisConfig


class HandEvidenceState(StrEnum):
    """Raw, context-conditioned per-hand evidence from Step 2."""

    REST = "rest"
    NON_REST = "non_rest"
    UNKNOWN = "unknown"


class HandLabel(StrEnum):
    """The only labels permitted on a positive G-unit."""

    LEFT = "LH"
    RIGHT = "RH"
    BOTH = "BH"


class JointActivityState(StrEnum):
    """Manual activity without requiring a participating-hand label."""

    PASSIVE = "passive"
    ACTIVE = "active"
    UNKNOWN = "unknown"


def _check_probability(value: float, name: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")


@dataclass(frozen=True, slots=True)
class AnalysisSample:
    """One half-open cell on the 10 Hz analysis grid."""

    index: int
    start_us: int
    end_us: int
    center_us: int
    source_references: tuple[str, ...] = ()
    source_gap_reason: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0 or self.start_us < 0:
            raise ValueError("sample index and start_us must be non-negative")
        if not self.start_us < self.end_us:
            raise ValueError("sample intervals must have positive duration")
        if not self.start_us <= self.center_us < self.end_us:
            raise ValueError("sample center_us must lie inside its interval")


@dataclass(frozen=True, slots=True)
class JointActivityEvidence:
    """Preserved joint observation and its independently auditable resolution.

    A known active state makes no claim that both hands participated. Source
    support retains native-time intervals; the enclosing decision supplies the
    separate canonical analysis interval. Confirmed discontinuities are barriers
    even when shorter than one canonical sample.
    """

    state: JointActivityState
    state_probabilities: Mapping[str, float]
    evidence_quality: float
    evidence_references: tuple[str, ...] = ()
    reason: str | None = None
    resolved_state: JointActivityState | None = None
    uncertainty_causes: tuple[Mapping[str, Any], ...] = ()
    source_support_intervals: tuple[Mapping[str, Any], ...] = ()
    pose_evidence_references: tuple[str, ...] = ()
    rgb_evidence_references: tuple[str, ...] = ()
    resolution_records: tuple[Mapping[str, Any], ...] = ()
    continuity_barrier: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, JointActivityState):
            raise ValueError("joint state must be a JointActivityState")
        if self.resolved_state is not None and (
            not isinstance(self.resolved_state, JointActivityState)
        ):
            raise ValueError("resolved joint state must be a JointActivityState")
        if set(self.state_probabilities) != {"passive", "active", "unknown"}:
            raise ValueError("joint probabilities must declare passive, active, unknown")
        for name, probability in self.state_probabilities.items():
            _check_probability(probability, f"joint state_probabilities[{name!r}]")
        if abs(sum(self.state_probabilities.values()) - 1.0) > 1e-06:
            raise ValueError("joint state_probabilities must sum to one")
        _check_probability(self.evidence_quality, "joint evidence_quality")
        if self.state is JointActivityState.UNKNOWN and (not self.reason):
            raise ValueError("unknown joint evidence must retain a reason")
        if self.resolved_state not in (None, self.state) and (not self.resolution_records):
            raise ValueError("changed joint state requires a resolution record")
        if type(self.continuity_barrier) is not bool:
            raise ValueError("continuity_barrier must be Boolean")
        for cause in self.uncertainty_causes:
            if not isinstance(cause, Mapping) or not isinstance(cause.get("cause"), str):
                raise ValueError("joint uncertainty cause requires its reported cause")
            if cause.get("status") not in {"unverified", "confirmed", "rejected"}:
                raise ValueError("joint uncertainty causes require a verification status")
        for interval in self.source_support_intervals:
            if (
                not isinstance(interval, Mapping)
                or type(interval.get("start_us")) is not int
                or type(interval.get("end_us")) is not int
                or (not 0 <= interval["start_us"] < interval["end_us"])
            ):
                raise ValueError("joint source support must use positive half-open intervals")
        for references in (
            self.evidence_references,
            self.pose_evidence_references,
            self.rgb_evidence_references,
        ):
            if any((not isinstance(item, str) or not item for item in references)):
                raise ValueError("joint evidence references must be nonempty strings")

    @property
    def effective_state(self) -> JointActivityState:
        if self.is_barrier:
            return JointActivityState.UNKNOWN
        return self.resolved_state or self.state

    @property
    def is_barrier(self) -> bool:
        return self.continuity_barrier or any(
            (
                cause["status"] == "confirmed"
                and cause["cause"]
                in {
                    "camera_cut",
                    "source_gap",
                    "decoder_gap",
                    "identity_switch",
                    "identity_change",
                    "target_absent",
                    "other_scene",
                }
                for cause in self.uncertainty_causes
            )
        )


@dataclass(frozen=True, slots=True)
class ActivityCandidate:
    """A broad, automatically generated Step 2 activity envelope."""

    candidate_id: str
    participant_id: str
    start_sample: int
    end_sample_exclusive: int
    start_us: int
    end_us: int
    start_observed: bool
    end_observed: bool
    proposal_basis: str = "contextual_non_rest_evidence"
    evidence_references: tuple[str, ...] = ()
    has_unresolved_gap: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.participant_id:
            raise ValueError("candidate and participant identifiers are required")
        if self.start_sample < 0 or self.end_sample_exclusive <= self.start_sample:
            raise ValueError("candidate sample range is invalid")
        if self.start_us < 0 or self.end_us <= self.start_us:
            raise ValueError("candidate time range is invalid")

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us

    @property
    def complete(self) -> bool:
        return self.start_observed and self.end_observed and (not self.has_unresolved_gap)


@dataclass(frozen=True, slots=True)
class ProcessingWindow:
    """A Step 3 request window whose edges never imply annotation edges."""

    window_id: str
    candidate_id: str
    start_us: int
    end_us: int
    contains_complete_candidate: bool
    may_finalize: bool
    requires_stitching: bool = False
    sequence_index: int = 0

    def __post_init__(self) -> None:
        if self.start_us < 0 or self.end_us <= self.start_us:
            raise ValueError("processing window range is invalid")
        if self.may_finalize and (not self.contains_complete_candidate):
            raise ValueError("a partial window cannot finalize a candidate")


@dataclass(frozen=True, slots=True)
class JointSampleDecision:
    """Complete Boolean detection, independent of optional hand attribution."""

    sample: AnalysisSample
    participant_id: str
    joint_evidence: JointActivityEvidence
    is_g_unit: bool
    articulator: HandLabel | None
    class_probabilities: Mapping[str, float]
    evidence_quality: float
    evidence_references: tuple[str, ...] = ()
    candidate_id: str | None = None
    other_reason: str | None = None
    decision_basis: str = "contextual_joint_candidate_decision"
    window_ids: tuple[str, ...] = ()
    raw_decision: str = "not_g_unit"
    policy_overrides: tuple[str, ...] = ()
    attribution_status: str = "not_included"

    def __post_init__(self) -> None:
        if not self.participant_id or type(self.is_g_unit) is not bool:
            raise ValueError("joint decision requires participant and Boolean membership")
        if self.attribution_status not in {"not_included", "resolved", "unresolved", "failed"}:
            raise ValueError("joint decision attribution_status is invalid")
        if self.articulator is not None and (
            not self.is_g_unit
            or self.attribution_status != "resolved"
            or self.articulator not in set(HandLabel)
        ):
            raise ValueError("only resolved positive samples can have an articulator")
        if self.attribution_status == "resolved" and self.articulator is None:
            raise ValueError("resolved attribution requires an articulator")
        if not self.is_g_unit and (not self.other_reason):
            raise ValueError("a negative joint decision must retain other_reason")
        if self.is_g_unit and self.other_reason is not None:
            raise ValueError("a positive joint decision cannot have other_reason")
        _check_probability(self.evidence_quality, "evidence_quality")
        if self.raw_decision not in {"g_unit", "not_g_unit"}:
            raise ValueError("joint raw_decision must be binary")
        if set(self.class_probabilities) != {"g_unit", "not_g_unit", "other"}:
            raise ValueError("joint sample probabilities must declare every class")
        for name, probability in self.class_probabilities.items():
            _check_probability(probability, f"class_probabilities[{name!r}]")
        if abs(sum(self.class_probabilities.values()) - 1.0) > 1e-06:
            raise ValueError("joint sample class_probabilities must sum to one")


@dataclass(frozen=True, slots=True)
class GUnit:
    """One complete G-unit on the canonical microsecond timeline."""

    g_unit_id: str
    candidate_id: str
    participant_id: str
    start_us: int
    end_us: int
    start_sample: int
    end_sample_exclusive: int
    articulator: HandLabel | None
    decision_basis: str = "contextual_binary_policy_v0.1"

    def __post_init__(self) -> None:
        if self.start_us < 0 or self.end_us <= self.start_us:
            raise ValueError("G-unit time range is invalid")
        if self.start_sample < 0 or self.end_sample_exclusive <= self.start_sample:
            raise ValueError("G-unit sample range is invalid")


@dataclass(frozen=True, slots=True)
class ParticipantProfile:
    """Label-free Step 1 context for the selected visible participant."""

    participant_id: str
    selection_policy: str
    modalities_supplied: tuple[str, ...]
    modalities_used: tuple[str, ...]
    summary: Mapping[str, Any]
    evidence_references: tuple[str, ...] = ()
    model_output: Mapping[str, Any] | None = None
    contract_normalized_output: Mapping[str, Any] | None = None
    contract_normalization: Mapping[str, Any] | None = None
    subject_id_canonicalization: Mapping[str, Any] | None = None


@dataclass(slots=True)
class AnalysisResult:
    """Canonical, comprehensive JSON result for one autonomous run."""

    schema_version: str
    run_id: str
    created_at: str
    capability: Mapping[str, Any]
    source: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    config: AnalysisConfig
    participant_profiles: list[ParticipantProfile]
    prompt_envelopes: list[Mapping[str, Any]]
    activity_candidates: list[ActivityCandidate]
    processing_windows: list[ProcessingWindow]
    window_responses: list[Mapping[str, Any]]
    candidate_assessments: list[Mapping[str, Any]]
    samples: list[JointSampleDecision]
    g_units: list[GUnit]
    warnings: list[str] = field(default_factory=list)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    evaluation: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if len(self.participant_profiles) != 1:
            raise ValueError("Milestone 1 requires exactly one selected participant")
        expected_index = 0
        previous_end = 0
        for decision in self.samples:
            if not isinstance(decision, JointSampleDecision):
                raise ValueError("joint analyses require native joint sample evidence")
            sample = decision.sample
            if sample.index != expected_index:
                raise ValueError("sample indices must be contiguous from zero")
            if expected_index and sample.start_us != previous_end:
                raise ValueError("analysis sample intervals must be contiguous")
            if sample.end_us - sample.start_us > self.config.sample_period_us:
                raise ValueError("analysis sample is wider than the 10 Hz grid")
            expected_index += 1
            previous_end = sample.end_us
        candidate_ids = {candidate.candidate_id for candidate in self.activity_candidates}
        if len(candidate_ids) != len(self.activity_candidates):
            raise ValueError("candidate identifiers must be unique")
        g_unit_ids: set[str] = set()
        for unit in self.g_units:
            if unit.g_unit_id in g_unit_ids:
                raise ValueError("G-unit identifiers must be unique")
            g_unit_ids.add(unit.g_unit_id)
            if unit.candidate_id not in candidate_ids:
                raise ValueError("every G-unit must refer to a known candidate")
            matching = [
                sample
                for sample in self.samples[unit.start_sample : unit.end_sample_exclusive]
                if sample.candidate_id == unit.candidate_id
            ]
            if not matching or not all((sample.is_g_unit for sample in matching)):
                raise ValueError("G-unit intervals must agree with the binary sample raster")


def to_json_value(value: Any) -> Any:
    """Convert domain objects to stable, JSON-compatible values."""
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {key: to_json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json_value(item) for item in value]
    return value
