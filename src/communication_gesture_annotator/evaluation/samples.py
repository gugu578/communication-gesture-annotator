# SPDX-License-Identifier: Apache-2.0
"""Reference parsing and canonical sample metrics on assessed reference time."""

from __future__ import annotations
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from .reference_scope import ReferenceScope


class EvaluationError(ValueError):
    """Invalid reference, decision, or evaluation input."""


class SampleDecision(Protocol):
    sample_index: int
    start_us: int
    center_us: int
    end_us: int
    final_is_g_unit: bool | None
    joint: Mapping[str, object] | None
    phase1_excluded: bool


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


DEFAULT_REFERENCE_TIER_ID = "Manual_GUnit"


EVALUATION_INTERVAL_US = 100_000


EVALUATION_OUTCOMES = ("tp", "fn", "fp", "tn")


EVALUATION_OUTCOME_NAMES = {
    "tp": "true_positive",
    "fn": "false_negative",
    "fp": "false_positive",
    "tn": "true_negative",
}


EVALUATION_CLASSIFICATION_RULE = (
    "reference-positive-when-sample-center-is-inside-aligned-half-open-tier-interval"
)


@dataclass(frozen=True)
class ReferenceGUnit:
    """One alignable interval from the selected reference G-unit tier."""

    annotation_id: str
    source_start_us: int
    source_end_us: int
    start_us: int
    end_us: int
    annotation_value: str
    source_tier_id: str | None = None


@dataclass(frozen=True)
class EvaluationInterval:
    """One canonical 10 Hz decision compared with the aligned reference tier."""

    sample_index: int
    start_us: int
    end_us: int
    predicted: bool
    reference: bool
    outcome: str


@dataclass(frozen=True)
class TemporalEvaluation:
    """Sample-level confusion metrics on the canonical 100 ms analysis grid."""

    intervals: tuple[EvaluationInterval, ...]
    counts: Mapping[str, int]
    durations_us: Mapping[str, int]
    percentages: Mapping[str, float]
    precision: float | None
    recall: float | None
    miss_rate: float | None
    false_positive_rate: float | None
    f1: float | None
    accuracy: float | None

    @property
    def duration_us(self) -> int:
        return sum(self.durations_us.values())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, local_name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child.tag) == local_name]


def _decoded_xml_for_declaration_scan(payload: bytes) -> str:
    """Decode supported XML Unicode encodings before scanning declarations."""

    if payload.startswith((b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")):
        encoding = "utf-32"
    elif payload.startswith((b"\xfe\xff", b"\xff\xfe")):
        encoding = "utf-16"
    elif payload.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    elif payload.startswith(b"\x00\x00\x00<"):
        encoding = "utf-32-be"
    elif payload.startswith(b"<\x00\x00\x00"):
        encoding = "utf-32-le"
    elif payload.startswith(b"\x00<"):
        encoding = "utf-16-be"
    elif payload.startswith(b"<\x00"):
        encoding = "utf-16-le"
    else:
        encoding = "utf-8"
    try:
        return payload.decode(encoding)
    except UnicodeError as exc:
        raise EvaluationError(
            "reference EAF must use UTF-8, UTF-16, or UTF-32 XML encoding"
        ) from exc


def read_reference_g_units(
    eaf_path: str | Path,
    *,
    tier_id: str = DEFAULT_REFERENCE_TIER_ID,
    alignment_offset_us: int = 0,
) -> tuple[Path, list[ReferenceGUnit]]:
    """Read every interval from one configured alignable reference G-unit tier."""

    if not tier_id.strip():
        raise EvaluationError("reference EAF tier ID cannot be blank")
    if type(alignment_offset_us) is not int:
        raise EvaluationError(
            "reference EAF alignment offset must be an integer number of microseconds"
        )
    source = Path(eaf_path)
    try:
        if not source.is_file():
            raise EvaluationError("reference EAF path is not a file")
        payload = source.read_bytes()
    except OSError as exc:
        raise EvaluationError("reference EAF cannot be read") from exc
    decoded_xml = _decoded_xml_for_declaration_scan(payload)
    if "<!DOCTYPE" in decoded_xml.upper() or "<!ENTITY" in decoded_xml.upper():
        raise EvaluationError("reference EAF must not contain a DTD or entity declaration")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise EvaluationError("reference EAF is not readable XML") from exc
    if _local_name(root.tag) != "ANNOTATION_DOCUMENT":
        raise EvaluationError("reference EAF root must be ANNOTATION_DOCUMENT")
    if root.get("FORMAT") != "3.0" or root.get("VERSION") != "3.0":
        raise EvaluationError("only reference EAF FORMAT/VERSION 3.0 is supported")

    headers = _children(root, "HEADER")
    if len(headers) != 1:
        raise EvaluationError("reference EAF must contain exactly one HEADER")
    if headers[0].get("TIME_UNITS", "milliseconds") != "milliseconds":
        raise EvaluationError("only millisecond reference EAF time units are supported")

    time_orders = _children(root, "TIME_ORDER")
    if len(time_orders) != 1:
        raise EvaluationError("reference EAF must contain exactly one TIME_ORDER")
    slots: dict[str, int] = {}
    seen_slot_ids: set[str] = set()
    for slot in _children(time_orders[0], "TIME_SLOT"):
        identifier = slot.get("TIME_SLOT_ID")
        if not identifier:
            raise EvaluationError("reference EAF contains a time slot without an ID")
        if identifier in seen_slot_ids:
            raise EvaluationError("reference EAF contains a duplicate time slot ID")
        seen_slot_ids.add(identifier)
        raw_value = slot.get("TIME_VALUE")
        if raw_value is None:
            continue
        try:
            time_ms = int(raw_value)
        except ValueError as exc:
            raise EvaluationError("reference EAF time values must be integers") from exc
        if time_ms < 0:
            raise EvaluationError("reference EAF time values must be non-negative")
        slots[identifier] = time_ms

    tiers = [tier for tier in _children(root, "TIER") if tier.get("TIER_ID") == tier_id]
    if len(tiers) != 1:
        detail = "does not exist" if not tiers else "is duplicated"
        raise EvaluationError(f"reference EAF tier {tier_id!r} {detail}")

    units: list[ReferenceGUnit] = []
    annotation_ids: set[str] = set()
    for wrapper in list(tiers[0]):
        if _local_name(wrapper.tag) != "ANNOTATION":
            raise EvaluationError(f"reference EAF tier {tier_id!r} contains an unsupported child")
        annotations = list(wrapper)
        if len(annotations) != 1:
            raise EvaluationError(
                f"reference EAF tier {tier_id!r} has a malformed ANNOTATION wrapper"
            )
        annotation = annotations[0]
        kind = _local_name(annotation.tag)
        if kind == "REF_ANNOTATION":
            raise EvaluationError(f"reference EAF tier {tier_id!r} contains a REF_ANNOTATION")
        if kind != "ALIGNABLE_ANNOTATION":
            raise EvaluationError(
                f"reference EAF tier {tier_id!r} contains an unsupported annotation type"
            )
        annotation_id = annotation.get("ANNOTATION_ID")
        if not annotation_id or annotation_id in annotation_ids:
            raise EvaluationError(
                f"reference EAF tier {tier_id!r} has a missing or duplicate annotation ID"
            )
        annotation_ids.add(annotation_id)
        start_ref = annotation.get("TIME_SLOT_REF1")
        end_ref = annotation.get("TIME_SLOT_REF2")
        if start_ref not in slots or end_ref not in slots:
            raise EvaluationError(f"reference EAF tier {tier_id!r} has an unresolved time slot")
        value_nodes = _children(annotation, "ANNOTATION_VALUE")
        if len(value_nodes) > 1:
            raise EvaluationError(f"reference EAF annotation {annotation_id!r} has multiple values")
        value = "" if not value_nodes else "".join(value_nodes[0].itertext()).strip()
        source_start_us = slots[start_ref] * 1_000
        source_end_us = slots[end_ref] * 1_000
        start_us = source_start_us + alignment_offset_us
        end_us = source_end_us + alignment_offset_us
        if source_end_us <= source_start_us or start_us < 0:
            raise EvaluationError(f"reference EAF tier {tier_id!r} has an invalid aligned interval")
        units.append(
            ReferenceGUnit(
                annotation_id=annotation_id,
                source_start_us=source_start_us,
                source_end_us=source_end_us,
                start_us=start_us,
                end_us=end_us,
                annotation_value=value,
                source_tier_id=tier_id,
            )
        )

    units.sort(key=lambda item: (item.start_us, item.end_us, item.annotation_id))
    previous_end_us = -1
    for unit in units:
        if unit.start_us < previous_end_us:
            raise EvaluationError(f"reference EAF tier {tier_id!r} contains overlapping intervals")
        previous_end_us = unit.end_us
    return source, units


def evaluate_final_decisions(
    samples: Sequence[SampleDecision],
    reference_units: Sequence[ReferenceGUnit],
    *, assessment_scope: ReferenceScope | None = None,
) -> TemporalEvaluation:
    """Compare canonical final decisions with aligned EAF intervals at 10 Hz.

    This deliberately matches the project's sample-level evaluator: a reference
    G-unit owns a sample exactly when the sample center lies inside its aligned
    half-open interval.  The final prediction is ``samples[].is_g_unit``; raw
    model classes and activity probabilities are not substituted for it.
    """

    if not samples:
        raise EvaluationError("at least one sample is required")
    if any(sample.final_is_g_unit is None for sample in samples):
        raise EvaluationError("sample-level evaluation requires is_g_unit for every sample")
    samples = [s for s in samples if not phase1_excluded(s)
               and (assessment_scope is None or assessment_scope.contains(s.center_us))]
    intervals: list[EvaluationInterval] = []
    counts = {outcome: 0 for outcome in EVALUATION_OUTCOMES}
    durations_us = {outcome: 0 for outcome in EVALUATION_OUTCOMES}
    reference_index = 0
    ordered_units = tuple(reference_units)
    for sample in samples:
        while (
            reference_index < len(ordered_units)
            and ordered_units[reference_index].end_us <= sample.center_us
        ):
            reference_index += 1
        reference = (
            reference_index < len(ordered_units)
            and ordered_units[reference_index].start_us
            <= sample.center_us
            < ordered_units[reference_index].end_us
        )
        predicted = bool(sample.final_is_g_unit)
        if predicted and reference:
            outcome = "tp"
        elif reference:
            outcome = "fn"
        elif predicted:
            outcome = "fp"
        else:
            outcome = "tn"
        intervals.append(
            EvaluationInterval(
                sample_index=sample.sample_index,
                start_us=sample.start_us,
                end_us=sample.end_us,
                predicted=predicted,
                reference=reference,
                outcome=outcome,
            )
        )
        counts[outcome] += 1
        durations_us[outcome] += sample.end_us - sample.start_us

    tp = counts["tp"]
    fn = counts["fn"]
    fp = counts["fp"]
    tn = counts["tn"]
    total_count = len(intervals)
    total_duration_us = sum(durations_us.values())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    miss_rate = _safe_ratio(fn, tp + fn)
    false_positive_rate = _safe_ratio(fp, fp + tn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = _safe_ratio(tp + tn, total_count)
    percentages = {
        outcome: (100.0 * durations_us[outcome] / total_duration_us if total_duration_us else 0.0)
        for outcome in EVALUATION_OUTCOMES
    }
    return TemporalEvaluation(
        intervals=tuple(intervals),
        counts=counts,
        durations_us=durations_us,
        percentages=percentages,
        precision=precision,
        recall=recall,
        miss_rate=miss_rate,
        false_positive_rate=false_positive_rate,
        f1=f1,
        accuracy=accuracy,
    )


def time_selection_excluded(sample: SampleDecision) -> bool:
    """A test interval is processing scope, not a negative model decision."""
    return sample.joint is not None and any(
        isinstance(record, Mapping)
        and record.get("policy_version") == "phase2-time-selection-v0.1"
        and record.get("exclude_from_detection") is True
        for record in sample.joint.get("resolution_records", ())
    )


def phase1_excluded(sample: SampleDecision) -> bool:
    """Only stored upstream exclusion decisions exclude model evidence."""
    if sample.phase1_excluded or time_selection_excluded(sample):
        return True
    if sample.joint is None:
        return False
    for record in sample.joint.get("resolution_records", ()):
        if isinstance(record, Mapping) and "exclude_from_detection" in record:
            value = record["exclude_from_detection"]
            if type(value) is not bool:
                raise EvaluationError("exclude_from_detection must be Boolean")
            if "sample_index" in record and record["sample_index"] != sample.sample_index:
                raise EvaluationError("exclusion record sample index mismatch")
            return value
    return False


def _evaluation_payload(
    evaluation: TemporalEvaluation | None,
    *,
    reference_available: bool,
    final_decision_available: bool,
) -> dict[str, object]:
    if evaluation is None:
        reason = (
            "reference_eaf_not_supplied"
            if not reference_available
            else "authoritative_final_decision_unavailable"
        )
        return {
            "available": False,
            "interval_us": EVALUATION_INTERVAL_US,
            "classification_rule": EVALUATION_CLASSIFICATION_RULE,
            "reason": reason,
        }
    return {
        "available": True,
        "interval_us": EVALUATION_INTERVAL_US,
        "classification_rule": EVALUATION_CLASSIFICATION_RULE,
        "exclusion_policy": "phase1-and-processing-scope-only-v1",
        "metric_basis": "canonical-analysis-sample-count",
        "duration_basis": "actual-canonical-sample-duration-us",
        "outcome_percentage_denominator": "evaluated-duration-us",
        "interval_count": len(evaluation.intervals),
        "duration_us": evaluation.duration_us,
        "confusion": {
            EVALUATION_OUTCOME_NAMES[outcome]: {
                "count": evaluation.counts[outcome],
                "duration_us": evaluation.durations_us[outcome],
                "percentage_of_evaluated_time": evaluation.percentages[outcome],
            }
            for outcome in EVALUATION_OUTCOMES
        },
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        "miss_rate": evaluation.miss_rate,
        "false_positive_rate": evaluation.false_positive_rate,
        "f1": evaluation.f1,
        "accuracy": evaluation.accuracy,
        "reference_available": reference_available,
        "final_decision_available": final_decision_available,
    }
