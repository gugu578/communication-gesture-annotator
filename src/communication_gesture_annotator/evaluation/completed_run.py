# SPDX-License-Identifier: Apache-2.0
"""Completed-run statistics with no inference or media dependency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4
from .reference_scope import DEFAULT_VISIBILITY_TIER, read_reference_scope

from ..export.completed_run import (
    check_outputs,
    eaf_destinations,
    export_loaded_run,
    load_completed_run,
    protected_run_paths,
)
from ..export.json_export import write_json
from ..pipeline.llm_projection import project_llm_output, decision_basis
from .samples import (
    EVALUATION_INTERVAL_US,
    _evaluation_payload,
    evaluate_final_decisions,
    phase1_excluded,
    read_reference_g_units,
    time_selection_excluded,
)


@dataclass(frozen=True)
class DecisionSample:
    sample_index: int
    start_us: int
    center_us: int
    end_us: int
    final_is_g_unit: bool
    joint: Mapping[str, object] | None
    phase1_excluded: bool = False


def decision_samples(document: Mapping[str, Any], *, membership: Sequence[bool] | None = None) -> list[DecisionSample]:
    """Read a Boolean raster, optionally using the shared LLM-only projection."""
    if membership is not None and len(membership) != len(document["samples"]):
        raise ValueError("one projected decision is required per sample")
    result = []
    for row in document["samples"]:
        sample = row["sample"]
        if type(row["is_g_unit"]) is not bool:
            raise ValueError("sample-level evaluation requires Boolean final decisions")
        joint = row.get("joint_evidence")
        if joint is not None and not isinstance(joint, Mapping):
            raise ValueError("joint evidence must be an object or null")
        result.append(
            DecisionSample(
                sample["index"],
                sample["start_us"],
                sample["center_us"],
                sample["end_us"],
                membership[len(result)] if membership is not None else row["is_g_unit"],
                joint,
            )
        )
    # Canonical grid cells are 100 ms, with a possibly shorter last cell.
    # Do not silently relabel other grids as 100 ms or invent interpolated labels.
    if not result or any(
        sample.start_us != index * EVALUATION_INTERVAL_US
        or sample.center_us != (sample.start_us + sample.end_us) // 2
        or not 0 < sample.end_us - sample.start_us <= EVALUATION_INTERVAL_US
        or (index < len(result) - 1 and sample.end_us - sample.start_us != EVALUATION_INTERVAL_US)
        for index, sample in enumerate(result)
    ):
        raise ValueError("completed-run evaluation requires the canonical 100 ms grid")
    return result


def phase1_enabled(
    document: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> bool | None:
    """Read saved settings; older enabled snapshots omit the default-true flag."""
    for configuration in (
        document.get("provenance", {}).get("research_configuration", {}),
        (manifest or {}).get("configuration", {}),
    ):
        if isinstance(configuration, Mapping):
            inference = configuration.get("inference")
            if isinstance(inference, Mapping):
                value = inference.get("phase1_enabled", True)
                if type(value) is bool:
                    return value
    return None


def render_text_report(report: Mapping[str, Any]) -> str:
    """Render summary statistics and exclusion counts."""
    metrics, scope = report["evaluation"], report["scope"]
    reference = report["reference"]
    phase1 = {True: "enabled", False: "disabled", None: "not recorded"}[scope["phase1_enabled"]]
    lines = [
        "Communication Gesture Annotator - completed-run statistics",
        f"Run: {report['run_id']}",
        f"Analysis: {report['analysis']['path']}",
        f"Reference: {reference['path']}",
        f"Reference tier: {reference['tier_id']}; positive value: {reference['positive_value']}",
        f"Reference alignment offset: {reference['alignment_offset_us']} us",
        "Grid: 100 ms; final Boolean decisions versus reference membership at sample centers.",
        f"Decision basis: {report['decision_basis']['score_source']}; local grouping; no model calls.",
        f"Grouping policies: {report['decision_basis'].get('candidate_policy_version', 'saved legacy')}; {report['decision_basis'].get('finalization_policy_version', 'saved legacy')}.",
        f"Phase1: {phase1}. Unlabelled reference visibility intervals are excluded; labelled visibility values do not mask detection.",
        f"Total: {scope['total_sample_count']} samples ({scope['total_duration_us'] / 1e6:.6f} s)",
        f"Evaluated: {metrics['interval_count']} samples ({metrics['duration_us'] / 1e6:.6f} s)",
        f"Excluded: {scope['excluded_sample_count']} samples ({scope['excluded_duration_us'] / 1e6:.6f} s)",
    ]
    for reason, values in scope["excluded_by_reason"].items():
        lines.append(f"  {reason}: {values['count']} samples ({values['duration_us'] / 1e6:.6f} s)")
    lines.extend(f"Warning: {warning}" for warning in report["warnings"])
    lines.extend(
        [
            "",
            "Outcome                       Samples     Duration (s)     % evaluated time",
        ]
    )
    for name, label in (
        ("true_positive", "True positives (hits)"),
        ("false_negative", "False negatives (misses)"),
        ("false_positive", "False alarms"),
        ("true_negative", "True negatives"),
    ):
        values = metrics["confusion"][name]
        lines.append(
            f"{label:<30} {values['count']:>7} {values['duration_us'] / 1e6:>16.6f} "
            f"{values['percentage_of_evaluated_time']:>20.2f}"
        )
    lines.extend(["", "Metrics (sample-count basis):"])
    for name in (
        "precision",
        "recall",
        "miss_rate",
        "false_positive_rate",
        "f1",
        "accuracy",
        "balanced_accuracy",
    ):
        value = metrics[name]
        formatted = "undefined (zero denominator)" if value is None else f"{100 * value:.2f}%"
        lines.append(f"  {name}: {formatted}; {report['metric_definitions'][name]}")
    if "exported_eaf" in report:
        lines.append(f"Exported EAF: {report['exported_eaf']}")
    return "\n".join(lines) + "\n"


def _write_text_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_completed_run(
    path: str | Path,
    reference_eaf: str | Path,
    *,
    reference_tier: str = "Manual_GUnit",
    positive_value: str = "G-unit",
    visibility_tier: str = DEFAULT_VISIBILITY_TIER,
    alignment_offset_us: int = 0,
    output: str | Path | None = None,
    export_eaf: bool = False,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Validate inputs and write 100 ms confusion statistics (and optionally EAF).

    Uses the activity figure's center-in-interval rule and locally grouped
    LLM-only decisions for probability runs. Recorded Phase1/time-selection exclusions and unlabelled reference visibility
    intervals are removed from denominators. Labelled visibility values do not mask detection. Writes
    JSON and text reports, replacing existing outputs by default. Requires only
    saved canonical results and their manifest, so compact runs are supported.
    """
    run = load_completed_run(path)
    reference = Path(reference_eaf).resolve()
    target = Path(output) if output is not None else run.path.parent / "reference-statistics.json"
    if target.suffix.lower() != ".json":
        raise ValueError("statistics output must use the .json extension")
    text_target = target.with_suffix(".txt")
    eaf_target = run.path.with_suffix(".eaf")
    destinations = (
        (target, text_target, *eaf_destinations(run, eaf_target))
        if export_eaf
        else (target, text_target)
    )
    check_outputs(
        destinations,
        protected=(*protected_run_paths(run), reference),
        overwrite=overwrite,
    )
    reference_digest = sha256(reference.read_bytes()).hexdigest()
    if not positive_value.strip():
        raise ValueError("reference G-unit value cannot be blank")
    _, units = read_reference_g_units(
        reference,
        tier_id=reference_tier,
        alignment_offset_us=alignment_offset_us,
    )
    units = [unit for unit in units if unit.annotation_value == positive_value]
    if sha256(reference.read_bytes()).hexdigest() != reference_digest:
        raise ValueError("reference EAF changed during evaluation")
    scope = read_reference_scope(
        reference, read_reference_g_units, tier_id=visibility_tier,
        alignment_offset_us=alignment_offset_us,
    )
    if sha256(reference.read_bytes()).hexdigest() != reference_digest:
        raise ValueError("reference EAF changed during evaluation")
    projection = project_llm_output(run.document)
    samples = decision_samples(run.document, membership=projection.membership if projection else None)
    evaluation = evaluate_final_decisions(samples, units, assessment_scope=scope)
    enabled = phase1_enabled(run.document, run.manifest)
    report_warnings = (
        [
            (
                "Phase1 visibility filtering was enabled. Its recorded exclusions and unlabelled reference intervals are removed; "
                "visibility-check errors in admitted samples may be included in these statistics."
            )
        ]
        if enabled is True
        else []
    )
    if not scope.tier_present:
        report_warnings.append(
            f"Phase1 cannot be scored: reference visibility tier {visibility_tier!r} is missing; "
            "no scorable reference data."
        )
    metrics = _evaluation_payload(
        evaluation, reference_available=True, final_decision_available=True
    )
    metrics["balanced_accuracy"] = (
        (metrics["recall"] + 1 - metrics["false_positive_rate"]) / 2
        if metrics["recall"] is not None and metrics["false_positive_rate"] is not None else None
    )
    metrics["available"] = bool(evaluation.intervals)
    if not evaluation.intervals:
        metrics["reason"] = "no_assessed_admitted_samples"
    metrics["exclusion_policy"] = "phase1-processing-and-labelled-reference-scope-v2"
    excluded: dict[str, dict[str, int]] = {
        name: {"count": 0, "duration_us": 0} for name in ("outside_selected_time", "unlabelled_reference", "phase1")
    }
    for sample in samples:
        # Disjoint accounting: a sample excluded for several reasons is counted once.
        reason = (
            "outside_selected_time"
            if time_selection_excluded(sample)
            else "unlabelled_reference"
            if not scope.contains(sample.center_us)
            else "phase1"
            if phase1_excluded(sample)
            else None
        )
        if reason is not None:
            excluded[reason]["count"] += 1
            excluded[reason]["duration_us"] += sample.end_us - sample.start_us
    report = {
        "schema_version": "3.2.0",
        "policy_version": "completed-run-reference-evaluation-v5",
        "decision_basis": decision_basis(projection),
        "run_id": run.document["run_id"],
        "analysis": {"path": str(run.path), "sha256": run.sha256},
        "reference": {
            "path": str(reference),
            "sha256": reference_digest,
            "tier_id": reference_tier,
            "positive_value": positive_value,
            "visibility_annotations_used": True,
            "visibility_tier_id": visibility_tier,
            "visibility_tier_present": scope.tier_present,
            "visibility_label_values_mask_detection": False,
            "alignment_offset_us": alignment_offset_us,
        },
        "scope": {
            "phase1_enabled": enabled,
            "total_sample_count": len(samples),
            "total_duration_us": sum(s.end_us - s.start_us for s in samples),
            "excluded_sample_count": len(samples) - len(evaluation.intervals),
            "excluded_duration_us": sum(v["duration_us"] for v in excluded.values()),
            "exclusion_precedence": list(excluded),
            "excluded_by_reason": excluded,
        },
        "evaluation": metrics,
        "warnings": report_warnings,
        "metric_definitions": {
            "precision": "TP / (TP + FP)",
            "recall": "TP / (TP + FN)",
            "miss_rate": "FN / (TP + FN)",
            "false_positive_rate": "FP / (FP + TN)",
            "f1": "2*TP / (2*TP + FP + FN)",
            "accuracy": "(TP + TN) / (TP + FN + FP + TN)",
            "balanced_accuracy": "(TP/(TP+FN) + TN/(TN+FP)) / 2; requires both reference classes",
            "undefined_denominator": "null",
            "ratio_scale": "0 to 1",
            "confusion_time_percentage": "100 * outcome_duration / evaluated_duration",
        },
        "model_calls": 0,
        "output_files": {
            "json": str(target.resolve()),
            "text": str(text_target.resolve()),
        },
    }
    if export_eaf:
        report["exported_eaf"] = str(export_loaded_run(run, eaf_target, overwrite=overwrite))
    text_report = render_text_report(report)
    write_json(target, report)
    _write_text_report(text_target, text_report)
    return report
