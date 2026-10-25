# SPDX-License-Identifier: Apache-2.0
"""Property-level modality admission, factual rechecks, and bounded recovery.

These helpers resolve availability, not manual state. A usable wrist measurement
does not establish rest, contact, target presence in a disputed scene, or a G-unit.
Model-authored explanations remain reports until independent evidence resolves
them; no substring in free text creates a continuity barrier.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal, Sequence

EVIDENCE_RESOLUTION_VERSION = "property-evidence-resolution-v0.1"
UNCERTAINTY_CAUSES = frozenset(
    {
        "insufficient_evidence",
        "occlusion",
        "out_of_frame",
        "target_association",
        "pose_quality",
        "rgb_quality",
        "timing_alignment",
        "camera_cut",
        "identity_change",
        "source_gap",
        "scene_exclusion",
        "conflicting_evidence",
    }
)
PropertyKind = Literal["position", "motion", "support", "contact", "scene", "target", "context"]


@dataclass(frozen=True)
class MeasurementEvidence:
    property_name: str
    modality: Literal["pose", "rgb"]
    available: bool
    target_associated: bool
    confidence: float
    geometry_valid: bool
    coverage_valid: bool
    timing_valid: bool
    evidence_refs: tuple[str, ...] = ()
    property_kind: PropertyKind = "position"

    def __post_init__(self) -> None:
        if not self.property_name or self.modality not in {"pose", "rgb"}:
            raise ValueError("measurement requires a property name and known modality")
        if self.property_kind not in {
            "position",
            "motion",
            "support",
            "contact",
            "scene",
            "target",
            "context",
        }:
            raise ValueError("measurement property kind is invalid")
        if (
            isinstance(self.confidence, bool)
            or not isfinite(self.confidence)
            or (not 0 <= self.confidence <= 1)
        ):
            raise ValueError("measurement confidence must be finite and in [0, 1]")
        if any(
            (
                type(value) is not bool
                for value in (
                    self.available,
                    self.target_associated,
                    self.geometry_valid,
                    self.coverage_valid,
                    self.timing_valid,
                )
            )
        ):
            raise ValueError("measurement validity fields must be Boolean")
        if any((not isinstance(ref, str) or not ref for ref in self.evidence_refs)):
            raise ValueError("measurement evidence references must be nonempty strings")


def _rejection_reasons(value: MeasurementEvidence | None, minimum_confidence: float) -> list[str]:
    if value is None:
        return ["not_supplied"]
    reasons = []
    for valid, reason in (
        (value.available, "measurement_unavailable"),
        (value.target_associated, "target_association_unresolved"),
        (value.confidence >= minimum_confidence, "confidence_below_threshold"),
        (value.geometry_valid, "geometry_invalid"),
        (value.coverage_valid, "coverage_insufficient"),
        (value.timing_valid, "timing_unverified"),
        (bool(value.evidence_refs), "evidence_reference_missing"),
    ):
        if not valid:
            reasons.append(reason)
    return reasons


def resolve_property(
    pose: MeasurementEvidence | None,
    rgb: MeasurementEvidence | None,
    *,
    minimum_confidence: float = 0.4,
) -> dict[str, Any]:
    """Prefer admitted pose for measurements and RGB for contextual properties.

    Neither confidence nor detector coverage is treated as physical visibility.
    Geometry/association/timing checks must be supplied by the caller. The
    nonpreferred modality remains in the audit record, including its rejection
    reasons; a missing modality never erases an admitted other modality.
    """
    if not isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum confidence must be in [0, 1]")
    if pose is None and rgb is None:
        raise ValueError("resolve_property requires at least one property record")
    if pose is not None and pose.modality != "pose" or (rgb is not None and rgb.modality != "rgb"):
        raise ValueError("property arguments must match their modality")
    supplied = [item for item in (pose, rgb) if item is not None]
    if len({(item.property_name, item.property_kind) for item in supplied}) != 1:
        raise ValueError("property records must describe the same property")
    first = supplied[0]
    reasons = {
        "pose": _rejection_reasons(pose, minimum_confidence),
        "rgb": _rejection_reasons(rgb, minimum_confidence),
    }
    if first.property_kind not in {"position", "motion"} and pose is not None:
        reasons["pose"].append("property_requires_rgb_context")
    order = ("pose", "rgb") if first.property_kind in {"position", "motion"} else ("rgb", "pose")
    selected = next((modality for modality in order if not reasons[modality]), None)
    selected_evidence = pose if selected == "pose" else rgb if selected == "rgb" else None
    return {
        "property_name": first.property_name,
        "property_kind": first.property_kind,
        "usable": selected is not None,
        "preferred_modality": selected,
        "confidence": selected_evidence.confidence if selected_evidence else 0.0,
        "evidence_refs": list(selected_evidence.evidence_refs) if selected_evidence else [],
        "modalities": {
            modality: {
                "admitted": not reasons[modality],
                "rejection_reasons": reasons[modality],
                "measurement": asdict(value) if value is not None else None,
            }
            for modality, value in (("pose", pose), ("rgb", rgb))
        },
    }


def resolve_sample_evidence(
    sample_index: int,
    pose_measurements: Sequence[MeasurementEvidence],
    rgb_measurements: Sequence[MeasurementEvidence],
    *,
    scene_excluded: bool = False,
    source_barrier: bool = False,
    minimum_confidence: float = 0.4,
) -> dict[str, Any]:
    """Admit independent properties and identify a factual scene contradiction.

    ``detection_usable`` only permits inference. It never labels rest or activity,
    and it stays false for a disputed scene until a factual recheck resolves it.
    Individual admitted measurements are retained while that conflict is open.
    """
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    by_modality: dict[str, dict[str, MeasurementEvidence]] = {"pose": {}, "rgb": {}}
    for modality, values in (("pose", pose_measurements), ("rgb", rgb_measurements)):
        for value in values:
            if value.modality != modality or value.property_name in by_modality[modality]:
                raise ValueError("each modality needs one record per property")
            by_modality[modality][value.property_name] = value
    properties = [
        resolve_property(
            by_modality["pose"].get(name),
            by_modality["rgb"].get(name),
            minimum_confidence=minimum_confidence,
        )
        for name in sorted(set(by_modality["pose"]) | set(by_modality["rgb"]))
    ]
    manual_usable = any(
        (
            value["usable"] and value["property_kind"] in {"position", "motion"}
            for value in properties
        )
    )
    return {
        "policy_version": EVIDENCE_RESOLUTION_VERSION,
        "sample_index": sample_index,
        "properties": properties,
        "manual_measurements_usable": manual_usable,
        "scene_excluded": scene_excluded,
        "source_barrier": source_barrier,
        "recheck_required": scene_excluded and manual_usable and (not source_barrier),
        "detection_usable": manual_usable and (not scene_excluded) and (not source_barrier),
    }


def reported_uncertainty(
    reported_causes: Sequence[str] = (),
    *,
    reason: str | None = None,
    evidence_refs: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Store typed reported causes as unverified; never classify free text."""
    causes = list(dict.fromkeys(reported_causes))
    if any((cause not in UNCERTAINTY_CAUSES for cause in causes)):
        raise ValueError("reported uncertainty cause is invalid")
    if not causes and reason is not None:
        causes = ["insufficient_evidence"]
    return [
        {
            "cause": cause,
            "status": "unverified",
            "reported_reason": reason,
            "evidence_refs": list(evidence_refs),
            "resolution_evidence_refs": [],
        }
        for cause in causes
    ]
