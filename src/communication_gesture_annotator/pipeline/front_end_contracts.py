# SPDX-License-Identifier: Apache-2.0
"""Contracts for scene discovery, dense usability, and optional rest profiling."""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Sequence
from .exemplar_normalization import (
    ExemplarValidationResult,
    normalize_exemplar_probabilities,
)
from .llm_contracts import (
    PROFILE_MAX_DESCRIPTION_CHARACTERS,
    PROFILE_MAX_IDENTIFIER_CHARACTERS,
    PROFILE_MAX_SELECTED_SUBJECTS,
    PROFILE_SCHEMA,
    ProfileValidationResult,
    TaskContractViolation,
    validate_profile_response,
)

OVERVIEW_PROMPT_VERSION = "recording-overview-v0.5"
PHASE0A_EXEMPLAR_CONTRACT_VERSION = "selected-live-target-exemplars-v0.1"
PHASE0A_EXEMPLAR_PROMPT_VERSION = "selected-live-target-exemplars-v0.2"
VISIBILITY_GATE_VERSION = "visibility-irrelevant-gate-v0.2"
_VISIBILITY = {"visible", "not_visible", "uncertain"}
_SCENE_TYPES = {
    "live_target",
    "slide",
    "static_picture",
    "embedded_video",
    "audience_or_cutaway",
    "other_live_scene",
    "transition",
    "uncertain",
}
_EXEMPLAR_CLASSES = {"positive_main_scene", "not_positive_main_scene", "uncertain"}
PHASE0A_EXEMPLAR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target_subject_ids", "samples"],
    "properties": {
        "target_subject_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": PROFILE_MAX_SELECTED_SUBJECTS,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": PROFILE_MAX_IDENTIFIER_CHARACTERS,
            },
        },
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "sample_index",
                    "scene_type",
                    "selected_target_visibility",
                    "classification",
                    "probabilities",
                    "reason",
                ],
                "properties": {
                    "sample_index": {"type": "integer"},
                    "scene_type": {"type": "string", "enum": sorted(_SCENE_TYPES)},
                    "selected_target_visibility": {"type": "string", "enum": sorted(_VISIBILITY)},
                    "classification": {"type": "string", "enum": sorted(_EXEMPLAR_CLASSES)},
                    "probabilities": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": sorted(_EXEMPLAR_CLASSES),
                        "properties": {
                            label: {"type": "number", "minimum": 0, "maximum": 1}
                            for label in sorted(_EXEMPLAR_CLASSES)
                        },
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": PROFILE_MAX_DESCRIPTION_CHARACTERS,
                    },
                },
            },
        },
    },
}


def phase0a_exemplar_schema_for_count(sample_count: int) -> dict[str, Any]:
    """Return the exact-cardinality Phase 0a exemplar-verification schema."""
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("Phase 0a exemplar sample_count must be a positive integer")
    schema = deepcopy(PHASE0A_EXEMPLAR_SCHEMA)
    samples = schema["properties"]["samples"]
    samples["minItems"] = sample_count
    samples["maxItems"] = sample_count
    return schema


def _derived_exemplar_classification(sample: Mapping[str, Any]) -> str:
    scene_type = sample["scene_type"]
    target_visibility = sample["selected_target_visibility"]
    if scene_type == "live_target" and target_visibility == "visible":
        return "positive_main_scene"
    if (
        scene_type
        in {"slide", "static_picture", "embedded_video", "audience_or_cutaway", "other_live_scene"}
        or target_visibility == "not_visible"
    ):
        return "not_positive_main_scene"
    return "uncertain"


def validate_phase0a_exemplar_response(
    value: Any, *, expected_indices: Sequence[int], expected_subject_ids: Sequence[str]
) -> ExemplarValidationResult:
    """Validate an exact independent decision for every final storyboard image."""
    if not isinstance(value, Mapping) or set(value) != {"target_subject_ids", "samples"}:
        raise ValueError("Phase 0a exemplar response has an invalid field set")
    if value.get("target_subject_ids") != list(expected_subject_ids):
        raise ValueError("Phase 0a exemplar target_subject_ids do not match")
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Phase 0a exemplar samples must be an array")
    required_indices = list(expected_indices)
    actual_indices = [
        sample.get("sample_index") if isinstance(sample, Mapping) else None for sample in samples
    ]
    if actual_indices != required_indices:
        raise ValueError(
            "Phase 0a exemplar response must cover every supplied sample exactly once in the supplied order"
        )
    required_fields = {
        "sample_index",
        "scene_type",
        "selected_target_visibility",
        "classification",
        "probabilities",
        "reason",
    }
    normalized: list[dict[str, Any]] = []
    normalization_events: list[dict[str, Any]] = []
    for sample in samples:
        assert isinstance(sample, Mapping)
        if set(sample) != required_fields:
            raise ValueError("Phase 0a exemplar sample has an invalid field set")
        if type(sample["sample_index"]) is not int:
            raise ValueError("Phase 0a exemplar sample_index must be an integer")
        if sample["scene_type"] not in _SCENE_TYPES:
            raise ValueError("Phase 0a exemplar scene_type is invalid")
        if sample["selected_target_visibility"] not in _VISIBILITY:
            raise ValueError("Phase 0a exemplar selected_target_visibility is invalid")
        classification = sample["classification"]
        if (
            classification not in _EXEMPLAR_CLASSES
            or classification != _derived_exemplar_classification(sample)
        ):
            raise ValueError(
                "Phase 0a exemplar classification contradicts scene type or selected-target visibility"
            )
        probabilities, event = normalize_exemplar_probabilities(
            sample["probabilities"], sample_index=sample["sample_index"]
        )
        if (
            sample["probabilities"][classification] != max(sample["probabilities"].values())
            or probabilities[classification] != max(probabilities.values())
        ):
            raise TaskContractViolation(
                "overview_exemplar.classification_not_maximum",
                "Phase 0a exemplar classification must be a maximum-probability class",
            )
        if event is not None:
            normalization_events.append(event)
        reason = sample["reason"]
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > PROFILE_MAX_DESCRIPTION_CHARACTERS
        ):
            raise ValueError("Phase 0a exemplar reason must be a concise nonempty string")
        normalized.append({**dict(sample), "probabilities": probabilities})
    return ExemplarValidationResult(
        {"target_subject_ids": list(expected_subject_ids), "samples": normalized},
        normalization_events=normalization_events,
    )


def _validated_direct_indices(direct_indices: Sequence[int]) -> list[int]:
    """Return canonical direct storyboard indices for positive exemplars."""
    values = list(direct_indices)
    if (
        not values
        or any((type(index) is not int or index < 0 for index in values))
        or values != sorted(set(values))
    ):
        raise ValueError(
            "positive direct evidence indices must be a non-empty sorted unique non-negative integer sequence"
        )
    return values


def final_overview_schema(direct_indices: Sequence[int]) -> dict[str, Any]:
    """Return the final single-target schema with bounded direct evidence.

    The directly supplied storyboard is uniform across the recording and can
    contain evidence for a candidate who is not selected as the recording-wide target.
    Root evidence supports target/profile consolidation only. A separate exact
    verification contract decides which direct images are positive exemplars.
    """
    direct = _validated_direct_indices(direct_indices)
    schema = deepcopy(PROFILE_SCHEMA)
    schema["properties"]["target_mode"] = {"type": "string", "const": "individual"}
    selected = schema["properties"]["selected_subjects"]
    selected["minItems"] = 1
    selected["maxItems"] = 1
    root_evidence = schema["properties"]["evidence_sample_indices"]
    root_evidence["minItems"] = 0
    root_evidence["maxItems"] = min(8, len(direct))
    root_evidence["items"] = {"type": "integer", "enum": direct}
    return schema


def validate_overview_response(
    value: Any,
    allowed_indices: set[int],
    *,
    allow_empty: bool = False,
    require_single_subject: bool = False,
    require_nested_evidence_in_root: bool = False,
    require_positive_direct_evidence: bool = False,
    require_direct_evidence: bool = False,
    allowed_direct_indices: set[int] | None = None,
) -> ProfileValidationResult:
    """Validate the target profile with non-overview fields empty."""
    validated = validate_profile_response(
        value,
        allowed_indices,
        allow_empty=allow_empty,
        require_nested_evidence_in_root=require_nested_evidence_in_root,
    )
    if validated["annotation_relevance_spans"]:
        raise ValueError("recording overview must not return annotation relevance spans")
    if any((subject["typical_rest_postures"] for subject in validated["selected_subjects"])):
        raise ValueError("recording overview must not return rest postures")
    if require_single_subject and (
        validated["target_mode"] != "individual" or len(validated["selected_subjects"]) != 1
    ):
        raise ValueError("final recording overview must select exactly one participant")
    if require_positive_direct_evidence or require_direct_evidence:
        if allowed_direct_indices is None:
            raise ValueError("direct evidence validation requires allowed direct indices")
        direct = _validated_direct_indices(sorted(allowed_direct_indices))
        if not set(direct).issubset(allowed_indices):
            raise ValueError("direct evidence indices must be allowed overview evidence")
        evidence = validated["evidence_sample_indices"]
        if len(evidence) > min(8, len(direct)) or not set(evidence).issubset(direct):
            raise ValueError(
                "final recording overview evidence must contain at most eight items selected only from direct storyboard evidence"
            )
    return validated


def overview_system_prompt() -> str:
    return "You are Phase 0a of Communication Gesture Annotator. Discover the recording-wide main speaker and the family of legitimate live views in which that same person appears. Intermediate segment summaries may contain several recurring candidates, but the final recording overview selects exactly one participant. Use all supplied sparse storyboard evidence. Link the target across camera angles using only within-recording clothing, body, stage, and temporal continuity cues. Do not identify or name a real person and do not use biometric recognition. Slides, static pictures, embedded videos, audience shots, and brief unrelated people are not alternative targets even when locally prevalent. Return concise identity_cues and recurring_views that Phase 0b can apply to dense frames. This phase does not decide exact relevance boundaries and does not infer rest postures: annotation_relevance_spans and every typical_rest_postures array must be empty. Aggregate recurring sightings rather than creating one subject per image or camera view. In the final recording overview, root evidence_sample_indices are ordinary direct visual evidence supporting the recording-wide target/profile. They are not positive exemplars and will not be sent to Phase 0b as such. A separate exact per-image verification step owns positive-exemplar decisions. Never add a citation merely to reach a count. Return only the requested JSON object."


def phase0a_exemplar_verification_system_prompt() -> str:
    return "You are the Phase 0a positive-exemplar verifier of Communication Gesture Annotator. The recording-wide selected target profile is fixed. Classify every supplied final storyboard image exactly once and in the supplied order. Compare the images as a set to the fixed within-recording clothing, body, stage, and continuity cues, but make an independent visual decision for each image; do not infer that an image is positive because nearby images are positive. A positive_main_scene image must physically show the selected live target in a legitimate live camera view. A live target standing beside a projected slide or appearing in a split view with a slide can be positive, but only while that selected live target is visibly present. A full-screen slide, static picture, audience/cutaway, unrelated person, or embedded video is not positive. A person depicted inside a slide, photograph, screen, or embedded clip is not the live selected target. Do not accept an image merely because it contains some speaker or a similar stage. Use uncertain when the selected target's live physical presence cannot be established. Return only the requested JSON object."
