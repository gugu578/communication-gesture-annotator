# SPDX-License-Identifier: Apache-2.0
"""Strict staged prompt and response contracts for Research Stage 2."""

from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

PROFILE_NORMALIZATION_VERSION = "profile-normalization-v0.1"
PROFILE_EVIDENCE_SCOPE_VERSION = "profile-evidence-scope-v0.1"
PROFILE_MAX_SELECTED_SUBJECTS = 8
PROFILE_MAX_IDENTITY_CUES = 8
PROFILE_MAX_RECURRING_VIEWS = 8
PROFILE_MAX_REST_POSTURES = 8
PROFILE_MAX_IGNORED_TRANSIENT_PEOPLE = 16
PROFILE_MAX_RELEVANCE_SPANS = 32
PROFILE_MAX_NESTED_EVIDENCE_INDICES = 16
PROFILE_MAX_ROOT_EVIDENCE_INDICES = 128
PROFILE_MAX_IDENTIFIER_CHARACTERS = 64
PROFILE_MAX_LABEL_CHARACTERS = 160
PROFILE_MAX_DESCRIPTION_CHARACTERS = 512
PROFILE_MAX_SCENE_SUMMARY_CHARACTERS = 1024


def validate_profile_evidence_scope(value: Any) -> dict[str, Any]:
    """Validate the closed consolidation evidence-scope provenance record."""
    if not isinstance(value, Mapping):
        raise ValueError("profile evidence scope must be an object")
    required = {
        "policy_version",
        "direct_storyboard_sample_indices",
        "inherited_profile_sample_indices",
        "allowed_evidence_sample_indices",
        "maximum_output_evidence_items",
        "irrelevant_gate_evidence_policy",
    }
    if set(value) != required:
        raise ValueError("profile evidence scope has an invalid field set")

    def ordered_indices(field: str) -> list[int]:
        indices = value.get(field)
        if (
            not isinstance(indices, list)
            or any((type(index) is not int or index < 0 for index in indices))
            or indices != sorted(set(indices))
        ):
            raise ValueError(f"profile evidence scope {field} is invalid")
        return list(indices)

    direct = ordered_indices("direct_storyboard_sample_indices")
    inherited = ordered_indices("inherited_profile_sample_indices")
    allowed = ordered_indices("allowed_evidence_sample_indices")
    if set(direct) & set(inherited):
        raise ValueError("profile evidence scope direct and inherited sets overlap")
    if allowed != sorted({*direct, *inherited}):
        raise ValueError("profile evidence scope allowed set is inconsistent")
    if (
        value.get("policy_version") != PROFILE_EVIDENCE_SCOPE_VERSION
        or value.get("maximum_output_evidence_items") != PROFILE_MAX_ROOT_EVIDENCE_INDICES
        or value.get("irrelevant_gate_evidence_policy") != "direct-final-storyboard-only"
    ):
        raise ValueError("profile evidence scope policy is unsupported")
    return {
        "policy_version": PROFILE_EVIDENCE_SCOPE_VERSION,
        "direct_storyboard_sample_indices": direct,
        "inherited_profile_sample_indices": inherited,
        "allowed_evidence_sample_indices": allowed,
        "maximum_output_evidence_items": PROFILE_MAX_ROOT_EVIDENCE_INDICES,
        "irrelevant_gate_evidence_policy": "direct-final-storyboard-only",
    }


_TASK_CONTRACT_PATH_BY_CODE = {
    "overview_exemplar.probabilities_invalid": "$.samples[].probabilities",
    "overview_exemplar.classification_not_maximum": "$.samples[].classification|probabilities",
    "main_scene_visibility.coverage_or_boolean": "$.samples",
    "activity.subject_mismatch": "$.subject_id",
    "activity.spans_not_array": "$.spans",
    "activity.spans_not_empty_for_empty_request": "$.spans",
    "activity.span_malformed": "$.spans[]",
    "activity.span_bounds_not_integer": "$.spans[].start_sample|end_sample_exclusive",
    "activity.span_start_not_contiguous": "$.spans[].start_sample",
    "activity.span_end_not_after_start": "$.spans[].end_sample_exclusive",
    "activity.span_end_after_required": "$.spans[].end_sample_exclusive",
    "activity.final_span_end_after_required": "$.spans[last].end_sample_exclusive",
    "activity.coverage_incomplete": "$.spans",
    "activity.hand_state_invalid": "$.spans[].left|right.state",
    "activity.hand_probabilities_invalid": "$.spans[].left|right.probabilities",
    "activity.hand_quality_invalid": "$.spans[].left|right.quality",
    "activity.unknown_reason_missing": "$.spans[].left|right.reason",
    "non_g_probability.invalid_scores": "$.non_g_unit_probability",
    "profile.root_evidence_not_ordered": "$.evidence_sample_indices",
    "profile.root_evidence_not_supplied": "$.evidence_sample_indices",
    "profile.target_mode_cardinality": "$.target_mode",
    "profile.subject_id_not_unique": "$.selected_subjects[].subject_id",
    "profile.posture_view_not_recurring": "$.selected_subjects[].typical_rest_postures[].view",
    "profile.posture_evidence_not_ordered": "$.selected_subjects[].typical_rest_postures[].evidence_sample_indices",
    "profile.posture_evidence_not_supplied": "$.selected_subjects[].typical_rest_postures[].evidence_sample_indices",
    "profile.posture_evidence_not_in_root": "$.selected_subjects[].typical_rest_postures[].evidence_sample_indices",
    "profile.relevance_bounds_not_supplied": "$.annotation_relevance_spans[].start_sample|end_sample_inclusive",
    "profile.relevance_bounds_reversed": "$.annotation_relevance_spans[]",
    "profile.relevance_spans_not_ordered": "$.annotation_relevance_spans",
    "profile.relevance_evidence_not_ordered": "$.annotation_relevance_spans[].evidence_sample_indices",
    "profile.relevance_evidence_not_supplied": "$.annotation_relevance_spans[].evidence_sample_indices",
    "profile.relevance_evidence_missing_bounds": "$.annotation_relevance_spans[].evidence_sample_indices",
    "profile.relevance_evidence_outside_span": "$.annotation_relevance_spans[].evidence_sample_indices",
    "profile.relevance_evidence_not_in_root": "$.annotation_relevance_spans[].evidence_sample_indices",
}


class ModelContextLengthError(RuntimeError):
    """A provider rejected a request because its model context was exceeded.

    Orchestration may retry with a smaller, complete evidence window. This
    contract does not classify transport, schema, or output-budget failures as
    context exhaustion.
    """


class TaskContractViolation(ValueError):
    """A domain-validation failure with content-free diagnostic metadata.

    The code determines the path through a closed allowlist. Adapters may expose
    these two fields in non-sensitive logs without copying arbitrary validator
    messages or model content.
    """

    def __init__(self, code: str, message: str) -> None:
        try:
            path = _TASK_CONTRACT_PATH_BY_CODE[code]
        except KeyError as exc:
            raise ValueError("task contract violation code is not allowlisted") from exc
        self.code = code
        self.path = path
        super().__init__(message)


def validate_profile_normalization_events(value: Any) -> list[dict[str, Any]]:
    """Validate the closed, content-free Pass 1 normalization ledger."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("profile normalization events must be an array")
    normalized: list[dict[str, Any]] = []
    relevance_values = {"relevant", "irrelevant", "uncertain"}
    target_modes = {"none", "individual", "group"}
    for raw_event in value:
        if not isinstance(raw_event, Mapping):
            raise ValueError("profile normalization event must be an object")
        event = dict(raw_event)
        if event.get("policy_version") != PROFILE_NORMALIZATION_VERSION:
            raise ValueError("profile normalization event policy is unsupported")
        event_name = event.get("event")
        if event_name == "relevance_evidence_bounds_inserted":
            required = {
                "policy_version",
                "event",
                "path",
                "span_ordinal",
                "inserted_sample_indices",
                "removed_interior_sample_indices",
                "raw_relevance",
                "canonical_relevance",
                "profile_irrelevant_gate_eligible",
            }
            ordinal = event.get("span_ordinal")
            inserted = event.get("inserted_sample_indices")
            removed = event.get("removed_interior_sample_indices")
            if (
                set(event) != required
                or type(ordinal) is not int
                or ordinal < 0
                or (
                    event.get("path")
                    != f"$.annotation_relevance_spans[{ordinal}].evidence_sample_indices"
                )
                or (not isinstance(inserted, list))
                or (not inserted)
                or any((type(index) is not int for index in inserted))
                or (inserted != sorted(set(inserted)))
                or (not isinstance(removed, list))
                or any((type(index) is not int for index in removed))
                or (removed != sorted(set(removed)))
                or (set(inserted) & set(removed))
                or (event.get("raw_relevance") not in relevance_values)
                or (
                    event.get("canonical_relevance")
                    != (
                        "uncertain"
                        if event.get("raw_relevance") == "irrelevant"
                        else event.get("raw_relevance")
                    )
                )
                or (event.get("profile_irrelevant_gate_eligible") is not False)
            ):
                raise ValueError("profile relevance normalization event is invalid")
        elif event_name == "target_mode_derived_from_subject_cardinality":
            required = {
                "policy_version",
                "event",
                "path",
                "raw_target_mode",
                "canonical_target_mode",
            }
            if (
                set(event) != required
                or event.get("path") != "$.target_mode"
                or event.get("raw_target_mode") not in target_modes
                or (event.get("canonical_target_mode") not in target_modes)
                or (event.get("raw_target_mode") == event.get("canonical_target_mode"))
            ):
                raise ValueError("profile target-mode normalization event is invalid")
        else:
            raise ValueError("profile normalization event type is unsupported")
        normalized.append(deepcopy(event))
    return normalized


@dataclass(frozen=True, slots=True)
class ModelCallResult:
    """One validated model result plus non-secret transport provenance."""

    payload: Mapping[str, Any]
    request_id: str
    response_sha256: str
    latency_seconds: float
    endpoint_mode: str
    repaired: bool = False
    trace_reference: Mapping[str, Any] | None = None
    model_payload: Mapping[str, Any] | None = None
    reasoning_observed: bool = False
    contract_normalization_events: tuple[Mapping[str, Any], ...] = ()
    model_payload_sha256: str | None = None
    payload_sha256: str | None = None
    contract_normalization_policy_version: str | None = None


class ProfileValidationResult(dict[str, Any]):
    """Validated profile plus deterministic contract-normalization events.

    The mapping is the canonical payload. The adapter separately preserves the
    schema-valid model payload, while these events describe the narrow,
    versioned relation repair without embedding model-authored prose.
    """

    normalization_events: list[dict[str, Any]]
    normalization_policy_version: str

    def __init__(
        self, values: Mapping[str, Any], *, normalization_events: Sequence[Mapping[str, Any]] = ()
    ) -> None:
        super().__init__(values)
        self.normalization_events = validate_profile_normalization_events(normalization_events)
        self.normalization_policy_version = PROFILE_NORMALIZATION_VERSION


class Phase2Model(Protocol):
    """Task-specific private model boundary used by the staged runner."""

    def check_capabilities(self) -> Mapping[str, Any]:
        """Verify endpoint, model, vision, and structured-output support."""

    def complete_json(
        self,
        *,
        request_id: str,
        task: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[Path],
        response_schema: Mapping[str, Any],
        trace_context: Mapping[str, Any],
        response_validator: Callable[[Any], Mapping[str, Any]] | None = None,
    ) -> ModelCallResult:
        """Return a locally validated JSON response for a domain task."""


PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "target_mode",
        "selected_subjects",
        "ignored_transient_people",
        "annotation_relevance_spans",
        "scene_summary",
        "confidence",
        "evidence_sample_indices",
    ],
    "properties": {
        "target_mode": {"type": "string", "enum": ["individual", "group"]},
        "selected_subjects": {
            "type": "array",
            "minItems": 1,
            "maxItems": PROFILE_MAX_SELECTED_SUBJECTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "subject_id",
                    "visual_description",
                    "identity_cues",
                    "recurring_views",
                    "typical_rest_postures",
                    "selection_reason",
                    "prevalence",
                ],
                "properties": {
                    "subject_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": PROFILE_MAX_IDENTIFIER_CHARACTERS,
                    },
                    "visual_description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": PROFILE_MAX_DESCRIPTION_CHARACTERS,
                    },
                    "identity_cues": {
                        "type": "array",
                        "maxItems": PROFILE_MAX_IDENTITY_CUES,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": PROFILE_MAX_LABEL_CHARACTERS,
                        },
                    },
                    "recurring_views": {
                        "type": "array",
                        "maxItems": PROFILE_MAX_RECURRING_VIEWS,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": PROFILE_MAX_LABEL_CHARACTERS,
                        },
                    },
                    "typical_rest_postures": {
                        "type": "array",
                        "maxItems": PROFILE_MAX_REST_POSTURES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "view",
                                "left_hand",
                                "right_hand",
                                "evidence_sample_indices",
                            ],
                            "properties": {
                                "view": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": PROFILE_MAX_LABEL_CHARACTERS,
                                },
                                "left_hand": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": PROFILE_MAX_DESCRIPTION_CHARACTERS,
                                },
                                "right_hand": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": PROFILE_MAX_DESCRIPTION_CHARACTERS,
                                },
                                "evidence_sample_indices": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": PROFILE_MAX_NESTED_EVIDENCE_INDICES,
                                    "uniqueItems": True,
                                    "items": {"type": "integer"},
                                },
                            },
                        },
                    },
                    "selection_reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": PROFILE_MAX_DESCRIPTION_CHARACTERS,
                    },
                    "prevalence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "ignored_transient_people": {
            "type": "array",
            "maxItems": PROFILE_MAX_IGNORED_TRANSIENT_PEOPLE,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "reason"],
                "properties": {
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": PROFILE_MAX_DESCRIPTION_CHARACTERS,
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": PROFILE_MAX_DESCRIPTION_CHARACTERS,
                    },
                },
            },
        },
        "annotation_relevance_spans": {
            "type": "array",
            "maxItems": PROFILE_MAX_RELEVANCE_SPANS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "start_sample",
                    "end_sample_inclusive",
                    "relevance",
                    "confidence",
                    "reason",
                    "evidence_sample_indices",
                ],
                "properties": {
                    "start_sample": {"type": "integer"},
                    "end_sample_inclusive": {"type": "integer"},
                    "relevance": {
                        "type": "string",
                        "enum": ["relevant", "irrelevant", "uncertain"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": PROFILE_MAX_DESCRIPTION_CHARACTERS,
                    },
                    "evidence_sample_indices": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": PROFILE_MAX_NESTED_EVIDENCE_INDICES,
                        "uniqueItems": True,
                        "items": {"type": "integer"},
                    },
                },
            },
        },
        "scene_summary": {"type": "string", "maxLength": PROFILE_MAX_SCENE_SUMMARY_CHARACTERS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_sample_indices": {
            "type": "array",
            "maxItems": PROFILE_MAX_ROOT_EVIDENCE_INDICES,
            "uniqueItems": True,
            "items": {"type": "integer"},
        },
    },
}
PROFILE_SEGMENT_SCHEMA: dict[str, Any] = deepcopy(PROFILE_SCHEMA)
PROFILE_SEGMENT_SCHEMA["properties"]["target_mode"]["enum"] = ["none", "individual", "group"]
PROFILE_SEGMENT_SCHEMA["properties"]["selected_subjects"]["minItems"] = 0


def _profile_evidence_indices(
    value: Any,
    *,
    allowed_indices: set[int],
    label: str,
    ordering_violation_code: str,
    supplied_violation_code: str,
    require_nonempty: bool = False,
    maximum_items: int = PROFILE_MAX_ROOT_EVIDENCE_INDICES,
) -> list[int]:
    if not isinstance(value, list) or any((type(item) is not int for item in value)):
        raise ValueError(f"{label} must be an integer array")
    if require_nonempty and (not value):
        raise ValueError(f"{label} must not be empty")
    if len(value) > maximum_items:
        raise ValueError(f"{label} has too many items")
    if value != sorted(set(value)):
        raise TaskContractViolation(
            ordering_violation_code, f"{label} must be strictly increasing and unique"
        )
    if not set(value) <= allowed_indices:
        raise TaskContractViolation(
            supplied_violation_code, f"{label} cites a sample that was not supplied"
        )
    return list(value)


def _profile_text(
    value: Any, *, label: str, maximum_characters: int, require_nonempty: bool = True
) -> str:
    if not isinstance(value, str) or (require_nonempty and (not value.strip())):
        raise ValueError(f"{label} is required")
    if len(value) > maximum_characters:
        raise ValueError(f"{label} is too long")
    return value


def validate_profile_response(
    value: Any,
    allowed_indices: set[int],
    *,
    allow_empty: bool = False,
    require_nested_evidence_in_root: bool = True,
) -> ProfileValidationResult:
    if not isinstance(value, Mapping):
        raise ValueError("profile response must be an object")
    required_profile_fields = {
        "target_mode",
        "selected_subjects",
        "ignored_transient_people",
        "annotation_relevance_spans",
        "scene_summary",
        "confidence",
        "evidence_sample_indices",
    }
    if set(value) != required_profile_fields:
        raise ValueError("profile response has an invalid field set")
    mode = value.get("target_mode")
    subjects = value.get("selected_subjects")
    allowed_modes = {"none", "individual", "group"} if allow_empty else {"individual", "group"}
    if mode not in allowed_modes or not isinstance(subjects, list):
        raise ValueError("profile must select an individual or non-empty group")
    if len(subjects) > PROFILE_MAX_SELECTED_SUBJECTS:
        raise ValueError("profile selected_subjects has too many items")
    evidence = _profile_evidence_indices(
        value.get("evidence_sample_indices"),
        allowed_indices=allowed_indices,
        label="profile evidence_sample_indices",
        ordering_violation_code="profile.root_evidence_not_ordered",
        supplied_violation_code="profile.root_evidence_not_supplied",
    )
    normalization_events: list[dict[str, Any]] = []
    if allow_empty:
        normalized_mode = (
            "none" if not subjects else "individual" if len(subjects) == 1 else "group"
        )
        if normalized_mode != mode:
            normalization_events.append(
                {
                    "policy_version": PROFILE_NORMALIZATION_VERSION,
                    "event": "target_mode_derived_from_subject_cardinality",
                    "path": "$.target_mode",
                    "raw_target_mode": mode,
                    "canonical_target_mode": normalized_mode,
                }
            )
    else:
        normalized_mode = mode
        if (mode == "none") != (not subjects):
            raise TaskContractViolation(
                "profile.target_mode_cardinality",
                "profile target_mode and selected_subjects disagree",
            )
        if mode == "individual" and len(subjects) != 1:
            raise TaskContractViolation(
                "profile.target_mode_cardinality",
                "individual profile mode must select exactly one subject",
            )
        if mode == "group" and len(subjects) < 2:
            raise TaskContractViolation(
                "profile.target_mode_cardinality",
                "group profile mode must select at least two subjects",
            )
        if not subjects:
            raise ValueError("profile must select an individual or non-empty group")
    normalized_subjects: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    required_subject_fields = {
        "subject_id",
        "visual_description",
        "identity_cues",
        "recurring_views",
        "typical_rest_postures",
        "selection_reason",
        "prevalence",
    }
    for subject in subjects:
        if not isinstance(subject, Mapping):
            raise ValueError("selected subject must be an object")
        if set(subject) != required_subject_fields:
            raise ValueError("selected subject has an invalid field set")
        identifier = subject.get("subject_id")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or len(identifier) > PROFILE_MAX_IDENTIFIER_CHARACTERS
        ):
            raise ValueError("selected subject identifiers must be non-empty and unique")
        if identifier in identifiers:
            raise TaskContractViolation(
                "profile.subject_id_not_unique",
                "selected subject identifiers must be non-empty and unique",
            )
        identifiers.add(identifier)
        prevalence = subject.get("prevalence")
        if (
            not isinstance(prevalence, (int, float))
            or isinstance(prevalence, bool)
            or (not 0 <= prevalence <= 1)
        ):
            raise ValueError("subject prevalence must be in [0, 1]")
        for field in ("visual_description", "selection_reason"):
            _profile_text(
                subject.get(field),
                label=f"selected subject {field}",
                maximum_characters=PROFILE_MAX_DESCRIPTION_CHARACTERS,
            )
        for field, maximum_items in (
            ("identity_cues", PROFILE_MAX_IDENTITY_CUES),
            ("recurring_views", PROFILE_MAX_RECURRING_VIEWS),
        ):
            items = subject.get(field)
            if not isinstance(items, list) or any(
                (
                    not isinstance(item, str)
                    or not item.strip()
                    or len(item) > PROFILE_MAX_LABEL_CHARACTERS
                    for item in items
                )
            ):
                raise ValueError(f"selected subject {field} must be a non-empty-string array")
            if len(items) > maximum_items:
                raise ValueError(f"selected subject {field} has too many items")
        normalized_subject = dict(subject)
        normalized_subject["typical_rest_postures"] = []
        normalized_subjects.append(normalized_subject)
    confidence = value.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or (not 0 <= confidence <= 1)
    ):
        raise ValueError("profile confidence must be in [0, 1]")
    ignored = value.get("ignored_transient_people")
    if not isinstance(ignored, list) or any((not isinstance(item, Mapping) for item in ignored)):
        raise ValueError("ignored_transient_people must be an object array")
    if len(ignored) > PROFILE_MAX_IGNORED_TRANSIENT_PEOPLE:
        raise ValueError("ignored_transient_people has too many items")
    normalized_ignored: list[dict[str, str]] = []
    for item in ignored:
        if set(item) != {"description", "reason"}:
            raise ValueError("ignored transient person has an invalid field set")
        normalized_ignored.append(
            {
                "description": _profile_text(
                    item.get("description"),
                    label="ignored transient person description",
                    maximum_characters=PROFILE_MAX_DESCRIPTION_CHARACTERS,
                ),
                "reason": _profile_text(
                    item.get("reason"),
                    label="ignored transient person reason",
                    maximum_characters=PROFILE_MAX_DESCRIPTION_CHARACTERS,
                ),
            }
        )
    _profile_text(
        value.get("scene_summary"),
        label="profile scene_summary",
        maximum_characters=PROFILE_MAX_SCENE_SUMMARY_CHARACTERS,
        require_nonempty=False,
    )
    normalized_relevance, relevance_normalization_events = ([], [])
    normalization_events.extend(relevance_normalization_events)
    result = dict(value)
    result["target_mode"] = normalized_mode
    result["selected_subjects"] = normalized_subjects
    result["ignored_transient_people"] = normalized_ignored
    result["annotation_relevance_spans"] = normalized_relevance
    result["evidence_sample_indices"] = evidence
    return ProfileValidationResult(result, normalization_events=normalization_events)
