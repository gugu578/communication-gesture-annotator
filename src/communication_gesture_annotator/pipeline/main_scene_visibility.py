# SPDX-License-Identifier: Apache-2.0
"""Small per-frame scene decisions, independent of manual activity assessment."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
from .llm_contracts import TaskContractViolation

TASK = "main_scene_visibility"
PROMPT_VERSION = "main-scene-per-frame-v0.2"
PROJECTION_VERSION = "main-scene-compatibility-fields-v0.1"
SYSTEM_PROMPT = "You classify whether each CURRENT frame belongs to the selected speaker's main communication scene in Communication Gesture Annotator.\nCompare EACH CURRENT frame with the fixed selected-speaker description and the separately labelled REFERENCE main-scene exemplars. Return one Boolean main_scene decision for each requested global sample_index, in the supplied order.\nEach contact-sheet cell is one frame, labelled CURRENT f<global_sample_index>. A sheet is only a container. REFERENCE cells are non-temporal examples of the speaker and scene family, never current observations or output targets. Do not restart frame numbering at a sheet boundary. Judge every CURRENT frame individually, even when adjacent decisions differ; never use the majority of the sheet.\nmain_scene=true means the CURRENT frame shows the selected live speaker in the main communication setting. A different camera angle, close-up, crop, or lighting change can still belong to this scene. The exemplars are not an exhaustive list of valid views. The selected live speaker beside slides or in a split view counts as true. Hands need not be visible and no movement is required.\nmain_scene=false means the frame does not establish that live speaker in the main scene: for example a full-screen slide, audience/cutaway, another speaker, unrelated scene, or an embedded photograph/video. A person depicted inside a slide, screen or photograph is not the live selected speaker. Do not infer presence from a REFERENCE frame. If the current frame cannot establish main-scene membership, return false.\nDo not assess hand usability, rest, gestures, G-units, movement, camera-view labels or probabilities. Do not return scene runs or transition intervals. Return only the requested JSON object; no decisions for REFERENCE cells."


def response_schema(count: int) -> dict[str, Any]:
    if type(count) is not int or count < 1:
        raise ValueError("main-scene assessment requires at least one frame")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["samples"],
        "properties": {
            "samples": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sample_index", "main_scene"],
                    "properties": {
                        "sample_index": {"type": "integer", "minimum": 0},
                        "main_scene": {"type": "boolean"},
                    },
                },
            }
        },
    }


def validate_response(value: Mapping[str, Any], expected: Sequence[int]) -> dict[str, Any]:
    rows = value.get("samples") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != {"samples"}
        or (not isinstance(rows, list))
        or (len(rows) != len(expected))
        or any(
            (
                not isinstance(row, dict)
                or set(row) != {"sample_index", "main_scene"}
                or type(row["sample_index"]) is not int
                or (row["sample_index"] != index)
                or (type(row["main_scene"]) is not bool)
                for row, index in zip(rows, expected)
            )
        )
    ):
        raise TaskContractViolation(
            "main_scene_visibility.coverage_or_boolean",
            "main-scene decisions must cover the exact requested global frames once in order",
        )
    return {"samples": [dict(row) for row in rows]}


def project_response(value: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the Boolean to saved gate fields, without inventing hand observations.

    One-hot numbers encode the classification for saved admission records;
    they are explicitly not LLM confidence estimates. Unassessed hand fields
    stay unknown and cannot be used to establish a rest exemplar.
    """
    return {
        "samples": [
            {
                "sample_index": row["sample_index"],
                "main_scene": row["main_scene"],
                "assessment_kind": PROMPT_VERSION,
                "assessment_status": "assessed",
                "scene_membership": "main_scene" if row["main_scene"] else "other_scene",
                "scene_type": "uncertain",
                "target_visibility": "visible" if row["main_scene"] else "uncertain",
                "left_hand_usability": "uncertain",
                "right_hand_usability": "uncertain",
                "view": None,
                "relevance": "relevant" if row["main_scene"] else "irrelevant",
                "probabilities": {
                    "relevant": float(row["main_scene"]),
                    "irrelevant": float(not row["main_scene"]),
                    "uncertain": 0.0,
                },
                "probabilities_source": "binary_encoding_not_model_confidence",
                "reason": "per-frame main-scene membership; hand usability and view were not assessed",
            }
            for row in value["samples"]
        ]
    }
