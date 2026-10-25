# SPDX-License-Identifier: Apache-2.0
"""Bound dense visibility to the selected targets and their temporal context."""

from __future__ import annotations
from typing import Any, Mapping
from ..config import ResearchConfig

VERSION = "selected-targets-and-visibility-context-v0.1"
NOT_ASSESSED = "outside_requested_visibility_context"


def visibility_scope(
    config: ResearchConfig, sample_count: int, *, processing: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    from .probability_pass import schedule_metadata, selected_span

    start, end = (0, sample_count)
    history = lookahead = 0
    start, end = selected_span(config.detection, sample_count)
    schedule = schedule_metadata(config.detection)
    history = schedule["history_us"] // config.analysis.sample_period_us
    lookahead = schedule["lookahead_us"] // config.analysis.sample_period_us
    first, stop = (max(0, start - history), min(sample_count, end + lookahead))
    if processing is not None:
        first, stop = (
            max(first, processing["sample_span"][0]),
            min(stop, processing["sample_span"][1]),
        )
    return {
        "policy_version": VERSION,
        "selected_sample_span": [start, end],
        "assessed_sample_span": [first, stop],
        "assessed_sample_count": stop - first,
        "not_assessed_sample_count": sample_count - (stop - first),
        "requested_history_us": history * config.analysis.sample_period_us,
        "requested_lookahead_us": lookahead * config.analysis.sample_period_us,
        "chronological_frames": "strict whole-program processing range; context shortened at selection edges"
        if processing
        else "selected interval plus required Phase2 history and lookahead; clipped to recording",
        "reference_exemplars": "separately labelled available identity references; not chronological context",
        **({"processing_scope": dict(processing)} if processing else {}),
    }


def unassessed_visibility_sample(index: int) -> dict[str, Any]:
    """A local coverage placeholder, never a model judgment or probability."""
    return {
        "sample_index": index,
        "assessment_status": "not_assessed",
        "scene_membership": "uncertain",
        "scene_type": "uncertain",
        "target_visibility": "uncertain",
        "left_hand_usability": "uncertain",
        "right_hand_usability": "uncertain",
        "view": None,
        "relevance": "uncertain",
        "probabilities": None,
        "reason": NOT_ASSESSED,
        "request_id": None,
        "response_sha256": None,
    }
