# SPDX-License-Identifier: Apache-2.0
"""Processing bounds on the original timeline; no source-time rebasing."""

from __future__ import annotations
from typing import Any, Mapping, Sequence

OUTSIDE = "outside_processing_range"
VERSION = "whole-program-time-range-v0.1"


def processing_scope(
    duration_us: int, start_us: int = 0, end_us: int | None = None, *, period_us: int = 100000
) -> dict[str, Any]:
    if end_us is not None and end_us > (duration_us + period_us - 1) // period_us * period_us:
        raise ValueError("processing range end exceeds the video analysis grid")
    end = min(duration_us, duration_us if end_us is None else end_us)
    if start_us < 0 or start_us >= end or start_us % period_us:
        raise ValueError("processing range must contain video and start on the analysis grid")
    if end != duration_us and end % period_us:
        raise ValueError("processing range must end on the analysis grid")
    return {
        "policy_version": VERSION,
        "start_us": start_us,
        "end_us": end,
        "sample_span": [start_us // period_us, (end + period_us - 1) // period_us],
        "timestamps": "original video timeline; excluded intervals contain no media or pose evidence",
    }


def apply_scope(
    grid: Sequence[Mapping[str, Any]], scope: Mapping[str, Any]
) -> list[dict[str, Any]]:
    first, stop = scope["sample_span"]
    return [
        dict(s) if first <= s["index"] < stop else dict(s, source_gap_reason=OUTSIDE) for s in grid
    ]


def excluded_observation(frame: Mapping[str, Any], width: int, height: int) -> dict[str, Any]:
    """An explicit unprocessed row, never a claim that pose inference ran."""
    from .pose import POSE_CONTRACT_VERSION

    return {
        "schema_version": POSE_CONTRACT_VERSION,
        "source_frame": {
            k: v
            for k, v in frame.items()
            if k not in {"source_frame_index", "analysis_sample_indices"}
        },
        **{
            k: frame.get(k)
            for k in (
                "source_frame_index",
                "decode_index",
                "pts",
                "best_effort_timestamp",
                "presentation_timestamp",
                "timestamp_source",
                "time_base",
                "canonical_time_us",
                "canonical_end_us",
                "endpoint_basis",
                "analysis_sample_indices",
            )
        },
        "source_frame_id": frame["frame_id"],
        "pose_inference_attempted": False,
        "processing_status": OUTSIDE,
        "pose_evidence_available": False,
        "people": [],
        "decoded_frame": {
            "width": width,
            "height": height,
            "sample_aspect_ratio": {"numerator": 1, "denominator": 1},
            "dimensions_only": True,
        },
    }
