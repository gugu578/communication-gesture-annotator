# SPDX-License-Identifier: Apache-2.0
"""Keep future edits and uncertain post-edit estimates out of target evidence."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .request_windows import RequestWindow, plan_target_windows

VERSION = "camera-bounded-future-and-supported-history-v0.1"


def shot_ranges(indices: Sequence[int], resolutions: Mapping[int, Mapping[str, Any]]) -> dict[int, tuple[int, int]]:
    """Camera geometry ranges within one already admitted speaker context."""
    result = {}
    begin = 0
    for end in range(1, len(indices)+1):
        if end < len(indices) and resolutions[indices[end]].get('camera_shot_id') == resolutions[indices[end-1]].get('camera_shot_id'):
            continue
        for i in indices[begin:end]:
            result[i] = (indices[begin], indices[end-1]+1)
        begin = end
    return result


def plan_camera_windows(available: Sequence[int], *, target_indices: Sequence[int],
                        target_items: int, history_items: int, lookahead_items: int,
                        ranges: Mapping[int, tuple[int, int]]) -> tuple[RequestWindow, ...]:
    """Split ownership at image-view changes, keep preceding RGB, bound future.

    This never changes output eligibility, rest thresholds or final grouping.
    Source-native cuts are mapped to the view of each delivered analysis image.
    """
    targets = list(target_indices)
    groups: list[list[int]] = []
    for index in targets:
        if not groups or ranges[index] != ranges[groups[-1][-1]]:
            groups.append([])
        groups[-1].append(index)
    result = []
    for group in groups:
        for window in plan_target_windows(available, target_indices=group, target_items=target_items,
                                          history_items=history_items, lookahead_items=lookahead_items):
            end = ranges[window.output_indices[0]][1]
            result.append(RequestWindow(len(result), tuple(i for i in window.input_indices if i < end), window.output_indices))
    return tuple(result)


def supported_estimates(estimates: Mapping[int, float], *, core: Sequence[int],
                        context: Sequence[int], ranges: Mapping[int, tuple[int, int]],
                        resolutions: Mapping[int, Mapping[str, Any]], history_items: int,
                        eligible_estimates: set[int]) -> tuple[dict[int, float], dict[str, Any]]:
    """A short-history answer is still an output, but cannot bootstrap memory.

    Retain the actual RGB history. Do not use automatic motion scores to reject
    priors: quiet communicative holds legitimately disagree with physical rest.
    """
    known_shot = resolutions[core[0]].get('camera_shot_id') is not None
    start = ranges[core[0]][0]
    selected = {i: v for i, v in estimates.items() if not known_shot or (i >= start and i in eligible_estimates)}
    supported = not known_shot or core[0]-max(start, context[0]) >= history_items
    metadata = {'policy_version': VERSION, 'camera_boundary_available': known_shot,
                'shot_context_start_sample': start, 'configured_history_frames': history_items,
                'current_output_eligible_for_future_estimates': supported,
                'withheld_prior_count': sum(i in context and i < core[0] and i not in selected for i in estimates)}
    return selected, metadata
