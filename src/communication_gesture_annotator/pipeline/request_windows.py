# SPDX-License-Identifier: Apache-2.0
"""Deterministic overlapping request windows with non-overlapping output ownership."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RequestWindow:
    """One model input window and the subset whose output it owns."""

    sequence_index: int
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.sequence_index < 0:
            raise ValueError("request-window sequence index must be non-negative")
        if not self.input_indices:
            raise ValueError("request window must contain input indices")
        if tuple(sorted(set(self.input_indices))) != self.input_indices:
            raise ValueError("request-window input indices must be sorted and unique")
        if tuple(sorted(set(self.output_indices))) != self.output_indices:
            raise ValueError("request-window output indices must be sorted and unique")
        if not set(self.output_indices).issubset(self.input_indices):
            raise ValueError("request-window output must be contained in its input")


def fixed_period_indices(
    sample_count: int, *, sample_period_us: int, selection_period_us: int
) -> tuple[int, ...]:
    """Select globally anchored samples for a lower-rate inference view."""
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(sample_period_us) is not int or sample_period_us <= 0:
        raise ValueError("sample_period_us must be a positive integer")
    if type(selection_period_us) is not int or selection_period_us <= 0:
        raise ValueError("selection_period_us must be a positive integer")
    if selection_period_us % sample_period_us:
        raise ValueError("selection period must align to the canonical sample grid")
    step = selection_period_us // sample_period_us
    return tuple(range(0, sample_count, step))


def plan_index_windows(
    indices: Sequence[int], *, items_per_window: int, overlap_items: int
) -> tuple[RequestWindow, ...]:
    """Plan full-sized overlapping inputs and partition their shared outputs.

    The last input is anchored at the end.  Boundaries through overlapping
    inputs are split at the overlap midpoint, so every supplied index is owned
    by exactly one request while remaining visible as context in its neighbor.
    """
    ordered = tuple(indices)
    if not ordered:
        raise ValueError("indices cannot be empty")
    if tuple(sorted(set(ordered))) != ordered:
        raise ValueError("indices must be sorted and unique")
    if type(items_per_window) is not int or items_per_window <= 0:
        raise ValueError("items_per_window must be a positive integer")
    if type(overlap_items) is not int or not 0 <= overlap_items < items_per_window:
        raise ValueError("overlap_items must be in [0, items_per_window)")
    count = len(ordered)
    if count <= items_per_window:
        return (RequestWindow(0, ordered, ordered),)
    stride = items_per_window - overlap_items
    starts = list(range(0, count - items_per_window + 1, stride))
    final_start = count - items_per_window
    if starts[-1] != final_start:
        starts.append(final_start)
    input_positions = [(start, min(count, start + items_per_window)) for start in starts]
    ownership_boundaries = [0]
    for previous, following in zip(input_positions, input_positions[1:]):
        overlap_start = following[0]
        overlap_end = previous[1]
        if overlap_end <= overlap_start:
            boundary = overlap_start
        else:
            boundary = overlap_start + (overlap_end - overlap_start) // 2
        ownership_boundaries.append(boundary)
    ownership_boundaries.append(count)
    windows: list[RequestWindow] = []
    for sequence, ((start, end), owned_start, owned_end) in enumerate(
        zip(input_positions, ownership_boundaries[:-1], ownership_boundaries[1:], strict=True)
    ):
        inputs = ordered[start:end]
        outputs = ordered[owned_start:owned_end]
        windows.append(RequestWindow(sequence, inputs, outputs))
    flattened = tuple((index for window in windows for index in window.output_indices))
    if flattened != ordered:
        raise RuntimeError("request-window ownership did not partition the input indices")
    return tuple(windows)


def plan_target_windows(
    available: Sequence[int],
    *,
    target_indices: Sequence[int],
    target_items: int,
    history_items: int,
    lookahead_items: int,
) -> tuple[RequestWindow, ...]:
    """Score short consecutive targets with asymmetric, clipped context.

    Both sequences must belong to one continuous admitted interval. Context may
    extend outside a user-selected output range, never outside admission. Missing
    history is not replaced by extra future frames (or vice versa).
    """
    support, targets = (tuple(available), tuple(target_indices))
    for name, value, minimum in (
        ("target_items", target_items, 1),
        ("history_items", history_items, 0),
        ("lookahead_items", lookahead_items, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not support or support != tuple(range(support[0], support[-1] + 1)):
        raise ValueError("available target context must be contiguous")
    if not targets:
        return ()
    if targets != tuple(range(targets[0], targets[-1] + 1)) or not set(targets).issubset(support):
        raise ValueError("targets must be contiguous and inside available context")
    windows = []
    for offset in range(0, len(targets), target_items):
        core = targets[offset : offset + target_items]
        context = tuple(
            range(
                max(support[0], core[0] - history_items),
                min(support[-1] + 1, core[-1] + 1 + lookahead_items),
            )
        )
        windows.append(RequestWindow(len(windows), context, core))
    return tuple(windows)
