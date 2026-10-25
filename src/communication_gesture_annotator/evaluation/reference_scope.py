# SPDX-License-Identifier: Apache-2.0
"""Reference annotation coverage, separate from visible/hidden classification."""

from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
import warnings

POLICY_VERSION = "labelled-reference-visibility-scope-v1"
DEFAULT_VISIBILITY_TIER = "Spker_vis"
T = TypeVar("T")


@dataclass(frozen=True)
class ReferenceScope:
    """Only nonblank, aligned visibility annotations establish assessed time."""

    intervals: tuple[tuple[int, int], ...]
    tier_id: str = DEFAULT_VISIBILITY_TIER
    tier_present: bool = True

    @classmethod
    def from_annotations(cls, annotations: Sequence[Any], *, tier_id: str = DEFAULT_VISIBILITY_TIER) -> "ReferenceScope":
        merged: list[tuple[int, int]] = []
        for item in sorted(annotations, key=lambda item: item.start_us):
            if not item.annotation_value.strip():
                continue
            start, end = item.start_us, item.end_us
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return cls(tuple(merged), tier_id)

    def contains(self, time_us: int) -> bool:
        index = bisect_right(self.intervals, (time_us, float("inf"))) - 1
        return index >= 0 and time_us < self.intervals[index][1]

    def covers(self, start_us: int, end_us: int) -> bool:
        """Motion summaries must be wholly inside assessed time."""
        index = bisect_right(self.intervals, (start_us, float("inf"))) - 1
        return index >= 0 and end_us <= self.intervals[index][1]

    def intersect(self, intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(max(start, left), min(end, right))
                for start, end in intervals for left, right in self.intervals
                if max(start, left) < min(end, right)]

    def gaps(self, start_us: int, end_us: int) -> list[tuple[int, int]]:
        result = []
        cursor = start_us
        for start, end in self.intersect([(start_us, end_us)]):
            if cursor < start:
                result.append((cursor, start))
            cursor = end
        if cursor < end_us:
            result.append((cursor, end_us))
        return result


class ScopedReference(list[T]):
    """Keep reference coverage with intervals without inventing negative labels."""

    def __init__(self, items: Iterable[T], scope: ReferenceScope) -> None:
        super().__init__(items)
        self.assessment_scope = scope


def reference_assessed(time_us: int, reference: Sequence[Any]) -> bool:
    scope = getattr(reference, "assessment_scope", None)
    return scope is None or scope.contains(time_us)


def read_reference_scope(
    path: str | Path, reader: Callable[..., tuple[Path, Sequence[Any]]], *,
    tier_id: str = DEFAULT_VISIBILITY_TIER, alignment_offset_us: int = 0,
) -> ReferenceScope:
    try:
        _, annotations = reader(path, tier_id=tier_id, alignment_offset_us=alignment_offset_us)
    except ValueError as exc:
        if str(exc) != f"reference EAF tier {tier_id!r} does not exist":
            raise
        warnings.warn(
            f"Phase1 cannot be scored: reference visibility tier {tier_id!r} is missing; "
            "no reference-assessed time is available.", stacklevel=2,
        )
        return ReferenceScope((), tier_id, False)
    return ReferenceScope.from_annotations(annotations, tier_id=tier_id)
