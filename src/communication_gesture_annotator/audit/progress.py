# SPDX-License-Identifier: Apache-2.0
"""Sparse console progress for a stage's existing logical request plan."""

from __future__ import annotations

from collections.abc import Mapping
from time import monotonic
from typing import Any, Protocol


class EventWriter(Protocol):
    def log_event(
        self, stage: str, event: str, status: str, details: Mapping[str, Any] | None = None
    ) -> Any: ...


class StepProgress:
    """Observe work without changing requests, retries, checkpoints or timings.

    Counts include checkpoint replay, exclude transport/repair attempts, and are
    scoped to this step. Intermediate updates occur only after completed work,
    at most once a minute. They are not a background heartbeat or an ETA.
    """

    def __init__(
        self,
        trace: EventWriter | None,
        stage: str,
        planned_calls: int,
        *,
        model: Any = None,
        **details: Any,
    ) -> None:
        self.trace = trace
        self.stage = stage
        self.planned_calls = planned_calls
        self.completed_calls = 0
        self.model = model
        self.details = details
        self.started = self.last_report = monotonic()
        self.initial_reused = self._reused_count()
        self._emit("started")

    def _reused_count(self) -> int | None:
        count = getattr(self.model, "reused_count", None)
        return count if type(count) is int else None

    def _emit(self, status: str, **details: Any) -> None:
        if self.trace is None:
            return
        counts: dict[str, Any] = {
            "planned_calls": self.planned_calls,
            "completed_calls": self.completed_calls,
            "remaining_calls": max(0, self.planned_calls - self.completed_calls),
            "call_count_basis": "logical_requests_including_reuse",
        }
        reused = self._reused_count()
        if reused is not None and self.initial_reused is not None:
            counts["reused_calls"] = reused - self.initial_reused
        if status != "started":
            counts["wall_seconds"] = monotonic() - self.started
        self.trace.log_event(self.stage, "step", status, {**self.details, **counts, **details})

    def advance(self) -> None:
        self.completed_calls += 1
        now = monotonic()
        if self.completed_calls < self.planned_calls and now - self.last_report >= 60:
            self._emit("progress")
            self.last_report = now

    def finish(self, **details: Any) -> None:
        self._emit("completed", **details)
