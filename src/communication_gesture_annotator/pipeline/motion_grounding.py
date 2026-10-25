# SPDX-License-Identifier: Apache-2.0
"""Versioned motion baseline and explicit contextual log-odds adjustment."""

from __future__ import annotations
from typing import Any, Mapping


def evidence_motion_record(evidence: Any) -> Mapping[str, Any] | None:
    """Read the typed additive audit record from domain objects or saved JSON."""
    records = (
        evidence.get("resolution_records", ())
        if isinstance(evidence, Mapping)
        else evidence.resolution_records
    )
    return next(
        (
            r["motion_grounding"]
            for r in reversed(records)
            if r.get("policy") == "single-pass-non-g-probability-v0.1" and "motion_grounding" in r
        ),
        None,
    )
