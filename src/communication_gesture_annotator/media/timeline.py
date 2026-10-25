# SPDX-License-Identifier: Apache-2.0
"""Exact timestamp normalization and construction of the 10 Hz grid."""

from __future__ import annotations
from fractions import Fraction
from ..domain import AnalysisSample


def round_fraction_half_even(value: Fraction) -> int:
    """Round an exact rational to the nearest integer, ties to even."""
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder < Fraction(1, 2):
        return floor
    if remainder > Fraction(1, 2):
        return floor + 1
    return floor if floor % 2 == 0 else floor + 1


def normalize_pts_us(
    pts: int, time_base_numerator: int, time_base_denominator: int, origin_seconds: Fraction
) -> int:
    """Map a raw presentation timestamp to canonical integer microseconds."""
    if time_base_numerator <= 0 or time_base_denominator <= 0:
        raise ValueError("time base must be a positive rational")
    source_seconds = Fraction(pts * time_base_numerator, time_base_denominator)
    canonical = round_fraction_half_even((source_seconds - origin_seconds) * 1000000)
    if canonical < 0:
        raise ValueError("timestamp precedes the recorded global origin")
    return canonical


def build_analysis_grid(duration_us: int, sample_rate_hz: int = 10) -> list[AnalysisSample]:
    """Cover a recording with contiguous half-open 10 Hz analysis samples."""
    if duration_us <= 0:
        raise ValueError("duration_us must be positive")
    if sample_rate_hz != 10:
        raise ValueError("version 0.1 supports only a 10 Hz analysis grid")
    period_us = 1000000 // sample_rate_hz
    samples: list[AnalysisSample] = []
    start_us = 0
    index = 0
    while start_us < duration_us:
        end_us = min(duration_us, start_us + period_us)
        samples.append(
            AnalysisSample(
                index=index,
                start_us=start_us,
                end_us=end_us,
                center_us=start_us + (end_us - start_us) // 2,
                source_references=(f"proxy-frame:{index}",),
            )
        )
        index += 1
        start_us = end_us
    return samples
