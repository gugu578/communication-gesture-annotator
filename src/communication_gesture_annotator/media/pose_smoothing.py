# SPDX-License-Identifier: Apache-2.0
"""Pure timestamp-aware smoothing primitives for scalar pose coordinates."""

from __future__ import annotations
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import math
from numbers import Real
from statistics import median
from typing import Literal, Sequence

DEFAULT_HALF_WINDOW_US = 100000
DEFAULT_POLYNOMIAL_DEGREE = 2
DEFAULT_MINIMUM_SUPPORT = 5
DEFAULT_MINIMUM_SIDE_SUPPORT = 2
_TUKEY_TUNING_CONSTANT = 4.685
_SINGULAR_RELATIVE_TOLERANCE = 1e-12
PoseSmoothingStatus = Literal[
    "smoothed",
    "raw_fallback_insufficient_support",
    "raw_fallback_asymmetric_support",
    "raw_fallback_singular_fit",
]


@dataclass(frozen=True, slots=True)
class PoseSmoothingEstimate:
    """One filtered scalar value and the evidence supporting it.

    ``support_count`` and the side counts include observations with positive
    clipped confidence and positive tricube weight. ``outlier_count`` counts
    observations rejected by the single Tukey bisquare reweighting pass.
    ``edge_asymmetric`` means that either side lacks the configured minimum
    support; unequal but sufficient side counts are valid for irregular PTS.
    """

    value: float
    status: PoseSmoothingStatus
    support_count: int
    left_support_count: int
    right_support_count: int
    outlier_count: int
    edge_asymmetric: bool


def robust_local_polynomial_series(
    timestamps_us: Sequence[int],
    values: Sequence[float],
    confidences: Sequence[float],
    *,
    half_window_us: int = DEFAULT_HALF_WINDOW_US,
    degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    minimum_support: int = DEFAULT_MINIMUM_SUPPORT,
    minimum_side_support: int = DEFAULT_MINIMUM_SIDE_SUPPORT,
) -> list[tuple[PoseSmoothingEstimate, int, int]]:
    """Estimate every observation using exact local support after full validation.

    The public scalar estimator remains the numerical reference. Boundary points
    have zero weight, so strict bisection preserves its support and arithmetic.
    Returned bounds include all local observations, also zero-confidence ones.
    """
    if not timestamps_us:
        raise ValueError("timestamps_us must not be empty")
    ts = _validate_timestamps(timestamps_us, timestamps_us[0])
    vs = _validate_finite_scalars(values, "values")
    cs = _validate_finite_scalars(confidences, "confidences")
    if not len(ts) == len(vs) == len(cs):
        raise ValueError("timestamps, values, and confidences must have equal length")
    _validate_parameters(
        center_time_us=ts[0],
        half_window_us=half_window_us,
        degree=degree,
        minimum_support=minimum_support,
        minimum_side_support=minimum_side_support,
    )
    result = []
    for center in ts:
        first = bisect_right(ts, center - half_window_us)
        last = bisect_left(ts, center + half_window_us)
        estimate = robust_local_polynomial_estimate(
            ts[first:last],
            vs[first:last],
            cs[first:last],
            center,
            half_window_us=half_window_us,
            degree=degree,
            minimum_support=minimum_support,
            minimum_side_support=minimum_side_support,
        )
        result.append((estimate, ts[first], ts[last - 1]))
    return result


def robust_local_polynomial_estimate(
    timestamps_us: Sequence[int],
    values: Sequence[float],
    confidences: Sequence[float],
    center_time_us: int,
    *,
    half_window_us: int = DEFAULT_HALF_WINDOW_US,
    degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    minimum_support: int = DEFAULT_MINIMUM_SUPPORT,
    minimum_side_support: int = DEFAULT_MINIMUM_SIDE_SUPPORT,
) -> PoseSmoothingEstimate:
    """Estimate one pose coordinate with a robust local polynomial.

    Timestamps remain exact integer microseconds for validation, selection,
    and side accounting. Time is normalized only inside the small weighted
    least-squares system. The target timestamp must occur exactly once so that
    every conservative fallback can return its unmodified input value.

    A local-constant pilot uses tricube time weights multiplied by confidence
    clipped to ``[0, 1]``. One Tukey bisquare residual reweight is then applied
    before the requested polynomial fit. The lower-order pilot prevents a
    single high-leverage target spike from bending a quadratic enough to mask
    its own residual. Samples exactly on the window boundary have zero tricube
    weight.
    """
    _validate_parameters(
        center_time_us=center_time_us,
        half_window_us=half_window_us,
        degree=degree,
        minimum_support=minimum_support,
        minimum_side_support=minimum_side_support,
    )
    checked_timestamps = _validate_timestamps(timestamps_us, center_time_us)
    checked_values = _validate_finite_scalars(values, "values")
    checked_confidences = _validate_finite_scalars(confidences, "confidences")
    if not len(checked_timestamps) == len(checked_values) == len(checked_confidences):
        raise ValueError("timestamps, values, and confidences must have equal length")
    center_index = checked_timestamps.index(center_time_us)
    raw_value = checked_values[center_index]
    support: list[tuple[float, float, float]] = []
    left_support_count = 0
    right_support_count = 0
    for timestamp_us, value, confidence in zip(
        checked_timestamps, checked_values, checked_confidences, strict=True
    ):
        delta_us = timestamp_us - center_time_us
        distance_ratio = abs(delta_us) / half_window_us
        if distance_ratio >= 1.0:
            continue
        time_weight = (1.0 - distance_ratio**3) ** 3
        confidence_weight = min(1.0, max(0.0, confidence))
        weight = time_weight * confidence_weight
        if weight <= 0.0:
            continue
        normalized_time = delta_us / half_window_us
        support.append((normalized_time, value, weight))
        if delta_us < 0:
            left_support_count += 1
        elif delta_us > 0:
            right_support_count += 1
    support_count = len(support)
    edge_asymmetric = (
        left_support_count < minimum_side_support or right_support_count < minimum_side_support
    )
    if support_count < minimum_support or support_count <= degree:
        return _fallback(
            raw_value,
            "raw_fallback_insufficient_support",
            support_count,
            left_support_count,
            right_support_count,
            edge_asymmetric,
        )
    if edge_asymmetric:
        return _fallback(
            raw_value,
            "raw_fallback_asymmetric_support",
            support_count,
            left_support_count,
            right_support_count,
            edge_asymmetric,
        )
    pilot_coefficients = _weighted_polynomial_fit(support, 0)
    if pilot_coefficients is None:
        return _fallback(
            raw_value,
            "raw_fallback_singular_fit",
            support_count,
            left_support_count,
            right_support_count,
            edge_asymmetric,
        )
    residuals = [
        value - _evaluate_polynomial(pilot_coefficients, normalized_time)
        for normalized_time, value, _weight in support
    ]
    robust_weights, outlier_count = _tukey_bisquare_weights(
        residuals, tuple((value for _normalized_time, value, _weight in support))
    )
    reweighted_support = [
        (normalized_time, value, weight * robust_weight)
        for (normalized_time, value, weight), robust_weight in zip(
            support, robust_weights, strict=True
        )
        if weight * robust_weight > 0.0
    ]
    final_coefficients = _weighted_polynomial_fit(reweighted_support, degree)
    if final_coefficients is None:
        return _fallback(
            raw_value,
            "raw_fallback_singular_fit",
            support_count,
            left_support_count,
            right_support_count,
            edge_asymmetric,
            outlier_count=outlier_count,
        )
    estimate = final_coefficients[0]
    if not math.isfinite(estimate):
        return _fallback(
            raw_value,
            "raw_fallback_singular_fit",
            support_count,
            left_support_count,
            right_support_count,
            edge_asymmetric,
            outlier_count=outlier_count,
        )
    return PoseSmoothingEstimate(
        value=estimate,
        status="smoothed",
        support_count=support_count,
        left_support_count=left_support_count,
        right_support_count=right_support_count,
        outlier_count=outlier_count,
        edge_asymmetric=False,
    )


def _validate_parameters(
    *,
    center_time_us: int,
    half_window_us: int,
    degree: int,
    minimum_support: int,
    minimum_side_support: int,
) -> None:
    for name, value in (
        ("center_time_us", center_time_us),
        ("half_window_us", half_window_us),
        ("degree", degree),
        ("minimum_support", minimum_support),
        ("minimum_side_support", minimum_side_support),
    ):
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
    if half_window_us <= 0:
        raise ValueError("half_window_us must be positive")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if minimum_support <= 0:
        raise ValueError("minimum_support must be positive")
    if minimum_side_support < 0:
        raise ValueError("minimum_side_support must be non-negative")


def _validate_timestamps(timestamps_us: Sequence[int], center_time_us: int) -> tuple[int, ...]:
    checked = tuple(timestamps_us)
    if not checked:
        raise ValueError("timestamps_us must not be empty")
    if any((type(timestamp_us) is not int for timestamp_us in checked)):
        raise ValueError("timestamps_us must contain integer microseconds")
    if any((current <= previous for previous, current in zip(checked, checked[1:]))):
        raise ValueError("timestamps_us must be strictly increasing")
    if center_time_us not in checked:
        raise ValueError("center_time_us must identify one input observation")
    return checked


def _validate_finite_scalars(values: Sequence[float], name: str) -> tuple[float, ...]:
    checked: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain finite real scalars")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must contain finite real scalars")
        checked.append(converted)
    return tuple(checked)


def _fallback(
    raw_value: float,
    status: PoseSmoothingStatus,
    support_count: int,
    left_support_count: int,
    right_support_count: int,
    edge_asymmetric: bool,
    *,
    outlier_count: int = 0,
) -> PoseSmoothingEstimate:
    return PoseSmoothingEstimate(
        value=raw_value,
        status=status,
        support_count=support_count,
        left_support_count=left_support_count,
        right_support_count=right_support_count,
        outlier_count=outlier_count,
        edge_asymmetric=edge_asymmetric,
    )


def _weighted_polynomial_fit(
    support: Sequence[tuple[float, float, float]], degree: int
) -> tuple[float, ...] | None:
    size = degree + 1
    if len(support) < size:
        return None
    matrix = [
        [
            math.fsum(
                (
                    weight * normalized_time ** (row + column)
                    for normalized_time, _value, weight in support
                )
            )
            for column in range(size)
        ]
        for row in range(size)
    ]
    right_hand_side = [
        math.fsum(
            (weight * value * normalized_time**row for normalized_time, value, weight in support)
        )
        for row in range(size)
    ]
    return _solve_linear_system(matrix, right_hand_side)


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], right_hand_side: Sequence[float]
) -> tuple[float, ...] | None:
    size = len(matrix)
    augmented = [
        [*map(float, row), float(right_hand_side[index])] for index, row in enumerate(matrix)
    ]
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    if scale == 0.0 or not math.isfinite(scale):
        return None
    tolerance = scale * _SINGULAR_RELATIVE_TOLERANCE
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        pivot = augmented[pivot_row][column]
        if not math.isfinite(pivot) or abs(pivot) <= tolerance:
            return None
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (augmented[pivot_row], augmented[column])
        pivot = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= pivot
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for entry in range(column, size + 1):
                augmented[row][entry] -= factor * augmented[column][entry]
    solution = tuple((augmented[row][size] for row in range(size)))
    if not all((math.isfinite(value) for value in solution)):
        return None
    return solution


def _evaluate_polynomial(coefficients: Sequence[float], normalized_time: float) -> float:
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * normalized_time + coefficient
    return result


def _tukey_bisquare_weights(
    residuals: Sequence[float], source_values: Sequence[float]
) -> tuple[tuple[float, ...], int]:
    residual_median = median(residuals)
    absolute_deviations = [abs(residual - residual_median) for residual in residuals]
    mad = median(absolute_deviations)
    value_scale = max(1.0, *(abs(value) for value in source_values))
    numerical_tolerance = value_scale * _SINGULAR_RELATIVE_TOLERANCE
    if mad <= numerical_tolerance:
        weights = tuple(
            (1.0 if deviation <= numerical_tolerance else 0.0 for deviation in absolute_deviations)
        )
        return (weights, sum((weight == 0.0 for weight in weights)))
    robust_scale = 1.4826 * mad
    cutoff = _TUKEY_TUNING_CONSTANT * robust_scale
    weights_list: list[float] = []
    for deviation in absolute_deviations:
        ratio = deviation / cutoff
        if ratio >= 1.0:
            weights_list.append(0.0)
        else:
            weights_list.append((1.0 - ratio * ratio) ** 2)
    weights = tuple(weights_list)
    return (weights, sum((weight == 0.0 for weight in weights)))
