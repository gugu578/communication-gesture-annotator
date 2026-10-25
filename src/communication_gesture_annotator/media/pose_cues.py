# SPDX-License-Identifier: Apache-2.0
"""Fixed, body-relative pose cues for diagnostic analysis.

The functions in this module are deliberately pure.  They do not fit a
recording baseline, cluster a person's postures, or choose thresholds from the
input video.  Coordinates are translated, rotated, and scaled independently
for each observation by a fixed trunk-frame construction.  Temporal motion is
then calculated *after* that transformation using exact presentation-time
deltas.

Nothing in this module is a rest/static classifier.  The resulting cues are
measurement evidence for analysis and future, explicitly versioned model
inputs.
"""

from __future__ import annotations
import math
from numbers import Real
from typing import Any, Sequence

BODY_RELATIVE_POSE_CUE_CONTRACT_VERSION = "0.1.0"
BODY_RELATIVE_POSE_CUE_POLICY_VERSION = "fixed-coco-trunk-frame-v0.1"
BODY_RELATIVE_MIN_KEYPOINT_CONFIDENCE = 0.0
BODY_RELATIVE_MIN_TORSO_LENGTH = 1e-06
BODY_RELATIVE_MIN_TRANSVERSE_SUPPORT_RATIO = 0.05
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8
_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_PALM_ANCHORS = (91, 96, 100, 104, 108)
_RIGHT_PALM_ANCHORS = (112, 117, 121, 125, 129)
_POINT_INDICES = {
    "left_wrist": _LEFT_WRIST,
    "right_wrist": _RIGHT_WRIST,
    "left_elbow": _LEFT_ELBOW,
    "right_elbow": _RIGHT_ELBOW,
}
_TEMPORAL_POINT_NAMES = (
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_palm_center",
    "right_palm_center",
)


def _is_real(value: Any) -> bool:
    return isinstance(value, Real) and (not isinstance(value, bool)) and math.isfinite(float(value))


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    result = round(float(value), 9)
    return 0.0 if result == 0.0 else result


def _point(value: Any) -> tuple[float, float] | None:
    if (
        isinstance(value, Sequence)
        and (not isinstance(value, (str, bytes)))
        and (len(value) == 2)
        and all((_is_real(axis) for axis in value))
    ):
        return (float(value[0]), float(value[1]))
    return None


def _confidence(value: Any) -> float | None:
    if not _is_real(value):
        return None
    return min(1.0, max(0.0, float(value)))


def _mean_point(values: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (
        math.fsum((point[0] for point in values)) / len(values),
        math.fsum((point[1] for point in values)) / len(values),
    )


def _empty_point(reason: str, confidence: float | None = None) -> dict[str, Any]:
    return {
        "xy_torso_lengths": None,
        "distance_to_trunk_center_torso_lengths": None,
        "confidence": _round(confidence),
        "unavailable_reason": reason,
    }


def _empty_bilateral(reason: str) -> dict[str, Any]:
    return {"value_torso_lengths": None, "confidence": None, "unavailable_reason": reason}


def derive_body_relative_pose_observation(
    points_xy_normalized: Sequence[Any],
    keypoint_confidences: Sequence[Any],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Derive one fixed trunk-frame observation from COCO-WholeBody-133 data.

    Image-normalized X and Y coordinates have different physical scales for a
    non-square frame.  They are first converted to isotropic image-height units
    before torso normalization.  The trunk origin is halfway between shoulder
    and hip midpoints; the longitudinal axis runs from shoulders to hips; and
    the transverse-axis sign follows anatomical left-to-right shoulder/hip
    evidence.  If any required trunk anchor or stable sign evidence is absent,
    every body-relative cue is explicitly unavailable—there is no fallback
    based on the bounding box, the image axes, or recording-specific data.
    """
    if len(points_xy_normalized) != 133 or len(keypoint_confidences) != 133:
        raise ValueError("body-relative pose cues require COCO-WholeBody-133 arrays")
    if type(image_width) is not int or type(image_height) is not int:
        raise ValueError("pose cue image dimensions must be integers")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("pose cue image dimensions must be positive")
    aspect = image_width / image_height
    points: list[tuple[float, float] | None] = []
    confidences: list[float | None] = []
    for raw_point, raw_confidence in zip(points_xy_normalized, keypoint_confidences, strict=True):
        parsed = _point(raw_point)
        points.append((parsed[0] * aspect, parsed[1]) if parsed is not None else None)
        confidences.append(_confidence(raw_confidence))
    anchor_indices = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
    missing_anchors = [
        index
        for index in anchor_indices
        if points[index] is None
        or confidences[index] is None
        or confidences[index] <= BODY_RELATIVE_MIN_KEYPOINT_CONFIDENCE
    ]
    point_names = (*_TEMPORAL_POINT_NAMES,)

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "schema_version": BODY_RELATIVE_POSE_CUE_CONTRACT_VERSION,
            "policy_version": BODY_RELATIVE_POSE_CUE_POLICY_VERSION,
            "trunk_frame": {
                "status": "unavailable",
                "unavailable_reason": reason,
                "anchor_confidence": None,
                "orientation_support_ratio": None,
                "origin_xy_image_height_units": None,
                "torso_length_image_height_units": None,
                "transverse_axis_xy": None,
                "longitudinal_axis_xy": None,
            },
            "points": {name: _empty_point("trunk_frame_unavailable") for name in point_names},
            "bilateral": {
                "wrist_separation": _empty_bilateral("trunk_frame_unavailable"),
                "palm_separation": _empty_bilateral("trunk_frame_unavailable"),
                "wrist_mirror_asymmetry": _empty_bilateral("trunk_frame_unavailable"),
                "palm_mirror_asymmetry": _empty_bilateral("trunk_frame_unavailable"),
            },
        }

    if missing_anchors:
        return unavailable("missing_or_nonpositive_confidence_trunk_anchor")
    left_shoulder = points[_LEFT_SHOULDER]
    right_shoulder = points[_RIGHT_SHOULDER]
    left_hip = points[_LEFT_HIP]
    right_hip = points[_RIGHT_HIP]
    assert left_shoulder is not None and right_shoulder is not None
    assert left_hip is not None and right_hip is not None
    shoulder_midpoint = _mean_point((left_shoulder, right_shoulder))
    hip_midpoint = _mean_point((left_hip, right_hip))
    longitudinal_vector = (
        hip_midpoint[0] - shoulder_midpoint[0],
        hip_midpoint[1] - shoulder_midpoint[1],
    )
    torso_length = math.hypot(*longitudinal_vector)
    if torso_length <= BODY_RELATIVE_MIN_TORSO_LENGTH:
        return unavailable("degenerate_torso_length")
    longitudinal = (longitudinal_vector[0] / torso_length, longitudinal_vector[1] / torso_length)
    transverse = (longitudinal[1], -longitudinal[0])
    anatomical_right = (
        (right_shoulder[0] - left_shoulder[0] + (right_hip[0] - left_hip[0])) / 2.0,
        (right_shoulder[1] - left_shoulder[1] + (right_hip[1] - left_hip[1])) / 2.0,
    )
    signed_support = anatomical_right[0] * transverse[0] + anatomical_right[1] * transverse[1]
    orientation_support_ratio = abs(signed_support) / torso_length
    if orientation_support_ratio < BODY_RELATIVE_MIN_TRANSVERSE_SUPPORT_RATIO:
        return unavailable("insufficient_anatomical_transverse_support")
    if signed_support < 0.0:
        transverse = (-transverse[0], -transverse[1])
    origin = _mean_point((shoulder_midpoint, hip_midpoint))
    anchor_confidence = min((confidences[index] or 0.0 for index in anchor_indices))

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        centered = (point[0] - origin[0], point[1] - origin[1])
        return (
            (centered[0] * transverse[0] + centered[1] * transverse[1]) / torso_length,
            (centered[0] * longitudinal[0] + centered[1] * longitudinal[1]) / torso_length,
        )

    output_points: dict[str, dict[str, Any]] = {}
    internal_points: dict[str, tuple[float, float]] = {}
    internal_confidences: dict[str, float] = {}
    for name, index in _POINT_INDICES.items():
        point = points[index]
        confidence = confidences[index]
        if (
            point is None
            or confidence is None
            or confidence <= BODY_RELATIVE_MIN_KEYPOINT_CONFIDENCE
        ):
            output_points[name] = _empty_point(
                "missing_or_nonpositive_confidence_keypoint", confidence
            )
            continue
        transformed = transform(point)
        internal_points[name] = transformed
        internal_confidences[name] = confidence
        output_points[name] = {
            "xy_torso_lengths": [_round(transformed[0]), _round(transformed[1])],
            "distance_to_trunk_center_torso_lengths": _round(math.hypot(*transformed)),
            "confidence": _round(confidence),
            "unavailable_reason": None,
        }
    for name, indices in (
        ("left_palm_center", _LEFT_PALM_ANCHORS),
        ("right_palm_center", _RIGHT_PALM_ANCHORS),
    ):
        palm_points = [points[index] for index in indices]
        palm_confidences = [confidences[index] for index in indices]
        if any(
            (
                point is None
                or confidence is None
                or confidence <= BODY_RELATIVE_MIN_KEYPOINT_CONFIDENCE
                for point, confidence in zip(palm_points, palm_confidences, strict=True)
            )
        ):
            finite_confidences = [value for value in palm_confidences if value is not None]
            output_points[name] = _empty_point(
                "incomplete_or_nonpositive_confidence_palm_anchors",
                min(finite_confidences) if finite_confidences else None,
            )
            continue
        complete_points = [point for point in palm_points if point is not None]
        complete_confidences = [
            confidence for confidence in palm_confidences if confidence is not None
        ]
        transformed = transform(_mean_point(complete_points))
        confidence = min(complete_confidences)
        internal_points[name] = transformed
        internal_confidences[name] = confidence
        output_points[name] = {
            "xy_torso_lengths": [_round(transformed[0]), _round(transformed[1])],
            "distance_to_trunk_center_torso_lengths": _round(math.hypot(*transformed)),
            "confidence": _round(confidence),
            "unavailable_reason": None,
        }

    def bilateral(left_name: str, right_name: str, *, mirror: bool) -> dict[str, Any]:
        left = internal_points.get(left_name)
        right = internal_points.get(right_name)
        if left is None or right is None:
            return _empty_bilateral("required_body_relative_points_unavailable")
        comparison_left = (-left[0], left[1]) if mirror else left
        return {
            "value_torso_lengths": _round(math.dist(comparison_left, right)),
            "confidence": _round(
                min(internal_confidences[left_name], internal_confidences[right_name])
            ),
            "unavailable_reason": None,
        }

    return {
        "schema_version": BODY_RELATIVE_POSE_CUE_CONTRACT_VERSION,
        "policy_version": BODY_RELATIVE_POSE_CUE_POLICY_VERSION,
        "trunk_frame": {
            "status": "available",
            "unavailable_reason": None,
            "anchor_confidence": _round(anchor_confidence),
            "orientation_support_ratio": _round(orientation_support_ratio),
            "origin_xy_image_height_units": [_round(origin[0]), _round(origin[1])],
            "torso_length_image_height_units": _round(torso_length),
            "transverse_axis_xy": [_round(transverse[0]), _round(transverse[1])],
            "longitudinal_axis_xy": [_round(longitudinal[0]), _round(longitudinal[1])],
        },
        "points": output_points,
        "bilateral": {
            "wrist_separation": bilateral("left_wrist", "right_wrist", mirror=False),
            "palm_separation": bilateral("left_palm_center", "right_palm_center", mirror=False),
            "wrist_mirror_asymmetry": bilateral("left_wrist", "right_wrist", mirror=True),
            "palm_mirror_asymmetry": bilateral(
                "left_palm_center", "right_palm_center", mirror=True
            ),
        },
    }
