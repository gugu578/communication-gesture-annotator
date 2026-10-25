# SPDX-License-Identifier: Apache-2.0
"""Geometry-qualified local motion derived from retained native observations.

No reference labels, activity thresholds or temporal resampling are used here.
Original image-relative geometry remains in the preprocessing artifact.
"""

from __future__ import annotations

from copy import deepcopy
import math
from statistics import median
from typing import Any, Mapping, Sequence

VERSION = "projected-hand-quality-relative-rotation-v0.1"
# Broad conditioning guards, not fitted communication/activity thresholds.
MIN_AXIS_TO_SCALE = 0.25
MAX_JOINT_RADIUS = 5.0
MAX_SCALE_RATIO = 3.0
MIN_CONFIDENCE = 0.3


def _point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(v) in (int, float) and math.isfinite(v) for v in value)
    )


def qualified_hand(hand: Mapping[str, Any], trunk: Mapping[str, Any]) -> dict[str, Any]:
    """Re-normalize with three MCP anchors and qualify channels independently."""
    result = deepcopy(dict(hand))
    features = result["features"] = {}
    reasons: dict[str, str] = {}
    result["quality_policy_version"] = VERSION
    result["channel_unavailable_reasons"] = reasons
    local = hand.get("local_joint_coordinates", ())
    result["reliable_finger_count"] = 0
    if (
        not hand.get("frame_available")
        or len(local) != 21
        or not all(_point(local[i]) for i in (0, 5, 9, 17))
    ):
        reasons.update(
            {
                k: "missing_palm_anchors"
                for k in ("orientation_speed", "articulation_speed", "extension_speed")
            }
        )
        return result
    scale = median(math.dist(local[0], local[i]) for i in (5, 9, 17))
    if not math.isfinite(scale) or scale <= 1e-6:
        reasons.update(
            {
                k: "degenerate_palm_scale"
                for k in ("orientation_speed", "articulation_speed", "extension_speed")
            }
        )
        return result
    normalized = [
        [(p[a] - local[0][a]) / scale for a in (0, 1)] if _point(p) else None for p in local
    ]
    result["local_joint_coordinates"] = normalized
    result["qualified_hand_scale_image_heights"] = hand.get("hand_scale_image_heights", 0.0) * scale
    width = math.dist(normalized[5], normalized[17])
    axis_supported = math.dist(normalized[0], normalized[9]) >= MIN_AXIS_TO_SCALE
    palm_supported = 0.1 <= width <= 4.0
    angle = hand.get("features", {}).get("palm_angle_rad")
    axis = trunk.get("longitudinal_axis_xy")
    if (
        axis_supported
        and palm_supported
        and angle
        and math.isfinite(angle[0])
        and trunk.get("status") == "available"
        and (trunk.get("anchor_confidence") or 0) >= MIN_CONFIDENCE
        and _point(axis)
        and math.hypot(*axis) > 0.5
    ):
        relative = angle[0] - math.atan2(axis[1], axis[0])
        features["palm_relative_angle_rad"] = [math.atan2(math.sin(relative), math.cos(relative))]
    else:
        reasons["orientation_speed"] = (
            "ill_conditioned_projected_palm_axis"
            if not axis_supported or not palm_supported
            else "unreliable_torso_rotation_reference"
        )
    from .hand_motion import FINGERS

    for n, finger in enumerate(FINGERS):
        joints = normalized[1 + n * 4 : 5 + n * 4]
        reliable = palm_supported and all(
            p is not None and math.hypot(*p) <= MAX_JOINT_RADIUS for p in joints
        )
        if not reliable:
            continue
        chain = sum(math.dist(a, b) for a, b in zip(joints, joints[1:]))
        if chain <= 1e-6:
            continue
        result["reliable_finger_count"] += 1
        if axis_supported:
            features[f"{finger}_articulation"] = [v / 2 for p in joints for v in p]
        features[f"{finger}_extension"] = [math.dist(joints[0], joints[-1]) / chain]
    if not any(k.endswith("_articulation") for k in features):
        reasons["articulation_speed"] = (
            "ill_conditioned_projected_palm_axis"
            if not axis_supported
            else "unreliable_finger_geometry"
        )
    if not any(k.endswith("_extension") for k in features):
        reasons["extension_speed"] = "unreliable_finger_geometry"
    return result


def condition_native_hands(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve every native timestamp; reject unstable articulation edges only."""
    output = []
    previous: dict[tuple[str, str, str], tuple[int, float]] = {}
    for row in records:
        people = []
        if row.get("target_association") != "unique_track_hypothesis":
            previous.clear()
        for person in row.get("people", ()):
            geometries = {}
            for curve, geometry in person["geometry"].items():
                hands = {}
                for side, hand in geometry.get("hands", {}).items():
                    value = qualified_hand(hand, geometry.get("trunk_frame", {}))
                    key = (person["continuity_segment_id"], curve, side)
                    scale = value.get("qualified_hand_scale_image_heights", 0.0)
                    prior = previous.pop(key, None)
                    if scale > 0 and row.get("target_association") == "unique_track_hypothesis":
                        previous[key] = (row["display_end_us"], scale)
                        if (
                            prior
                            and prior[0] >= row["display_start_us"]
                            and max(scale / prior[1], prior[1] / scale) > MAX_SCALE_RATIO
                        ):
                            # The local coordinate scale changed too much to
                            # interpret joint displacement on this edge. Do not
                            # erase independent extension ratios or palm motion.
                            value["blocked_derivative_suffixes"] = ["_articulation"]
                            value["channel_unavailable_reasons"]["articulation_speed"] = (
                                "unstable_projected_hand_scale"
                            )
                    hands[side] = value
                geometries[curve] = {**geometry, "hands": hands}
            people.append({**person, "geometry": geometries})
        output.append({**row, "people": people})
    return output
