# SPDX-License-Identifier: Apache-2.0
"""Projected palm and finger measurements; no communication classifier."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

HAND_MOTION_POLICY_VERSION = "hand-relative-motion-v0.1"
MINIMUM_LANDMARK_CONFIDENCE = 0.3
FINGERS = ("thumb", "index", "middle", "ring", "little")


def hand_geometry(
    points: Sequence[Any], confidences: Sequence[Any], *, side: str, aspect: float
) -> dict[str, Any]:
    """Normalize local articulation while separately retaining palm rotation."""
    offset = {"left": 91, "right": 112}[side]
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("image aspect must be positive")
    positions: list[list[float] | None] = []
    quality: list[float | None] = []
    for index in range(offset, offset + 21):
        point, confidence = points[index], confidences[index]
        valid = (
            point is not None
            and len(point) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
            and isinstance(confidence, (int, float))
            and MINIMUM_LANDMARK_CONFIDENCE <= confidence <= 1
        )
        positions.append([float(point[0]) * aspect, float(point[1])] if valid else None)
        quality.append(float(confidence) if valid else None)
    wrist, middle, index_mcp, little_mcp = (positions[i] for i in (0, 9, 5, 17))
    result: dict[str, Any] = {
        "policy_version": HAND_MOTION_POLICY_VERSION,
        "projected_2d": True,
        "minimum_landmark_confidence": MINIMUM_LANDMARK_CONFIDENCE,
        "local_joint_coordinates": [None] * 21,
        "joint_confidences": quality,
        "frame_available": False,
        "features": {},
        "unavailable_reason": "missing_palm_anchors",
    }
    if any(value is None for value in (wrist, middle, index_mcp, little_mcp)):
        return result
    axis = [middle[i] - wrist[i] for i in (0, 1)]
    scale = math.hypot(*axis)
    width = math.dist(index_mcp, little_mcp)
    if scale < 1e-6 or width < 1e-6:
        result["unavailable_reason"] = "degenerate_projected_hand_frame"
        return result
    axis = [v / scale for v in axis]
    local = [
        None
        if p is None
        else [
            ((p[0] - wrist[0]) * axis[1] - (p[1] - wrist[1]) * axis[0]) / scale,
            ((p[0] - wrist[0]) * axis[0] + (p[1] - wrist[1]) * axis[1]) / scale,
        ]
        for p in positions
    ]
    features: dict[str, Any] = {
        "palm_angle_rad": [math.atan2(axis[1], axis[0])],
        "palm_spread": [width / scale],
    }
    for number, finger in enumerate(FINGERS):
        joints = local[1 + number * 4 : 5 + number * 4]
        # Keep each joint in the durable stream; the temporal metric includes
        # all four joints and cannot miss PIP/DIP articulation with a fixed tip.
        features[f"{finger}_articulation"] = (
            [v / 2 for p in joints for v in p] if all(p is not None for p in joints) else None
        )
        if all(p is not None for p in joints):
            chain = sum(math.dist(a, b) for a, b in zip(joints, joints[1:]))
            features[f"{finger}_extension"] = (
                [math.dist(joints[0], joints[-1]) / chain] if chain > 1e-6 else None
            )
        else:
            features[f"{finger}_extension"] = None
        if number:
            features[f"thumb_{finger}_pinch"] = (
                [math.dist(local[4], joints[-1])]
                if local[4] is not None and joints[-1] is not None
                else None
            )
    result.update(
        frame_available=True,
        unavailable_reason=None,
        local_joint_coordinates=local,
        hand_scale_image_heights=scale,
        frame_confidence=min(quality[i] for i in (0, 5, 9, 17)),
        features=features,
    )
    return result


def summarize_hand_motion(
    records: Sequence[Mapping[str, Any]], start_us: int, end_us: int
) -> dict[str, Any]:
    """Integrate only adjacent, supported, same-target measurements at native PTS."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    previous: dict[tuple[str, str, str, str], tuple[Mapping[str, Any], Sequence[float]]] = {}
    for row in records:
        if row["target_association"] != "unique_track_hypothesis":
            previous.clear()
            continue
        for person in row["people"]:
            for curve, geometry in person["geometry"].items():
                for side, hand in geometry.get("hands", {}).items():
                    key = (person["continuity_segment_id"], curve, side)
                    if row["display_start_us"] < end_us and row["display_end_us"] > start_us:
                        group = groups.setdefault(
                            key,
                            {
                                "continuity_segment_id": key[0],
                                "curve": curve,
                                "side": side,
                                "observed_frames": 0,
                                "usable_hand_frames": 0,
                                "features": {},
                                "confidence_sum": 0.0,
                            },
                        )
                        group["observed_frames"] += 1
                        group["usable_hand_frames"] += int(hand["frame_available"])
                        group["confidence_sum"] += hand.get("frame_confidence", 0.0)
                    for name in sorted(
                        set(hand["features"]) | {k[3] for k in previous if k[:3] == key}
                    ):
                        feature_key = (*key, name)
                        value = hand["features"].get(name)
                        prior = previous.pop(feature_key, None)
                        if value is None:
                            continue
                        previous[feature_key] = (row, value)
                        if prior is None:
                            continue
                        if any(
                            name.endswith(suffix)
                            for suffix in hand.get("blocked_derivative_suffixes", ())
                        ):
                            continue
                        old, old_value = prior
                        delta = row["observation_time_us"] - old["observation_time_us"]
                        support = max(
                            0,
                            min(end_us, row["observation_time_us"])
                            - max(start_us, old["observation_time_us"]),
                        )
                        if (
                            delta <= 0
                            or support == 0
                            or old["display_end_us"] < row["display_start_us"]
                        ):
                            continue
                        # Support-only next frames may supply an edge into the
                        # window even though their own display starts outside it.
                        group = groups.get(key)
                        if group is None:
                            continue
                        change = (
                            [
                                math.atan2(
                                    math.sin(value[0] - old_value[0]),
                                    math.cos(value[0] - old_value[0]),
                                )
                            ]
                            if name.endswith("angle_rad")
                            else [b - a for a, b in zip(old_value, value)]
                        )
                        distance = math.sqrt(sum(v * v for v in change))
                        first_fraction = max(0, start_us - old["observation_time_us"]) / delta
                        last_fraction = min(delta, end_us - old["observation_time_us"]) / delta
                        supported_first = [
                            a + first_fraction * d for a, d in zip(old_value, change)
                        ]
                        supported_last = [a + last_fraction * d for a, d in zip(old_value, change)]
                        speed = distance * 1e6 / delta
                        stats = group["features"].setdefault(
                            name,
                            {
                                "path": 0.0,
                                "supported_us": 0,
                                "peak_speed": 0.0,
                                "peak_support_us": None,
                                "peak_source_frame_indices": None,
                                "start_value": supported_first,
                                "end_value": supported_last,
                                "last_support_end_us": old["observation_time_us"],
                                "discontinuous_support": False,
                            },
                        )
                        stats["discontinuous_support"] |= (
                            stats["last_support_end_us"] != old["observation_time_us"]
                        )
                        stats["last_support_end_us"] = row["observation_time_us"]
                        stats["path"] += distance * support / delta
                        stats["supported_us"] += support
                        stats["end_value"] = supported_last
                        if stats["peak_support_us"] is None or speed > stats["peak_speed"]:
                            stats.update(
                                peak_speed=speed,
                                peak_support_us=[
                                    old["observation_time_us"],
                                    row["observation_time_us"],
                                ],
                                peak_source_frame_indices=[
                                    old["source_frame_index"],
                                    row["source_frame_index"],
                                ],
                            )
    for group in groups.values():
        group["mean_frame_confidence"] = group.pop("confidence_sum") / group["observed_frames"]
        for name, stats in group["features"].items():
            stats["mean_speed"] = stats["path"] * 1e6 / stats["supported_us"]
            stats["coverage"] = stats["supported_us"] / (end_us - start_us)
            # Net displacement is deliberately distinct from travelled path.
            first, last = stats.pop("start_value"), stats.pop("end_value")
            stats["net_change"] = (
                abs(math.atan2(math.sin(last[0] - first[0]), math.cos(last[0] - first[0])))
                if name.endswith("angle_rad")
                else math.dist(first, last)
            )
            stats.pop("last_support_end_us")
            if stats["discontinuous_support"]:
                stats["net_change"] = None
    return {
        "policy_version": HAND_MOTION_POLICY_VERSION,
        "start_us": start_us,
        "end_us": end_us,
        "units": "hand lengths; articulation is RMS displacement of four finger joints; angle in radians; speeds per second",
        "scope": "native-derived projected palm/finger summaries; full joints remain in native artifact; missing support is not zero motion",
        "groups": list(groups.values()),
    }
