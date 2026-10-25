# SPDX-License-Identifier: Apache-2.0
"""Compact native-derived measurements; no reference labels or rest templates."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .hand_motion import summarize_hand_motion

MOTION_TIMELINE_VERSION = "native-hand-timeline-v0.3"
MINIMUM_CONFIDENCE = 0.3
MINIMUM_COVERAGE = 0.8
HAND_COLUMNS = [
    "palm_speed",
    "wrist_speed",
    "raw_palm_speed",
    "palm_path",
    "palm_supported_us",
    "palm_xy",
    "palm_excursion",
    "palm_quality",
    "orientation_speed",
    "extension_speed",
    "articulation_speed",
    "hand_motion_coverage",
    "palm_speed_300ms",
    "palm_speed_700ms",
    "palm_net_displacement",
    "palm_path_efficiency",
]


def _qualified(point: Mapping[str, Any], geometry: Mapping[str, Any]) -> bool:
    return (
        point.get("xy_torso_lengths") is not None
        and float(point.get("confidence") or 0) >= MINIMUM_CONFIDENCE
        and float(geometry.get("points", {}).get("trunk_center", {}).get("confidence") or 0)
        >= MINIMUM_CONFIDENCE
    )


def _point_summary(
    records: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
    segment: str,
    curve: str,
    name: str,
    *,
    with_displacement: bool = False,
) -> dict[str, Any]:
    path = 0.0
    covered = 0
    positions: list[tuple[list[float], int, float]] = []
    previous: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
    first_position = last_position = None
    last_support_end: int | None = None
    continuous = True
    for row in records:
        people = row.get("people", [])
        if row.get("target_association") != "unique_track_hypothesis" or len(people) != 1:
            previous = None
            continue
        person = people[0]
        if person["continuity_segment_id"] != segment:
            previous = None
            continue
        geometry = person["geometry"][curve]
        point = geometry["points"].get(name, {})
        if not _qualified(point, geometry):
            previous = None
            continue
        duration = max(0, min(end, row["display_end_us"]) - max(start, row["display_start_us"]))
        if duration:
            positions.append((point["xy_torso_lengths"], duration, float(point["confidence"])))
        # Use actual adjacent observation PTS, including support-only endpoints.
        if previous is not None:
            old, old_point = previous
            delta = row["observation_time_us"] - old["observation_time_us"]
            support = max(
                0, min(end, row["observation_time_us"]) - max(start, old["observation_time_us"])
            )
            if delta > 0 and support and old["display_end_us"] >= row["display_start_us"]:
                path += (
                    math.dist(old_point["xy_torso_lengths"], point["xy_torso_lengths"])
                    * support
                    / delta
                )
                covered += support
                if with_displacement:
                    a, b = (
                        max(start, old["observation_time_us"]),
                        min(end, row["observation_time_us"]),
                    )
                    continuous &= last_support_end is None or last_support_end == a
                    change = [
                        point["xy_torso_lengths"][axis] - old_point["xy_torso_lengths"][axis]
                        for axis in (0, 1)
                    ]
                    if first_position is None:
                        first_position = [
                            old_point["xy_torso_lengths"][axis]
                            + change[axis] * (a - old["observation_time_us"]) / delta
                            for axis in (0, 1)
                        ]
                    last_position = [
                        old_point["xy_torso_lengths"][axis]
                        + change[axis] * (b - old["observation_time_us"]) / delta
                        for axis in (0, 1)
                    ]
                    last_support_end = b
        previous = (row, point)
    enough = covered >= (end - start) * MINIMUM_COVERAGE
    position_support = sum(item[1] for item in positions)
    xy = (
        [
            sum(p[axis] * duration for p, duration, _ in positions) / position_support
            for axis in (0, 1)
        ]
        if position_support >= (end - start) * MINIMUM_COVERAGE
        else None
    )
    excursion = (
        math.hypot(
            *(
                max(p[a] for p, _, _ in positions) - min(p[a] for p, _, _ in positions)
                for a in (0, 1)
            )
        )
        if xy is not None
        else None
    )
    net = (
        math.dist(first_position, last_position)
        if enough and continuous and first_position is not None
        else None
    )
    return {
        "speed": path * 1e6 / covered if enough else None,
        "path": path if covered else None,
        "supported_us": covered,
        "xy": xy,
        "excursion": excursion,
        "net_displacement": net,
        "path_efficiency": min(1.0, net / path) if net is not None and path > 0 else None,
        "quality": min((q for _, _, q in positions), default=None),
    }


def build_motion_timeline(
    records: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    start_us: int,
    end_us: int,
) -> dict[str, Any]:
    """Integrate all supported native edges; never bridge association/source gaps.

    Compact arrays are lossless with respect to these computed summaries. Native
    coordinates remain in the hash-bound source artifact. Null is unavailable.
    """
    rows: list[list[Any]] = []
    quality: dict[int, dict[str, Any]] = {}
    for sample in samples:
        start, end = max(start_us, sample["start_us"]), min(end_us, sample["end_us"])
        if end <= start:
            continue
        overlapping = [
            i
            for i, r in enumerate(records)
            if r["display_start_us"] < end and r["display_end_us"] > start
        ]
        local = records[max(0, overlapping[0] - 1) : overlapping[-1] + 2] if overlapping else []
        visible = [records[i] for i in overlapping]
        segments = {p["continuity_segment_id"] for r in visible for p in r.get("people", [])}
        unique = (
            bool(visible)
            and len(segments) == 1
            and all(
                r.get("target_association") == "unique_track_hypothesis"
                and len(r.get("people", [])) == 1
                for r in visible
            )
        )
        segment = next(iter(segments)) if unique else None
        # A gap inside a bin is not an invitation to interpolate across it.
        cursor = start
        for r in visible:
            if r["display_start_us"] > cursor:
                unique = False
            cursor = max(cursor, r["display_end_us"])
        unique = unique and cursor >= end
        segment = segment if unique else None
        hand_summary = summarize_hand_motion(local, start, end) if unique else {"groups": []}
        hands = []
        sample_quality: dict[str, Any] = {}
        for side in ("left", "right"):
            if not unique:
                hands.append([None] * len(HAND_COLUMNS))
                continue
            palm = _point_summary(
                local,
                start,
                end,
                segment,
                "smoothed",
                side + "_palm_center",
                with_displacement=True,
            )
            wrist = _point_summary(local, start, end, segment, "smoothed", side + "_wrist")
            raw = _point_summary(local, start, end, segment, "raw", side + "_palm_center")
            features = next(
                (
                    g["features"]
                    for g in hand_summary["groups"]
                    if g["curve"] == "smoothed"
                    and g["side"] == side
                    and g["continuity_segment_id"] == segment
                ),
                {},
            )

            def rate(suffix: str) -> float | None:
                values = [
                    v["mean_speed"]
                    for k, v in features.items()
                    if k.endswith(suffix)
                    and v["coverage"] >= MINIMUM_COVERAGE
                    and not v["discontinuous_support"]
                ]
                return max(values) if values else None

            hand_coverage = max((v["coverage"] for v in features.values()), default=0.0)
            native_hands = [
                p["geometry"]["smoothed"].get("hands", {}).get(side, {})
                for r in visible
                for p in r.get("people", ())
            ]
            qualified = any(h.get("quality_policy_version") for h in native_hands)
            rotation_name = "palm_relative_angle_rad" if qualified else "palm_angle_rad"
            if qualified:
                sample_quality[side] = {
                    "rotation_reference": "projected_torso_longitudinal_axis",
                    "reliable_finger_count": min(
                        (h.get("reliable_finger_count", 0) for h in native_hands), default=0
                    ),
                    "channel_unavailable_reasons": {
                        field: reason
                        for h in native_hands
                        for field, reason in h.get("channel_unavailable_reasons", {}).items()
                        if rate(
                            {
                                "orientation_speed": rotation_name,
                                "articulation_speed": "_articulation",
                                "extension_speed": "_extension",
                            }[field]
                        )
                        is None
                    },
                }
            hands.append(
                [
                    palm["speed"],
                    wrist["speed"],
                    raw["speed"],
                    palm["path"],
                    palm["supported_us"],
                    palm["xy"],
                    palm["excursion"],
                    palm["quality"],
                    rate(rotation_name),
                    rate("_extension"),
                    rate("_articulation"),
                    hand_coverage,
                    None,
                    None,
                    palm["net_displacement"],
                    palm["path_efficiency"],
                ]
            )
        index = sample.get("sample_index", sample.get("index"))
        if index is None and sample["start_us"] % 100_000 == 0:
            # The public packet API also accepts index-free canonical bins.
            index = sample["start_us"] // 100_000
        if type(index) is not int:
            raise ValueError("motion timeline needs an explicit canonical sample index")
        rows.append([index, start, end, segment, *hands])
        quality[index] = sample_quality
    for i, row in enumerate(rows):
        for side in (4, 5):
            for radius, output in ((1, 12), (3, 13)):
                # Demand continuous support throughout the local measurement window.
                window = rows[max(0, i - radius) : i + radius + 1]
                if (
                    row[3] is not None
                    and all(r[3] == row[3] and r[side][0] is not None for r in window)
                    and all(a[2] == b[1] for a, b in zip(window, window[1:]))
                ):
                    duration = sum(r[2] - r[1] for r in window)
                    row[side][output] = sum(r[side][0] * (r[2] - r[1]) for r in window) / duration
    return {
        "policy_version": MOTION_TIMELINE_VERSION,
        "columns": ["sample_index", "start_us", "end_us", "continuity_segment", "left", "right"],
        "hand_columns": HAND_COLUMNS,
        "rows": rows,
        "hand_quality": quality,
        "units": "transport/position: torso lengths; projected angle: radians; extension: unitless ratio; rates per second",
        "minimum_confidence": MINIMUM_CONFIDENCE,
        "minimum_coverage": MINIMUM_COVERAGE,
        "local_windows": "centered 300/700ms, clipped at delivered context edges; null across missing support",
        "interpretation": "Physical measurements only. Null is unknown, never zero; scene/target admission remains required.",
    }
