# SPDX-License-Identifier: Apache-2.0
"""Reference-free motion screening, computed once before probability requests."""

from __future__ import annotations
from bisect import bisect_left, bisect_right
from copy import deepcopy
import math
from statistics import median
from typing import Any, Mapping, Sequence
from ..config import DetectionConfig
from ..domain import AnalysisSample

VERSION = "motion-quality-v0.3"
SCORE_VERSION = "quality-aware-motion-v0.3"
LOCAL_DISPLAY_CAP = 8.0
TRANSPORT = (
    "palm_speed",
    "raw_palm_speed",
    "palm_path",
    "palm_xy",
    "palm_excursion",
    "palm_speed_300ms",
    "palm_speed_700ms",
    "palm_net_displacement",
    "palm_path_efficiency",
)
LOCAL = ("orientation_speed", "extension_speed", "articulation_speed")


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0)


def conditioned_timeline(conditioning: Any, samples: Sequence[AnalysisSample]) -> dict[str, Any]:
    """Build canonical bins in bounded chunks, without constructing LLM packets.

    Three-bin halos make the original centered 300/700 ms measurements stable
    at chunk edges. Native observations and exact timestamps remain unchanged.
    """
    from ..media.motion_timeline import build_motion_timeline, HAND_COLUMNS
    from ..media.hand_quality import condition_native_hands

    records = conditioning.native_feature_records
    starts = [r["display_start_us"] for r in records]
    ends = [r["display_end_us"] for r in records]
    rows = []
    quality = {}
    for offset in range(0, len(samples), 100):
        core = samples[offset : offset + 100]
        context = samples[max(0, offset - 3) : offset + 103]
        a, b = (context[0].start_us, context[-1].end_us)
        lo, hi = (max(0, bisect_right(ends, a) - 1), min(len(records), bisect_left(starts, b) + 1))
        bins = [
            {"sample_index": s.index, "start_us": s.start_us, "end_us": s.end_us} for s in context
        ]
        qualified_records = condition_native_hands(records[max(0, lo - 1) : hi])
        chunk = build_motion_timeline(qualified_records, bins, a, b)
        rows.extend((r for r in chunk["rows"] if core[0].index <= r[0] <= core[-1].index))
        quality.update(
            {i: q for i, q in chunk["hand_quality"].items() if core[0].index <= i <= core[-1].index}
        )
    return {"hand_columns": HAND_COLUMNS, "rows": rows, "hand_quality": quality}


def screen_timeline(
    timeline: Mapping[str, Any],
    resolutions: Mapping[int, Mapping[str, Any]],
    config: DetectionConfig,
) -> dict[int, dict[str, Any]]:
    """Reject broken support; flag only isolated robust high outliers as potential.

    Statistical rarity never establishes that movement is false. Suspect values
    remain in flags/raw records, but are not used in automatic physical scores.
    """
    output: dict[int, dict[str, Any]] = {}
    runs: list[list[int]] = []
    previous = None
    for row in timeline["rows"]:
        index, start, end, segment = row[:4]
        admitted = segment is not None and (not resolutions[index].get("exclude_from_detection"))
        if admitted:
            if (
                previous is None
                or previous[2] != start
                or previous[3] != segment
                or (previous[0] + 1 != index)
            ):
                runs.append([])
            runs[-1].append(index)
        previous = row if admitted else None
        properties = {v["property_name"]: v for v in resolutions[index].get("properties", ())}
        entry: dict[str, Any] = {"start_us": start, "end_us": end, "hands": {}}
        for side, values in zip(("left", "right"), row[4:6], strict=True):
            prop = properties.get(side + "_motion", {})
            if (
                not admitted
                or not prop.get("usable")
                or (not prop.get("modalities", {}).get("pose", {}).get("admitted"))
            ):
                continue
            raw = dict(zip(timeline["hand_columns"], values, strict=True))
            quality_info = timeline.get("hand_quality", {}).get(index, {}).get(side, {})
            hand = {
                "raw": deepcopy(raw),
                "values": deepcopy(raw),
                "flags": [],
                "quality": deepcopy(quality_info),
            }
            entry["hands"][side] = hand
            for field, reason in quality_info.get("channel_unavailable_reasons", {}).items():
                hand["flags"].append(
                    {
                        "field": field,
                        "status": "unreliable",
                        "reason": reason,
                        "raw_value": None,
                        "used_for_motion_score": False,
                    }
                )
                hand["values"][field] = None
            for field, value in raw.items():
                if value is None:
                    continue
                valid = (
                    isinstance(value, (list, tuple))
                    and len(value) == 2
                    and all((type(v) in (int, float) and math.isfinite(v) for v in value))
                    if field == "palm_xy"
                    else finite(value)
                )
                if not valid:
                    flag(hand, field, "unreliable", "invalid_numeric_value")
                    hand["raw"][field] = None
            quality = raw.get("palm_quality")
            support = raw.get("palm_supported_us")
            if (
                quality is not None
                and (not finite(quality) or not 0.3 <= quality <= 1)
                or (
                    support is not None
                    and (not finite(support) or not 0.8 * (end - start) <= support <= end - start)
                )
            ):
                for field in TRANSPORT:
                    flag(hand, field, "unreliable", "insufficient_palm_confidence_or_support")
            coverage = raw.get("hand_motion_coverage")
            if coverage is not None and (not finite(coverage) or not 0.8 <= coverage <= 1):
                for field in LOCAL:
                    flag(hand, field, "unreliable", "insufficient_local_hand_support")
        output[index] = entry
    scales = {
        "palm_speed": config.motion_transport_scale,
        "wrist_speed": config.motion_transport_scale,
        "orientation_speed": config.motion_orientation_scale,
        "extension_speed": config.motion_finger_scale,
        "articulation_speed": config.motion_articulation_scale,
    }
    radius = config.motion_outlier_window_us // 200000
    for run in runs:
        for side in ("left", "right"):
            for field, scale in scales.items():
                values = [
                    output[i]["hands"].get(side, {}).get("values", {}).get(field) for i in run
                ]
                suspects: dict[int, float] = {}
                for n, value in enumerate(values):
                    if not finite(value) or value < 3 * scale:
                        continue
                    peers = [
                        v
                        for k in range(max(0, n - radius), min(len(run), n + radius + 1))
                        if abs(k - n) > 1 and finite((v := values[k]))
                    ]
                    if len(peers) < 6:
                        continue
                    logs = [math.log1p(v / scale) for v in peers]
                    mid = median(logs)
                    spread = max(0.15, 1.4826 * median((abs(v - mid) for v in logs)))
                    if (
                        math.log1p(value / scale) > mid + config.motion_outlier_z * spread
                        and value > config.motion_outlier_ratio * max(scale * 0.1, median(peers))
                    ):
                        suspects[n] = mid + config.motion_outlier_z * spread
                for n in suspects:
                    a = n
                    while a - 1 in suspects:
                        a -= 1
                    b = n
                    while b + 1 in suspects:
                        b += 1
                    if b - a + 1 > 2 or a == 0 or b + 1 == len(values):
                        continue
                    if not finite(values[a - 1]) or not finite(values[b + 1]):
                        continue
                    hand = output[run[n]]["hands"][side]
                    flag(hand, field, "potential_outlier", "isolated_robust_spike")
                    if field == "palm_speed":
                        for related in TRANSPORT:
                            flag(
                                hand,
                                related,
                                "potential_outlier",
                                "same_sample_as_suspect_palm_transport",
                            )
    for run in runs:
        for n, index in enumerate(run):
            for side, hand in output[index]["hands"].items():
                for radius, field in ((1, "palm_speed_300ms"), (3, "palm_speed_700ms")):
                    if field not in hand["values"]:
                        continue
                    neighbours = [
                        output[i]["hands"].get(side, {}).get("values", {}).get("palm_speed")
                        for i in run[max(0, n - radius) : n + radius + 1]
                    ]
                    if not all((finite(v) for v in neighbours)):
                        flag(
                            hand,
                            field,
                            "unreliable",
                            "window_contains_unavailable_or_flagged_transport",
                        )
                        hand["values"][field] = None
                    else:
                        hand["values"][field] = sum(neighbours) / len(neighbours)
    return output


def flag(hand: dict[str, Any], field: str, status: str, reason: str) -> None:
    value = hand["raw"].get(field)
    if value is not None and (not any((f["field"] == field for f in hand["flags"]))):
        safe = value if _json_finite(value) else None
        hand["flags"].append(
            {
                "field": field,
                "status": status,
                "reason": reason,
                "raw_value": safe,
                "used_for_motion_score": False,
            }
        )
    if field in hand["values"]:
        hand["values"][field] = None


def _json_finite(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        or (isinstance(value, (list, tuple)) and all((_json_finite(v) for v in value)))
    )


# Incomplete quiet evidence cannot establish rest. Reliable movement may still help.
def score_measurement(
    entry: Mapping[str, Any],
    config: DetectionConfig,
    artifact_sha256: str | None,
    *,
    policy_version: str = SCORE_VERSION,
) -> dict[str, Any]:
    if policy_version != SCORE_VERSION:
        raise ValueError("unsupported physical measurement policy")
    hands, components, flags, quality = ({}, {}, {}, {})
    complete = []
    for side, item in entry["hands"].items():
        h = item["values"]
        flags[side] = deepcopy(item["flags"])
        quality[side] = deepcopy(item.get("quality", {}))
        supported = h.get("palm_supported_us")
        path_rate = (
            h["palm_path"] * 1000000.0 / supported
            if finite(h.get("palm_path"))
            and finite(supported)
            and (supported >= 0.8 * (entry["end_us"] - entry["start_us"]))
            else None
        )
        transport = next(
            (v for v in (path_rate, h.get("palm_speed"), h.get("wrist_speed")) if finite(v)), None
        )
        rotation = h.get("orientation_speed")
        finger = [
            h[k] / scale
            for k, scale in (
                ("extension_speed", config.motion_finger_scale),
                ("articulation_speed", config.motion_articulation_scale),
            )
            if finite(h.get(k))
        ]
        rotation = (
            min(LOCAL_DISPLAY_CAP, rotation / config.motion_orientation_scale)
            if finite(rotation)
            else None
        )
        finger_value = min(LOCAL_DISPLAY_CAP, max(finger)) if finger else None
        local = (
            math.sqrt(config.motion_hand_weight) * math.hypot(rotation or 0.0, finger_value or 0.0)
            if rotation is not None or finger_value is not None
            else None
        )
        transport = transport / config.motion_transport_scale if transport is not None else None
        if transport is None and local is None:
            continue
        hands[side] = deepcopy(h)
        magnitude = math.hypot(transport or 0.0, local or 0.0)
        components[side] = {
            "transport": transport,
            "rotation": rotation,
            "fingers": finger_value,
            "local": local,
            "magnitude": magnitude,
        }
        complete.append(
            transport is not None
            and (not item["flags"])
            and (
                all((finite(h.get(k)) for k in LOCAL))
                and quality[side].get("reliable_finger_count", 5) == 5
            )
        )
    magnitude = max((c["magnitude"] for c in components.values()), default=None)
    baseline = None if magnitude is None else magnitude / (1 + magnitude)
    rest_supported = len(hands) == 2 and all(complete)
    q = None if baseline is None or (not rest_supported and 1 - baseline >= 0.5) else 1 - baseline
    result = {
        "motion_baseline": baseline,
        "normalized_motion": magnitude,
        "motion_non_g_probability": q,
        "admitted_hands": list(hands),
        "hand_measurements": hands,
        "measurement_status": "available" if magnitude is not None else "unavailable",
        "native_artifact_sha256": artifact_sha256,
        "motion_measurement_policy_version": policy_version,
        "motion_sample_duration_us": entry["end_us"] - entry["start_us"],
        "motion_quality_policy_version": VERSION,
        "quality_flags": flags,
        "hand_quality": quality,
        "score_components": components,
        "motion_scales": {
            "transport": config.motion_transport_scale,
            "rotation": config.motion_orientation_scale,
            "extension": config.motion_finger_scale,
            "articulation": config.motion_articulation_scale,
            "local_weight": config.motion_hand_weight,
        },
        "rest_motion_supported": rest_supported,
        "motion_probability_status": "available" if q is not None else "insufficient_support",
    }
    return result
