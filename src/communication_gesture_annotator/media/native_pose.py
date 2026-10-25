# SPDX-License-Identifier: Apache-2.0
"""Native-time measurement packets; no motion threshold or posture classifier.

The durable stream keeps raw and smoothed positions, missing observations and
derivative support separately. A request is never silently temporally thinned.
"""

from __future__ import annotations
from collections import defaultdict
import json
import math
from typing import Any, Iterator, Mapping, Sequence

NATIVE_POSE_CONTRACT_VERSION = "0.2.0"
NATIVE_POSE_PACKET_POLICY_VERSION = "native-complete-with-hand-timeline-v0.3"
SOURCE_FRAME_SELECTION_POLICY_VERSION = "display-covers-center-latest-start-lowest-index-v0.1"
POINT_INDICES = {
    "left_wrist": (9,),
    "right_wrist": (10,),
    "left_elbow": (7,),
    "right_elbow": (8,),
    "left_palm_center": (91, 96, 100, 104, 108),
    "right_palm_center": (112, 117, 121, 125, 129),
    "trunk_center": (5, 6, 11, 12),
}
PACKET_POINT_NAMES = (
    "left_wrist",
    "right_wrist",
    "left_palm_center",
    "right_palm_center",
    "trunk_center",
)


class PosePacketCapacityError(ValueError):
    """The caller must split with overlap, or report a capacity failure."""


def select_source_observation(
    rows: Sequence[Mapping[str, Any]], center_us: int
) -> Mapping[str, Any] | None:
    """Choose actual display coverage, never a nearest frame across a gap."""
    covering = [
        row for row in rows if row["canonical_time_us"] <= center_us < row["canonical_end_us"]
    ]
    return (
        min(covering, key=lambda row: (-row["canonical_time_us"], row["source_frame_index"]))
        if covering
        else None
    )


def _position(
    points: Sequence[Any], confidences: Sequence[Any], indices: Sequence[int]
) -> dict[str, Any]:
    valid = all(
        (
            points[index] is not None
            and len(points[index]) == 2
            and all((isinstance(axis, (int, float)) for axis in points[index]))
            and (confidences[index] is not None)
            and (confidences[index] > 0)
            for index in indices
        )
    )
    return {
        "xy_normalized": [
            round(sum((points[index][axis] for index in indices)) / len(indices), 9)
            for axis in (0, 1)
        ]
        if valid
        else None,
        "confidence": round(min(1.0, min((float(confidences[index]) for index in indices))), 9)
        if valid
        else None,
        "unavailable_reason": None if valid else "missing_or_nonpositive_confidence_anchor",
    }


def iter_native_records(
    rows: Sequence[Mapping[str, Any]], people: Sequence[Any],
) -> Iterator[dict[str, Any]]:
    """Retain every source observation, including frames without a target.

    Track identity is an association hypothesis. Multiple sequential tracks are
    separately retained; a frame with concurrent people has ambiguous target
    association. Neither an image visibility judgment nor missing palm anchors
    erases a measured wrist.
    """

    # Lazy import avoids the diagnostic module's existing validation imports.
    from .pose_cues import derive_body_relative_pose_observation

    by_frame: dict[int, list[Any]] = defaultdict(list)
    for person in people:
        by_frame[person.source_frame_index].append(person)
    previous: dict[tuple[str, str, str], tuple[int, int, dict[str, Any]]] = {}
    for row in rows:
        frame_people = by_frame.get(row["source_frame_index"], [])
        raw_people = {person["track_id"]: person for person in row["people"]}
        record: dict[str, Any] = {
            "schema_version": NATIVE_POSE_CONTRACT_VERSION,
            "source_frame_id": row["source_frame_id"],
            "source_frame_index": row["source_frame_index"],
            "source_frame": dict(row.get("source_frame", {})),
            "observation_time_us": row["canonical_time_us"],
            "display_start_us": row["canonical_time_us"],
            "display_end_us": row["canonical_end_us"],
            "analysis_sample_indices": list(row["analysis_sample_indices"]),
            "target_association": "unique_track_hypothesis" if len(frame_people) == 1 else
                                  "ambiguous_concurrent_people" if frame_people else "no_pose_observation",
            "people": [],
        }
        if "camera_shot_id" in row:
            record.update(camera_shot_id=row["camera_shot_id"], camera_cut_before=row.get("camera_cut_before", False))
        for person in frame_people:
            raw = raw_people[person.track_id]
            raw_points = raw["keypoints_image_xy_normalized"]
            geometry: dict[str, Any] = {}
            for curve, points in (("raw", raw_points), ("smoothed", person.points)):
                from .hand_motion import hand_geometry
                relative = derive_body_relative_pose_observation(
                    points, person.confidences, image_width=row["decoded_frame"]["width"],
                    image_height=row["decoded_frame"]["height"],
                )
                measurements: dict[str, Any] = {}
                for name, indices in POINT_INDICES.items():
                    value = _position(points, person.confidences, indices)
                    body_point = relative.get("points", {}).get(name, {})
                    value["xy_torso_lengths"] = body_point.get("xy_torso_lengths")
                    value["body_relative_unavailable_reason"] = body_point.get("unavailable_reason")
                    if name == "trunk_center":
                        value["xy_torso_lengths"] = [0.0, 0.0] if relative.get("trunk_frame", {}).get("status") == "available" else None
                    key = (person.segment_id, curve, name)
                    previous_value = previous.get(key)
                    value["motion"] = None
                    if len(frame_people) == 1 and value["xy_normalized"] is not None:
                        if previous_value is not None:
                            previous_time, previous_end, prior = previous_value
                            delta = row["canonical_time_us"] - previous_time
                            if delta > 0 and previous_end >= row["canonical_time_us"]:
                                vector = [value["xy_normalized"][axis] - prior["xy_normalized"][axis] for axis in (0, 1)]
                                body_vector = [value["xy_torso_lengths"][axis] - prior["xy_torso_lengths"][axis] for axis in (0, 1)] if value["xy_torso_lengths"] is not None and prior["xy_torso_lengths"] is not None else None
                                value["motion"] = {
                                    "support_start_us": previous_time, "support_end_us": row["canonical_time_us"],
                                    "delta_time_us": delta, "displacement_xy_normalized": vector,
                                    "speed_normalized_per_s": math.hypot(*vector) * 1_000_000 / delta,
                                    "displacement_xy_torso_lengths": body_vector,
                                    "speed_torso_lengths_per_s": math.hypot(*body_vector) * 1_000_000 / delta if body_vector else None,
                                }
                        previous[key] = (row["canonical_time_us"], row["canonical_end_us"], value)
                    else:
                        previous.pop(key, None)
                    measurements[name] = value
                geometry[curve] = {"points": measurements, "trunk_frame": relative.get("trunk_frame"),
                    "hands": {side: hand_geometry(points, person.confidences, side=side,
                        aspect=row["decoded_frame"]["width"] / row["decoded_frame"]["height"]) for side in ("left", "right")}}
            record["people"].append({
                "track_id": person.track_id, "continuity_segment_id": person.segment_id,
                "bbox_xyxy_normalized": list(person.bbox),
                "target_association": record["target_association"],
                "geometry": geometry,
                "smoothing_support": {str(index): value for index, value in enumerate(person.smoothing_support) if value is not None},
            })
        yield record


def summarize_native_window(
    records: Sequence[Mapping[str, Any]], start_us: int, end_us: int
) -> dict[str, Any]:
    """Measure per-continuity-segment paths without joining unknown intervals."""
    tracks: dict[tuple[str, str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    missing: list[dict[str, Any]] = []
    selected = [
        row
        for row in records
        if row["display_start_us"] < end_us and row["display_end_us"] > start_us
    ]
    for row in records:
        overlapping = row["display_start_us"] < end_us and row["display_end_us"] > start_us
        has_edge = any(
            (
                value["motion"] is not None
                and value["motion"]["support_start_us"] < end_us
                and (value["motion"]["support_end_us"] > start_us)
                for person in row["people"]
                for curve in ("raw", "smoothed")
                for value in person["geometry"][curve]["points"].values()
            )
        )
        if not overlapping and (not has_edge):
            continue
        if overlapping and row["target_association"] != "unique_track_hypothesis":
            missing.append(
                {
                    "start_us": max(start_us, row["display_start_us"]),
                    "end_us": min(end_us, row["display_end_us"]),
                    "reason": row["target_association"],
                    "source_frame_id": row["source_frame_id"],
                }
            )
        for person in row["people"]:
            for curve in ("raw", "smoothed"):
                for name, value in person["geometry"][curve]["points"].items():
                    tracks[person["continuity_segment_id"], curve, name].append((row, value))
    cursor = start_us
    for row in selected:
        if row["display_start_us"] > cursor:
            missing.append(
                {
                    "start_us": cursor,
                    "end_us": min(end_us, row["display_start_us"]),
                    "reason": "source_coverage_gap",
                    "source_frame_id": None,
                }
            )
        cursor = max(cursor, min(end_us, row["display_end_us"]))
    if cursor < end_us:
        missing.append(
            {
                "start_us": cursor,
                "end_us": end_us,
                "reason": "source_coverage_gap",
                "source_frame_id": None,
            }
        )
    output = []
    for (segment, curve, name), values in sorted(tracks.items()):
        supported = [
            (row, value)
            for row, value in values
            if value["xy_normalized"] is not None
            and row["display_start_us"] < end_us
            and (row["display_end_us"] > start_us)
        ]
        edges = [
            (
                row,
                value["motion"],
                max(
                    0,
                    min(end_us, value["motion"]["support_end_us"])
                    - max(start_us, value["motion"]["support_start_us"]),
                ),
            )
            for row, value in values
            if value["motion"] is not None
        ]
        edges = [edge for edge in edges if edge[2] > 0]
        covered = sum((edge[2] for edge in edges))
        peak = max(edges, key=lambda item: item[1]["speed_normalized_per_s"], default=None)
        body_edges = [edge for edge in edges if edge[1]["speed_torso_lengths_per_s"] is not None]
        body_covered = sum((edge[2] for edge in body_edges))
        body_path = sum(
            (edge[1]["speed_torso_lengths_per_s"] * edge[2] / 1000000 for edge in body_edges)
        )
        body_peak = max(
            body_edges, key=lambda item: item[1]["speed_torso_lengths_per_s"], default=None
        )
        extrema = []
        for axis in (0, 1):
            for operation, choose in (("minimum", min), ("maximum", max)):
                if supported:
                    row, value = choose(supported, key=lambda item: item[1]["xy_normalized"][axis])
                    extrema.append(
                        {
                            "axis": axis,
                            "kind": operation,
                            "value": value["xy_normalized"][axis],
                            "time_us": row["observation_time_us"],
                            "source_frame_id": row["source_frame_id"],
                        }
                    )
        reversals = []
        for previous_edge, edge in zip(edges, edges[1:]):
            if previous_edge[1]["support_end_us"] != edge[1]["support_start_us"]:
                continue
            if (
                sum(
                    (
                        a * b
                        for a, b in zip(
                            previous_edge[1]["displacement_xy_normalized"],
                            edge[1]["displacement_xy_normalized"],
                            strict=True,
                        )
                    )
                )
                < 0
            ):
                reversals.append(
                    {
                        "time_us": edge[1]["support_start_us"],
                        "support_start_us": previous_edge[1]["support_start_us"],
                        "support_end_us": edge[1]["support_end_us"],
                    }
                )

        def endpoint(item: tuple[Mapping[str, Any], Mapping[str, Any]]) -> dict[str, Any]:
            return {
                "time_us": item[0]["observation_time_us"],
                "source_frame_id": item[0]["source_frame_id"],
                "xy_normalized": item[1]["xy_normalized"],
                "xy_torso_lengths": item[1]["xy_torso_lengths"],
            }

        path = sum((edge[1]["speed_normalized_per_s"] * edge[2] / 1000000 for edge in edges))
        output.append(
            {
                "continuity_segment_id": segment,
                "curve": curve,
                "point": name,
                "first": endpoint(supported[0]) if supported else None,
                "last": endpoint(supported[-1]) if supported else None,
                "extrema": extrema,
                "path_length_normalized": path if edges else None,
                "mean_speed_normalized_per_s": path * 1000000 / covered if covered else None,
                "peak_speed": dict(peak[1]) if peak else None,
                "reversals": reversals,
                "motion_covered_duration_us": covered,
                "body_relative_path_length_torso_lengths": body_path if body_edges else None,
                "body_relative_mean_speed_torso_lengths_per_s": body_path * 1000000 / body_covered
                if body_covered
                else None,
                "body_relative_peak_speed": {
                    "speed_torso_lengths_per_s": body_peak[1]["speed_torso_lengths_per_s"],
                    "support_start_us": body_peak[1]["support_start_us"],
                    "support_end_us": body_peak[1]["support_end_us"],
                }
                if body_peak
                else None,
                "body_relative_motion_covered_duration_us": body_covered,
                "missing_source_frame_ids": [
                    row["source_frame_id"]
                    for row, value in values
                    if value["xy_normalized"] is None
                    and row["display_start_us"] < end_us
                    and (row["display_end_us"] > start_us)
                ],
            }
        )
    disagreements = []
    for row in selected:
        for person in row["people"]:
            for name in POINT_INDICES:
                raw = person["geometry"]["raw"]["points"][name]["xy_normalized"]
                smoothed = person["geometry"]["smoothed"]["points"][name]["xy_normalized"]
                if raw is not None and smoothed is not None:
                    disagreements.append(
                        (math.dist(raw, smoothed), row["observation_time_us"], name)
                    )
    return {
        "start_us": start_us,
        "end_us": end_us,
        "tracks": output,
        "missing_spans": missing,
        "raw_smoothed_max_disagreement": {
            "distance_normalized": max(disagreements)[0],
            "time_us": max(disagreements)[1],
            "point": max(disagreements)[2],
        }
        if disagreements
        else None,
    }


def native_pose_packet(
    records: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    start_us: int,
    end_us: int,
    artifact_sha256: str,
    maximum_chars: int | None = None,
) -> dict[str, Any]:
    """Pack every native observation; ask for a smaller overlapping window on overflow.

    Capacity failure is deliberate: this version does not make an unvalidated
    error-bound claim for lossy temporal compression.
    """
    if type(start_us) is not int or type(end_us) is not int or end_us <= start_us:
        raise ValueError("pose packet interval must be a positive integer-microsecond interval")
    if maximum_chars is not None and (type(maximum_chars) is not int or maximum_chars <= 0):
        raise ValueError("pose packet maximum_chars must be positive")
    selected_indices = [
        index
        for index, row in enumerate(records)
        if row["display_start_us"] < end_us and row["display_end_us"] > start_us
    ]
    selected = (
        list(records[max(0, selected_indices[0] - 1) : min(len(records), selected_indices[-1] + 2)])
        if selected_indices
        else []
    )
    observations = []
    segments = sorted(
        {person["continuity_segment_id"] for row in selected for person in row["people"]}
    )
    for row in selected:
        people = []
        for person in row["people"]:
            point_values = []
            for name in PACKET_POINT_NAMES:
                raw = person["geometry"]["raw"]["points"][name]
                smooth = person["geometry"]["smoothed"]["points"][name]
                point_values.append(
                    [
                        raw["xy_normalized"],
                        "raw"
                        if smooth["xy_normalized"] == raw["xy_normalized"]
                        else smooth["xy_normalized"],
                        raw["xy_torso_lengths"],
                        "raw"
                        if smooth["xy_torso_lengths"] == raw["xy_torso_lengths"]
                        else smooth["xy_torso_lengths"],
                        raw["confidence"],
                    ]
                )
            people.append(
                [person["track_id"], segments.index(person["continuity_segment_id"]), point_values]
            )
        support_only = not (row["display_start_us"] < end_us and row["display_end_us"] > start_us)
        observations.append(
            [
                row["source_frame_index"],
                row["observation_time_us"],
                row["display_end_us"],
                support_only,
                row["target_association"],
                people,
            ]
        )
    times = [row["observation_time_us"] for row in selected]
    packet = {
        "contract_version": NATIVE_POSE_CONTRACT_VERSION,
        "packing_policy_version": NATIVE_POSE_PACKET_POLICY_VERSION,
        "artifact_sha256": artifact_sha256,
        "start_us": start_us,
        "end_us": end_us,
        "representation": "native_complete_compact",
        "source_observation_count": len(selected),
        "transmitted_observation_count": len(selected),
        "window_observation_count": len(selected_indices),
        "derivative_support_observation_count": len(selected) - len(selected_indices),
        "additional_packing_position_error_bound_normalized": 0.0,
        "native_point_scope": "wrist, stable palm center and trunk; full elbow and fit-support dimensions remain in the hash-bound native artifact",
        "maximum_observation_gap_us": max((b - a for a, b in zip(times, times[1:])), default=0),
        "temporal_reduction": False,
        "point_names": list(PACKET_POINT_NAMES),
        "continuity_segments": segments,
        "motion_derivation": "Adjacent supported native points in one continuity segment determine displacement and actual-delta speed; null points and ambiguous association interrupt support. The string raw in a smoothed coordinate means exactly equal to its raw coordinate. Canonical summaries explicitly supply measured path, speed, peak support and reversals.",
        "observation_columns": [
            "source_frame_index",
            "observation_time_us",
            "display_end_us",
            "derivative_support_only",
            "target_association",
            "people",
        ],
        "person_columns": ["track_id", "continuity_segment_index", "points"],
        "point_columns": [
            "raw_xy_normalized",
            "smoothed_xy_normalized_or_raw",
            "raw_xy_torso_lengths",
            "smoothed_xy_torso_lengths_or_raw",
            "confidence",
        ],
        "observations": observations,
        "canonical_summaries": [
            summarize_native_window(
                selected, max(start_us, sample["start_us"]), min(end_us, sample["end_us"])
            )
            for sample in samples
            if sample["start_us"] < end_us and sample["end_us"] > start_us
        ],
    }
    from .hand_motion import summarize_hand_motion
    from .motion_timeline import build_motion_timeline

    packet["motion_timeline"] = build_motion_timeline(selected, samples, start_us, end_us)
    packet["hand_motion"] = summarize_hand_motion(selected, start_us, end_us)
    packet["window_motion_summary"] = summarize_native_window(selected, start_us, end_us)
    packet["hand_motion_subintervals"] = [
        summarize_hand_motion(selected, t, min(t + 1000000, end_us))
        for t in range(start_us, end_us, 1000000)
    ]
    packet["summary_track_columns"] = [
        "continuity_segment_index",
        "curve_index",
        "point_index",
        "first",
        "last",
        "extrema",
        "path_length_normalized",
        "mean_speed_normalized_per_s",
        "peak_speed",
        "reversals",
        "motion_covered_duration_us",
        "missing_source_frame_indices",
        "body_relative_path_length_torso_lengths",
        "body_relative_mean_speed_torso_lengths_per_s",
        "body_relative_peak_speed",
        "body_relative_motion_covered_duration_us",
    ]
    packet["curve_names"] = ["raw", "smoothed"]
    packet["endpoint_columns"] = [
        "time_us",
        "source_frame_index",
        "xy_normalized",
        "xy_torso_lengths",
    ]
    packet["extremum_columns"] = ["axis", "kind_index", "value", "time_us", "source_frame_index"]
    packet["extremum_kind_names"] = ["minimum", "maximum"]
    packet["peak_speed_columns"] = ["speed_normalized_per_s", "support_start_us", "support_end_us"]
    packet["body_relative_peak_speed_columns"] = [
        "speed_torso_lengths_per_s",
        "support_start_us",
        "support_end_us",
    ]
    packet["reversal_columns"] = ["time_us", "support_start_us", "support_end_us"]
    for summary in packet["canonical_summaries"]:
        compact = []
        for track in summary["tracks"]:
            if track["point"] not in PACKET_POINT_NAMES:
                continue
            track["continuity_segment_index"] = segments.index(track.pop("continuity_segment_id"))
            track["curve_index"] = packet["curve_names"].index(track.pop("curve"))
            track["point_index"] = packet["point_names"].index(track.pop("point"))
            for key in ("first", "last"):
                if track[key]:
                    track[key]["source_frame_index"] = int(
                        track[key].pop("source_frame_id").split(":")[-1]
                    )
                track[key] = (
                    [track[key][column] for column in packet["endpoint_columns"]]
                    if track[key]
                    else None
                )
            for extremum in track["extrema"]:
                extremum["source_frame_index"] = int(extremum.pop("source_frame_id").split(":")[-1])
                extremum["kind_index"] = packet["extremum_kind_names"].index(extremum.pop("kind"))
            track["extrema"] = [
                [extremum[column] for column in packet["extremum_columns"]]
                for extremum in track["extrema"]
            ]
            track["peak_speed"] = (
                [track["peak_speed"][column] for column in packet["peak_speed_columns"]]
                if track["peak_speed"]
                else None
            )
            track["body_relative_peak_speed"] = (
                [
                    track["body_relative_peak_speed"][column]
                    for column in packet["body_relative_peak_speed_columns"]
                ]
                if track["body_relative_peak_speed"]
                else None
            )
            track["reversals"] = [
                [reversal[column] for column in packet["reversal_columns"]]
                for reversal in track["reversals"]
            ]
            track["missing_source_frame_indices"] = [
                int(value.split(":")[-1]) for value in track.pop("missing_source_frame_ids")
            ]
            compact.append([track[column] for column in packet["summary_track_columns"]])
        summary["tracks"] = compact
    if (
        maximum_chars is not None
        and len(json.dumps(packet, separators=(",", ":"), allow_nan=False)) > maximum_chars
    ):
        raise PosePacketCapacityError(
            "native pose packet exceeds budget; split the temporal request with overlap; minimum-window overflow is a capacity failure"
        )
    return packet
