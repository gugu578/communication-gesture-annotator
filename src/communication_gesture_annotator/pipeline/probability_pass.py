# SPDX-License-Identifier: Apache-2.0
"""One chronological probability assessment; local motion fusion and grouping."""

from __future__ import annotations
from .motion_quality import SCORE_VERSION
from ..media.pose_conditioning import PoseConditioningResult
from .score_candidates import ScoreCandidateResult

# Public orchestration records the same measurement policy as the motion scorer.
MOTION_SCORE_VERSION = SCORE_VERSION
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from ..config import AnalysisConfig, DetectionConfig, ResearchConfig
from ..domain import AnalysisSample, JointActivityEvidence
from .llm_contracts import Phase2Model, TaskContractViolation
from .motion_quality import VERSION as MOTION_QUALITY_VERSION
from ..media.hand_quality import VERSION as HAND_QUALITY_VERSION

VERSION = "single-pass-non-g-probability-v0.1"
SCORE_SELECTION_VERSION = "non-g-score-selection-v0.1"
PROMPT_VERSION = "target-g-unit-phrase-membership-v0.13"
INPUT_VERSION = "target-motion-tables-v0.8"
MOTION_INPUT_POLICY = "camera-qualified-relative-motion-v0.2"
OMITTED_HAND_FIELDS = frozenset({"palm_net_displacement", "palm_path_efficiency"})
SCHEDULE_VERSION = "asymmetric-probability-targets-v0.3"
HISTORY_VERSION = "previous-llm-estimates-v0.3"
TASK = "non_g_probability"
USER_PREFIX = "Score only TARGET samples in output_slot order. Read HISTORY and LOOK-AHEAD to establish phrase linkage; they receive no outputs. Input:\n"
# Gesture concepts and annotation guidance are credited in docs/pipeline.md.
# Prompt, motion fusion and automatic grouping are this application's choices.
SYSTEM_PROMPT_TEMPLATE = """For the Phase1-selected speaker, score only the visibly marked TARGET samples. Return exactly {"non_g_unit_probability":[...]} in output_slot order. Let p be belief that this moment is OUTSIDE a communicative G-unit phrase: p=0 means membership; p=1 means non-membership. It is not movement intensity or confidence in a chosen class. Scene/speaker selection is already done; return no classes, hand labels, reasons, spans or boundaries.
A G-unit is speaking with the hands: one manual communicative phrase or a succession perceived to belong together through movement dynamics. It begins when participating hands leave contextually supported full or partial rest, includes preparation, expression, related repetitions, communicative holds and return toward rest, and ends on supported rest. A phrase usually lasts at least about 0.5 seconds; related phrases may continue for seconds, a minute or longer. These are expectations, not hard limits. A request edge never ends a phrase.
Read the supplied sequence, then focus on the TARGET interval. The configured maximum is $target_frames separate $sample_ms ms samples in a $target_ms ms target, using up to $history_frames frames ($history_seconds seconds) of HISTORY and $future_frames frames ($future_seconds seconds) of LOOK-AHEAD. Actual durations are in target_context; window settings are configurable. Confirmed scene/speaker exclusions and clip edges shorten context; pose-only gaps do not reset same-speaker visual history; never invent unseen history or future. Do not rate the entire $context_seconds-second context and copy that judgment across the target. For each target time, relate its image and motion row to preceding departures, linked dynamics and following settling/resumption. Equal probabilities are valid when membership evidence stays stable; change them when that evidence changes.
previous_estimates contains earlier LLM probabilities for already-scored HISTORY samples only. These are fallible estimates, NOT evidence, ground truth or confirmed phrase boundaries. Read the current RGB and measured motion first. Use previous estimates only as a tentative reminder of possible phrase continuation or rest, checking them against the supplied sequence. They may be wrong and must be disregarded when contradicted by the evidence. Never copy or average them into target scores, enforce smoothness, or assume a phrase/rest must continue because previous probabilities were confident. Typical durations are soft expectations, not a timer or proof of membership. Earlier estimates used overlapping images/motion and sometimes look-ahead, so they are correlated with this request, not independent corroboration. Missing rows mean no estimate was supplied, not rest or 0.5. Do not revise or return HISTORY estimates; score only the TARGET. No previous combined scores, candidate labels or reference annotations are supplied.
RGB pictures are clean person crops or full-frame fallbacks when a crop is unsupported. Their crop geometry can vary; use body-relative measurements for physical displacement. The continuity object labels pose/view discontinuities and context-only samples. Across those breaks, compare the same speaker visually but do not compute cross-break velocities, infer rest from missing measurements, or assume a track change ends a phrase. A camera cut can abruptly change the apparent position, size or orientation of the hands without actual hand movement. Do not treat that visual jump as evidence of movement, a G-unit onset, or a return to rest. Evaluate movement within each continuous camera shot. A genuine G-unit or non-G-unit may continue across a cut when the surrounding hand behavior supports that interpretation. Context-only samples receive no outputs. When preceding context is missing or a view discontinuity is marked, do not assume an unseen phrase or use earlier estimates alone to establish continuation. Clearly relaxed hands can receive high p without prior footage; use p near 0.5 only when the available hand evidence leaves membership unresolved.
For a visible onset, seek an abrupt departure/change relative to preceding rest in body-relative palm transport or finger/palm motion. It need not be large or fast; shoulder or torso movement without communicative hand use is not a manual phrase. For an ongoing phrase, do not demand its onset inside this request. Rhythms, directions, hand configurations and subsequent resumption may link quiet holds to the phrase.
Non-G time includes supported relaxed/default hand rest, often near the body, separating pauses, and incidental noncommunicative hand use. Rest requires contextual settling, not a speed minimum. Rest concerns the hands, not whole-body stillness: relaxed hands can remain at rest while being carried with the torso or shoulders. Configured separating rest is $rest_ms ms; pauses are often 0.5 seconds or much longer. Duration alone never makes a communicative hold into rest. Talking, clasped hands or holding an object alone does not establish a hold. Communication is plausible in these speaking scenarios, but not every hand posture is communicative. Neither symmetry nor asymmetry establishes rest or communication.
Semantic probability anchors (all intermediate values allowed):
p=0.00: virtually certain phrase membership, including linked preparation, hold or return.
p=0.10: strong membership/continuation evidence, even with low instantaneous motion.
p=0.25: probably part of the phrase; non-membership remains plausible.
p=0.50: balanced or insufficient evidence; never a default for quiet hands, small motion, context overlap, request tails or an unseen onset.
p=0.75: probably outside; supported rest/settling or incidental action is more plausible, with possible phrase linkage.
p=0.90: strong evidence of rest or noncommunicative action, little phrase linkage.
p=1.00: virtually certain non-membership across the available sequence.
Establish physical movement from the accepted pose measurements first: supported palm path/transport, rotation and finger articulation. Use these measurements even when movement is too small to see clearly in RGB. Reliable measured movement must not be described as stillness merely because the contact-sheet images look similar. RGB determines communicative purpose, object interaction and phrase linkage; movement alone does not prove communication. When motion_trace is supplied, read it together with matching frame times, then inspect exact numeric rows around the TARGET and changes from HISTORY. The trace is physical data, not a G-unit prediction. Its target band, global times and frame labels align with RGB. Transport is path per supported second; local rotation/finger traces are normalized by supplied scales and capped at $local_cap for display/scoring. Exact uncapped accepted rates remain in JSON.
motion_quality_flags identify unreliable data or potential statistical outliers. Unreliable values must not support a movement/rest inference. Potential outliers may be real brief movements: inspect their preserved original_value and RGB, but never treat them as reliable solely because they are large. Flagged values are excluded from the automatic physical score; native palm coordinates may be suppressed for affected intervals. A red X marks a flagged channel and a trace gap means unavailable/omitted, never zero motion or rest. Do not restore missing movement from previous estimates. A real sustained or abrupt movement is not an outlier just because it is large. Unflagged data still have measurement limitations.
No acceleration channel is supplied: inspect successive positions/speeds for changes, without inventing derivatives. Large fast incidental motion can have high p; a quiet communicative hold or small finger gesture can have low p. motion_non_g_probability is an uncalibrated physical clue, not a phrase label to copy. Local motion/context combination happens after your response.
Images are chronological frames/contact sheets, not an encoded video. Labels show HISTORY, TARGET or LOOK-AHEAD, global analysis frame and time in seconds; TARGET includes its output slot. frames.columns defines each numeric row; hand_columns defines left/right hand arrays. Image references identify image ordinal and cell. Read units and missingness explicitly. Native palm rows retain source timestamps/frame indices, distinct from analysis indices and output slots. Context-only rows have output_slot=null and receive no output. Return exactly output_count finite [0,1] numbers, including the last slots. Express uncertainty numerically, never null."""
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["non_g_unit_probability"],
    "properties": {
        "non_g_unit_probability": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "number", "minimum": 0, "maximum": 1},
        }
    },
}


def validate_response(value: Mapping[str, Any], core: Sequence[int]) -> dict[str, Any]:
    values = value.get("non_g_unit_probability") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != {"non_g_unit_probability"}
        or (not isinstance(values, list))
        or (len(values) != len(core))
        or any(
            (
                type(v) not in (int, float) or not math.isfinite(v) or (not 0 <= v <= 1)
                for v in values
            )
        )
    ):
        raise TaskContractViolation(
            "non_g_probability.invalid_scores",
            "return exactly one finite [0,1] probability per output_slot, with no nulls or other fields",
        )
    return {"non_g_unit_probability": list(values)}


def repair_instruction(trace: Mapping[str, Any] | None) -> str:
    core = list((trace or {}).get("core_sample_indices", []))
    return (
        'Return only {"non_g_unit_probability":[...]} with exactly '
        + str(len(core))
        + " numerical values in [0,1]. Reassess the same supplied evidence. No nulls, spans or text. Array slots map to these global indices, not the first context indices: "
        + json.dumps(core)
    )


def _finite_motion(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0)


def prepare_motion(
    samples: Sequence[AnalysisSample],
    resolutions: Mapping[int, Mapping[str, Any]],
    conditioning: Any,
    config: DetectionConfig,
    *,
    artifact_dir: Path | None = None,
    source_rights: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[dict[int, dict[str, Any]], str]:
    from .motion_quality import (
        VERSION as quality_version,
        SCORE_VERSION,
        conditioned_timeline,
        screen_timeline,
        score_measurement,
    )
    from .llm_runner import _digest
    from ..export.json_export import write_json
    from ..rights import write_rights_sidecar

    native = conditioning
    native_hash = (
        getattr(native, "manifest", {})
        .get("artifacts", {})
        .get("native_features", {})
        .get("sha256")
        or _digest(native.native_feature_records)
        if native is not None
        else None
    )
    identity = _digest(
        {
            "quality_version": quality_version,
            "score_version": SCORE_VERSION,
            "hand_geometry_policy_version": HAND_QUALITY_VERSION,
            "selected_pose_projection_sha256": getattr(native, "projection_sha256", None),
            "native_artifact_sha256": native_hash,
            "config": config.model_input_dict(),
            "resolutions": resolutions,
            "sample_times": [[s.index, s.start_us, s.end_us] for s in samples],
            "source_rights": source_rights,
        }
    )
    path = artifact_dir / "phase2-motion-quality.json" if artifact_dir is not None else None
    if path is not None and path.is_file() and (not overwrite):
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["input_sha256"] != identity or saved["data_sha256"] != _digest(saved["samples"]):
            raise ValueError("motion-quality artifact differs from current inputs or was modified")
        return ({int(i): v for i, v in saved["samples"].items()}, saved["data_sha256"])
    screened = (
        screen_timeline(conditioned_timeline(native, samples), resolutions, config)
        if native is not None
        else {}
    )
    measured = {
        s.index: score_measurement(
            screened.get(s.index, {"start_us": s.start_us, "end_us": s.end_us, "hands": {}}),
            config,
            native_hash,
        )
        for s in samples
    }
    serial = {str(i): v for i, v in measured.items()}
    digest = _digest(serial)
    if path is not None:
        write_json(
            path,
            {
                "policy_version": quality_version,
                "measurement_policy_version": SCORE_VERSION,
                "hand_geometry_policy_version": HAND_QUALITY_VERSION,
                "input_sha256": identity,
                "data_sha256": digest,
                "source_rights": source_rights,
                "parameters": {
                    "window_us": config.motion_outlier_window_us,
                    "robust_z": config.motion_outlier_z,
                    "minimum_ratio": config.motion_outlier_ratio,
                },
                "samples": serial,
            },
        )
        if source_rights is not None:
            write_rights_sidecar(path, source_rights, overwrite=True)
    return (measured, digest)


# Motion supplies a physical anchor; the model supplies communicative context.
def combine_probability(motion: float | None, contextual: float, strength: float) -> float:
    """A motion anchor with a bounded contextual log-odds correction, not calibration."""
    if motion is None:
        return contextual
    anchor = min(0.95, max(0.05, motion))
    log_odds = math.log(anchor / (1 - anchor)) + strength * (2 * contextual - 1)
    return 1 / (1 + math.exp(-log_odds))


def candidate_policy(config: DetectionConfig) -> dict[str, Any]:
    return {
        "rest_threshold": config.communication_rest_threshold,
        "rest_seed_threshold": config.communication_rest_seed_threshold,
        "minimum_departure_us": config.communication_minimum_departure_us,
    }


def probability_record(measurement: Mapping[str, Any], contextual: float, config: DetectionConfig) -> dict[str, Any]:
    llm_only = config.probability_score_source == "llm"
    q = contextual if llm_only else combine_probability(
        measurement["motion_non_g_probability"], contextual, config.probability_context_strength)
    result = {"policy_version": VERSION, **deepcopy(dict(measurement)),
        "llm_non_g_unit_probability": contextual, "non_g_unit_probability": q,
        "communicative_probability": 1 - q, "context_strength": config.probability_context_strength,
        "method": "motion_plus_context_probability" if measurement["motion_non_g_probability"] is not None else "visual_context_only",
        "context": {"kind": "probability_only"}, "reason": "Single-pass contextual probability with local motion combination.",
        "physical_state_encoding": "derived_from_combined_probability_not_model_classification",
        "candidate_policy": candidate_policy(config), "calibrated": False}
    if llm_only:
        result.update(score_selection_policy_version=SCORE_SELECTION_VERSION,
            score_source="llm", method="llm_probability",
            reason="Single-pass LLM probability used directly for local candidate creation.",
            physical_state_encoding="derived_from_selected_probability_not_model_classification")
    # Fusion retains its historical record format so immutable results remain
    # reproducible. The selected source is also saved in the run configuration.
    return result


def compatibility_value(record: Mapping[str, Any]) -> dict[str, Any]:
    """An internal compatibility projection, never an extra model output."""
    q = record["non_g_unit_probability"]
    state = "passive" if q >= record["candidate_policy"]["rest_threshold"] else "active"
    return {
        "state": state,
        "probabilities": {s: float(s == state) for s in ("passive", "active", "unknown")},
        "quality": 1.0,
        "reason": record["reason"],
    }


def native_palm_view(
    packet: Mapping[str, Any] | None,
    admitted_intervals: Sequence[tuple[int, int, Sequence[str]]] | None = None,
) -> dict[str, Any] | None:
    """Retain source cadence and exact positions; omit redundant raw summary trees."""
    if packet is None:
        return None
    names = packet.get("point_names", [])
    rows = []
    for observation in packet.get("observations", []):
        source, time, end, support, association, people = observation
        admitted = (
            {"left", "right"}
            if admitted_intervals is None
            else {
                side
                for start_us, end_us, sides in admitted_intervals
                if start_us < end and time < end_us
                for side in sides
            }
        )
        people_view = []
        qualified_people = (
            people if len(people) == 1 and association == "unique_track_hypothesis" else []
        )
        for person in qualified_people:
            track, segment, points = person
            trunk = points[names.index("trunk_center")] if "trunk_center" in names else None
            if (
                trunk is None
                or not _finite_motion(trunk[4])
                or trunk[4] < 0.3
                or (trunk[2] is None)
            ):
                continue
            palms = {}
            for side in ("left", "right"):
                name = side + "_palm_center"
                if name in names and side in admitted:
                    point = points[names.index(name)]
                    if not _finite_motion(point[4]) or point[4] < 0.3 or point[2] is None:
                        continue
                    palms[side] = {
                        "raw_xy_torso_lengths": point[2],
                        "smoothed_xy_torso_lengths": point[2] if point[3] == "raw" else point[3],
                        "confidence": point[4],
                    }
            if palms:
                people_view.append({"track_id": track, "segment_index": segment, "palms": palms})
        rows.append(
            {
                "source_frame_index": source,
                "time_us": time,
                "display_end_us": end,
                "derivative_support_only": support,
                "target_association": association,
                "people": people_view,
            }
        )
    return {
        "source_artifact_sha256": packet.get("artifact_sha256"),
        "representation": "native_palm_positions_with_per_frame_hand_motion_summaries",
        "all_source_observation_times_retained": True,
        "scope": "Exact qualified native palm coordinates; rejected or outside-admission coordinates are omitted but timestamps remain. Finger/palm rotation and path are integrated per analysis frame. Full reconstruction stays in the source artifact.",
        "rows": rows,
    }


def schedule_metadata(
    config: DetectionConfig, analysis: AnalysisConfig | None = None
) -> dict[str, Any]:
    period = (analysis or AnalysisConfig()).sample_period_us
    target = config.probability_target_frames * period
    history = config.probability_history_frames * period
    future = config.probability_future_frames * period
    return {
        "policy_version": SCHEDULE_VERSION,
        "target_us": target,
        "history_us": history,
        "lookahead_us": future,
        "maximum_context_us": target + history + future,
        "context_clipping": "Phase1 same-speaker visual-context edges; retain preceding RGB across camera/pose breaks; targets and future stop at the next camera view; no replacement with extra opposite-side context",
    }


def render_system_prompt(config: DetectionConfig, analysis: AnalysisConfig | None = None) -> str:
    """Render the prompt with the actual configured schedule and rest duration."""
    from string import Template
    from .motion_quality import LOCAL_DISPLAY_CAP

    analysis = analysis or AnalysisConfig()
    schedule = schedule_metadata(config, analysis)
    period = analysis.sample_period_us
    return Template(SYSTEM_PROMPT_TEMPLATE).substitute(
        target_frames=schedule["target_us"] // period,
        sample_ms=f"{period / 1000:g}",
        target_ms=f"{schedule['target_us'] / 1000:g}",
        history_frames=schedule["history_us"] // period,
        history_seconds=f"{schedule['history_us'] / 1000000.0:g}",
        future_frames=schedule["lookahead_us"] // period,
        future_seconds=f"{schedule['lookahead_us'] / 1000000.0:g}",
        context_seconds=f"{schedule['maximum_context_us'] / 1000000.0:g}",
        rest_ms=f"{analysis.minimum_separating_rest_us / 1000:g}",
        local_cap=f"{LOCAL_DISPLAY_CAP:g}",
    )


def selected_span(config: DetectionConfig, sample_count: int) -> tuple[int, int]:
    """The requested output range, preserving global indices and timestamps."""
    start = config.phase2_start_us // 100000
    end = config.phase2_end_us // 100000 if config.phase2_end_us is not None else sample_count
    if not 0 <= start < end <= sample_count:
        raise ValueError(
            f"Phase2 time range lies outside the {sample_count} available analysis frames"
        )
    return (start, end)


def frame_role(index: int, core: Sequence[int]) -> str:
    return "HISTORY" if index < core[0] else "LOOK-AHEAD" if index > core[-1] else "TARGET"


def visual_labels(
    samples: Sequence[AnalysisSample], core: Sequence[int], context: Sequence[int]
) -> dict[int, str]:
    slots = {i: n for n, i in enumerate(core)}
    return {
        i: f"{frame_role(i, core)} f{i} {samples[i].start_us / 1000000:.3f}s"
        + (f" slot{slots[i]}" if i in slots else "")
        for i in context
    }


def _native_palm_table(view: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if view is None:
        return None
    result = {k: v for k, v in view.items() if k != "rows"}
    result.update(
        columns=[
            "source_frame_index",
            "time_us",
            "display_end_us",
            "derivative_support_only",
            "target_association",
            "people",
        ],
        people_columns=["track_id", "segment_index", "left_palm", "right_palm"],
        palm_columns=["raw_xy_torso_lengths", "smoothed_xy_torso_lengths", "confidence"],
        rows=[],
    )
    for row in view["rows"]:
        people = [
            [
                person["track_id"],
                person["segment_index"],
                *[
                    [person["palms"][side][key] for key in result["palm_columns"]]
                    if side in person["palms"]
                    else None
                    for side in ("left", "right")
                ],
            ]
            for person in row["people"]
        ]
        result["rows"].append([*[row[k] for k in result["columns"][:-1]], people])
    return result


def previous_estimate_context(
    samples: Sequence[AnalysisSample],
    core: Sequence[int],
    context: Sequence[int],
    estimates: Mapping[int, float],
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Bound prior outputs to delivered HISTORY; never fabricate missing estimates."""
    indices = [i for i in context if i < core[0] and i in estimates] if enabled else []
    if indices:
        validate_response({"non_g_unit_probability": [estimates[i] for i in indices]}, indices)
    return {
        "policy_version": HISTORY_VERSION,
        "enabled": enabled,
        "epistemic_status": "fallible_model_estimates_not_evidence",
        "source": "validated_previous_llm_outputs_from_this_pass",
        "scope": "already-scored HISTORY samples in the same Phase1-confirmed speaker context; retain across pose-only gaps, never across scene/identity exclusions or test-range starts",
        "missing": "no estimate supplied; never infer rest or a numerical probability",
        "columns": ["sample_index", "start_us", "end_us", "llm_non_g_unit_probability"],
        "rows": [[i, samples[i].start_us, samples[i].end_us, estimates[i]] for i in indices],
    }


def build_prompt(*, subject: Mapping[str, Any], samples: Sequence[AnalysisSample], core: Sequence[int],
                 context: Sequence[int], synchronization: Mapping[str, Any],
                 measures: Mapping[int, Mapping[str, Any]], packet: Mapping[str, Any] | None,
                 previous_estimates: Mapping[int, float] | None = None, history_enabled: bool = True,
                 motion_trace: Mapping[str, Any] | None = None,
                 resolutions: Mapping[int, Mapping[str, Any]] | None = None) -> str:
    slots = {index: slot for slot, index in enumerate(core)}
    mapping = {v["sample_index"]: v for v in synchronization["sample_to_visual_mapping"]}
    hand_columns = sorted({key for i in context for hand in measures[i]["hand_measurements"].values()
                           for key in hand if key not in OMITTED_HAND_FIELDS})
    frame_columns = ["sample_index", "role", "output_slot", "time_us", "end_us", "source_frame",
        "image_references", "motion_non_g_probability", "left_hand", "right_hand", "motion_quality_flags",
        "hand_quality", "motion_probability_status"]
    frames = []
    for index in context:
        visual = mapping[index]
        if not visual["image_references"]:
            raise ValueError(f"probability frame {index} has no delivered image mapping")
        measurement = measures[index]
        image_refs = [[r["image_ordinal"], r["cell"].get("ordinal", 0),
            r["cell"].get("row", 0), r["cell"].get("column", 0)] for r in visual["image_references"]]
        hands = measurement["hand_measurements"]
        frames.append([index, frame_role(index, core), slots.get(index), samples[index].start_us, samples[index].end_us,
            visual["source_frame"], image_refs, measurement["motion_non_g_probability"],
            *[[hands[side].get(k) for k in hand_columns] if side in hands else None for side in ("left", "right")],
            [[side, f['field'], f['status'], f['reason'], f['raw_value'] if f['status'] == 'potential_outlier' else None]
             for side, flags in measurement.get('quality_flags', {}).items() for f in flags
             if f['field'] not in OMITTED_HAND_FIELDS], measurement.get('hand_quality', {}),
             measurement.get('motion_probability_status', 'available' if measurement['motion_non_g_probability'] is not None else 'insufficient_support')])
    payload = {"contract_version": VERSION, "subject_id": subject["subject_id"],
        "input_version": INPUT_VERSION, "selected_speaker": subject.get("visual_description"), "output_count": len(core),
        "output_slot_to_sample_index": list(core), "context_sample_indices": list(context),
        "target_context": {"target_start_us": samples[core[0]].start_us, "target_end_us": samples[core[-1]].end_us,
            "context_start_us": samples[context[0]].start_us, "context_end_us": samples[context[-1]].end_us,
            "available_history_us": samples[core[0]].start_us - samples[context[0]].start_us,
            "available_lookahead_us": samples[context[-1]].end_us - samples[core[-1]].end_us,
            "actual_target_frames": len(core), "actual_history_frames": sum(i < core[0] for i in context),
            "actual_future_frames": sum(i > core[-1] for i in context)},
        "units": {"time": "integer microseconds, half-open intervals", "transport": "torso lengths per second",
            "palm_path": "torso lengths traveled in this sample", "palm_xy": "torso-length coordinates",
            "orientation_speed": "projected hand rotation relative to the torso longitudinal axis, radians per second", "extension_speed": "extension ratio per second",
            "articulation_speed": "hand lengths per second", "palm_supported_us": "microseconds",
            "coverage_and_quality": "fractions in [0,1]", "missing": "null = unavailable, not zero"},
        "frames": {"columns": frame_columns, "hand_columns": hand_columns,
            "quality_flag_columns": ['hand', 'field', 'status', 'reason', 'original_value'],
            "image_reference_columns": ["image_ordinal", "cell_ordinal", "row", "column"], "rows": frames},
        "motion_trace": dict(motion_trace) if motion_trace is not None else None,
        "motion_input_policy": MOTION_INPUT_POLICY,
        "continuity": {"policy": "same-speaker RGB context may cross a pose break; never derive motion across track/view discontinuities or treat missing motion as rest",
            "context_only_sample_indices": [i for i in context if resolutions and resolutions[i].get("exclude_from_detection")],
            "motion_discontinuities": [{"sample_index": i, "reasons": resolutions[i].get("pose_motion_discontinuity_reasons", [])}
                for i in context if resolutions and resolutions[i].get("pose_motion_discontinuity")],
            "camera_edits": [{"sample_index": i, "times_us": resolutions[i]["camera_cut_times_us"],
                "camera_shot_id": resolutions[i].get("camera_shot_id")}
                for i in context if resolutions and resolutions[i].get("camera_cut_times_us")]},
        "motion_quality": {'policy_version': MOTION_QUALITY_VERSION,
            'hand_geometry_policy_version': HAND_QUALITY_VERSION,
            'normalization_scales': next((measures[i]['motion_scales'] for i in context if 'motion_scales' in measures[i]), None),
            'rule': 'Unreliable values are null; originals remain only in local diagnostic artifacts. Potential statistical outliers retain finite originals in flags for contextual inspection, excluded from the physical score. Nulls and flagged gaps never establish rest. Reliability is not a multiplier on speed or probability.',
            'hand_quality_fields': 'Per hand: rotation_reference; reliable_finger_count (0..5 anatomical support, distinct from temporal coverage); channel_unavailable_reasons. Fewer reliable fingers means partial observation, never stillness.',
            'probability_missingness': 'motion_probability_status=insufficient_support and motion_non_g_probability=null mean no supported whole-hand physical anchor, not a measured 0.5. Reliable component measurements remain available.',
            'native_rule': 'Native palm coordinates are omitted in intervals with flagged palm transport; original source artifact is preserved.'},
        "previous_estimates": previous_estimate_context(samples, core, context, previous_estimates or {}, enabled=history_enabled),
        "native_palm_trajectory": _native_palm_table(native_palm_view(packet,
            [(samples[i].start_us, samples[i].end_us, [side for side in measures[i]["admitted_hands"]
                if measures[i]['hand_measurements'][side].get('palm_xy') is not None
                and not any(f['field'] in ('palm_speed', 'palm_xy', 'palm_path') for f in measures[i].get('quality_flags', {}).get(side, []))]) for i in context]))}
    return USER_PREFIX + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# Group locally from the saved combined score: no transition-review model pass.
def candidate_diagnostics(
    samples: Sequence[AnalysisSample],
    evidence: Sequence[JointActivityEvidence],
    analysis: AnalysisConfig,
    *,
    participant_id: str = "participant-1",
    candidate_policy_version: str = "contextual-score-candidates-v0.3",
) -> ScoreCandidateResult | None:
    from .motion_grounding import evidence_motion_record
    from .grouping import joint_unknown_is_bridgeable
    from .score_candidates import ScoreEvidence, ScoreCandidatePolicy, propose_score_candidates

    records = [evidence_motion_record(e) for e in evidence]
    policies = [r["candidate_policy"] for r in records if r and r.get("policy_version") == VERSION]
    if not policies:
        return None
    if any((p != policies[0] for p in policies)):
        raise ValueError("mixed single-pass candidate policies")
    rows = []
    for item, record in zip(evidence, records, strict=True):
        if record is not None and record.get("policy_version") != VERSION:
            raise ValueError("incompatible probability assessment contracts")
        if record is None and item.state.value != "unknown":
            raise ValueError("resolved sample missing probability evidence")
        excluded = any((r.get("exclude_from_detection", False) for r in item.resolution_records))
        rows.append(
            ScoreEvidence(
                non_g_score=record["non_g_unit_probability"] if record else None,
                excluded=excluded,
                barrier=item.is_barrier,
                bridgeable=joint_unknown_is_bridgeable(item),
                references=item.evidence_references
                + item.pose_evidence_references
                + item.rgb_evidence_references,
            )
        )
    return propose_score_candidates(
        samples, rows, analysis, ScoreCandidatePolicy(**policies[0]), participant_id=participant_id, policy_version=candidate_policy_version
    )


def run_pass(*, samples: Sequence[AnalysisSample], subject: Mapping[str, Any],
             resolutions: Mapping[int, Mapping[str, Any]], frames: Mapping[int, Mapping[str, Any]],
             config: ResearchConfig, model: Phase2Model, conditioning: PoseConditioningResult | None, visuals: Callable[..., Any],
             slots: list[JointActivityEvidence | None], trace: Any, artifact_dir: Path | None = None,
             source_rights: Mapping[str, Any] | None = None, overwrite: bool = False) -> list[dict[str, Any]]:
    """One assessment per owned sample; no scene, recovery or boundary model calls."""
    from . import llm_runner as runtime
    from .joint_runner import _joint_evidence
    from .motion_context import synchronized_evidence
    from ..audit.progress import StepProgress
    from .camera_context import shot_ranges, plan_camera_windows, supported_estimates
    eligible = {s.index for s in samples if not resolutions[s.index]["exclude_from_detection"]}
    context_eligible = [s.index for s in samples if resolutions[s.index].get(
        "phase2_context_usable", not resolutions[s.index]["exclude_from_detection"])]
    start, end = selected_span(config.detection, len(samples))
    schedule = schedule_metadata(config.detection, config.analysis)
    system_prompt = render_system_prompt(config.detection, config.analysis)
    # Count the very same windows we execute, including partial targets and
    # admission gaps. This is bookkeeping, not another model pass.
    window_groups = []
    camera_ranges = {}
    for run in runtime._contiguous_runs(context_eligible, label="Phase1 visual context"):
        camera_ranges.update(shot_ranges(run, resolutions))
        window_groups.append([w for target_run in runtime._contiguous_runs(
            [i for i in run if i in eligible and start <= i < end], label="Phase2 output ownership")
            for w in plan_camera_windows(run, target_indices=target_run, ranges=camera_ranges,
                target_items=schedule["target_us"] // config.analysis.sample_period_us,
                history_items=schedule["history_us"] // config.analysis.sample_period_us,
                lookahead_items=schedule["lookahead_us"] // config.analysis.sample_period_us)])
    runtime._record_event(trace, 'activity', 'motion_quality', 'started')
    all_measures, quality_hash = prepare_motion(samples, resolutions, conditioning, config.detection,
        artifact_dir=artifact_dir, source_rights=source_rights, overwrite=overwrite)
    runtime._record_event(trace, 'activity', 'motion_quality', 'completed',
        sample_count=len(all_measures), flagged_sample_count=sum(any(v['quality_flags'].values()) for v in all_measures.values()))
    from .probability_resume import ProbabilityReplay
    from ..media.phase2_rgb import VERSION as rgb_version
    from ..media.motion_trace import VERSION as chart_version
    replay = ProbabilityReplay(model, [
        f"activity-probability-v13-{subject['subject_id']}-{w.output_indices[0]:08d}-{w.output_indices[-1]+1:08d}"
        for windows in window_groups for w in windows], {
        "system_prompt": system_prompt, "input_version": INPUT_VERSION,
        "schema": SCHEMA, "rgb_version": rgb_version, "chart_version": chart_version,
        "motion_quality_sha256": quality_hash, "subject": subject,
        "frames": frames, "resolutions_sha256": runtime._digest(resolutions),
        "schedule": schedule, "config": config.detection.model_input_dict()})
    progress = StepProgress(trace, "activity", sum(map(len, window_groups)), model=model,
        cached_calls_available=len(replay.prefix),
        planned_uncached_calls=sum(map(len, window_groups))-len(replay.prefix),
        subject_id=subject["subject_id"], sample_count=sum(start <= i < end for i in eligible),
        target_frames=schedule["target_us"] // config.analysis.sample_period_us,
        history_frames=schedule["history_us"] // config.analysis.sample_period_us,
        future_frames=schedule["lookahead_us"] // config.analysis.sample_period_us)
    traces = []
    # Context survives measurement-only gaps; output ownership still excludes
    # them. Previous estimates persist within this same-speaker context only.
    for windows in window_groups:
        previous_outputs: dict[int, float] = {}
        eligible_estimates: set[int] = set()
        for window in windows:
            core, context = list(window.output_indices), list(window.input_indices)
            priors, prior_policy = supported_estimates(previous_outputs, core=core, context=context,
                ranges=camera_ranges, resolutions=resolutions,
                history_items=schedule["history_us"] // config.analysis.sample_period_us,
                eligible_estimates=eligible_estimates)
            measured = {i: all_measures[i] for i in context}
            request_id = f"activity-probability-v13-{subject['subject_id']}-{core[0]:08d}-{core[-1]+1:08d}"
            history = previous_estimate_context(samples, core, context, priors,
                enabled=config.detection.probability_history_enabled)
            replay_key = replay.key(core=core, context=context, history=history, policy=prior_policy)
            restored = replay.restore(request_id, replay_key,
                lambda value, owned=core: validate_response(value, owned))
            if restored is not None:
                call, restored_trace = restored
                parsed = validate_response(call.payload, core)
                packet = None
                prompt = None
                motion_trace = restored_trace.get("motion_trace")
            else:
                packet = (conditioning.temporal_packet(samples[context[0]].start_us, samples[context[-1]].end_us)
                    if conditioning is not None else None)
                images, refs = visuals(request_id, context, "activity", visible_labels=visual_labels(samples, core, context))
                motion_trace = None
                if artifact_dir is not None and config.detection.probability_motion_trace_enabled:
                    from ..media.motion_trace import write_motion_trace
                    from ..rights import unrestricted_or_unspecified_rights
                    if len(images) >= config.inference.maximum_images_per_request:
                        raise ValueError('no image slot available for the required motion trace')
                    path, motion_trace = write_motion_trace(artifact_dir/'motion-traces', samples, core, context, measured,
                        camera_cut_times_us=sorted({t for i in context for t in resolutions[i].get('camera_cut_times_us', [])}),
                        image_ordinal=len(images), rights=source_rights if source_rights is not None else unrestricted_or_unspecified_rights(),
                        width=min(1400, config.inference.maximum_image_width), height=min(780, config.inference.maximum_image_height))
                    images = [*images, path]
                synchronization = synchronized_evidence(context, core, frames, refs, packet)
                prompt = build_prompt(subject=subject, samples=samples, core=core, context=context,
                    synchronization=synchronization, measures=measured, packet=packet,
                    previous_estimates=priors, history_enabled=config.detection.probability_history_enabled,
                    motion_trace=motion_trace, resolutions=resolutions)
                schema = deepcopy(SCHEMA)
                schema["properties"]["non_g_unit_probability"].update(minItems=len(core), maxItems=len(core))
                if len(system_prompt) + len(prompt) + len(json.dumps(schema)) > config.inference.maximum_prompt_characters:
                    raise ValueError("single-pass probability prompt exceeds configured capacity; increase the limit or reduce the configured target/history/future frame counts explicitly")
                trace_context = {"prompt_version": PROMPT_VERSION, "subject_id": subject["subject_id"],
                    "input_version": INPUT_VERSION, "target_schedule": schedule, "selected_sample_span": [start, end],
                    "system_prompt_sha256": runtime._digest(system_prompt),
                    "core_sample_indices": core, "context_sample_indices": context, "image_refs": refs,
                    "phase1_admission_sha256": runtime._digest([resolutions[i] for i in context]),
                    "previous_estimates_policy_version": HISTORY_VERSION,
                    "previous_estimates_sha256": runtime._digest(history),
                    "previous_estimates_count": len(history["rows"]),
                    "previous_estimate_admission": prior_policy,
                    "motion_view_sha256": runtime._digest(measured)}
                trace_context.update(motion_quality_sha256=quality_hash, motion_trace=motion_trace,
                    motion_input_policy=MOTION_INPUT_POLICY)
                call = runtime._model_call(model, trace, "activity", request_id=request_id, task=TASK,
                    system_prompt=system_prompt, user_prompt=prompt, images=images, response_schema=schema,
                    trace_context=trace_context, response_validator=lambda value, owned=core: validate_response(value, owned))
                parsed = validate_response(call.payload, core)
                # The model wrapper has persisted its validated checkpoint by now.
                # These packets can be reproduced byte-for-byte from retained RGB
                # and motion evidence, including when replaying a resumed run.
                from ..storage import compact_enabled, release_image
                if artifact_dir is not None and compact_enabled():
                    for image in images:
                        path = Path(image)
                        for cache in (artifact_dir / "phase2-rgb" / "sheets", artifact_dir / "motion-traces"):
                            if path.resolve().is_relative_to(cache.resolve()):
                                release_image(path, cache)
                                break
            for index, contextual in zip(core, parsed["non_g_unit_probability"], strict=True):
                record = probability_record(measured[index], contextual, config.detection)
                if slots[index] is not None:
                    raise ValueError("single-pass probability ownership overlaps")
                slots[index] = _joint_evidence(samples[index], compatibility_value(record), request_id,
                    call.response_sha256, resolutions[index], motion_record=record)
            # Only validated outputs from completed earlier targets enter memory.
            # Checkpoint replay reconstructs the same chronological mapping.
            previous_outputs.update(zip(core, parsed["non_g_unit_probability"], strict=True))
            if prior_policy['current_output_eligible_for_future_estimates']:
                eligible_estimates.update(core)
            traces.append({"request_id": request_id, "core_sample_indices": core, "context_sample_indices": context,
                "response_sha256": call.response_sha256, "model_output": deepcopy(call.model_payload or call.payload),
                "validated_output": parsed, "motion_measurements": measured,
                "native_input_artifact_sha256": (packet or {}).get("artifact_sha256"),
                "model_view_sha256": runtime._digest(prompt) if prompt is not None else None, "pass_version": VERSION,
                "input_version": INPUT_VERSION, "target_schedule": schedule,
                "previous_estimates": history,
                "previous_estimate_admission": prior_policy,
                "motion_quality_sha256": quality_hash, "motion_trace": motion_trace,
                "motion_input_policy": MOTION_INPUT_POLICY,
                "prompt_version": PROMPT_VERSION})
            if restored is not None:
                traces[-1].update(restored_trace)
            replay.remember(request_id, replay_key, traces[-1])
            if artifact_dir is not None:
                from ..adapters.image_recovery import release_checkpointed_images
                release_checkpointed_images(artifact_dir, request_id)
            progress.advance()
    progress.finish()
    return traces
