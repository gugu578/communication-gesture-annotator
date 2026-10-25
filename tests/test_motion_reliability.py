# SPDX-License-Identifier: Apache-2.0
"""Synthetic motion geometry, speaker association and configured prompt checks."""

from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from communication_gesture_annotator.config import AnalysisConfig
from communication_gesture_annotator.media.hand_motion import hand_geometry, summarize_hand_motion
from communication_gesture_annotator.media.hand_quality import qualified_hand, condition_native_hands
from communication_gesture_annotator.pipeline.selected_pose import select_segments, SelectedPoseView
from communication_gesture_annotator.pipeline import probability_pass as p
from communication_gesture_annotator.pipeline import motion_quality as q
from communication_gesture_annotator.domain import AnalysisSample


def _samples(count):
    return [
        AnalysisSample(i, i * 100000, (i + 1) * 100000, i * 100000 + 50000) for i in range(count)
    ]


from communication_gesture_annotator.config import ResearchConfig, OpenWebUIConfig


def configuration():
    return ResearchConfig(open_webui=OpenWebUIConfig(model="synthetic"))


def hand(angle: float = 0.0) -> dict:
    points = [[0.0, 0.0] for _ in range(133)]
    # Fully synthetic palm; each finger has four separate, supported joints.
    for offset in (91, 112):
        points[offset] = [0.5, 0.5]
        for n in range(5):
            for joint in range(4):
                x, y = (n - 2) * 0.02, 0.04 + joint * 0.02
                points[offset + 1 + n * 4 + joint] = [
                    0.5 + x * math.cos(angle) - y * math.sin(angle),
                    0.5 + x * math.sin(angle) + y * math.cos(angle),
                ]
    return hand_geometry(points, [0.9] * 133, side="left", aspect=1.0)


def trunk(angle: float = 0.0) -> dict:
    return {
        "status": "available",
        "anchor_confidence": 0.9,
        "longitudinal_axis_xy": [-math.sin(angle), math.cos(angle)],
    }


def native(angles: list[tuple[float, float]]) -> list[dict]:
    return [
        {
            "source_frame_index": i,
            "observation_time_us": i * 40000,
            "display_start_us": i * 40000,
            "display_end_us": (i + 1) * 40000,
            "target_association": "unique_track_hypothesis",
            "people": [
                {
                    "continuity_segment_id": "speaker",
                    "geometry": {
                        "smoothed": {"trunk_frame": trunk(body), "hands": {"left": hand(palm)}}
                    },
                }
            ],
        }
        for i, (palm, body) in enumerate(angles)
    ]


class MotionReliabilityTest(unittest.TestCase):
    def test_common_rotation_cancels_and_hand_rotation_remains(self):
        for angles, expected in (([(0, 0), (0.4, 0.4)], 0.0), ([(0, 0), (0.4, 0)], 10.0)):
            raw = native(angles)
            snapshot = deepcopy(raw)
            conditioned = condition_native_hands(raw)
            summary = summarize_hand_motion(conditioned, 0, 40000)
            rotation = summary["groups"][0]["features"]["palm_relative_angle_rad"]["mean_speed"]
            self.assertAlmostEqual(rotation, expected)
            self.assertEqual(raw, snapshot)
            self.assertEqual([r["observation_time_us"] for r in conditioned], [0, 40000])

    def test_angle_wrap_and_native_time_are_used(self):
        rows = condition_native_hands(native([(math.pi - 0.01, 0), (-math.pi + 0.01, 0)]))
        value = summarize_hand_motion(rows, 0, 40000)["groups"][0]["features"][
            "palm_relative_angle_rad"
        ]
        self.assertAlmostEqual(value["mean_speed"], 0.5)

    def test_bad_torso_rejects_rotation_without_removing_fingers(self):
        body = trunk()
        body["anchor_confidence"] = 0.05
        result = qualified_hand(hand(), body)
        self.assertNotIn("palm_relative_angle_rad", result["features"])
        self.assertIn("middle_articulation", result["features"])
        self.assertEqual(
            result["channel_unavailable_reasons"]["orientation_speed"],
            "unreliable_torso_rotation_reference",
        )

    def test_collapsed_middle_axis_does_not_amplify_finger_displacement(self):
        raw = hand()
        # Re-express the same geometry with an artificially collapsed old scale.
        raw["local_joint_coordinates"] = [
            [v * 100 for v in point] for point in raw["local_joint_coordinates"]
        ]
        raw["local_joint_coordinates"][9] = [0.0, 1.0]
        raw["hand_scale_image_heights"] /= 100
        result = qualified_hand(raw, trunk())
        self.assertNotIn("palm_relative_angle_rad", result["features"])
        self.assertFalse(any(k.endswith("_articulation") for k in result["features"]))
        self.assertTrue(any(k.endswith("_extension") for k in result["features"]))
        self.assertLess(max(math.hypot(*point) for point in result["local_joint_coordinates"]), 5)

    def test_bad_finger_does_not_erase_other_fingers(self):
        raw = hand()
        raw["local_joint_coordinates"][8] = [100.0, 0.0]
        result = qualified_hand(raw, trunk())
        self.assertEqual(result["reliable_finger_count"], 4)
        self.assertNotIn("index_articulation", result["features"])
        self.assertIn("middle_articulation", result["features"])

    def test_scale_jump_blocks_articulation_edge_and_preserves_independent_channels(self):
        rows = native([(0, 0), (0, 0), (0, 0)])
        rows[1]["people"][0]["geometry"]["smoothed"]["hands"]["left"][
            "hand_scale_image_heights"
        ] *= 10
        result = condition_native_hands(rows)
        self.assertEqual(
            result[1]["people"][0]["geometry"]["smoothed"]["hands"]["left"][
                "blocked_derivative_suffixes"
            ],
            ["_articulation"],
        )
        summary = summarize_hand_motion(result, 0, 80000)["groups"][0]["features"]
        self.assertNotIn("middle_articulation", summary)
        self.assertIn("middle_extension", summary)
        self.assertIn("palm_relative_angle_rad", summary)

    def test_dynamic_prompt_changes_only_configured_guidance(self):
        cfg = configuration().detection
        cfg = replace(
            cfg,
            probability_target_frames=1,
            probability_history_frames=0,
            probability_future_frames=0,
        )
        rendered = p.render_system_prompt(
            cfg, replace(AnalysisConfig(), minimum_separating_rest_us=700000)
        )
        self.assertIn("1 separate 100 ms samples in a 100 ms target", rendered)
        self.assertIn("0 frames (0 seconds) of HISTORY", rendered)
        self.assertIn("0 frames (0 seconds) of LOOK-AHEAD", rendered)
        self.assertIn("Configured separating rest is 700 ms", rendered)
        self.assertNotIn("skeleton overlay", rendered)
        self.assertIn("A phrase usually lasts at least about 0.5 seconds", rendered)
        self.assertNotIn("$", rendered)

    def test_incomplete_quiet_motion_has_no_physical_rest_anchor(self):
        cfg = configuration().detection
        entry = {
            "start_us": 0,
            "end_us": 100000,
            "hands": {"left": {"values": {"palm_speed": 0.1}, "flags": []}},
        }
        current = q.score_measurement(entry, cfg, "a" * 64)
        self.assertIsNone(current["motion_non_g_probability"])
        self.assertEqual(p.probability_record(current, 0.8, cfg)["non_g_unit_probability"], 0.8)

    def test_weak_native_trunk_cannot_leak_unqualified_palm_coordinates(self):
        from synthetic import SyntheticNativeConditioning

        packet = SyntheticNativeConditioning(2).temporal_packet(0, 200000)
        index = packet["point_names"].index("trunk_center")
        for row in packet["observations"]:
            for person in row[5]:
                person[2][index][4] = 0.05
        result = p.native_palm_view(packet)
        self.assertEqual(len(result["rows"]), len(packet["observations"]))
        self.assertTrue(all(not row["people"] for row in result["rows"]))

    def test_bounded_qualification_matches_whole_native_timeline(self):
        from synthetic import SyntheticNativeConditioning
        from communication_gesture_annotator.media.motion_timeline import build_motion_timeline

        source = SyntheticNativeConditioning(105)
        for row in source.native_feature_records:
            for person in row["people"]:
                for geometry in person["geometry"].values():
                    geometry["hands"] = {
                        side: hand(row["observation_time_us"] / 1e6 * 0.2)
                        for side in ("left", "right")
                    }
        # A normalization discontinuity close to a chunk boundary must not vanish.
        for geometry in source.native_feature_records[298]["people"][0]["geometry"].values():
            geometry["hands"]["left"]["hand_scale_image_heights"] *= 10
        samples = _samples(105)
        bins = [
            {"sample_index": s.index, "start_us": s.start_us, "end_us": s.end_us} for s in samples
        ]
        full = build_motion_timeline(
            condition_native_hands(source.native_feature_records), bins, 0, 10500000
        )
        chunked = q.conditioned_timeline(source, samples)
        self.assertEqual(full["rows"], chunked["rows"])
        self.assertEqual(full["hand_quality"], chunked["hand_quality"])


class SelectedSpeakerTest(unittest.TestCase):
    def fixture(self, tracks: list[list[str]]) -> tuple:
        samples = _samples(len(tracks))
        features = [
            {
                "sample_index": i,
                "start_us": i * 100000,
                "end_us": (i + 1) * 100000,
                "candidates": [
                    {"continuity_segment_id": t, "track_id": t, "coverage_fraction": 1.0}
                    for t in entries
                ],
            }
            for i, entries in enumerate(tracks)
        ]
        visibility = {
            "samples": [
                {
                    "sample_index": i,
                    "scene_membership": "main_scene",
                    "target_visibility": "visible",
                }
                for i in range(len(tracks))
            ]
        }
        return samples, visibility, features

    def test_additional_detection_does_not_remove_anchored_speaker(self):
        values = self.fixture(
            [["speaker"], ["speaker", "slide"], ["speaker", "slide"], ["speaker"]]
        )
        selected = select_segments(*values)
        self.assertEqual([r["continuity_segment_id"] for r in selected.values()], ["speaker"] * 4)

    def test_unanchored_or_conflicting_people_are_unresolved(self):
        for tracks in ([["a", "b"]] * 4, [["a"], ["a", "b"], ["b"]]):
            with self.subTest(tracks=tracks):
                selected = select_segments(*self.fixture(tracks))
                self.assertIsNone(selected[1]["continuity_segment_id"])

    def test_partial_measurement_coverage_does_not_erase_person_identity(self):
        samples, visibility, features = self.fixture([["speaker"]])
        features[0]["candidates"][0]["coverage_fraction"] = 0.4
        self.assertEqual(
            select_segments(samples, visibility, features)[0]["continuity_segment_id"], "speaker"
        )

    def test_no_association_across_exclusion_gap_or_track_switch(self):
        for change in ("scene", "source", "segment"):
            samples, visibility, features = self.fixture(
                [["speaker"], ["speaker", "slide"], ["speaker", "slide"]]
            )
            if change == "scene":
                visibility["samples"][1]["scene_membership"] = "other_scene"
            elif change == "source":
                features[1]["continuity_barrier"] = True
            else:
                for feature in features[1:]:
                    feature["candidates"][0]["continuity_segment_id"] = "new-speaker-segment"
            self.assertIsNone(
                select_segments(samples, visibility, features)[2]["continuity_segment_id"]
            )

    def test_projection_uses_same_track_for_crop_and_native_motion(self):
        samples, visibility, features = self.fixture([["speaker"], ["speaker", "slide"]])
        frames, records = [], []
        for i, feature in enumerate(features):
            frames.append(
                {
                    "sample_index": i,
                    "selected_source_frame": {"source_frame_index": i},
                    "pose_rgb_pairing": "verified_same_source_frame",
                    "transform": {"input_width": 100, "input_height": 100},
                }
            )
            records.append(
                {
                    "source_frame_index": i,
                    "analysis_sample_indices": [i],
                    "people": [
                        {
                            "continuity_segment_id": c["continuity_segment_id"],
                            "bbox_xyxy_normalized": [0.6, 0.2, 0.8, 0.9]
                            if c["track_id"] == "speaker"
                            else [0.1, 0.1, 0.3, 0.4],
                        }
                        for c in feature["candidates"]
                    ],
                }
            )
        source = SimpleNamespace(
            feature_records=features,
            focus_frame_records=frames,
            native_feature_records=records,
            manifest={"artifacts": {"native_features": {"sha256": "a" * 64}}},
        )
        original = deepcopy(source)
        view = SelectedPoseView(source, samples, visibility)
        self.assertEqual(len(view.native_feature_records[1]["people"]), 1)
        self.assertEqual(
            view.native_feature_records[1]["people"][0]["continuity_segment_id"], "speaker"
        )
        self.assertEqual(view.focus_frame_records[1]["conditioning_mode"], "pose_guided_crop")
        self.assertGreater(view.focus_frame_records[1]["transform"]["crop_xyxy_pixels"][0], 50)
        self.assertEqual(source, original)

    def test_projected_pose_reaches_single_pass_with_relative_motion(self):
        from synthetic import SyntheticNativeConditioning
        from synthetic import admission, ProbabilityModel

        samples, visibility, features = self.fixture(
            [["speaker"], ["speaker", "slide"], ["speaker"]]
        )
        source = SyntheticNativeConditioning(3)
        for row in source.native_feature_records:
            person = row["people"][0]
            person["continuity_segment_id"] = "speaker"
            for geometry in person["geometry"].values():
                geometry["hands"] = {
                    side: hand(row["observation_time_us"] / 1e6) for side in ("left", "right")
                }
            if row["analysis_sample_indices"] == [1]:
                slide = deepcopy(person)
                slide.update(track_id=99, continuity_segment_id="slide")
                row["people"].append(slide)
                row["target_association"] = "ambiguous_concurrent_people"
        source.feature_records = features
        source.manifest = {"artifacts": {"native_features": {"sha256": "a" * 64}}}
        source.model_input_identity = lambda: {"native_features_sha256": "a" * 64}
        source.focus_frame_records = [
            {
                "sample_index": i,
                "selected_source_frame": {"source_frame_index": i * 3 + 1},
                "pose_rgb_pairing": "verified_same_source_frame",
                "transform": {"input_width": 100, "input_height": 100},
            }
            for i in range(3)
        ]
        view = SelectedPoseView(source, samples, visibility)
        packet = view.temporal_packet(0, 300000)
        self.assertEqual(len(packet["observations"]), 9)
        self.assertTrue(
            all(len(row[5]) == 1 and row[5][0][0] == 0 for row in packet["observations"])
        )
        cfg = configuration()
        model = ProbabilityModel()
        frames = {
            s.index: {"sample_index": s.index, "start_us": s.start_us, "end_us": s.end_us}
            for s in samples
        }

        def visuals(request_id, indices, stage, **kwargs):
            return [], [
                {"kind": "individual", "sample_index": i, "image_ordinal": n}
                for n, i in enumerate(indices)
            ]

        traces = p.run_pass(
            samples=samples,
            subject={"subject_id": "synthetic"},
            resolutions=admission(3, cfg),
            frames=frames,
            config=cfg,
            model=model,
            conditioning=view,
            visuals=visuals,
            slots=[None] * 3,
            trace=None,
        )
        self.assertEqual(len(model.calls), 1)
        measured = traces[0]["motion_measurements"][1]
        self.assertEqual(
            measured["hand_quality"]["left"]["rotation_reference"],
            "projected_torso_longitudinal_axis",
        )
        self.assertAlmostEqual(measured["hand_measurements"]["left"]["orientation_speed"], 1.0)


if __name__ == "__main__":
    unittest.main()
