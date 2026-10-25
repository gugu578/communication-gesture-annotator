# SPDX-License-Identifier: Apache-2.0
"""Synthetic camera boundaries, request history and frozen grouping behavior."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from communication_gesture_annotator.config import ResearchConfig, OpenWebUIConfig
from communication_gesture_annotator.media.camera_cuts import detect_changes, scan_source, segment_observations, WIDTH, HEIGHT
from communication_gesture_annotator.media.pose_conditioning import _SmoothedPerson, _smooth_people, _motion_edges
from communication_gesture_annotator.pipeline import probability_pass as probability
from communication_gesture_annotator.pipeline.camera_context import shot_ranges, plan_camera_windows
from communication_gesture_annotator.pipeline.phase1_admission import build_admission
from communication_gesture_annotator.pipeline.grouping import propose_joint_activity_candidates
from communication_gesture_annotator.pipeline.probability_finalization import finalize_probability_candidates
from communication_gesture_annotator.pipeline.llm_runner import _CheckpointingPhase2Model
from synthetic import _samples, ProbabilityModel, scene, feature


def configuration() -> ResearchConfig:
    return ResearchConfig(open_webui=OpenWebUIConfig(model="synthetic"))


class CameraCutsTest(unittest.TestCase):
    def test_geometry_jump_does_not_enter_smoothing_or_derivatives(self) -> None:
        rows = [{"source_frame_index": i} for i in range(16)]
        people = [
            _SmoothedPerson(
                source_frame_index=i, source_frame_id=f"source-frame:{i}",
                canonical_time_us=i * 33333, canonical_end_us=(i + 1) * 33333,
                track_id=0, segment_id="track-0-segment-0", bbox=(0, 0, 1, 1),
                detector_confidence=0.9,
                points=[(0.2 if i < 8 else 0.7, 0.4)] * 133,
                confidences=(0.9,) * 133, smoothing_statuses=[None] * 133,
                smoothing_support=[None] * 133,
            )
            for i in range(16)
        ]
        segment_observations(rows, people, [{"source_frame_index": 8, "time_us": 8 * 33333}])
        _smooth_people(people)
        self.assertNotEqual(people[7].segment_id, people[8].segment_id)
        self.assertTrue(rows[8]["camera_cut_before"])
        for person in people:
            self.assertAlmostEqual(person.points[91][0], 0.2 if person.source_frame_index < 8 else 0.7)
        edges = _motion_edges(people)
        self.assertEqual(len(edges), 2)
        for segment in edges.values():
            for edge in segment:
                self.assertFalse(edge.start_us < 8 * 33333 <= edge.end_us)
                self.assertTrue(all(v is None or abs(v) < 1e-8 for v in edge.displacements.values()))

    def test_abrupt_broad_change_is_detected_but_local_motion_and_gradual_change_are_not(self) -> None:
        base = np.full((HEIGHT, WIDTH, 3), 20, np.uint8)
        images = [base.copy() for _ in range(7)]
        images[1][20:25, 20:30] = 240  # Local moving hand, not a shot change.
        for i in range(3, 7):
            images[i][:] = 170+(i-3)*5
        times = [0, 33_400, 67_500, 101_200, 135_800, 170_100, 205_000]
        changes = detect_changes((i, t, frame) for i, (t, frame) in enumerate(zip(times, images)))
        self.assertEqual([(c['source_frame_index'], c['time_us']) for c in changes], [(3, 101_200)])
        self.assertEqual(detect_changes([(0, 0, base), (8, 1_000_000, images[-1])]), [])


    def test_edit_with_large_unchanged_dark_background_is_still_detected(self) -> None:
        before = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        after = before.copy()
        after[13:41, 24:72] = 210
        cuts = detect_changes([(0, 0, before), (1, 33_333, before), (2, 66_667, after)])
        self.assertEqual([c['source_frame_index'] for c in cuts], [2])


    def test_camera_clarification_preserves_the_phrase_prompt(self) -> None:
        clarification = (
            "A camera cut can abruptly change the apparent position, size or orientation "
            "of the hands without actual hand movement. Do not treat that visual jump "
            "as evidence of movement, a G-unit onset, or a return to rest. Evaluate "
            "movement within each continuous camera shot. A genuine G-unit or non-G-unit "
            "may continue across a cut when the surrounding hand behavior supports that "
            "interpretation. "
        )
        self.assertEqual(probability.SYSTEM_PROMPT_TEMPLATE.count(clarification), 1)
        self.assertIn(clarification, probability.render_system_prompt(configuration().detection))
        baseline = probability.SYSTEM_PROMPT_TEMPLATE.replace(clarification, "")
        self.assertEqual(sha256(baseline.encode()).hexdigest(),
            '04e0ab7865ec4c6e762bf9c51412a5d0fd454434022d4033360a9708c5891cae')


    def test_motion_trace_marks_exact_cut_time_and_binds_it_to_cache_identity(self) -> None:
        from communication_gesture_annotator.media.motion_trace import write_motion_trace
        from communication_gesture_annotator.pipeline.motion_quality import score_measurement
        from communication_gesture_annotator.rights import unrestricted_or_unspecified_rights
        samples = _samples(3)
        measures = {s.index: score_measurement({'start_us': s.start_us, 'end_us': s.end_us, 'hands': {}},
                    configuration().detection, None) for s in samples}
        with TemporaryDirectory() as directory:
            args = (Path(directory), samples, [1], [0, 1, 2], measures)
            kwargs = {'image_ordinal': 1, 'rights': unrestricted_or_unspecified_rights()}
            plain, _ = write_motion_trace(*args, **kwargs)
            marked, metadata = write_motion_trace(*args, **kwargs, camera_cut_times_us=[110_000])
            self.assertNotEqual(plain, marked)
            self.assertEqual(metadata['camera_cut_times_us'], [110_000])
            image = cv2.imdecode(np.frombuffer(marked.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            x = round(98+(1400-35-98)*110_000/300_000)
            self.assertEqual(image[150, x].tolist(), [160, 40, 140])


    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for native decode coverage')
    def test_source_scan_respects_range_and_uses_supplied_pts(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 10, (WIDTH, HEIGHT))
            self.assertTrue(writer.isOpened())
            for i in range(10):
                writer.write(np.full((HEIGHT, WIDTH, 3), 20 if i < 4 else 180, np.uint8))
            writer.release()
            rows = [{'source_frame_index': i, 'canonical_time_us': 100_000*i,
                     'source_frame': {'processing_status': 'outside_processing_range' if i < 2 or i >= 8 else 'included'}}
                    for i in range(10)]
            cuts = scan_source(path, rows, ffmpeg='ffmpeg')
            self.assertEqual([(c['source_frame_index'], c['time_us']) for c in cuts], [(4, 400_000)])
            for r in rows[:5]:
                r['source_frame']['processing_status'] = 'outside_processing_range'
            self.assertEqual(scan_source(path, rows, ffmpeg='ffmpeg'), [])


    def test_camera_window_splits_target_and_clips_future_without_losing_history_or_ownership(self) -> None:
        indices = list(range(30))
        resolutions = {i: {'camera_shot_id': int(i >= 11)} for i in indices}
        ranges = shot_ranges(indices, resolutions)
        windows = plan_camera_windows(indices, target_indices=indices, target_items=3,
            history_items=8, lookahead_items=5, ranges=ranges)
        self.assertEqual([i for w in windows for i in w.output_indices], indices)
        self.assertTrue(all(len({resolutions[i]['camera_shot_id'] for i in w.output_indices}) == 1 for w in windows))
        self.assertTrue(all(max(w.input_indices) < 11 for w in windows if w.output_indices[-1] < 11))
        after = next(w for w in windows if w.output_indices[0] == 11)
        self.assertEqual(after.input_indices[0], 3)
        self.assertEqual(list(after.output_indices), [11, 12, 13])


    def test_cut_metadata_and_prior_quarantine_do_not_change_prompt_or_final_grouping(self) -> None:
        cfg = configuration()
        cfg = replace(cfg, detection=replace(cfg.detection, probability_target_frames=3,
            probability_history_frames=4, probability_future_frames=3))
        samples = _samples(24)
        pose = {i: {**feature(), 'camera_shot_id': int(i >= 11),
                    'camera_cut_times_us': [1_100_000] if i == 11 else []} for i in range(24)}
        resolutions = build_admission(samples=samples, visibility={'samples': [scene(i) for i in range(24)]},
            pose_features=pose, config=cfg, frames={}, visibility_hash='a'*64)
        self.assertFalse(resolutions[11]['exclude_from_detection'])
        self.assertFalse(resolutions[11]['continuity_barrier'])
        self.assertTrue(resolutions[11]['pose_motion_discontinuity'])
        model = ProbabilityModel(); model.active_range = (0, 24)
        slots = [None]*24
        traces = probability.run_pass(samples=samples, subject={'subject_id': 's'}, resolutions=resolutions,
            frames={s.index: {'start_us': s.start_us, 'end_us': s.end_us} for s in samples},
            config=cfg, model=model, conditioning=None, slots=slots, trace=None,
            visuals=lambda rid, indices, stage, **kw: ([], [{'kind': 'individual_frame', 'sample_index': i,
                'image_ordinal': n} for n, i in enumerate(indices)]))
        for trace, call in zip(traces, model.calls, strict=True):
            self.assertEqual(call['system_prompt'], probability.render_system_prompt(cfg.detection, cfg.analysis))
            core = trace['core_sample_indices'][0]
            prior = trace['previous_estimates']['rows']
            if 11 <= core <= 17:
                self.assertEqual(prior, [])
            if core == 20:
                self.assertEqual([r[0] for r in prior], [17, 18, 19])
            if core == 11:
                payload = json.loads(call['user_prompt'].removeprefix(probability.USER_PREFIX))
                self.assertEqual(payload['continuity']['camera_edits'][0]['times_us'], [1_100_000])
                self.assertEqual(trace['context_sample_indices'][0], 7)
        candidates, _ = propose_joint_activity_candidates(samples, slots, cfg.analysis)
        _, units = finalize_probability_candidates(samples, slots, candidates, cfg.analysis)
        self.assertEqual([(u.start_sample, u.end_sample_exclusive) for u in units], [(0, 24)])


    def test_camera_context_and_estimate_quarantine_replay_identically(self) -> None:
        from synthetic import admission
        cfg = configuration()
        cfg = replace(cfg, detection=replace(cfg.detection, probability_target_frames=3,
            probability_history_frames=4, probability_future_frames=3))
        samples = _samples(24)
        resolutions = admission(24, cfg, excluded=[10])
        for i, record in resolutions.items():
            record['camera_shot_id'] = int(i >= 11)
        inputs = dict(samples=samples, subject={'subject_id': 's'}, resolutions=resolutions,
            frames={s.index: {'start_us': s.start_us, 'end_us': s.end_us} for s in samples},
            config=cfg, conditioning=None, trace=None,
            visuals=lambda rid, indices, stage, **kw: ([], [{'kind': 'individual_frame', 'sample_index': i,
                'image_ordinal': n} for n, i in enumerate(indices)]))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _CheckpointingPhase2Model(ProbabilityModel(), run_root=root, scope={'test': 'camera-context'}, resume=False, trace=None)
            expected = probability.run_pass(**inputs, model=first, slots=[None]*24)
            # New admitted interval, no history: early outputs must not be used
            # as priors even once they move into a later full-sized context.
            after = [t for t in expected if t['core_sample_indices'][0] >= 11]
            self.assertEqual(after[0]['context_sample_indices'][0], 11)
            self.assertTrue(all(not t['previous_estimates']['rows'] for t in after[:3]))
            underlying = ProbabilityModel()
            replay = _CheckpointingPhase2Model(underlying, run_root=root, scope={'test': 'camera-context'}, resume=True, trace=None)
            self.assertEqual(probability.run_pass(**inputs, model=replay, slots=[None]*24), expected)
            self.assertEqual(underlying.calls, [])
