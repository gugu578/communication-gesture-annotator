# SPDX-License-Identifier: Apache-2.0
"""Offline checks with generated media and deterministic model responses."""

from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import unittest
from unittest.mock import patch
from communication_gesture_annotator.cli import _parser
from communication_gesture_annotator.storage import storage_mode, compact_enabled, open_data
from communication_gesture_annotator.config import (
    DetectionConfig,
    InferenceConfig,
    OpenWebUIConfig,
    ResearchConfig,
    load_research_config,
)
from communication_gesture_annotator.domain import AnalysisSample
from communication_gesture_annotator.export.json_export import (
    read_json,
    write_json,
    validate_analysis_document,
)
from communication_gesture_annotator.media.ffmpeg import (
    file_sha256,
    preprocess_video,
    probe_video_frames,
)
from communication_gesture_annotator.media.frame_extract import extract_analysis_frames
from communication_gesture_annotator.media.pose import run_pose_preprocessing
from communication_gesture_annotator.media.pose_conditioning import run_pose_conditioning
from communication_gesture_annotator.media.processing_scope import OUTSIDE, excluded_observation
from communication_gesture_annotator.pipeline.llm_contracts import (
    ModelCallResult,
    TaskContractViolation,
)
from communication_gesture_annotator.pipeline.llm_runner import run_phase2_pipeline
from communication_gesture_annotator.pipeline import main_scene_visibility, probability_pass


class ScriptedModel:
    """Test transport only: never contacts a model service."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def check_capabilities(self) -> dict[str, Any]:
        return {"adapter": "test-local-model", "resolved_model": "synthetic", "vision": True}

    def complete_json(self, **kwargs: Any) -> ModelCallResult:
        self.calls.append(kwargs)
        task = kwargs["task"]
        trace = kwargs["trace_context"]
        if task in {"overview", "overview_segment", "overview_reduce"}:
            evidence = trace.get("allowed_evidence_sample_indices")
            if evidence is None:
                evidence = trace.get("input_sample_indices", [])
            if task == "overview":
                evidence = trace["profile_evidence_scope"]["direct_storyboard_sample_indices"]
            payload = {
                "target_mode": "individual",
                "selected_subjects": [
                    {
                        "subject_id": "synthetic-person",
                        "visual_description": "central person",
                        "identity_cues": ["same clothing"],
                        "recurring_views": ["frontal"],
                        "typical_rest_postures": [],
                        "selection_reason": "recurs in the supplied frames",
                        "prevalence": 0.9,
                    }
                ],
                "ignored_transient_people": [],
                "annotation_relevance_spans": [],
                "scene_summary": "one speaking scene",
                "confidence": 0.9,
                "evidence_sample_indices": sorted(set(evidence))[:8],
            }
        elif task == "overview_exemplar_verification":
            payload = {
                "target_subject_ids": list(trace["target_subject_ids"]),
                "samples": [
                    {
                        "sample_index": i,
                        "scene_type": "live_target",
                        "selected_target_visibility": "visible",
                        "classification": "positive_main_scene",
                        "probabilities": {
                            "positive_main_scene": 0.95,
                            "not_positive_main_scene": 0.03,
                            "uncertain": 0.02,
                        },
                        "reason": "synthetic test scene",
                    }
                    for i in trace["candidate_sample_indices"]
                ],
            }
        elif task == "main_scene_visibility":
            payload = {
                "samples": [
                    {"sample_index": i, "main_scene": i != 15} for i in trace["core_sample_indices"]
                ]
            }
        elif task == "non_g_probability":
            payload = {
                "non_g_unit_probability": [
                    0.05 if i < 14 or i >= 17 else 0.95 for i in trace["core_sample_indices"]
                ]
            }
        else:
            raise AssertionError(f"Unexpected pass: {task}")
        parsed = kwargs["response_validator"](payload)
        return ModelCallResult(
            payload=parsed,
            model_payload=payload,
            request_id=kwargs["request_id"],
            response_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
            latency_seconds=0.001,
            endpoint_mode="test",
            trace_reference={"trace_id": kwargs["request_id"], "artifacts": {}},
            contract_normalization_events=tuple(getattr(parsed, "normalization_events", ())),
            contract_normalization_policy_version=getattr(
                parsed, "normalization_policy_version", None
            ),
        )


def synthetic_pose_worker(request_path: Path) -> None:
    """Emit correctly timestamped empty pose observations; no weights are downloaded."""
    request = read_json(request_path)
    output = Path(request["output_directory"])
    frames = request["source_frames"]
    rows = []
    selected = []
    for i, frame in enumerate(frames):
        row = excluded_observation(frame, 160, 100)
        if frame.get("processing_status") != OUTSIDE:
            row["pose_inference_attempted"] = True
            row["decoded_frame"].pop("dimensions_only", None)
            selected.append(i)
        rows.append(row)
    observations = output / (
        "observations.jsonl.gz"
        if request.get("storage_mode") == "compact"
        else "observations.jsonl"
    )
    with open_data(observations, "w") as handle:
        handle.write("".join((json.dumps(r) + "\n" for r in rows)))
    if request.get("storage_mode") != "compact":
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                request["source_path"],
                "-an",
                "-vf",
                f"trim=start_frame={selected[0]}:end_frame={selected[-1] + 1},setpts=PTS-STARTPTS",
                "-frames:v",
                str(len(selected)),
                "-c:v",
                "libx264",
                str(output / "overlay.mp4"),
            ],
            check=True,
        )
    write_json(
        output / ".worker-result.json",
        {
            "schema_version": "0.2.0",
            "frame_count": len(frames),
            "frame_width": 160,
            "frame_height": 100,
            "processed_frame_span": [selected[0], selected[-1] + 1],
            "processed_frame_count": len(selected),
            "frames_with_people": 0,
            "person_observation_count": 0,
            "track_count": 0,
            "overlay_frame_rate": {
                "numerator": 25,
                "denominator": 1,
                "decimal": 25.0,
                "source_timing": "constant_cadence",
            },
            "overlay_timing_semantics": "synthetic source-timed overlay",
            "runtime": {"rtmlib_version": "synthetic-test"},
            "models": {
                "detector": {"loaded_file_sha256": "a" * 64},
                "pose": {"loaded_file_sha256": "b" * 64},
            },
        },
    )


class RuntimeTests(unittest.TestCase):
    def test_range_and_current_schedule(self) -> None:
        args = _parser().parse_args(
            [
                "run-llm",
                "video.mp4",
                "--run-dir",
                "run",
                "--phase2-config",
                "config.toml",
                "--from-second",
                "20.44",
                "--to-second",
                "61.3",
                "--skip-phase1",
            ]
        )
        self.assertEqual((args.from_seconds, args.to_seconds), (20400000, 61300000))
        self.assertEqual(
            probability_pass.schedule_metadata(DetectionConfig())["maximum_context_us"], 6000000
        )
        self.assertEqual(
            probability_pass.schedule_metadata(
                DetectionConfig(probability_history_frames=0, probability_future_frames=0)
            )["maximum_context_us"],
            300000,
        )
        with self.assertRaises(ValueError):
            DetectionConfig(probability_target_frames=0)
        self.assertEqual(InferenceConfig().visibility_max_frames, 5)

    def test_probability_contract_and_fusion(self) -> None:
        self.assertAlmostEqual(probability_pass.combine_probability(0.5, 0.5, 4), 0.5)
        self.assertEqual(probability_pass.combine_probability(None, 0.8, 4), 0.8)
        self.assertGreater(probability_pass.combine_probability(0.8, 0.9, 4), 0.8)
        for scores in ([0.5], [0.5, None], [0.5, float("nan")], [0.5, 1.1]):
            with self.assertRaises(TaskContractViolation):
                probability_pass.validate_response({"non_g_unit_probability": scores}, [7, 8])
        with self.assertRaises(TaskContractViolation):
            main_scene_visibility.validate_response(
                {"samples": [{"sample_index": 9, "main_scene": True}]}, [8]
            )

    def test_config_rejects_removed_options_and_blank_connection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[open_webui]\nbase_url=""\nmodel=""\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base_url"):
                load_research_config(path)
            path.write_text(
                '[open_webui]\nmodel="synthetic"\n[detection]\nactivity_assessment="old"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_research_config(path)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_scoped_pipeline_resume_and_presence_eaf(self) -> None:
        cfg = ResearchConfig(
            open_webui=OpenWebUIConfig(model="synthetic"),
            detection=DetectionConfig(phase2_start_us=1000000, phase2_end_us=2200000,
                **getattr(self, "schedule_overrides", {})),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, run = (root / "source.mp4", root / "run")
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x100:rate=25:duration=3",
                    "-an",
                    "-c:v",
                    "libx264",
                    str(source),
                ],
                check=True,
            )
            manifest = preprocess_video(
                source,
                run,
                config=cfg.analysis,
                include_audio=False,
                start_us=1000000,
                end_us=2200000,
            )
            self.assertEqual(len(probe_video_frames(run / "model-proxy.mp4")["frames"]), 12)
            samples = [AnalysisSample(**r) for r in manifest["proxy"]["frame_mapping"]]
            frames = extract_analysis_frames(
                run / "model-proxy.mp4", samples, run / "analysis-frames"
            )
            # Resume must not probe/transcode the source or decode its frames again.
            original_mtimes = {p: p.stat().st_mtime_ns for p in
                [run/"preprocessing.json", run/"model-proxy.mp4", run/"analysis-frames/sample-00000010.jpg"]}
            with patch("communication_gesture_annotator.media.ffmpeg.probe_media",
                       side_effect=AssertionError("cached source probed")), patch(
                    "communication_gesture_annotator.media.frame_extract._run_extraction",
                    side_effect=AssertionError("cached frames extracted")):
                restored = preprocess_video(source, run, config=cfg.analysis, include_audio=False,
                    start_us=1000000, end_us=2200000, overwrite=True, resume=True)
                restored_frames = extract_analysis_frames(run/"model-proxy.mp4", samples,
                    run/"analysis-frames", overwrite=True, resume=True)
            self.assertEqual(restored, manifest)
            self.assertEqual(restored_frames, frames)
            self.assertEqual(original_mtimes, {p: p.stat().st_mtime_ns for p in original_mtimes})
            with patch(
                "communication_gesture_annotator.media.pose._run_worker",
                side_effect=synthetic_pose_worker,
            ):
                pose = run_pose_preprocessing(
                    source,
                    manifest["source"]["video_frames"],
                    samples,
                    run,
                    config=cfg.pose_preprocessing,
                    rights=manifest["source"]["rights"],
                    expected_source_sha256=manifest["source"]["sha256"],
                    expected_source_byte_size=source.stat().st_size,
                    preprocessing_sha256=file_sha256(run / "preprocessing.json"),
                )
            conditioned = run_pose_conditioning(
                pose, samples, frames, run, rights=manifest["source"]["rights"], source_path=source
            )
            if compact_enabled():
                self.assertEqual(pose.observations_path.suffix, ".gz")
                self.assertEqual(conditioned.native_features_path.suffix, ".gz")
                self.assertFalse(pose.overlay_path.exists())
                self.assertFalse(conditioned.conditioned_overlay_path.exists())
            common = dict(
                run_dir=run,
                source=manifest["source"],
                preprocessing=manifest,
                analysis_samples=samples,
                extracted_frames=frames,
                config=cfg,
                audio_supplied=False,
                pose_conditioning=conditioned,
            )
            model = ScriptedModel()
            run_phase2_pipeline(**common, model=model, stop_after="phase2")
            self.assertIn("main_scene_visibility", {c["task"] for c in model.calls})
            self.assertIn("non_g_probability", {c["task"] for c in model.calls})
            for call in model.calls:
                trace = call["trace_context"]
                for key in (
                    "core_sample_indices",
                    "context_sample_indices",
                    "input_sample_indices",
                ):
                    self.assertTrue(set(trace.get(key, [])) <= set(range(10, 22)))
                if call["task"] == "main_scene_visibility":
                    self.assertLessEqual(len(trace["core_sample_indices"]), 5)
                if call["task"] == "non_g_probability":
                    self.assertNotIn(15, trace["context_sample_indices"])
            checkpoint_bytes = {p: p.read_bytes() for p in (run / "model-checkpoints").glob("checkpoint-*.json")}
            interrupted = ScriptedModel()
            with patch("communication_gesture_annotator.pipeline.joint_runner.write_analysis",
                       side_effect=OSError("synthetic interrupted annotation save")):
                with self.assertRaisesRegex(OSError, "interrupted annotation save"):
                    run_phase2_pipeline(**common, model=interrupted, resume=True, export_eaf=True)
            self.assertEqual(interrupted.calls, [])
            self.assertTrue((run / "source.json").is_file())
            self.assertFalse((run / "annotations.json").exists())

            def interrupt_completion(path: Path, payload: Any) -> Path:
                if Path(path).name == "completion.json":
                    raise OSError("synthetic interrupted completion save")
                return write_json(path, payload)

            interrupted = ScriptedModel()
            with patch("communication_gesture_annotator.pipeline.joint_runner.write_json",
                       side_effect=interrupt_completion):
                with self.assertRaisesRegex(OSError, "interrupted completion save"):
                    run_phase2_pipeline(**common, model=interrupted, resume=True, export_eaf=True)
            self.assertEqual(interrupted.calls, [])
            saved_units = read_json(run / "annotations.json")["g_units"]
            resumed = ScriptedModel()
            with patch("communication_gesture_annotator.media.image_packaging._run_packaging",
                       side_effect=AssertionError("cached visibility sheets rendered")), patch(
                    "communication_gesture_annotator.media.motion_trace.write_motion_trace",
                    side_effect=AssertionError("cached motion chart rendered")):
                result = run_phase2_pipeline(**common, model=resumed, resume=True, export_eaf=True)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(checkpoint_bytes, {p: p.read_bytes() for p in checkpoint_bytes})
            document = validate_analysis_document(read_json(result.analysis_paths[0]))
            self.assertEqual(document["provenance"]["software"], "communication-gesture-annotator")
            self.assertEqual(document["g_units"], saved_units)
            self.assertEqual(read_json(run / "pipeline-status.json")["completed_stage"], "phase3")
            self.assertEqual(
                document["provenance"]["phase3_finalization"]["model_request_count"], 0
            )
            self.assertTrue((run / "annotations.eaf").is_file())
            from communication_gesture_annotator.cli import main

            with patch(
                "communication_gesture_annotator.cli.OpenWebUIClient",
                side_effect=AssertionError("export must not call a model"),
            ):
                self.assertEqual(main(["export-eaf", str(run), "--force"]), 0)
                self.assertEqual(
                    main(
                        [
                            "export-eaf",
                            str(run),
                            "--output",
                            str(run / "annotations.json"),
                            "--force",
                        ]
                    ),
                    2,
                )
            skipped_model = ScriptedModel()
            skipped = run_phase2_pipeline(
                **{
                    **common,
                    "run_dir": root / "skip-phase1",
                    "config": replace(cfg, inference=replace(cfg.inference, phase1_enabled=False)),
                },
                model=skipped_model,
                export_eaf=True,
            )
            self.assertEqual({c["task"] for c in skipped_model.calls}, {"non_g_probability"})
            validate_analysis_document(read_json(skipped.analysis_paths[0]))

    def test_compact_pipeline_resume_and_presence_eaf(self) -> None:
        with storage_mode("compact"):
            self.test_scoped_pipeline_resume_and_presence_eaf()

    def test_minimal_pipeline_resume_and_presence_eaf(self) -> None:
        with storage_mode("minimal"):
            self.test_scoped_pipeline_resume_and_presence_eaf()

    def test_minimal_zero_history_future_and_one_target(self) -> None:
        self.schedule_overrides = dict(probability_target_frames=1,
            probability_history_frames=0, probability_future_frames=0)
        with storage_mode("minimal"):
            self.test_scoped_pipeline_resume_and_presence_eaf()


if __name__ == "__main__":
    unittest.main()
