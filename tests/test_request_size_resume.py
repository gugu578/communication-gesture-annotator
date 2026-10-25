# SPDX-License-Identifier: Apache-2.0
"""Offline recovery checks: preserve pixels, evidence and successful checkpoints."""

from base64 import b64decode
from io import BytesIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import cv2
import numpy as np

from communication_gesture_annotator.adapters.open_webui import OpenWebUIClient, OpenWebUIError
from communication_gesture_annotator.audit import TraceWriter
from communication_gesture_annotator.config import InferenceConfig, OpenWebUIConfig
from communication_gesture_annotator.export.json_export import write_json
from communication_gesture_annotator.pipeline.llm_runner import (
    _CheckpointingPhase2Model,
    _MODEL_RUN_INTERMEDIATE_ARTIFACT_NAMES,
    validate_model_run_resume_state,
    write_pipeline_status,
)
from communication_gesture_annotator.pipeline.probability_pass import SCHEMA
from communication_gesture_annotator.rights import restricted_research_rights, write_rights_sidecar


def http_error(status: int = 413) -> HTTPError:
    return HTTPError(
        "http://127.0.0.1:1/api/chat/completions", status, "synthetic failure",
        {}, BytesIO(b"synthetic request rejection"),
    )


def response() -> MagicMock:
    result = MagicMock()
    result.__enter__.return_value = result
    result.__exit__.return_value = False
    result.headers = {}
    result.status = 200
    result.read.return_value = json.dumps({
        "choices": [{"message": {"content": json.dumps({"non_g_unit_probability": [0.7]})},
                     "finish_reason": "stop"}],
    }).encode()
    return result


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RequestSizeRecoveryTests(unittest.TestCase):
    def request(self, root: Path) -> tuple[OpenWebUIClient, dict]:
        # Generated smooth RGB images make compression gains deterministic.
        images = []
        for ordinal in range(2):
            pixels = np.zeros((100, 160, 3), dtype=np.uint8)
            pixels[:, :, ordinal] = np.arange(160, dtype=np.uint8)
            ok, png = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.assertTrue(ok)
            image = root / f"synthetic-{ordinal}.png"
            image.write_bytes(png.tobytes())
            write_rights_sidecar(image, restricted_research_rights(basis="synthetic fixture"))
            images.append(image)
        client = OpenWebUIClient(
            OpenWebUIConfig(base_url="http://127.0.0.1:1", model="synthetic",
                            maximum_transient_http_retries=0, inter_request_cooldown_seconds=0),
            InferenceConfig(maximum_repair_attempts=0),
            environ={"GESTURE_ANNOTATOR_OPEN_WEBUI_API_KEY": "synthetic-key"},
            trace_writer=TraceWriter(root, append_exchanges=True),
        )
        client._capability_snapshot = {"vision": True}
        return client, {
            "request_id": "activity-synthetic", "task": "non_g_probability",
            "system_prompt": "unchanged prompt", "user_prompt": "unchanged motion evidence",
            "images": images, "response_schema": SCHEMA,
            "trace_context": {"core_sample_indices": [2], "context_sample_indices": [0, 1, 2, 3]},
            "response_validator": lambda value: value,
        }

    def test_minimal_recovery_cleanup_waits_for_checkpoint_and_replays(self) -> None:
        from communication_gesture_annotator.storage import storage_mode
        with TemporaryDirectory() as directory, storage_mode("minimal"):
            root = Path(directory)
            client, args = self.request(root)
            wrapper = _CheckpointingPhase2Model(client, run_root=root, scope={"synthetic": 1}, resume=False, trace=None)
            with patch.object(client._opener, "open", side_effect=[http_error(), response()]), patch(
                    "communication_gesture_annotator.pipeline.llm_runner.write_json",
                    side_effect=OSError("synthetic checkpoint disk full")):
                with self.assertRaisesRegex(OSError, "checkpoint disk full"):
                    wrapper.complete_json(**args)
            recovery = root / "model-image-recovery"
            self.assertTrue(list(recovery.rglob("image-*.png")))
            self.assertTrue(list(recovery.rglob("decision.json")))
            resumed = _CheckpointingPhase2Model(client, run_root=root, scope={"synthetic": 1}, resume=True, trace=None)
            # The saved recovery decision skips the known rejected original request.
            with patch.object(client._opener, "open", return_value=response()) as send:
                result = resumed.complete_json(**args)
            self.assertEqual(send.call_count, 1)
            self.assertFalse(list(recovery.rglob("image-*.png")))
            self.assertFalse(list(recovery.rglob("images.json")))
            self.assertTrue(list(recovery.rglob("decision.json")))
            self.assertTrue(all(p.is_file() for p in args["images"]))
            replay = _CheckpointingPhase2Model(client, run_root=root, scope={"synthetic": 1}, resume=True, trace=None)
            with patch.object(client._opener, "open", side_effect=AssertionError("checkpoint must replay")):
                self.assertEqual(replay.complete_json(**args).payload, result.payload)

    def test_smaller_request_preserves_pixels_text_and_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, args = self.request(root)
            wrapper = _CheckpointingPhase2Model(
                client, run_root=root, scope={"synthetic": 1}, resume=False, trace=None,
            )
            with patch.object(client._opener, "open", side_effect=[http_error(), response()]) as send:
                result = wrapper.complete_json(**args)
            self.assertEqual(send.call_count, 2)
            first, second = [c.args[0].data for c in send.call_args_list]
            self.assertLess(len(second), len(first))
            original, compressed = json.loads(first), json.loads(second)
            old_content = original["messages"][1]["content"]
            new_content = compressed["messages"][1]["content"]
            self.assertEqual(len(old_content), len(new_content))
            for old, new in zip(old_content[1:], new_content[1:], strict=True):
                before = b64decode(old["image_url"]["url"].split(",", 1)[1])
                after = b64decode(new["image_url"]["url"].split(",", 1)[1])
                self.assertTrue(new["image_url"]["url"].startswith("data:image/png;"))
                np.testing.assert_array_equal(
                    cv2.imdecode(np.frombuffer(before, np.uint8), cv2.IMREAD_UNCHANGED),
                    cv2.imdecode(np.frombuffer(after, np.uint8), cv2.IMREAD_UNCHANGED),
                )
            original["messages"][1]["content"] = old_content[:1]
            compressed["messages"][1]["content"] = new_content[:1]
            self.assertEqual(original, compressed)
            self.assertEqual(result.request_id, args["request_id"])
            self.assertTrue(result.trace_reference["request_id"].endswith("-png-size-v1"))
            resumed = _CheckpointingPhase2Model(
                client, run_root=root, scope={"synthetic": 1}, resume=True, trace=None,
            )
            with patch.object(client._opener, "open", side_effect=AssertionError("must reuse")):
                self.assertEqual(resumed.complete_json(**args).payload, result.payload)
            self.assertEqual(resumed.reused_count, 1)

    def test_spatial_recovery_preserves_context_chart_and_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, args = self.request(root)
            args["trace_context"]["motion_trace"] = {"image_ordinal": 1}
            args["trace_context"]["image_refs"] = [{"image_ordinal": 0, "width_px": 160,
                "height_px": 100, "cells": [{"sample_index": 2, "start_us": 200000,
                    "cell_xyxy_pixels": [0, 0, 160, 100], "rgb_size_px": [160, 100]}]}]
            wrapper = _CheckpointingPhase2Model(client, run_root=root, scope={"synthetic": 1}, resume=False, trace=None)
            with patch.object(client._opener, "open", side_effect=[http_error(), http_error(), response()]) as send:
                result = wrapper.complete_json(**args)
            self.assertEqual(send.call_count, 3)
            self.assertEqual(result.request_id, args["request_id"])
            self.assertTrue(result.trace_reference["request_id"].endswith("-png-spatial-v1-85"))
            original = json.loads(send.call_args_list[0].args[0].data)
            resized = json.loads(send.call_args_list[-1].args[0].data)
            content = resized["messages"][1]["content"]
            for ordinal, (block, path) in enumerate(zip(content[1:], args["images"], strict=True)):
                pixels = cv2.imdecode(np.frombuffer(b64decode(block["image_url"]["url"].split(",")[1]), np.uint8), cv2.IMREAD_UNCHANGED)
                source = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
                expected = source if ordinal == 1 else cv2.resize(source, (136, 85), interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(pixels, expected)
            for payload in (original, resized):
                payload["messages"][1]["content"] = payload["messages"][1]["content"][:1]
            self.assertEqual(original, resized)
            manifest = next(json.loads(p.read_text()) for p in (root/"model-image-recovery").rglob("images.json")
                if "spatial" in json.loads(p.read_text())["policy_version"])
            self.assertFalse(manifest["images"][0]["decoded_pixels_identical"])
            self.assertTrue(manifest["images"][1]["decoded_pixels_identical"])
            from communication_gesture_annotator.adapters.image_recovery import recovery_trace_context
            trace = recovery_trace_context(args["trace_context"], manifest, args["request_id"])
            self.assertEqual(trace["context_sample_indices"], [0, 1, 2, 3])
            self.assertEqual(trace["image_refs"][0]["cells"][0]["cell_xyxy_pixels"], [0, 0, 136, 85])
            self.assertEqual(trace["image_refs"][0]["source_geometry"]["width_px"], 160)
            resumed = _CheckpointingPhase2Model(client, run_root=root, scope={"synthetic": 1}, resume=True, trace=None)
            with patch.object(client._opener, "open", side_effect=AssertionError("must reuse")):
                self.assertEqual(resumed.complete_json(**args).payload, result.payload)
            self.assertEqual(resumed.reused_count, 1)

    def test_repeat_413_is_bounded_and_resume_skips_rejected_levels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, args = self.request(root)
            # Synthetic texture also exercises skipping a level whose PNG is
            # larger despite fewer pixels (compression gains are content dependent).
            rng = np.random.default_rng(431)
            for path in args["images"]:
                pixels = rng.integers(0, 16, (100, 160, 3), dtype=np.uint8) * 16
                _, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                path.write_bytes(encoded.tobytes())
                write_rights_sidecar(path, restricted_research_rights(basis="synthetic fixture"), overwrite=True)
            with patch.object(client._opener, "open", side_effect=http_error()) as send:
                with self.assertRaisesRegex(OpenWebUIError, "bounded PNG size recovery exhausted"):
                    client.complete_json(**args)
            self.assertIn(send.call_count, (3, 4))
            sizes = [len(c.args[0].data) for c in send.call_args_list]
            self.assertTrue(all(a > b for a, b in zip(sizes, sizes[1:])))
            reduced_body = send.call_args.args[0].data
            with patch.object(client._opener, "open", return_value=response()) as resumed:
                client.complete_json(**args)
            self.assertEqual(resumed.call_count, 1)
            self.assertEqual(resumed.call_args.args[0].data, reduced_body)
            with patch.object(client._opener, "open", return_value=response()) as changed:
                client.complete_json(**{**args, "user_prompt": "changed evidence"})
            self.assertGreater(len(changed.call_args.args[0].data), len(reduced_body))

    def test_non_413_does_not_trigger_compression(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, args = self.request(root)
            with patch.object(client._opener, "open", side_effect=http_error(400)) as send:
                with self.assertRaises(OpenWebUIError):
                    client.complete_json(**args)
            self.assertEqual(send.call_count, 1)
            self.assertFalse((root / "model-image-recovery").exists())

    def test_no_lossless_gain_proceeds_without_repeating_identical_request(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, args = self.request(root)
            def unchanged(source: Path, width: int, height: int) -> tuple[bytes, str]:
                return source.read_bytes(), "a" * 64
            with (
                patch.object(client._opener, "open", side_effect=[http_error(), response()]) as send,
                patch("communication_gesture_annotator.adapters.image_recovery.optimize_png",
                      side_effect=unchanged),
            ):
                result = client.complete_json(**args)
            self.assertEqual(send.call_count, 2)
            self.assertTrue(result.trace_reference["request_id"].endswith("-png-spatial-v1-85"))
            self.assertLess(len(send.call_args_list[1].args[0].data), len(send.call_args_list[0].args[0].data))

    def test_other_errors_after_spatial_retry_do_not_reduce_again(self) -> None:
        with TemporaryDirectory() as directory:
            client, args = self.request(Path(directory))
            with patch.object(client._opener, "open", side_effect=[http_error(), http_error(), http_error(400)]) as send:
                with self.assertRaises(OpenWebUIError):
                    client.complete_json(**args)
            self.assertEqual(send.call_count, 3)
            with patch.object(client._opener, "open", return_value=response()) as resumed:
                client.complete_json(**args)
            self.assertEqual(resumed.call_count, 1)
            self.assertEqual(resumed.call_args.args[0].data, send.call_args.args[0].data)

    def test_spatial_cache_corruption_prevents_dispatch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, args = self.request(root)
            with patch.object(client._opener, "open", side_effect=[http_error(), http_error(), response()]):
                client.complete_json(**args)
            manifest_path = next(p for p in (root/"model-image-recovery").rglob("images.json")
                if "spatial" in json.loads(p.read_text())["policy_version"])
            (manifest_path.parent/"image-000.png").write_bytes(b"corrupt")
            with patch.object(client._opener, "open") as send:
                with self.assertRaisesRegex(OpenWebUIError, "preparation failed"):
                    client.complete_json(**args)
                send.assert_not_called()

    def test_corrupt_cache_is_rejected_before_sending(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client, args = self.request(root)
            with patch.object(client._opener, "open", side_effect=[http_error(), response()]):
                client.complete_json(**args)
            next((root / "model-image-recovery").rglob("*.png")).write_bytes(b"corrupt")
            with patch.object(client._opener, "open") as send:
                with self.assertRaisesRegex(OpenWebUIError, "preparation failed"):
                    client.complete_json(**args)
                send.assert_not_called()


class ResumeStageTests(unittest.TestCase):
    def create_prefix(self, root: Path) -> list[str]:
        prefix = []
        for name in _MODEL_RUN_INTERMEDIATE_ARTIFACT_NAMES:
            if name == "phase2-evidence.json":
                break
            write_json(root / name, {})
            prefix.append(name)
        write_json(root / "model-checkpoints/checkpoint-synthetic.json", {})
        return prefix

    def test_resume_after_phase1_accepts_actual_markers(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = self.create_prefix(root)
            write_pipeline_status(root, requested_stop_after="phase3", completed_stage="phase1",
                                  artifacts=[root / name for name in prefix])
            self.assertEqual(validate_model_run_resume_state(root), tuple(prefix))

    def test_status_ahead_of_actual_markers_is_still_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = self.create_prefix(root)
            (root / "model-capabilities.json").unlink()
            write_pipeline_status(root, requested_stop_after="phase3", completed_stage="phase1",
                                  artifacts=[root / name for name in prefix[:-1]])
            with self.assertRaisesRegex(ValueError, "beyond the durable artifact prefix"):
                validate_model_run_resume_state(root)

    def test_artifact_hash_checks_are_preserved(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = self.create_prefix(root)
            write_pipeline_status(root, requested_stop_after="phase3", completed_stage="phase1",
                                  artifacts=[root / name for name in prefix])
            write_json(root / prefix[0], {"unexpected": "change"})
            with self.assertRaisesRegex(ValueError, "artifact identity does not match"):
                validate_model_run_resume_state(root)
