# SPDX-License-Identifier: Apache-2.0
"""Exercise installed module and child-process entry points without model calls."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import communication_gesture_annotator
from communication_gesture_annotator.adapters.open_webui import _classify_http_error
from communication_gesture_annotator.media.ffmpeg import MediaRuntimeError
from communication_gesture_annotator.media.pose import _run_worker


class EntryPointTests(unittest.TestCase):
    def test_module_command_outside_source_directory(self) -> None:
        package_root = Path(communication_gesture_annotator.__file__).parent.parent
        environment = dict(os.environ, PYTHONPATH=str(package_root))
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "communication_gesture_annotator", "--help"],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=30, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gesture-annotator", result.stdout)
        for command in ("run-llm", "evaluate-run", "export-eaf"):
            self.assertIn(command, result.stdout)

    def test_pose_launcher_reaches_packaged_worker(self) -> None:
        # A missing request exercises module startup without loading pose weights.
        package_root = Path(communication_gesture_annotator.__file__).parent.parent
        processes: list[subprocess.Popen[str]] = []
        popen = subprocess.Popen

        def start(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
            process = popen(*args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PYTHONPATH": str(package_root)}
        ), patch(
            "communication_gesture_annotator.media.pose.subprocess.Popen",
            side_effect=start,
        ) as launch:
            try:
                with self.assertRaisesRegex(MediaRuntimeError, "pose preprocessing failed"):
                    _run_worker(Path(directory) / "absent-request.json")
            finally:
                for process in processes:
                    if process.stderr is not None:
                        process.stderr.close()
            self.assertEqual(
                launch.call_args.args[0][1:3],
                ["-m", "communication_gesture_annotator.media.pose_worker"],
            )

    def test_visual_capacity_errors_still_have_a_distinct_category(self) -> None:
        for body in (
            b"image input limit exceeded", b"too many vision inputs",
            b"modality input length exceeds maximum",
        ):
            with self.subTest(body=body):
                self.assertEqual(_classify_http_error(400, body), "vision_input_limit")
        self.assertEqual(_classify_http_error(413, b"image body limit"), "request_too_large")
        self.assertEqual(_classify_http_error(400, b"model not found"), "model_not_found")
        self.assertEqual(_classify_http_error(400, b"unknown error"), "generic_http_error")


if __name__ == "__main__":
    unittest.main()
