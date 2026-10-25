# SPDX-License-Identifier: Apache-2.0
"""Resume rejects another video before touching cached work."""

from __future__ import annotations

from contextlib import redirect_stderr
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from communication_gesture_annotator.cli import main
from communication_gesture_annotator.media.resume_guard import validate_resume_source


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _saved_run(root: Path, contents: bytes = b"original-video") -> tuple[Path, Path]:
    source = root / "source.mp4"
    source.write_bytes(contents)
    run = root / "run"
    digest = sha256(contents).hexdigest()
    _write(run / "pipeline-status.json", {"source_sha256": digest})
    _write(run / "pose" / "manifest.json", {
        "input": {"source_sha256": digest, "source_byte_size": len(contents)}
    })
    return run, source


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


class ResumeSourceTest(unittest.TestCase):
    def test_wrong_same_size_video_rejected_without_changes(self) -> None:
        with TemporaryDirectory() as directory:
            run, source = _saved_run(Path(directory))
            source.write_bytes(b"different-file")
            before = _snapshot(run)
            with self.assertRaisesRegex(ValueError, "use the original video or a new --run-dir"):
                validate_resume_source(run, source)
            self.assertEqual(_snapshot(run), before)

    def test_same_source_can_move_or_be_renamed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run, source = _saved_run(root)
            moved = root / "renamed.mp4"
            source.rename(moved)
            before = _snapshot(run)
            validate_resume_source(run, moved)
            self.assertEqual(_snapshot(run), before)

    def test_durable_identity_survives_overwritten_preprocessing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run, source = _saved_run(root)
            other = root / "other.mp4"
            other.write_bytes(b"a shortened video")
            _write(run / "preprocessing.json", {"source": {
                "sha256": sha256(other.read_bytes()).hexdigest(), "byte_size": other.stat().st_size
            }})
            validate_resume_source(run, source)
            with self.assertRaisesRegex(ValueError, "source video does not match"):
                validate_resume_source(run, other)

    def test_inconsistent_durable_identities_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            run, source = _saved_run(Path(directory))
            _write(run / "pipeline-status.json", {"source_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "identities disagree"):
                validate_resume_source(run, source)

    def test_saved_size_must_also_match(self) -> None:
        with TemporaryDirectory() as directory:
            run, source = _saved_run(Path(directory))
            _write(run / "pose" / "manifest.json", {"input": {
                "source_sha256": sha256(source.read_bytes()).hexdigest(), "source_byte_size": 999
            }})
            with self.assertRaisesRegex(ValueError, "source video does not match"):
                validate_resume_source(run, source)

    def test_legacy_preprocessing_identity_is_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"original")
            run = root / "run"
            _write(run / "preprocessing.json", {"source": {
                "sha256": sha256(source.read_bytes()).hexdigest(), "byte_size": 8
            }})
            validate_resume_source(run, source)
            source.write_bytes(b"replaced")
            with self.assertRaisesRegex(ValueError, "source video does not match"):
                validate_resume_source(run, source)

    def test_no_identity_leaves_lifecycle_validation_in_charge(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            validate_resume_source(root / "nonexistent-run", root / "nonexistent-source.mp4")

    def test_cli_rejects_before_clients_lifecycle_or_preprocessing(self) -> None:
        with TemporaryDirectory() as directory:
            run, source = _saved_run(Path(directory))
            source.write_bytes(b"different-file")
            before = _snapshot(run)
            config = SimpleNamespace(
                storage=SimpleNamespace(mode="compact"),
                detection=SimpleNamespace(phase2_start_us=0, phase2_end_us=None),
                inference=SimpleNamespace(phase1_enabled=True),
                validate_export=lambda *args, **kwargs: None,
            )
            stderr = StringIO()
            with (
                patch("communication_gesture_annotator.cli.load_research_config", return_value=config),
                patch("communication_gesture_annotator.cli.OpenWebUIClient") as client,
                patch("communication_gesture_annotator.cli.preprocess_video") as preprocessing,
                patch("communication_gesture_annotator.cli.validate_model_run_resume_state") as lifecycle,
                patch("communication_gesture_annotator.cli.TraceWriter") as trace,
                redirect_stderr(stderr),
            ):
                result = main(["run-llm", str(source), "--run-dir", str(run),
                               "--phase2-config", "unused.toml", "--resume", "--log-level", "ERROR"])
            self.assertEqual(result, 2)
            self.assertIn("original video", stderr.getvalue())
            client.assert_not_called()
            preprocessing.assert_not_called()
            lifecycle.assert_not_called()
            trace.assert_not_called()
            self.assertEqual(_snapshot(run), before)


if __name__ == "__main__":
    unittest.main()
