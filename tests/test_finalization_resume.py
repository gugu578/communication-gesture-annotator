# SPDX-License-Identifier: Apache-2.0
"""Interrupted final output saves do not invalidate completed model work."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from communication_gesture_annotator.export.json_export import write_json
from communication_gesture_annotator.pipeline.llm_runner import (
    _MODEL_RUN_FINAL_ARTIFACT_NAMES,
    _MODEL_RUN_INTERMEDIATE_ARTIFACT_NAMES,
    validate_model_run_resume_state,
    write_pipeline_status,
)


class FinalizationResumeTest(unittest.TestCase):
    def prepare(self, root: Path, stage: str | None = "phase1") -> tuple[str, ...]:
        prefix = tuple(_MODEL_RUN_INTERMEDIATE_ARTIFACT_NAMES)
        for name in prefix:
            write_json(root / name, {})
        write_json(root / "model-checkpoints" / "checkpoint-test.json", {"saved": True})
        if stage is not None:
            write_pipeline_status(root, requested_stop_after="phase3", completed_stage=stage,
                artifacts=[root / name for name in prefix if name != "phase2-evidence.json"])
        return prefix

    def test_partial_outputs_are_allowed_without_reading_or_changing_them(self) -> None:
        for stage in ("phase1", "phase2"):
            with self.subTest(stage=stage), TemporaryDirectory() as directory:
                root = Path(directory)
                prefix = self.prepare(root, stage)
                for name in _MODEL_RUN_FINAL_ARTIFACT_NAMES:
                    # Partial outputs are rebuilt, never trusted as checkpoints.
                    (root / name).write_bytes(b"unfinished output")
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                          for p in root.rglob("*") if p.is_file()}
                self.assertEqual(validate_model_run_resume_state(root), prefix)
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_completed_run_remains_protected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, "phase3")
            write_json(root / "source.json", {})
            with self.assertRaisesRegex(ValueError, "completed Phase 3"):
                validate_model_run_resume_state(root)

    def test_legacy_final_outputs_without_status_remain_protected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, None)
            write_json(root / "annotations.json", {})
            with self.assertRaisesRegex(FileExistsError, "incomplete Phase2 lifecycle"):
                validate_model_run_resume_state(root)

    def test_partial_outputs_require_phase2_marker(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            (root / "phase2-evidence.json").unlink()
            write_json(root / "source.json", {})
            with self.assertRaisesRegex(FileExistsError, "incomplete Phase2 lifecycle"):
                validate_model_run_resume_state(root)

    def test_non_file_output_is_not_replaced(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            (root / "source.json").mkdir()
            with self.assertRaisesRegex(ValueError, "final artifact must be a file"):
                validate_model_run_resume_state(root)

    def test_partial_outputs_do_not_bypass_checkpoint_requirement(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            (root / "model-checkpoints" / "checkpoint-test.json").unlink()
            write_json(root / "source.json", {})
            with self.assertRaisesRegex(ValueError, "model checkpoint"):
                validate_model_run_resume_state(root)


if __name__ == "__main__":
    unittest.main()
