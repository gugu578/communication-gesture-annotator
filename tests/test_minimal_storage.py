# SPDX-License-Identifier: Apache-2.0
"""Minimal retention must preserve bytes, durable answers and reusable source RGB."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from communication_gesture_annotator.adapters.image_recovery import (
    ImageRecovery, release_checkpointed_images,
)
from communication_gesture_annotator.export.json_export import read_json, read_json_snapshot, write_json
from communication_gesture_annotator.storage import (
    compact_enabled, current_mode, data_path, minimal_enabled, phase_result_path,
    release_cache_files, release_phase1_images, storage_mode,
)


class MinimalStorageTests(unittest.TestCase):
    def test_phase_results_round_trip_and_mode_changes_keep_existing_encoding(self) -> None:
        payload = {"motion": [0.12345678901234567] * 1000, "timestamp_us": 1234567}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with storage_mode("minimal"):
                self.assertTrue(compact_enabled() and minimal_enabled())
                self.assertEqual(current_mode(), "minimal")
                path = write_json(phase_result_path(root / "phase2-evidence.json"), payload)
                self.assertEqual(path.suffix, ".gz")
                self.assertEqual(read_json(root / "phase2-evidence.json"), payload)
                self.assertEqual(read_json_snapshot(path)[1], sha256(path.read_bytes()).hexdigest())
                other = write_json(root / "other.json.gz", payload)
                self.assertEqual(path.read_bytes(), other.read_bytes())
                self.assertLess(path.stat().st_size, len(json.dumps(payload)) / 10)
                plain = write_json(root / "existing.json", payload)
                self.assertEqual(phase_result_path(plain), plain)
            with storage_mode("debug"):
                self.assertEqual(phase_result_path(root / "phase2-evidence.json"), path)
            (root / "phase2-evidence.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                data_path(root / "phase2-evidence.json")

    def test_interrupted_gzip_write_preserves_previous_phase_result(self) -> None:
        def interrupt(*args, **kwargs):
            yield "{"
            raise OSError("synthetic disk full")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = write_json(root / "phase2-evidence.json.gz", {"saved": True})
            original = target.read_bytes()
            with patch.object(json.JSONEncoder, "iterencode", interrupt):
                with self.assertRaisesRegex(OSError, "disk full"):
                    write_json(target, {"saved": False})
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [target])
            target.write_bytes(b"not a gzip stream")
            with self.assertRaisesRegex(ValueError, "gzip artifact is invalid"):
                read_json(target)
            target.write_bytes(original[:15])
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_json(target)

    def test_crop_cache_is_bounded_and_regeneration_keeps_pixels_and_sources(self) -> None:
        import cv2
        import numpy as np
        from communication_gesture_annotator.media.phase2_rgb import CleanPhase2Frames, compact_sheets
        from communication_gesture_annotator.media.image_packaging import ContactSheetLayout
        from communication_gesture_annotator.rights import unrestricted_or_unspecified_rights
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames = {}
            for i in range(8):
                path = root / f"source-{i}.png"
                pixels = np.full((90, 160, 3), i * 30, np.uint8)
                path.write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
                frames[i] = dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest(),
                    sample_index=i, start_us=i*100000, end_us=(i+1)*100000)
            # Duplicate pixels can legitimately share a single derived path.
            frames[7] = {**frames[1], "sample_index": 7, "start_us": 700000, "end_us": 800000}
            rights = unrestricted_or_unspecified_rights()
            clean = CleanPhase2Frames(frames, [], root / "phase2-rgb", rights)
            with storage_mode("minimal"):
                clean.prepare([0, 1, 2])
                first = {i: Path(clean.frames[i]["path"]).read_bytes() for i in (0, 1, 2)}
                clean.prepare([1, 2, 3, 7])
                self.assertLessEqual(len(list((root / "phase2-rgb/frames").glob("*.png"))), 3)
                clean.prepare([7])
                self.assertEqual(Path(clean.frames[7]["path"]).read_bytes(), first[1])
                clean.prepare([0, 1, 2])
                self.assertEqual(first, {i: Path(clean.frames[i]["path"]).read_bytes() for i in first})
                layout = ContactSheetLayout(columns=3, rows=1, tile_width_px=160, tile_height_px=100)
                labels = {i: f"TARGET f{i} {i/10}s" for i in first}
                sheet, _ = compact_sheets(clean.frames, list(first), labels, layout, root / "sheets", rights, 1)
                before = sheet[0].read_bytes()
                clean.release_except(set())
                clean.prepare(list(first))
                sheet2, _ = compact_sheets(clean.frames, list(first), labels, layout, root / "regenerated", rights, 1)
                self.assertEqual(before, sheet2[0].read_bytes())
                clean.release_except(set())
            self.assertFalse(list((root / "phase2-rgb/frames").glob("*")))
            self.assertTrue(all(Path(frame["path"]).is_file() for frame in frames.values()))

    def test_recovery_cleanup_keeps_retry_decisions_and_unrelated_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = ImageRecovery.request_directory(root, "test") / ("a"*64)
            cache.mkdir(parents=True)
            names = ["decision.json", "size-rejected.json", "keep.txt", "images.json",
                "images.json.rights.json", "image-000.png", "image-000.png.rights.json"]
            for name in names:
                (cache / name).write_text("saved")
            with storage_mode("compact"):
                release_checkpointed_images(root, "test")
                self.assertEqual(len(list(cache.iterdir())), len(names))
            with storage_mode("minimal"):
                release_checkpointed_images(root, "test")
                release_checkpointed_images(root, "test")
            self.assertEqual({p.name for p in cache.iterdir()}, {"decision.json", "size-rejected.json", "keep.txt"})
            outside = root / "source.png"
            outside.write_bytes(b"source")
            with self.assertRaisesRegex(ValueError, "outside"):
                release_cache_files(cache, [outside])
            self.assertTrue(outside.exists())

    def test_phase1_cleanup_only_removes_known_package_outputs(self) -> None:
        with TemporaryDirectory() as directory, storage_mode("minimal"):
            root = Path(directory)
            cache = root / "model-inputs" / "visibility-0000"
            cache.mkdir(parents=True)
            for name in ("contact-sheet-0000-abcdef123456.jpg", "contact-sheet-0000-abcdef123456.jpg.rights.json",
                         "image-packages.json", "rights-manifest.json", "keep.jpg"):
                (cache / name).write_text("saved")
            release_phase1_images(root)
            self.assertEqual({p.name for p in cache.iterdir()}, {"keep.jpg"})


if __name__ == "__main__":
    unittest.main()
