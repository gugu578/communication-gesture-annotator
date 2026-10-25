# SPDX-License-Identifier: Apache-2.0
"""Selected preprocessing reuse is complete or falls back without partial copying."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from communication_gesture_annotator.export.json_export import write_json
from communication_gesture_annotator.media.scoped_reuse import reuse_analysis_slice
from communication_gesture_annotator.rights import unrestricted_or_unspecified_rights


class ReuseTests(unittest.TestCase):
    def test_only_selected_frames_are_required_and_missing_data_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, staging = root / "old", root / "new", root / "staging"
            for path in (old / "analysis-frames", new, staging):
                path.mkdir(parents=True)
            rights = unrestricted_or_unspecified_rights()
            manifest = {
                "source": {"sha256": "a" * 64},
                "proxy": {
                    "sampling_rate_hz": 10,
                    "maximum_width": 960,
                    "rights": rights,
                    "frame_mapping": [{"sample_index": i} for i in range(6)],
                },
            }
            write_json(old / "preprocessing.json", manifest)
            write_json(new / "preprocessing.json", manifest)
            write_json(
                old / "inference-plan.json",
                {"model_limits": {"frame_width": 640, "jpeg_quality": 85}},
            )
            artifacts = []
            for index in (2, 3):
                path = old / "analysis-frames" / f"sample-{index:08d}.jpg"
                content = f"synthetic-image-{index}".encode()
                path.write_bytes(content)
                artifacts.append(
                    {"filename": path.name, "sha256": hashlib.sha256(content).hexdigest()}
                )
            write_json(
                old / "analysis-frames" / "rights-manifest.json",
                {"rights": rights, "artifacts": artifacts},
            )
            self.assertTrue(
                reuse_analysis_slice(old, new, [2, 3], staging, maximum_width=640, jpeg_quality=85)
            )
            self.assertEqual(
                sorted(p.name for p in staging.iterdir()),
                ["frame-00000000.jpg", "frame-00000001.jpg"],
            )
            missing = root / "missing"
            missing.mkdir()
            self.assertFalse(
                reuse_analysis_slice(
                    old, new, [2, 3, 4], missing, maximum_width=640, jpeg_quality=85
                )
            )
            self.assertEqual(list(missing.iterdir()), [])
            changed = deepcopy(manifest)
            changed["source"]["sha256"] = "b" * 64
            write_json(new / "preprocessing.json", changed)
            self.assertFalse(
                reuse_analysis_slice(old, new, [2, 3], missing, maximum_width=640, jpeg_quality=85)
            )


if __name__ == "__main__":
    unittest.main()
