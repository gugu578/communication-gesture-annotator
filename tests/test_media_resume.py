# SPDX-License-Identifier: Apache-2.0
"""Retained media must remain byte-identical and require no new decoder work."""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from communication_gesture_annotator.config import AnalysisConfig
from communication_gesture_annotator.domain import AnalysisSample
from communication_gesture_annotator.media.ffmpeg import preprocess_video
from communication_gesture_annotator.media.frame_extract import extract_analysis_frames
from communication_gesture_annotator.media.image_packaging import (
    FrameImage, ContactSheetLayout, plan_image_packages, render_image_packages,
)
from communication_gesture_annotator.rights import unrestricted_or_unspecified_rights


class MediaResumeTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_scope_proxy_frames_settings_and_integrity_without_decoding(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, run = root/'source.mp4', root/'run'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=160x100:rate=25:duration=2', '-an', '-c:v', 'libx264', str(source)], check=True)
            options = dict(config=AnalysisConfig(), include_audio=False, start_us=400000, end_us=1300000)
            first = preprocess_video(source, run, **options)
            samples = [AnalysisSample(**r) for r in first['proxy']['frame_mapping']]
            frames = extract_analysis_frames(run/'model-proxy.mp4', samples, run/'analysis-frames',
                                            maximum_width=120, jpeg_quality=75)
            with patch('communication_gesture_annotator.media.ffmpeg.probe_media',
                       side_effect=AssertionError('resume probed media')), patch(
                    'communication_gesture_annotator.media.frame_extract._run_extraction',
                    side_effect=AssertionError('resume decoded frames')):
                self.assertEqual(preprocess_video(source, run, **options, resume=True, overwrite=True), first)
                restored = extract_analysis_frames(run/'model-proxy.mp4', [asdict(s) for s in samples],
                    run/'analysis-frames', maximum_width=120, jpeg_quality=75, resume=True, overwrite=True)
                self.assertEqual(restored, frames)
                with self.assertRaisesRegex(ValueError, 'settings'):
                    extract_analysis_frames(run/'model-proxy.mp4', samples, run/'analysis-frames',
                        maximum_width=160, jpeg_quality=75, resume=True, overwrite=True)
                with self.assertRaisesRegex(ValueError, 'time range'):
                    preprocess_video(source, run, **{**options, 'start_us': 500000}, resume=True, overwrite=True)
                target = Path(frames[5]['path'])
                target.write_bytes(target.read_bytes()+b'corruption')
                with self.assertRaisesRegex(ValueError, 'rights manifest'):
                    extract_analysis_frames(run/'model-proxy.mp4', samples, run/'analysis-frames',
                        maximum_width=120, jpeg_quality=75, resume=True, overwrite=True)
                source.write_bytes(source.read_bytes()+b'another-video')
                with self.assertRaisesRegex(ValueError, 'different source'):
                    preprocess_video(source, run, **options, resume=True, overwrite=True)

    def test_visibility_sheet_is_reused_only_for_identical_plan_and_source(self) -> None:
        def render(command: list[str]) -> None:
            count = int(command[command.index('-frames:v')+1])
            for i in range(1, count+1):
                Path(command[-1].replace('%04d', f'{i:04d}')).write_bytes(b'synthetic-render')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.jpg'
            source.write_bytes(b'synthetic-input')
            frames = [FrameImage(sample_index=5, start_us=500000, end_us=600000,
                path=source, sha256=sha256(source.read_bytes()).hexdigest(),
                rights=unrestricted_or_unspecified_rights())]
            layout = ContactSheetLayout(columns=2, rows=1, tile_width_px=160, tile_height_px=90)
            plans = plan_image_packages(frames, layout)
            with patch('communication_gesture_annotator.media.image_packaging._run_packaging', side_effect=render) as renderer:
                initial = render_image_packages(plans, root/'sheets', ffmpeg=sys.executable)
                saved_time = initial.images[0].path.stat().st_mtime_ns
                resumed = render_image_packages(plans, root/'sheets', ffmpeg=sys.executable,
                                                 overwrite=True, reuse_existing=True)
                self.assertEqual(renderer.call_count, 1)
                self.assertEqual(resumed, initial)
                self.assertEqual(resumed.images[0].path.stat().st_mtime_ns, saved_time)
                changed = plan_image_packages(frames, replace(layout, tile_width_px=180))
                render_image_packages(changed, root/'sheets', ffmpeg=sys.executable,
                                      overwrite=True, reuse_existing=True)
                self.assertEqual(renderer.call_count, 2)
                source.write_bytes(b'changed source')
                with self.assertRaises(ValueError):
                    render_image_packages(changed, root/'sheets', ffmpeg=sys.executable,
                                          overwrite=True, reuse_existing=True)


if __name__ == '__main__':
    unittest.main()
