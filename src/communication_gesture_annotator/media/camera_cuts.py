# SPDX-License-Identifier: Apache-2.0
"""Local image discontinuities, separate from speaker identity and G-unit rest."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from statistics import median
import subprocess
from tempfile import TemporaryFile
from typing import Any, Iterable, Mapping, Sequence

from .ffmpeg import MediaRuntimeError, _resolve_executable

VERSION = "native-image-discontinuities-v0.1"
WIDTH, HEIGHT = 96, 54
POLICY = {"version": VERSION, "thumbnail_width": WIDTH, "thumbnail_height": HEIGHT,
          "minimum_mean_difference": 18.0, "minimum_changed_fraction": .20,
          "minimum_changed_tiles": 4, "minimum_change_ratio": 2.5,
          "meaning": "camera geometry boundary; never a speaker exclusion or a rest label"}


def detect_changes(frames: Iterable[tuple[int, int, Any]]) -> list[dict[str, Any]]:
    """Compare native adjacent images; timestamps come from validated source PTS.

    Broad, abrupt change is required so ordinary local hand movement and gradual
    image drift do not normally produce a boundary. This is a conservative edit
    detector, not a semantic scene classifier. No model or reference labels.
    """
    import numpy as np
    previous = None
    recent: deque[float] = deque(maxlen=5)
    cuts = []
    for index, time_us, image in frames:
        if image.shape != (HEIGHT, WIDTH, 3):
            raise ValueError("camera-cut thumbnail has unexpected dimensions")
        if previous is not None:
            old_index, old_time, old = previous
            if time_us <= old_time or index <= old_index:
                raise ValueError("camera-cut observations must be strictly ordered")
            if index != old_index + 1:
                recent.clear()
            else:
                delta = np.abs(image.astype(np.int16) - old.astype(np.int16)).mean(axis=2)
                mean = float(delta.mean())
                fraction = float((delta >= 20).mean())
                tiles = sum(float(tile.mean()) >= 12 for band in np.array_split(delta, 3)
                            for tile in np.array_split(band, 3, axis=1))
                background = max(1., median(recent)) if recent else 1.
                if (mean >= POLICY['minimum_mean_difference']
                        and fraction >= POLICY['minimum_changed_fraction']
                        and tiles >= POLICY['minimum_changed_tiles']
                        and mean >= POLICY['minimum_change_ratio'] * background):
                    cuts.append({"source_frame_index": index, "time_us": time_us,
                                 "previous_time_us": old_time, "mean_difference": mean,
                                 "changed_fraction": fraction, "changed_tiles": tiles,
                                 "change_ratio": mean/background})
                recent.append(mean)
        previous = (index, time_us, image)
    return cuts


def scan_source(source: Path, rows: Sequence[Mapping[str, Any]], *, ffmpeg: str) -> list[dict[str, Any]]:
    """Stream tiny native-rate images; keep neither decoded video nor thumbnails.

    Decoded ordinal selection matches exact-source RGB extraction. The raw pose
    records provide PTS; no annotation time is inferred from nominal frame rate.
    Only observations in the requested processing range are compared.
    """
    import numpy as np
    included = [r for r in rows if r.get('source_frame', {}).get('processing_status') != 'outside_processing_range']
    if not included:
        return []
    first, last = included[0]['source_frame_index'], included[-1]['source_frame_index']
    wanted = {r['source_frame_index']: r['canonical_time_us'] for r in included}
    count = last-first+1
    command = [_resolve_executable(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin',
               '-copyts', '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
               '-vf', f'trim=start_frame={first}:end_frame={last+1},scale={WIDTH}:{HEIGHT}:flags=area,setsar=1',
               '-fps_mode', 'passthrough', '-frames:v', str(count), '-threads', '1',
               '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    with TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            assert process.stdout is not None
            def frames() -> Iterable[tuple[int, int, Any]]:
                size = WIDTH*HEIGHT*3
                for index in range(first, last+1):
                    data = bytearray()
                    while len(data) < size:
                        block = process.stdout.read(size-len(data))
                        if not block:
                            raise MediaRuntimeError("camera-cut decoding did not preserve native frame coverage")
                        data.extend(block)
                    if index in wanted:
                        yield index, wanted[index], np.frombuffer(data, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
                if process.stdout.read(1):
                    raise MediaRuntimeError("camera-cut decoder returned unexpected extra frames")
            cuts = detect_changes(frames())
            if process.wait() != 0:
                errors.seek(0)
                raise MediaRuntimeError("camera-cut source decoding failed: " + errors.read(2000).decode('utf-8', 'replace'))
            return cuts
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()


def segment_observations(rows: Sequence[dict[str, Any]], people: Sequence[Any],
                         cuts: Sequence[Mapping[str, Any]]) -> None:
    """Split geometry before smoothing, without modifying raw pose files."""
    indices = {c['source_frame_index'] for c in cuts}
    if not indices.issubset({r['source_frame_index'] for r in rows}):
        raise ValueError("camera cut references an unknown source observation")
    shot = 0
    shots = {}
    for row in rows:
        index = row['source_frame_index']
        if index in indices:
            shot += 1
        shots[index] = shot
        row['camera_shot_id'] = shot
        row['camera_cut_before'] = index in indices
    for person in people:
        # A track can identify the same speaker across edits. Its measurement
        # segment cannot: all later derivatives key on this segment identity.
        if shots[person.source_frame_index]:
            person.segment_id += f"-camera-{shots[person.source_frame_index]}"
