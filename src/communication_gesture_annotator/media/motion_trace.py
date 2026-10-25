# SPDX-License-Identifier: Apache-2.0
"""A deterministic PNG of accepted motion, aligned with the delivered RGB times."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..domain import AnalysisSample
from ..rights import validate_rights, write_rights_sidecar, read_rights_sidecar
from ..export.json_export import write_json

VERSION = 'phase2-motion-trace-v0.6'


def render_motion_trace(samples: Sequence[AnalysisSample], core: Sequence[int], context: Sequence[int],
                        measures: Mapping[int, Mapping[str, Any]], *, width: int = 1400, height: int = 780,
                        transition_cues: Mapping[str, Any] | None = None,
                        camera_cut_times_us: Sequence[int] = ()) -> bytes:
    import cv2
    import numpy as np
    from ..pipeline.motion_quality import LOCAL_DISPLAY_CAP
    if width < 900 or height < 650:
        raise ValueError('motion trace requires at least 900x650 pixels for readable labels')
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    ink, gray = (40, 40, 40), (120, 120, 120)
    colors = {'left': (195, 95, 25), 'right': (20, 125, 220)}
    left, right = 98, width-35
    start, end = samples[context[0]].start_us, samples[context[-1]].end_us
    def x(time: int) -> int:
        return round(left+(right-left)*(time-start)/(end-start))
    def text(label: str, at: tuple[int, int], color: tuple[int, int, int] = ink, scale: float = .52) -> None:
        length = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        scale *= min(1., (width-at[0]-15)/max(1, length))
        cv2.putText(canvas, label, at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    text('PHYSICAL HAND MOTION - use RGB for communicative meaning', (25, 27), scale=.7)
    text('LH (solid)', (98, 51), colors['left'], .55)
    text('RH (dashed)', (240, 51), colors['right'], .55)
    text('red X = flagged / omitted', (420, 51), (30, 30, 195))
    if camera_cut_times_us:
        text('purple line = camera edit', (770, 51), (160, 40, 140))
    sample_ms = max(samples[i].end_us-samples[i].start_us for i in core)/1000
    text(f'TARGET f{core[0]}..f{core[-1]} | slots 0..{len(core)-1} | up to {sample_ms:g} ms each | gaps are unavailable, never zero', (25, 75))
    ta, tb = x(samples[core[0]].start_us), x(samples[core[-1]].end_us)
    cv2.rectangle(canvas, (ta, 90), (tb, height-90), (216, 245, 255), -1)
    for label, a, b in (('HISTORY', left, ta), ('TARGET', ta, tb), ('LOOK-AHEAD', tb, right)):
        if b > a:
            text(label, (max(left, (a+b)//2-35), 106), scale=.46)
    scales = measures[context[0]].get('motion_scales', {})
    panels = [('transport', f"Palm/wrist transport (torso lengths/s) / {scales.get('transport', 1):g}", {'palm_speed', 'wrist_speed', 'palm_path', 'palm_xy', 'palm_speed_300ms', 'palm_speed_700ms'}),
              ('rotation', f"Torso-relative palm rotation (rad/s) / {scales.get('rotation', 3):g}; capped at {LOCAL_DISPLAY_CAP:g}", {'orientation_speed'}),
              ('fingers', f"Max extension/{scales.get('extension', 1):g}, articulation/{scales.get('articulation', 5):g}; capped at {LOCAL_DISPLAY_CAP:g}", {'extension_speed', 'articulation_speed'})]
    recent = {}
    if transition_cues is not None:
        column = transition_cues['hand_columns'].index('palm_shift_from_recent_position')
        recent = {r[0]: {side: r[n][column] for n, side in ((3, 'left'), (4, 'right'))}
                  for r in transition_cues['rows']}
        panels.append(('recent_shift', 'Palm shift from preceding 0.5 s median (torso lengths); NOT confirmed rest', {'palm_xy'}))
    def value_at(i: int, side: str, key: str) -> float | None:
        return (recent.get(i, {}).get(side) if key == 'recent_shift'
            else measures[i].get('score_components', {}).get(side, {}).get(key))
    panel_height = (height-230)//len(panels)
    for n, (key, title, flag_fields) in enumerate(panels):
        top, bottom = 138+n*panel_height, 138+n*panel_height+panel_height-45
        text(title+' (log1p axis)', (left, top-10))
        values = [value_at(i, side, key) for i in context for side in ('left', 'right')]
        maximum = max(1., max((v for v in values if v is not None), default=1.))
        def y(value: float) -> int:
            return round(bottom-(bottom-top)*math.log1p(value)/math.log1p(maximum))
        ticks = [0., .25, 1., 3., 8., maximum]
        used = []
        for value in sorted(set(v for v in ticks if v <= maximum)):
            yy = y(value)
            if any(abs(yy-last) < 16 for last in used):
                continue
            used.append(yy)
            cv2.line(canvas, (left, yy), (right, yy), (224, 224, 224), 1)
            text(f'{value:.2g}', (28, yy+5), gray, .45)
        for side in ('left', 'right'):
            last = None
            for i in context:
                if any(samples[i].start_us <= t < samples[i].end_us for t in camera_cut_times_us):
                    last = None
                value = value_at(i, side, key)
                if value is None:
                    last = None
                else:
                    a, b = (x(samples[i].start_us), y(value)), (x(samples[i].end_us), y(value))
                    if last is not None:
                        cv2.line(canvas, last, a, colors[side], 1, cv2.LINE_AA)
                    if side == 'left':
                        cv2.line(canvas, a, b, colors[side], 2, cv2.LINE_AA)
                    else:
                        for xx in range(a[0], b[0]+1, 10):
                            cv2.line(canvas, (xx, a[1]), (min(xx+5, b[0]), b[1]), colors[side], 2, cv2.LINE_AA)
                    last = b
                if any(f['field'] in flag_fields for f in measures[i].get('quality_flags', {}).get(side, [])):
                    xx, yy = x(samples[i].center_us), top+7+(10 if side == 'right' else 0)
                    cv2.drawMarker(canvas, (xx, yy), (30, 30, 195), cv2.MARKER_TILTED_CROSS, 7, 1)
        cv2.rectangle(canvas, (left, top), (right, bottom), (160, 160, 160), 1)
        for time in camera_cut_times_us:
            if start <= time < end:
                cv2.line(canvas, (x(time), top), (x(time), bottom), (160, 40, 140), 1)
    # Global times and indices tie the trace to the contact-sheet labels.
    for k in range(7):
        time = start+(end-start)*k//6
        xx = x(time)
        text(f'{time/1e6:.2f}s', (min(right-48, max(left, xx-22)), height-67), gray, .48)
    text('Global video time (seconds); frame numbers match RGB labels', (left, height-40))
    text('Flags are potential errors, not proof that motion did not occur. Exact values and reasons are in JSON.', (left, height-17), gray, .45)
    ok, png = cv2.imencode('.png', canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError('motion trace PNG encoding failed')
    return bytes(png)


def write_motion_trace(directory: Path, samples: Sequence[AnalysisSample], core: Sequence[int], context: Sequence[int],
                       measures: Mapping[int, Mapping[str, Any]], *, image_ordinal: int,
                       transition_cues: Mapping[str, Any] | None = None,
                       camera_cut_times_us: Sequence[int] = (),
                       rights: Mapping[str, Any], width: int = 1400, height: int = 780) -> tuple[Path, dict[str, Any]]:
    rights = validate_rights(rights)
    data = {'policy_version': VERSION, 'core': list(core), 'context': list(context),
        'times': [[samples[i].start_us, samples[i].end_us] for i in context],
        'measurements': [measures[i] for i in context], 'transition_cues': transition_cues,
        'camera_cut_times_us': list(camera_cut_times_us),
        'rights': rights, 'width': width, 'height': height}
    digest = sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'trace-{digest[:24]}.png'
    manifest = path.with_suffix('.json')
    if manifest.is_file() and path.is_file():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['input_sha256'] != digest or saved['image_sha256'] != sha256(path.read_bytes()).hexdigest() or read_rights_sidecar(path) != rights:
            raise ValueError('cached motion trace failed identity validation')
    else:
        png = render_motion_trace(samples, core, context, measures, width=width, height=height,
                                  transition_cues=transition_cues, camera_cut_times_us=camera_cut_times_us)
        path.write_bytes(png)
        write_rights_sidecar(path, rights, overwrite=True)
        saved = {'policy_version': VERSION, 'input_sha256': digest, 'image_sha256': sha256(png).hexdigest()}
        write_json(manifest, saved)
    return path, {**saved, 'image_ordinal': image_ordinal, 'width': width, 'height': height,
        'artifact_path': 'motion-traces/'+path.name,
        'role': 'physical_motion_trace_not_rgb_or_g_unit_labels', 'target_sample_indices': list(core),
        'context_time_us': [samples[context[0]].start_us, samples[context[-1]].end_us],
        'camera_cut_times_us': list(camera_cut_times_us),
        'axes': 'global video seconds; log1p of normalized rates and unscaled torso-length position shift; linear values in JSON',
        'transition_cue_policy_version': transition_cues.get('policy_version') if transition_cues is not None else None,
        'position_panel': 'shift from preceding 500 ms median palm position, in torso lengths; not a rest classifier' if transition_cues is not None else None,
        'colors': {'left': 'blue', 'right': 'orange', 'flags': 'red X', 'target': 'yellow'},
        'missing': 'gaps are unavailable or excluded measurements, not stillness'}
