# SPDX-License-Identifier: Apache-2.0
"""Evaluation and EAF must share LLM-only grouping without changing saved evidence."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from communication_gesture_annotator.pipeline.llm_projection import project_llm_output
from communication_gesture_annotator.export.eaf import export_eaf_document
from communication_gesture_annotator.export.completed_run import export_completed_run
from communication_gesture_annotator.evaluation.completed_run import evaluate_completed_run
from communication_gesture_annotator.export.json_export import write_json
from test_output_guards import document


def contrasting_document():
    doc = document([.9]*12)
    for i, row in enumerate(doc['samples']):
        row['joint_evidence']['resolution_records'][-1]['motion_grounding']['llm_non_g_unit_probability'] = .1 if i < 6 else .9
    return doc


def reference(root):
    doc = ET.Element('ANNOTATION_DOCUMENT', FORMAT='3.0', VERSION='3.0')
    ET.SubElement(doc, 'HEADER', TIME_UNITS='milliseconds')
    order = ET.SubElement(doc, 'TIME_ORDER')
    for t in (0,600,1200):
        ET.SubElement(order, 'TIME_SLOT', TIME_SLOT_ID=f't{t}', TIME_VALUE=str(t))
    for name,end,value in (('Manual_GUnit',600,'G-unit'),('Spker_vis',1200,'Half Body Shot')):
        tier = ET.SubElement(doc, 'TIER', TIER_ID=name)
        wrapper = ET.SubElement(tier, 'ANNOTATION')
        ann = ET.SubElement(wrapper, 'ALIGNABLE_ANNOTATION', ANNOTATION_ID=name,
            TIME_SLOT_REF1='t0', TIME_SLOT_REF2=f't{end}')
        ET.SubElement(ann, 'ANNOTATION_VALUE').text = value
    path = root/'reference.eaf'
    ET.ElementTree(doc).write(path,encoding='utf8',xml_declaration=True)
    return path


class LLMProjectionTest(unittest.TestCase):
    def test_uses_raw_llm_and_leaves_fused_source_unchanged(self):
        doc = contrasting_document()
        before = deepcopy(doc)
        with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
            projected = project_llm_output(doc)
        self.assertEqual(projected.membership, (True,)*6+(False,)*6)
        self.assertEqual([(u['start_us'],u['end_us']) for u in projected.units], [(0,600000)])
        self.assertEqual(projected.metadata['changed_sample_count'],6)
        self.assertEqual(doc,before)

    def test_existing_llm_output_is_identical_and_barriers_remain(self):
        for kwargs in ({}, {'excluded':(4,5)}, {'barrier':(4,)}):
            doc = document([.1]*4+[None]*2+[.1]*4,**kwargs)
            projected = project_llm_output(doc)
            self.assertEqual(projected.membership,tuple(r['is_g_unit'] for r in doc['samples']))
            self.assertEqual(list(projected.units),doc['g_units'])
        self.assertFalse(any(project_llm_output(document([.6]*20)).membership))

    def test_missing_or_invalid_raw_estimate_never_uses_fused_value(self):
        for invalid in (None,float('nan'),True,1.1):
            doc = contrasting_document()
            doc['samples'][0]['joint_evidence']['resolution_records'][-1]['motion_grounding']['llm_non_g_unit_probability'] = invalid
            with self.assertRaisesRegex(ValueError,'raw LLM estimate'):
                project_llm_output(doc)

    def test_missing_probability_records_are_rejected_even_when_all_are_absent(self) -> None:
        for score in (.1, .9):
            for indices in ((0,), tuple(range(12))):
                with self.subTest(score=score, missing=indices):
                    doc = document([score] * 12)
                    for i in indices:
                        doc['samples'][i]['joint_evidence']['resolution_records'] = []
                    with self.assertRaisesRegex(ValueError, 'raw LLM estimate'):
                        project_llm_output(doc)

    def test_all_unscored_or_excluded_samples_remain_valid(self) -> None:
        for kwargs in ({}, {'excluded': tuple(range(12))}, {'barrier': tuple(range(12))}):
            with self.subTest(kwargs=kwargs):
                projected = project_llm_output(document([None] * 12, **kwargs))
                self.assertEqual(projected.membership, (False,) * 12)
                self.assertEqual(projected.units, ())

    def test_missing_records_stop_evaluation_and_export_before_writing(self) -> None:
        for style in ('standalone', 'public', 'artifact_map'):
            with self.subTest(style=style), TemporaryDirectory() as temporary:
                root = Path(temporary)
                doc = contrasting_document()
                for row in doc['samples']:
                    row['joint_evidence']['resolution_records'] = []
                path = write_json(root / 'annotations.json', doc)
                digest = sha256(path.read_bytes()).hexdigest()
                if style != 'standalone':
                    manifest = {'run_id': doc['run_id'], 'status': 'completed'}
                    if style == 'public':
                        manifest.update(annotations=path.name, annotations_sha256=digest)
                    else:
                        manifest.update(artifacts=[path.name], artifact_sha256={path.name: digest})
                    write_json(root / 'run.json', manifest)
                ref = reference(root)
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                source = path if style == 'standalone' else root
                with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
                    for action in (
                        lambda: evaluate_completed_run(source, ref, export_eaf=True),
                        lambda: export_completed_run(source),
                        lambda: export_eaf_document(doc, root / 'direct.eaf'),
                    ):
                        with self.assertRaisesRegex(ValueError, 'raw LLM estimate'):
                            action()
                self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_evaluation_and_all_eaf_routes_agree_for_both_manifest_formats(self):
        for style in ('public','artifact_map'):
            with self.subTest(style=style), TemporaryDirectory() as temporary:
                root = Path(temporary); doc = contrasting_document()
                path = write_json(root/'annotations.json',doc)
                digest = sha256(path.read_bytes()).hexdigest()
                manifest = {'run_id':doc['run_id'],'status':'completed'}
                if style=='public':
                    manifest.update(annotations=path.name,annotations_sha256=digest)
                else:
                    manifest.update(artifacts=[path.name],artifact_sha256={path.name:digest})
                write_json(root/'run.json',manifest)
                original = path.read_bytes()
                with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
                    report = evaluate_completed_run(root,reference(root),export_eaf=True)
                    standalone = export_completed_run(root,root/'standalone.eaf')
                    direct = export_eaf_document(doc,root/'direct.eaf')
                self.assertEqual(report['evaluation']['f1'],1)
                self.assertEqual(report['evaluation']['balanced_accuracy'],1)
                self.assertEqual(report['policy_version'],'completed-run-reference-evaluation-v5')
                self.assertEqual(report['decision_basis']['score_source'],'llm')
                for eaf in (root/'annotations.eaf',standalone,direct):
                    tree = ET.parse(eaf)
                    slots = [int(s.get('TIME_VALUE')) for s in tree.findall('./TIME_ORDER/TIME_SLOT')]
                    self.assertEqual(slots,[0,600])
                    basis = json.loads(tree.findtext("./HEADER/PROPERTY[@NAME='decision_basis']"))
                    self.assertEqual(basis, report['decision_basis'])
                self.assertEqual(path.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
