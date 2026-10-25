# SPDX-License-Identifier: Apache-2.0
"""Assessment coverage tests with synthetic EAF labels and saved decisions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from communication_gesture_annotator.evaluation import completed_run as scoring
from communication_gesture_annotator.evaluation.reference_scope import (
    ReferenceScope, read_reference_scope,
)
from communication_gesture_annotator.export.completed_run import CompletedRun


def reference(root: Path, *, missing: bool = False, empty: bool = False) -> Path:
    doc = ET.Element("ANNOTATION_DOCUMENT", FORMAT="3.0", VERSION="3.0")
    ET.SubElement(doc, "HEADER", TIME_UNITS="milliseconds")
    order = ET.SubElement(doc, "TIME_ORDER")
    for t in range(0, 801, 100):
        ET.SubElement(order, "TIME_SLOT", TIME_SLOT_ID=f"t{t}", TIME_VALUE=str(t))
    tiers = {"Manual_GUnit": [(0, 300, "G-unit"), (500, 800, "G-unit")]}
    if not missing:
        tiers["Spker_vis"] = [] if empty else [
            (100, 300, "Visible"), (300, 400, " \t "),
            (500, 700, "No visible data"),
        ]
    for name, intervals in tiers.items():
        tier = ET.SubElement(doc, "TIER", TIER_ID=name)
        for index, (start, end, value) in enumerate(intervals):
            wrapper = ET.SubElement(tier, "ANNOTATION")
            span = ET.SubElement(wrapper, "ALIGNABLE_ANNOTATION", ANNOTATION_ID=f"{name}-{index}",
                                 TIME_SLOT_REF1=f"t{start}", TIME_SLOT_REF2=f"t{end}")
            ET.SubElement(span, "ANNOTATION_VALUE").text = value
    path = root / "synthetic.eaf"
    ET.ElementTree(doc).write(path, encoding="utf-8", xml_declaration=True)
    return path


def completed(root: Path) -> CompletedRun:
    return CompletedRun(root / "annotations.json", {
        "run_id": "synthetic", "provenance": {"research_configuration": {
            "inference": {"phase1_enabled": False}}},
        "samples": [{"sample": {"index": i, "start_us": i * 100000,
                                  "end_us": (i + 1) * 100000, "center_us": i * 100000 + 50000},
                     "is_g_unit": i in (0, 1, 4, 5, 7),
                     "joint_evidence": {"resolution_records": []}} for i in range(8)],
    }, "a" * 64, None)


class ReferenceScopeTests(unittest.TestCase):
    def test_scope_omits_leading_trailing_internal_gaps_and_blank_labels(self) -> None:
        with TemporaryDirectory() as directory:
            path = reference(Path(directory))
            scope = read_reference_scope(path, scoring.read_reference_g_units)
            self.assertEqual([i for i in range(8) if scope.contains(i * 100000 + 50000)], [1, 2, 5, 6])
            self.assertFalse(scope.contains(300000))
            self.assertTrue(scope.contains(100000))
            self.assertEqual(scope.gaps(0, 800000), [(0, 100000), (300000, 500000), (700000, 800000)])
            self.assertFalse(scope.covers(250000, 350000))
            shifted = read_reference_scope(path, scoring.read_reference_g_units, alignment_offset_us=100000)
            self.assertEqual([i for i in range(8) if shifted.contains(i * 100000 + 50000)], [2, 3, 6, 7])

    def test_adjacent_labels_cover_motion_interval_without_artificial_gap(self) -> None:
        scope = ReferenceScope.from_annotations([
            SimpleNamespace(start_us=0, end_us=100, annotation_value="Visible"),
            SimpleNamespace(start_us=100, end_us=200, annotation_value="No visible data"),
        ])
        self.assertTrue(scope.covers(50, 150))
        self.assertFalse(scope.contains(200))

    def test_completed_report_excludes_unlabelled_but_keeps_explicit_hidden_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = reference(root)
            run = completed(root)
            with patch.object(scoring, "load_completed_run", return_value=run):
                report = scoring.evaluate_completed_run(root, path)
            self.assertEqual(report["evaluation"]["interval_count"], 4)
            self.assertEqual(report["evaluation"]["confusion"]["true_positive"]["count"], 2)
            self.assertEqual(report["evaluation"]["confusion"]["false_negative"]["count"], 2)
            self.assertEqual(report["evaluation"]["confusion"]["false_positive"]["count"], 0)
            self.assertEqual(report["scope"]["excluded_by_reason"]["unlabelled_reference"],
                             {"count": 4, "duration_us": 400000})
            self.assertEqual(report["scope"]["excluded_duration_us"], 400000)
            self.assertIn("unlabelled_reference", (root / "reference-statistics.txt").read_text())

    def test_empty_or_missing_tier_has_no_invented_negative_samples(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for missing, empty in ((True, False), (False, True)):
                with self.subTest(missing=missing, empty=empty):
                    path = reference(root, missing=missing, empty=empty)
                    with patch.object(scoring, "load_completed_run", return_value=completed(root)):
                        report = scoring.evaluate_completed_run(root, path)
                    self.assertEqual(report["evaluation"]["interval_count"], 0)
                    self.assertIsNone(report["evaluation"]["accuracy"])
                    self.assertEqual(report["scope"]["excluded_sample_count"], 8)
                    if missing:
                        self.assertTrue(any("Phase1 cannot be scored" in warning for warning in report["warnings"]))
                        self.assertIn("Phase1 cannot be scored", (root / "reference-statistics.txt").read_text())
