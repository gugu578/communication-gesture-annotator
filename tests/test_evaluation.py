# SPDX-License-Identifier: Apache-2.0
"""Offline reference metrics and CLI integration, using synthetic annotations only."""

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from communication_gesture_annotator.cli import main
from communication_gesture_annotator.evaluation.completed_run import evaluate_completed_run
from communication_gesture_annotator.evaluation.samples import read_reference_g_units
from communication_gesture_annotator.export.json_export import read_json, write_json
from communication_gesture_annotator.rights import unrestricted_or_unspecified_rights


def reference_file(root: Path, *, tier: str = "Manual_GUnit") -> Path:
    document = ET.Element("ANNOTATION_DOCUMENT", FORMAT="3.0", VERSION="3.0")
    ET.SubElement(document, "HEADER", TIME_UNITS="milliseconds")
    order = ET.SubElement(document, "TIME_ORDER")
    for i, time in enumerate((0, 100, 200, 350, 500, 700, 800)):
        ET.SubElement(order, "TIME_SLOT", TIME_SLOT_ID=f"t{i}", TIME_VALUE=str(time))
    units = ET.SubElement(document, "TIER", TIER_ID=tier)
    hidden = ET.SubElement(document, "TIER", TIER_ID="Spker_vis")
    for parent, name, start, end, value in (
        (units, "g1", 0, 1, "G-unit"),
        (units, "g2", 2, 3, "G-unit"),
        (units, "other", 4, 5, "Other"),
        (hidden, "visible", 0, 5, "Visible"),
        (hidden, "hidden", 5, 6, "No visible data"),
    ):
        wrapper = ET.SubElement(parent, "ANNOTATION")
        annotation = ET.SubElement(
            wrapper,
            "ALIGNABLE_ANNOTATION",
            ANNOTATION_ID=name,
            TIME_SLOT_REF1=f"t{start}",
            TIME_SLOT_REF2=f"t{end}",
        )
        ET.SubElement(annotation, "ANNOTATION_VALUE").text = value
    target = root / "reference.eaf"
    ET.ElementTree(document).write(target, encoding="utf-8", xml_declaration=True)
    return target


def completed_run(
    root: Path, *, excluded: tuple[int, ...] = (), enabled: bool = True, terminal_us: int = 800000
) -> Path:
    run = root / "run"
    run.mkdir()
    rows = []
    for i, prediction in enumerate((True, True, False, False, True, False, False, True)):
        end = min((i + 1) * 100000, terminal_us)
        records = (
            [
                {
                    "sample_index": i,
                    "exclude_from_detection": True,
                    "policy_version": "phase2-time-selection-v0.1"
                    if i == 1
                    else "phase1-admission-v0.1",
                }
            ]
            if i in excluded
            else []
        )
        rows.append(
            {
                "sample": {
                    "index": i,
                    "start_us": i * 100000,
                    "end_us": end,
                    "center_us": (i * 100000 + end) // 2,
                },
                "is_g_unit": prediction,
                "joint_evidence": {"resolution_records": records},
            }
        )
    intervals = [(0, 2), (4, 5), (7, 8)]
    document = {
        "schema_version": "0.2.0",
        "run_id": "synthetic-evaluation",
        "source": {"rights": unrestricted_or_unspecified_rights()},
        "preprocessing": {},
        "config": {},
        "participant_profiles": [{"participant_id": "subject-0001"}],
        "samples": rows,
        "activity_candidates": [{"candidate_id": f"c{i}"} for i in range(len(intervals))],
        "g_units": [
            {
                "g_unit_id": f"g{i}",
                "candidate_id": f"c{i}",
                "start_sample": a,
                "end_sample_exclusive": b,
                "start_us": a * 100000,
                "end_us": rows[b - 1]["sample"]["end_us"],
            }
            for i, (a, b) in enumerate(intervals)
        ],
        "provenance": {"research_configuration": {"inference": {"phase1_enabled": enabled}}},
    }
    write_json(run / "annotations.json", document)
    write_json(
        run / "run.json",
        {
            "run_id": document["run_id"],
            "status": "completed",
            "annotations": "annotations.json",
            "annotations_sha256": sha256((run / "annotations.json").read_bytes()).hexdigest(),
        },
    )
    return run


class EvaluationTests(unittest.TestCase):
    def test_cli_reports_known_counts_and_exports_without_media_or_model(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run, reference = completed_run(root), reference_file(root)
            inputs = {
                path: path.read_bytes()
                for path in (reference, run / "annotations.json", run / "run.json")
            }
            output, errors = StringIO(), StringIO()
            with (
                patch(
                    "communication_gesture_annotator.cli.OpenWebUIClient",
                    side_effect=AssertionError("model called"),
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                for _ in range(2):
                    self.assertEqual(
                        main(["evaluate-run", str(run), str(reference), "--export-eaf"]), 0
                    )
            report = read_json(run / "reference-statistics.json")
            confusion = report["evaluation"]["confusion"]
            self.assertEqual(
                {name: item["count"] for name, item in confusion.items()},
                {"true_positive": 1, "false_negative": 1, "false_positive": 3, "true_negative": 3},
            )
            self.assertAlmostEqual(report["evaluation"]["f1"], 1 / 3)
            self.assertEqual(report["evaluation"]["recall"], 0.5)
            self.assertEqual(report["evaluation"]["balanced_accuracy"], 0.5)
            self.assertEqual(report["model_calls"], 0)
            self.assertIn("visibility-check errors", errors.getvalue())
            self.assertIn("f1: 33.33%", output.getvalue())
            text = (run / "reference-statistics.txt").read_text(encoding="utf-8")
            self.assertIn("False alarms", text)
            self.assertIn("False negatives", text)
            self.assertNotIn("Error intervals", text)
            self.assertNotIn("error_intervals", report)
            self.assertTrue((run / "annotations.eaf.provenance.json").is_file())
            self.assertEqual(
                {node.text for node in ET.parse(run / "annotations.eaf").iter("ANNOTATION_VALUE")},
                {"G-unit"},
            )
            for path, before in inputs.items():
                self.assertEqual(path.read_bytes(), before)

    def test_only_phase1_and_time_scope_are_excluded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run, reference = completed_run(root, excluded=(1, 5)), reference_file(root)
            report = evaluate_completed_run(run, reference)
            self.assertEqual(report["evaluation"]["interval_count"], 6)
            self.assertEqual(
                report["scope"]["excluded_by_reason"],
                {
                    "outside_selected_time": {"count": 1, "duration_us": 100000},
                    "phase1": {"count": 1, "duration_us": 100000},
                    "unlabelled_reference": {"count": 0, "duration_us": 0},
                },
            )
            # Visibility labels, including labels placed in the G-unit tier, never mask samples.
            tree = ET.parse(reference)
            tree.find(".//ALIGNABLE_ANNOTATION[@ANNOTATION_ID='visible']/ANNOTATION_VALUE").text = "Other visible view"
            tree.find(
                ".//ALIGNABLE_ANNOTATION[@ANNOTATION_ID='other']/ANNOTATION_VALUE"
            ).text = "No visible data"
            tree.write(reference, encoding="utf-8", xml_declaration=True)
            self.assertEqual(
                evaluate_completed_run(run, reference)["evaluation"], report["evaluation"]
            )

    def test_all_excluded_and_short_last_cell(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = completed_run(root, excluded=tuple(range(8)), enabled=False, terminal_us=770000)
            report = evaluate_completed_run(run, reference_file(root))
            self.assertEqual(report["scope"]["excluded_duration_us"], 770000)
            self.assertEqual(report["evaluation"]["interval_count"], 0)
            for metric in ("precision", "recall", "f1", "accuracy", "false_positive_rate"):
                self.assertIsNone(report["evaluation"][metric])
            self.assertEqual(report["warnings"], [])

    def test_custom_mapping_offset_and_standalone_annotations(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run, reference = completed_run(root, enabled=False), reference_file(root, tier="Custom")
            (run / "run.json").unlink()
            report = evaluate_completed_run(
                run / "annotations.json",
                reference,
                reference_tier="Custom",
                positive_value="Other",
                alignment_offset_us=100000,
                output=root / "scores.json",
            )
            confusion = report["evaluation"]["confusion"]
            self.assertEqual(confusion["true_positive"]["count"], 1)
            self.assertEqual(confusion["false_negative"]["count"], 1)
            self.assertTrue((root / "scores.txt").is_file())
            self.assertEqual(report["warnings"], [])

    def test_incomplete_or_changed_run_and_protected_outputs_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run, reference = completed_run(root), reference_file(root)
            manifest = read_json(run / "run.json")
            manifest["status"] = "running"
            write_json(run / "run.json", manifest)
            with self.assertRaisesRegex(ValueError, "completed"):
                evaluate_completed_run(run, reference)
            manifest["status"] = "completed"
            manifest["annotations_sha256"] = "0" * 64
            write_json(run / "run.json", manifest)
            with self.assertRaisesRegex(ValueError, "hash"):
                evaluate_completed_run(run, reference)
            manifest["annotations_sha256"] = sha256(
                (run / "annotations.json").read_bytes()
            ).hexdigest()
            write_json(run / "run.json", manifest)
            for protected in (run / "annotations.json", run / "run.json"):
                before = protected.read_bytes()
                with self.assertRaises(ValueError):
                    evaluate_completed_run(run, reference, output=protected)
                self.assertEqual(protected.read_bytes(), before)
            # Validate all output destinations before optional export writes anything.
            renamed_reference = root / "report.txt"
            renamed_reference.write_bytes(reference.read_bytes())
            with self.assertRaises(ValueError):
                evaluate_completed_run(
                    run, renamed_reference, output=root / "report.json", export_eaf=True
                )
            self.assertFalse((run / "annotations.eaf").exists())

    def test_reference_mapping_and_unsafe_xml_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run, reference = completed_run(root), reference_file(root)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                evaluate_completed_run(run, reference, reference_tier="wrong")
            reference.write_text(
                '<!DOCTYPE a [<!ENTITY x "value">]><ANNOTATION_DOCUMENT/>', encoding="utf-16"
            )
            with self.assertRaisesRegex(ValueError, "DTD"):
                read_reference_g_units(reference)
