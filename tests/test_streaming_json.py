# SPDX-License-Identifier: Apache-2.0
"""Large artifact writes keep compatible bytes without materializing JSON text."""

from dataclasses import dataclass
from enum import StrEnum
import errno
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from communication_gesture_annotator.domain import to_json_value
from communication_gesture_annotator.export.json_export import read_json, write_json


class Kind(StrEnum):
    ACTIVE = "active"


@dataclass
class Evidence:
    kind: Kind
    scores: tuple[float, ...]


class StreamingJsonTests(unittest.TestCase):
    def test_bytes_match_previous_writer_and_report_order(self) -> None:
        payload = {"z": [Evidence(Kind.ACTIVE, (0.1, 0.9)), None, True],
                   "a": {2: "Ľuboš\nquoted: \"yes\"", False: []}}
        for sort_keys in (True, False):
            with self.subTest(sort_keys=sort_keys), TemporaryDirectory() as directory:
                root = Path(directory)
                expected = root / "previous.json"
                expected.write_text(json.dumps(to_json_value(payload), indent=2,
                    sort_keys=sort_keys, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
                actual = write_json(root / "current.json", payload, sort_keys=sort_keys)
                self.assertEqual(actual.read_bytes(), expected.read_bytes())
                self.assertEqual(read_json(actual), to_json_value(payload))

    def test_large_artifact_never_uses_whole_document_encoder(self) -> None:
        payload = {"samples": [{"i": i, "motion": [0.1] * 12} for i in range(5000)]}
        with TemporaryDirectory() as directory:
            target = Path(directory) / "phase2-evidence.json"
            with patch.object(json, "dumps", side_effect=AssertionError("whole JSON string")), \
                 patch.object(json.JSONEncoder, "encode", side_effect=MemoryError("whole JSON")):
                write_json(target, payload)
            self.assertEqual(read_json(target), payload)

    def test_invalid_data_preserves_existing_result_and_cleans_temporary_file(self) -> None:
        for invalid in (float("nan"), float("inf"), object()):
            for existing in (True, False):
                with self.subTest(invalid=invalid, existing=existing), TemporaryDirectory() as directory:
                    root = Path(directory)
                    target = root / "result.json"
                    if existing:
                        target.write_bytes(b"previous valid artifact\n")
                    with self.assertRaises((TypeError, ValueError)):
                        write_json(target, {"rows": [1, 2, invalid]})
                    self.assertEqual(target.exists(), existing)
                    if existing:
                        self.assertEqual(target.read_bytes(), b"previous valid artifact\n")
                    self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_interrupted_serialization_preserves_previous_result(self) -> None:
        def fail_after_first_chunk(encoder, value, *args, **kwargs):
            yield "{\n"
            raise MemoryError("synthetic interrupted encoding")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "result.json"
            target.write_bytes(b"previous valid artifact\n")
            with patch.object(json.JSONEncoder, "iterencode", fail_after_first_chunk):
                with self.assertRaises(MemoryError):
                    write_json(target, {"samples": []})
            self.assertEqual(target.read_bytes(), b"previous valid artifact\n")
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_disk_full_during_write_keeps_previous_result(self) -> None:
        real_open = Path.open

        class FullDisk:
            def __init__(self, path: Path, *args, **kwargs) -> None:
                self.handle = real_open(path, *args, **kwargs)
                self.writes = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.handle.close()

            def write(self, text: str) -> int:
                self.writes += 1
                if self.writes > 3:
                    raise OSError(errno.ENOSPC, "synthetic full disk")
                return self.handle.write(text)

        def open_with_full_disk(path, *args, **kwargs):
            return FullDisk(path, *args, **kwargs) if path.suffix == ".tmp" else real_open(path, *args, **kwargs)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "result.json"
            target.write_bytes(b"previous valid artifact\n")
            with patch.object(Path, "open", open_with_full_disk):
                with self.assertRaises(OSError) as failure:
                    write_json(target, {"rows": list(range(20))})
            self.assertEqual(failure.exception.errno, errno.ENOSPC)
            self.assertEqual(target.read_bytes(), b"previous valid artifact\n")
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
