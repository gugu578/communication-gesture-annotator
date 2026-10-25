# SPDX-License-Identifier: Apache-2.0
"""Read errors identify the artifact and cause without exposing its contents."""

from __future__ import annotations

import errno
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from communication_gesture_annotator.export.json_export import read_json_snapshot


class JsonReadErrorTests(unittest.TestCase):
    def test_os_failure_keeps_filename_errno_and_original_cause(self) -> None:
        path = Path("model-checkpoints/checkpoint-synthetic.json")
        for failure in (FileNotFoundError(errno.ENOENT, "Missing artifact"),
                        PermissionError(errno.EACCES, "Access denied"),
                        OSError(errno.ENOMEM, "Insufficient resources")):
            with self.subTest(failure=type(failure).__name__), \
                 patch.object(Path, "read_bytes", side_effect=failure):
                with self.assertRaises(ValueError) as result:
                    read_json_snapshot(path)
                message = str(result.exception)
                self.assertIn(str(path), message)
                self.assertIn(f"errno={failure.errno}", message)
                self.assertIn(type(failure).__name__, message)
                self.assertIn("file-read failure", message)
                self.assertIs(result.exception.__cause__, failure)

    def test_windows_error_number_is_not_discarded(self) -> None:
        failure = OSError(errno.ENOMEM, "Paging file is too small")
        failure.winerror = 1455
        with patch.object(Path, "read_bytes", side_effect=failure):
            with self.assertRaisesRegex(ValueError, "winerror=1455"):
                read_json_snapshot("replay-synthetic.json")

    def test_invalid_encoding_reports_offset_without_source_contents(self) -> None:
        private = b'{"private":"source material not for logs", "broken":"\xff"}'
        with patch.object(Path, "read_bytes", return_value=private):
            with self.assertRaises(ValueError) as result:
                read_json_snapshot("bad-encoding.json")
        message = str(result.exception)
        self.assertIn("bad-encoding.json", message)
        self.assertIn("invalid UTF-8", message)
        self.assertIn(f"byte_offset={private.index(bytes([255]))}", message)
        self.assertNotIn("source material", message)
        self.assertIsInstance(result.exception.__cause__, UnicodeDecodeError)

    def test_invalid_json_reports_position_without_source_contents(self) -> None:
        private = b'{\n"private":"source material not for logs",\n"unfinished":'
        with patch.object(Path, "read_bytes", return_value=private):
            with self.assertRaises(ValueError) as result:
                read_json_snapshot("bad-syntax.json")
        message = str(result.exception)
        self.assertIn("bad-syntax.json", message)
        self.assertIn("line=3", message)
        self.assertIn("column=", message)
        self.assertNotIn("source material", message)
        self.assertIsInstance(result.exception.__cause__, json.JSONDecodeError)

    def test_nonfinite_numbers_still_fail_with_filename(self) -> None:
        with patch.object(Path, "read_bytes", return_value=b'{"score":NaN}'):
            with self.assertRaisesRegex(ValueError, r"bad-number.json.*non-finite"):
                read_json_snapshot("bad-number.json")

    def test_valid_read_uses_one_snapshot_and_preserves_hash(self) -> None:
        data = b'{"score":0.5}\r\n'
        with patch.object(Path, "read_bytes", return_value=data) as read:
            result = read_json_snapshot("valid.json")
        read.assert_called_once()
        self.assertEqual(result, ({"score": 0.5}, sha256(data).hexdigest()))


if __name__ == "__main__":
    unittest.main()
