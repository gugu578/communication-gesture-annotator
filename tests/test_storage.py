# SPDX-License-Identifier: Apache-2.0
"""Compact storage preserves measurements and validated request reuse."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from communication_gesture_annotator.cli import main, _parser
from communication_gesture_annotator.config import ResearchConfig, OpenWebUIConfig, StorageConfig, load_research_config
from communication_gesture_annotator.pipeline.llm_contracts import ModelCallResult
from communication_gesture_annotator.pipeline.llm_runner import _CheckpointingPhase2Model
from communication_gesture_annotator.storage import (
    current_mode,
    data_path,
    open_data,
    output_data_path,
    release_image,
    storage_mode,
)


class StorageTests(unittest.TestCase):
    def test_storage_is_operational_and_cli_defaults_are_minimal_without_audio(self):
        config = ResearchConfig(open_webui=OpenWebUIConfig(model="synthetic"))
        self.assertEqual(config.storage.mode, "minimal")
        debug = replace(config, storage=StorageConfig(mode="debug"))
        self.assertEqual(config.model_input_dict(), debug.model_input_dict())
        self.assertEqual(config.detection_dict(), debug.detection_dict())
        args = [
            "run-llm",
            "synthetic.mp4",
            "--phase2-config",
            "unused.toml",
            "--run-dir",
            "unused-run",
        ]
        self.assertTrue(_parser().parse_args(args).no_audio)
        seen = []

        def run(args):
            seen.append(current_mode())
            return 0

        with (
            patch("communication_gesture_annotator.cli.load_research_config", return_value=config),
            patch("communication_gesture_annotator.cli._run_command", side_effect=run),
        ):
            self.assertEqual(main(args), 0)
            self.assertEqual(main([*args, "--storage-mode", "compact"]), 0)
            self.assertEqual(main([*args, "--storage-mode", "debug"]), 0)
            self.assertEqual(main([*args, "--storage-mode", "minimal"]), 0)
        self.assertEqual(seen, ["minimal", "compact", "debug", "minimal"])

    def test_missing_storage_table_and_explicit_toml_with_cli_override(self) -> None:
        connection = '[open_webui]\nbase_url = "http://127.0.0.1:3000"\nmodel = "synthetic"\n'
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(connection, encoding="utf8")
            self.assertEqual(load_research_config(path).storage.mode, "minimal")
            argv = ["run-llm", "synthetic.mp4", "--phase2-config", str(path), "--run-dir", str(Path(directory)/"run")]
            seen: list[str] = []

            def run(args: object) -> int:
                seen.append(current_mode())
                return 0

            with patch("communication_gesture_annotator.cli._run_command", side_effect=run):
                self.assertEqual(main(argv), 0)
                for mode in ("compact", "debug", "minimal"):
                    path.write_text(connection + f'\n[storage]\nmode = "{mode}"\n', encoding="utf8")
                    self.assertEqual(main(argv), 0)
                    self.assertEqual(main([*argv, "--storage-mode", "minimal"]), 0)
            self.assertEqual(seen, ["minimal", "compact", "minimal", "debug", "minimal", "minimal", "minimal"])

    def test_compression_is_lossless_deterministic_and_unambiguous(self):
        content = '{"motion":0.12345678901234567,"time_us":1234567}\n' * 50
        with TemporaryDirectory() as directory, storage_mode("compact"):
            root = Path(directory)
            paths = [output_data_path(root / name) for name in ("one.jsonl", "two.jsonl")]
            for path in paths:
                with open_data(path, "w") as handle:
                    handle.write(content)
                with open_data(path) as handle:
                    self.assertEqual(handle.read(), content)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertEqual(data_path(root / "one.jsonl"), paths[0])
            (root / "one.jsonl").write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                data_path(root / "one.jsonl")

    def test_cleanup_removes_only_owned_regenerable_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "sheets"
            cache.mkdir()
            image = cache / ("a" * 64 + ".png")
            for p in (
                image,
                image.with_suffix(".json"),
                image.with_name(image.name + ".rights.json"),
            ):
                p.write_text("temporary", encoding="utf-8")
            keep = cache / "keep.txt"
            keep.write_text("keep", encoding="utf-8")
            release_image(image, cache)
            self.assertEqual(list(cache.iterdir()), [keep])
            with self.assertRaises(ValueError):
                release_image(root / image.name, cache)

    def test_checkpoint_order_is_stable_and_changed_inputs_or_results_are_rejected(self):
        class Model:
            def complete_json(self, **kwargs):
                return ModelCallResult(
                    payload={"z": 1, "a": {"y": 2, "b": 3}},
                    model_payload={"z": 1, "a": {"y": 2, "b": 3}},
                    request_id=kwargs["request_id"],
                    response_sha256="a" * 64,
                    latency_seconds=0.01,
                    endpoint_mode="synthetic",
                )

        request = dict(
            request_id="test",
            task="synthetic",
            system_prompt="instructions",
            user_prompt="measurements",
            images=[],
            response_schema={"type": "object"},
            trace_context={},
        )
        with TemporaryDirectory() as directory, storage_mode("compact"):
            root = Path(directory)
            live = _CheckpointingPhase2Model(
                Model(), run_root=root, scope={}, resume=False, trace=None
            )
            original = live.complete_json(**request)
            replay = _CheckpointingPhase2Model(
                None, run_root=root, scope={}, resume=True, trace=None
            )
            resumed = replay.complete_json(**request)
            self.assertEqual(json.dumps(original.payload), json.dumps(resumed.payload))
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                replay.complete_json(**{**request, "user_prompt": "changed"})
            path = next((root / "model-checkpoints").glob("*.json"))
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["call"]["payload"]["z"] = 42
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                replay.complete_json(**request)
