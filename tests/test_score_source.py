# SPDX-License-Identifier: Apache-2.0
"""LLM default and optional motion fusion preserve model inputs."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from communication_gesture_annotator.config import DetectionConfig, load_research_config
from communication_gesture_annotator.pipeline.probability_pass import (
    SCORE_SELECTION_VERSION, candidate_policy, combine_probability, probability_record,
)


class ScoreSourceTests(unittest.TestCase):
    def test_default_uses_llm_exactly_while_retaining_motion(self) -> None:
        config = DetectionConfig()
        self.assertEqual(config.probability_score_source, "llm")
        for llm in (0., .2, .5, 1.):
            for motion in (None, 0., .3, .95, 1.):
                with self.subTest(llm=llm, motion=motion):
                    record = probability_record({"motion_non_g_probability": motion}, llm, config)
                    self.assertEqual(record["non_g_unit_probability"], llm)
                    self.assertEqual(record["communicative_probability"], 1-llm)
                    self.assertEqual(record["motion_non_g_probability"], motion)
                    self.assertEqual(record["score_source"], "llm")
                    self.assertEqual(record["score_selection_policy_version"], SCORE_SELECTION_VERSION)

    def test_optional_fusion_combines_scores_and_falls_back_when_motion_is_missing(self) -> None:
        config = DetectionConfig(probability_score_source="motion_fusion")
        self.assertEqual(config.probability_score_source, "motion_fusion")
        for llm, motion in ((.25, .95), (.75, .05), (.5, .8), (.3, None)):
            record = probability_record({"motion_non_g_probability": motion}, llm, config)
            self.assertEqual(record["non_g_unit_probability"], combine_probability(motion, llm, 4))
        self.assertGreater(probability_record({"motion_non_g_probability": .95}, .25, config)["non_g_unit_probability"], .5)

    def test_local_choice_is_serialized_but_does_not_change_model_identity_or_grouping(self) -> None:
        config = DetectionConfig(probability_score_source="llm")
        fusion = replace(config, probability_score_source="motion_fusion")
        self.assertNotEqual(config.to_dict(), fusion.to_dict())
        self.assertEqual(config.model_input_dict(), fusion.model_input_dict())
        self.assertEqual(candidate_policy(config), candidate_policy(fusion))
        for invalid in (None, True, 1, "motion", "", [], {}):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                replace(config, probability_score_source=invalid)

    def test_toml_omission_and_explicit_selection(self) -> None:
        base = '[open_webui]\nbase_url = "http://127.0.0.1:3000"\nmodel = "test"\n'
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            for selection in (None, "llm", "motion_fusion"):
                with self.subTest(selection=selection):
                    text = base
                    if selection is not None:
                        text += f'\n[detection]\nprobability_score_source = "{selection}"\n'
                    path.write_text(text, encoding="utf-8")
                    self.assertEqual(load_research_config(path).detection.probability_score_source,
                                     selection or "llm")


if __name__ == "__main__":
    unittest.main()
