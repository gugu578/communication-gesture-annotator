# SPDX-License-Identifier: Apache-2.0
"""Cross-stage regression cases for output-only probability policies."""
import unittest

from communication_gesture_annotator.config import AnalysisConfig
from communication_gesture_annotator.domain import AnalysisSample, JointActivityEvidence, JointActivityState, to_json_value
from communication_gesture_annotator.pipeline.probability_pass import candidate_diagnostics, VERSION
from communication_gesture_annotator.pipeline.probability_finalization import (
    POLICY_VERSION, MERGE_POLICY_VERSION, finalize_probability_candidates, finalization_metadata,
)
from communication_gesture_annotator.rights import unrestricted_or_unspecified_rights


def fixture(values, *, excluded=(), barrier=(), policy_version=POLICY_VERSION, gap=300):
    config = AnalysisConfig(maximum_bridged_missing_samples=2)
    samples = [AnalysisSample(i, i*100000, (i+1)*100000, i*100000+50000) for i in range(len(values))]
    evidence = []
    for i, q in enumerate(values):
        state = "unknown" if q is None else "passive" if q >= .5 else "active"
        records = [{"exclude_from_detection": True}] if i in excluded else []
        if q is not None:
            records.append({"policy": VERSION, "motion_grounding": {
                "policy_version": VERSION, "llm_non_g_unit_probability": q,
                "motion_non_g_probability": None, "non_g_unit_probability": q,
                "candidate_policy": {"rest_threshold": .5, "rest_seed_threshold": .65, "minimum_departure_us": 300000},
            }})
        evidence.append(JointActivityEvidence(state=JointActivityState(state),
            state_probabilities={k: float(k == state) for k in ("passive", "active", "unknown")},
            evidence_quality=0. if q is None else 1., reason="missing_evidence" if q is None else None,
            continuity_barrier=i in barrier, resolution_records=tuple(records)))
    diagnostics = candidate_diagnostics(samples, evidence, config, participant_id="subject-0001",
        candidate_policy_version="contextual-score-candidates-v0.3" if policy_version == POLICY_VERSION else "contextual-score-candidates-v0.2")
    candidates = diagnostics.candidates if diagnostics is not None else ()
    decisions, units = finalize_probability_candidates(samples, evidence, candidates, config,
        participant_id="subject-0001", policy_version=policy_version, merge_gap_ms=gap)
    return samples, evidence, diagnostics, decisions, units


def document(values, **kwargs):
    samples, evidence, diagnostics, decisions, units = fixture(values, **kwargs)
    version = kwargs.get("policy_version", POLICY_VERSION)
    config = AnalysisConfig(maximum_bridged_missing_samples=2)
    config = {k: getattr(config, k) for k in ("sample_rate_hz", "minimum_separating_rest_us",
        "maximum_bridged_missing_samples", "maximum_proxy_width", "eaf_tier_id")}
    return {"schema_version": "0.2.0", "run_id": "synthetic", "created_at": "2026-01-01T00:00:00Z",
        "source": {"path": "synthetic.mp4", "duration_us": len(samples)*100000, "sha256": "a"*64,
                   "rights": unrestricted_or_unspecified_rights()},
        "preprocessing": {}, "config": config, "participant_profiles": [{"participant_id": "subject-0001"}],
        "samples": to_json_value(decisions), "activity_candidates": to_json_value(diagnostics.candidates if diagnostics else ()),
        "g_units": to_json_value(units), "processing_windows": [], "window_responses": [],
        "candidate_assessments": [], "prompt_envelopes": [], "warnings": [], "evaluation": None,
        "provenance": {"software": "communication-gesture-annotator", "policy_version": version,
            "phase3_finalization": finalization_metadata(version, merge_gap_ms=kwargs.get("gap", 300)),
            "research_configuration": {"analysis": config, "inference": {"phase1_enabled": True}}}}


class OutputGuardsTest(unittest.TestCase):
    def test_exclusions_and_barriers_survive_final_merging(self):
        for kwargs in ({"excluded": (4, 5)}, {"barrier": (4,)}):
            with self.subTest(kwargs=kwargs):
                _, _, _, decisions, units = fixture([.1]*4+[None]*2+[.1]*4, **kwargs)
                self.assertEqual([(u.start_sample,u.end_sample_exclusive) for u in units], [(0,4),(6,10)])
                self.assertFalse(any(d.is_g_unit for d in decisions[4:6]))
        # Historical v0.3 is still reconstructible, including its old defect.
        _, _, _, decisions, units = fixture([.1]*4+[None]*2+[.1]*4,
            excluded=(4,5), policy_version=MERGE_POLICY_VERSION)
        self.assertEqual(len(units), 1)
        self.assertTrue(all(d.is_g_unit for d in decisions[4:6]))

    def test_uncertainty_cannot_bootstrap_a_phrase(self):
        for q in (.5, .6, .64, .65):
            with self.subTest(q=q):
                _, _, _, decisions, units = fixture([q]*100)
                self.assertFalse(any(d.is_g_unit for d in decisions))
                self.assertEqual(units, [])
        _, _, _, _, old = fixture([.6]*100, policy_version=MERGE_POLICY_VERSION)
        self.assertEqual(len(old), 1)

    def test_supported_hold_and_clip_fragment_are_preserved(self):
        _, _, _, decisions, units = fixture([.6]*5+[.1]*3+[.6]*20+[.1]*3+[.9]*5)
        self.assertEqual([(u.start_sample,u.end_sample_exclusive) for u in units], [(5,31)])
        self.assertEqual(decisions[8].raw_decision, "not_g_unit")
        self.assertIn("no_sustained_rest_seed", decisions[8].policy_overrides)
        for values in ([.1]*4+[.9]*3, [.1]*8, [.9]*3+[.1]*4):
            self.assertEqual(len(fixture(values)[-1]), 1)

    def test_short_missingness_is_preserved_but_long_gap_never_remerged(self):
        self.assertEqual(len(fixture([.1]*4+[None]*2+[.1]*4)[-1]), 1)
        for gap in (300, 1000):
            self.assertEqual(len(fixture([.1]*4+[None]*3+[.1]*4, gap=gap)[-1]), 2)
        self.assertEqual(len(fixture([.1]*4+[.9]*3+[.1]*4, gap=300)[-1]), 1)
        self.assertEqual(len(fixture([.1]*4+[.9]*3+[.1]*4, gap=299)[-1]), 2)
        self.assertEqual(len(fixture([.1]*4+[.9]*3+[.1]*4, gap=0)[-1]), 2)

    def test_new_canonical_validator_rejects_excluded_positive(self):
        from communication_gesture_annotator.export.json_export import validate_analysis_document
        doc = document([.1]*4+[None]*2+[.1]*4, excluded=(4,5), policy_version=MERGE_POLICY_VERSION)
        validate_analysis_document(doc)
        doc["provenance"]["policy_version"] = POLICY_VERSION
        with self.assertRaisesRegex(ValueError, "excluded|barrier"):
            validate_analysis_document(doc)


if __name__ == "__main__":
    unittest.main()
