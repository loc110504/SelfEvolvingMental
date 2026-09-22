"""Deterministic counterfactual band computation for the Evidence-Criterion Map.

``scripts/run_proposed_assessment.py::_counterfactual`` is read straight off
the fixed PHQ-8 rubric's day-count bands — no model call — so this checks the
boundary arithmetic rather than anything that could hallucinate.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "run_proposed_assessment", PROJECT_ROOT / "scripts" / "run_proposed_assessment.py"
)
assert _spec and _spec.loader
runner = importlib.util.module_from_spec(_spec)
sys.modules["run_proposed_assessment"] = runner
_spec.loader.exec_module(runner)


class CounterfactualTest(unittest.TestCase):
    def test_abstained_item_names_what_evidence_would_resolve_it(self) -> None:
        text = runner._counterfactual(None, False)
        self.assertIn("no evidence was found", text)

    def test_score_zero_only_offers_an_upward_move(self) -> None:
        text = runner._counterfactual(0, True)
        self.assertIn("raise the score to 1", text)
        self.assertNotIn("lower the score", text)

    def test_score_three_only_offers_a_downward_move(self) -> None:
        text = runner._counterfactual(3, True)
        self.assertIn("lower the score to 2", text)
        self.assertNotIn("raise the score", text)

    def test_a_middle_score_offers_both_directions(self) -> None:
        text = runner._counterfactual(2, True)
        self.assertIn("raise the score to 3", text)
        self.assertIn("lower the score to 1", text)

    def test_the_named_band_matches_the_rubric(self) -> None:
        # Score 1 -> 2 requires "7-11 days" per configs/scales/scoring_standards.json.
        text = runner._counterfactual(1, True)
        self.assertIn("7-11 days", text)


class CriterionMapShapeTest(unittest.TestCase):
    def test_one_row_per_assessed_topic(self) -> None:
        from psyvec.memory.patient_evidence import PatientEvidenceMemory

        assessed = {
            "Sleep Problems": {
                "score": 2, "abstained": False, "present": "yes", "quote": "x",
            },
            "Depressed Mood": {
                "score": None, "abstained": True, "present": "unclear", "quote": "",
            },
        }
        memory = PatientEvidenceMemory(participant_id="1")
        rows = runner._build_criterion_map(assessed, memory)
        statuses = {row["criterion"]: row["status"] for row in rows}
        self.assertEqual(statuses["Sleep Problems"], "supported")
        self.assertEqual(statuses["Depressed Mood"], "unresolved")


if __name__ == "__main__":
    unittest.main()
