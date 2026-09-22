"""End-to-end v3 assessment against a scripted backend, no network.

Locks in the behaviors the v2 pipeline got wrong on real participants: an item
the interview never covered abstains rather than scoring 0, an abstention
reaches the total through the imputer, a fabricated quote is flagged, and the
ablation flags actually change the pipeline they claim to.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.imputation import fit_from_references  # noqa: E402
from psyvec.model.llm_backend import GenerationRequest  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_assessment", PROJECT_ROOT / "scripts" / "run_assessment.py"
)
assert _spec and _spec.loader
runner = importlib.util.module_from_spec(_spec)
# Register before exec: the module defines dataclasses, and @dataclass resolves
# annotations through sys.modules[cls.__module__].
sys.modules["run_assessment"] = runner
_spec.loader.exec_module(runner)

ITEM_KEYS = tuple(runner.TOPIC_TO_ITEM_KEY.values())

TRANSCRIPT = {
    "Participant_ID": "999",
    "phq8_scores": {"PHQ8_Score": 9, "items": dict.fromkeys(ITEM_KEYS, 1)},
    "real_interview": [
        {
            "roleName": "Ellie",
            "content": "easy_sleep (how easy is it to get a good night's sleep)",
        },
        {"roleName": "Participant", "content": "not easy i lay awake nearly"},
        {"roleName": "Participant", "content": "every night for the past two weeks"},
        {
            "roleName": "Ellie",
            "content": "feel_lately (how have you been feeling lately)",
        },
        {"roleName": "Participant", "content": "pretty low most days honestly"},
        {
            "roleName": "Ellie",
            "content": "Ellie17Dec2012_08 (what are you most proud of in your life)",
        },
        {"roleName": "Participant", "content": "my kids and my work honestly"},
        {"roleName": "Ellie", "content": "where are you from originally"},
        {"roleName": "Participant", "content": "born and raised in los angeles"},
    ],
}


def make_args(**overrides: object) -> argparse.Namespace:
    defaults = dict(
        no_imputation=False, imputation_model=None, calibrator=None, lessons=None,
        max_rounds=2, evidence_limit=8, no_turn_merge=False, no_polarity=False,
        no_store_evidence=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class ScriptedBackend:
    """Answers sufficiency / expansion / scoring calls by prompt shape."""

    model = "scripted"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, request: GenerationRequest) -> str:
        prompt = request.user
        self.prompts.append(prompt)
        if "Do these excerpts contain enough" in prompt:
            covered = "lay awake" in prompt or "pretty low" in prompt
            return json.dumps({"sufficient": covered, "missing": "any mention"})
        if "Propose 6 NEW short search phrases" in prompt:
            return json.dumps({"queries": ["zzzz nonexistent phrase"]})
        if "EVIDENCE: none" in prompt:
            return json.dumps(
                {"present": "unclear", "frequency_basis": "none", "sufficient": False,
                 "score": None, "quote": "", "turn_ids": [], "reasoning": "not raised"}
            )
        quote = (
            "lay awake nearly" if "lay awake" in prompt else "pretty low most days"
        )
        return json.dumps(
            {"present": "yes", "frequency_basis": "ongoing_state", "sufficient": True,
             "score": 2, "quote": quote, "turn_ids": ["turn-1"],
             "reasoning": "reported as an ongoing state"}
        )


class V3PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = PROJECT_ROOT / "tests" / "fixtures" / "_v3_tmp_999.json"
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self.tmp.write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
        self.imputer_path = self.tmp.parent / "_v3_tmp_imputer.json"
        self.imputer_path.write_text(
            fit_from_references(
                [dict.fromkeys(ITEM_KEYS, level) for level in (0, 1, 2, 3, 1)],
                ITEM_KEYS,
            ).to_json(),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.unlink(missing_ok=True)
        self.imputer_path.unlink(missing_ok=True)

    def _run(self, **overrides: object) -> dict:
        overrides.setdefault("imputation_model", self.imputer_path)
        config = runner.PipelineConfig(make_args(**overrides))
        return runner.assess_participant(ScriptedBackend(), self.tmp, config)

    def test_uncovered_items_abstain_instead_of_scoring_zero(self) -> None:
        record = self._run()
        self.assertGreater(record["abstained_item_count"], 0)
        for info in record["topics"].values():
            if info["abstained"]:
                self.assertIsNone(info["score"])

    def test_abstentions_reach_the_total_through_the_imputer(self) -> None:
        with_imputer = self._run()
        without = self._run(no_imputation=True)
        self.assertTrue(with_imputer["imputation_applied"])
        self.assertGreater(
            with_imputer["raw_total_score"], without["raw_total_score"]
        )

    def test_evidence_backed_items_are_scored_and_quote_grounded(self) -> None:
        record = self._run()
        scored = [i for i in record["topics"].values() if not i["abstained"]]
        self.assertTrue(scored)
        self.assertTrue(all(info["quote_grounded"] for info in scored))

    def test_a_quote_absent_from_the_evidence_is_flagged(self) -> None:
        """The anti-fabrication check must fire on an invented citation."""

        class FabricatingBackend(ScriptedBackend):
            def generate(self, request: GenerationRequest) -> str:
                if "Do these excerpts contain enough" in request.user:
                    return json.dumps({"sufficient": True, "missing": ""})
                return json.dumps(
                    {"present": "no", "frequency_basis": "none", "sufficient": True,
                     "score": 0, "turn_ids": [],
                     "quote": "i have never felt down a single day in my life",
                     "reasoning": "denied"}
                )

        config = runner.PipelineConfig(
            make_args(imputation_model=self.imputer_path)
        )
        record = runner.assess_participant(FabricatingBackend(), self.tmp, config)
        self.assertEqual(
            record["evidence_diagnostics"]["ungrounded_quotes"],
            len(record["topics"]),
        )

    def test_split_answers_are_merged_before_scoring(self) -> None:
        merged = self._run()
        split = self._run(no_turn_merge=True)
        sleep_merged = merged["topics"]["Sleep Problems"]["evidence_text"]
        sleep_split = split["topics"]["Sleep Problems"]["evidence_text"]
        self.assertIn("lay awake nearly every night", sleep_merged)
        self.assertNotIn("lay awake nearly every night", sleep_split)

    def test_reverse_polarity_is_labelled_and_the_ablation_removes_it(self) -> None:
        with_polarity = self._run()["topics"]["Low Self-Worth"]["evidence_text"]
        without = self._run(no_polarity=True)["topics"]["Low Self-Worth"][
            "evidence_text"
        ]
        self.assertIn("REVERSE POLARITY", with_polarity)
        self.assertNotIn("REVERSE POLARITY", without)

    def test_max_rounds_one_issues_no_expansion_queries(self) -> None:
        record = self._run(max_rounds=1)
        for info in record["topics"].values():
            self.assertEqual(info["rounds"], 1)
            self.assertEqual(info["queries_issued"], [])

    def test_record_carries_the_reference_and_a_category(self) -> None:
        record = self._run()
        self.assertEqual(record["ground_truth_total"], 9)
        self.assertIn("depression", record["predicted_category"].lower())

    def test_diagnostics_count_every_item_exactly_once(self) -> None:
        tiers = self._run()["evidence_diagnostics"]
        self.assertEqual(
            sum(tiers[name] for name in runner.TIER_NAMES), len(ITEM_KEYS)
        )


if __name__ == "__main__":
    unittest.main()
