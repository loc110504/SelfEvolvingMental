"""End-to-end proposed-method assessment against a scripted backend, no network.

Locks in the behaviors that are new relative to ``run_assessment.py``: a
contradiction between an early denial and a later disclosure survives into
the Patient Evidence Memory instead of being smoothed into one paraphrase,
the scheduler spends its shared expansion-round budget across topics rather
than exhausting it on one, abstention still never becomes a forced zero, and
the Evidence-Criterion Map is present and internally consistent with the
score it explains.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.imputation import fit_from_references  # noqa: E402
from psyvec.model.llm_backend import GenerationRequest  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_proposed_assessment", PROJECT_ROOT / "scripts" / "run_proposed_assessment.py"
)
assert _spec and _spec.loader
runner = importlib.util.module_from_spec(_spec)
sys.modules["run_proposed_assessment"] = runner
_spec.loader.exec_module(runner)

ITEM_KEYS = tuple(runner.TOPIC_TO_ITEM_KEY.values())
_TURN_ID_RE = re.compile(r"turn-\d+")

#: A tag-anchored seed pass (per configs/scales/topic_evidence_map.json,
#: "easy_sleep" and "sleep_affects" both anchor to Sleep Problems) retrieves
#: BOTH turns together in one round, so the contradiction is present from
#: round 0 with no dependency on the scheduler ever expanding this topic.
TRANSCRIPT = {
    "Participant_ID": "998",
    "phq8_scores": {"PHQ8_Score": 10, "items": dict.fromkeys(ITEM_KEYS, 1)},
    "real_interview": [
        {
            "roleName": "Ellie",
            "content": "easy_sleep (how easy is it to get a good night's sleep)",
        },
        {"roleName": "Participant", "content": "i sleep just fine most nights"},
        {"roleName": "Ellie", "content": "how have you been feeling lately"},
        {
            "roleName": "Participant",
            "content": "pretty good honestly nothing much going on",
        },
        {
            "roleName": "Ellie",
            "content": "sleep_affects (what are you like when you dont sleep well)",
        },
        {
            "roleName": "Participant",
            "content": (
                "actually i wake up around three am most nights now and "
                "cant get back to sleep"
            ),
        },
        {"roleName": "Ellie", "content": "where are you from originally"},
        {"roleName": "Participant", "content": "born and raised in los angeles"},
    ],
}

_DENIAL_QUOTE = "i sleep just fine most nights"
_DISCLOSURE_QUOTE = (
    "actually i wake up around three am most nights now and cant get back to sleep"
)


def make_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = dict(
        imputation_model=None, calibrator=None, lessons=None, interview_lessons=None,
        global_budget=5, max_rounds_per_topic=3, evidence_limit=8,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _turn_ids_before(prompt: str, quote: str) -> list[str]:
    """The bracketed ``[turn-N]`` id(s) immediately preceding ``quote`` in a
    ``format_tiered_evidence``-rendered excerpt block."""
    prefix = prompt.split(quote)[0][-40:]
    return _TURN_ID_RE.findall(prefix + quote)


class ScriptedBackend:
    """Answers extraction / sufficiency / expansion / scoring calls by prompt shape."""

    model = "scripted"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, request: GenerationRequest) -> str:
        prompt = request.user
        self.prompts.append(prompt)
        if "List every distinct claim" in prompt:
            return self._extraction(prompt)
        if "Do these excerpts contain enough" in prompt:
            return self._sufficiency(prompt)
        if "Propose 6 NEW short search phrases" in prompt:
            return self._expansion(prompt)
        return self._scorer(prompt)

    def _extraction(self, prompt: str) -> str:
        claims = []
        if _DENIAL_QUOTE in prompt:
            claims.append({
                "turn_ids": _turn_ids_before(prompt, _DENIAL_QUOTE),
                "quote": _DENIAL_QUOTE, "polarity": "refutes",
                "frequency_basis": "none", "confidence": 0.8,
            })
        if _DISCLOSURE_QUOTE in prompt:
            claims.append({
                "turn_ids": _turn_ids_before(prompt, _DISCLOSURE_QUOTE),
                "quote": _DISCLOSURE_QUOTE, "polarity": "supports",
                "frequency_basis": "ongoing_state", "confidence": 0.9,
            })
        return json.dumps({"claims": claims})

    def _sufficiency(self, prompt: str) -> str:
        if "Sleep Problems" not in prompt:
            return json.dumps({"sufficient": False, "missing": "any mention"})
        return json.dumps({"sufficient": True, "missing": ""})

    def _expansion(self, prompt: str) -> str:
        return json.dumps({"queries": []})

    def _scorer(self, prompt: str) -> str:
        if _DISCLOSURE_QUOTE in prompt:
            return json.dumps({
                "present": "yes", "frequency_basis": "ongoing_state",
                "sufficient": True, "score": 2, "quote": _DISCLOSURE_QUOTE,
                "turn_ids": _TURN_ID_RE.findall(prompt),
                "reasoning": "reported as an ongoing state",
            })
        return json.dumps({
            "present": "unclear", "frequency_basis": "none", "sufficient": False,
            "score": None, "quote": "", "turn_ids": [], "reasoning": "not raised",
        })


class ProposedPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = PROJECT_ROOT / "tests" / "fixtures" / "_proposed_tmp_998.json"
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self.tmp.write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
        self.imputer_path = self.tmp.parent / "_proposed_tmp_imputer.json"
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

    def test_contradictory_evidence_is_preserved_not_merged(self) -> None:
        record = self._run()
        contradictions = record["evidence_memory"]["contradictions"]
        sleep_edges = [c for c in contradictions if c["topic"] == "Sleep Problems"]
        self.assertEqual(len(sleep_edges), 1)
        all_records = record["evidence_memory"]["records"]
        records_by_id = {r["record_id"]: r for r in all_records}
        quotes = {
            records_by_id[sleep_edges[0]["record_id_a"]]["quote"],
            records_by_id[sleep_edges[0]["record_id_b"]]["quote"],
        }
        self.assertEqual(quotes, {_DENIAL_QUOTE, _DISCLOSURE_QUOTE})

    def test_neither_conflicting_record_is_dropped(self) -> None:
        record = self._run()
        quotes = {r["quote"] for r in record["evidence_memory"]["records"]}
        self.assertIn(_DENIAL_QUOTE, quotes)
        self.assertIn(_DISCLOSURE_QUOTE, quotes)

    def test_criterion_map_flags_the_contradiction_and_matches_the_score(self) -> None:
        record = self._run()
        rows = {row["criterion"]: row for row in record["criterion_map"]}
        sleep_row = rows["Sleep Problems"]
        self.assertTrue(sleep_row["contradiction"])
        self.assertEqual(sleep_row["status"], "supported")
        self.assertEqual(sleep_row["score"], 2)
        self.assertTrue(sleep_row["counterfactual"])
        self.assertEqual(len(record["criterion_map"]), 8)

    def test_a_resolved_topic_needs_no_expansion_rounds(self) -> None:
        record = self._run()
        self.assertEqual(record["topics"]["Sleep Problems"]["acquisition_trace"], [])
        self.assertFalse(record["topics"]["Sleep Problems"]["abstained"])

    def test_the_global_budget_is_spent_exactly_once_each(self) -> None:
        """7 topics tie on need with Sleep Problems already resolved: the
        scheduler round-robins the shared budget across them one at a time."""
        record = self._run(global_budget=5)
        total_rounds = sum(
            len(info["acquisition_trace"]) for info in record["topics"].values()
        )
        self.assertEqual(total_rounds, 5)

    def test_a_zero_budget_still_produces_a_valid_record(self) -> None:
        record = self._run(global_budget=0)
        self.assertTrue(record["assessment_valid"])
        self.assertEqual(
            sum(len(info["acquisition_trace"]) for info in record["topics"].values()), 0
        )

    def test_uncovered_topics_abstain_instead_of_scoring_zero(self) -> None:
        record = self._run()
        others = [
            info for topic, info in record["topics"].items()
            if topic != "Sleep Problems"
        ]
        self.assertTrue(any(info["abstained"] for info in others))
        for info in others:
            if info["abstained"]:
                self.assertIsNone(info["score"])

    def test_abstentions_reach_the_total_through_the_imputer(self) -> None:
        with_imputer = self._run()
        without = self._run(imputation_model=None)
        self.assertTrue(with_imputer["imputation_applied"])
        self.assertGreaterEqual(
            with_imputer["raw_total_score"], without["raw_total_score"]
        )

    def test_record_carries_the_reference_and_a_category(self) -> None:
        record = self._run()
        self.assertEqual(record["ground_truth_total"], 10)
        self.assertIn("depression", record["predicted_category"].lower())

    def test_pipeline_name_identifies_the_proposed_method(self) -> None:
        record = self._run()
        self.assertEqual(
            record["pipeline"], "psyvec-proposed-evidence-utility-adaptive"
        )


if __name__ == "__main__":
    unittest.main()
