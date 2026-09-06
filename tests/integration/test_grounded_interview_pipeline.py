"""End-to-end wiring check for the Grounded Interview Loop (Plan_Improve.md).

``scripts/run_qwen_sample.py`` hard-imports ``torch``/``transformers`` at
module level (it is a script meant to run where those are installed), which
this repo's test environment does not have. Rather than skip verifying the
rewritten pipeline entirely, this test stubs those two modules just enough
to import the script, then drives ``run_full_sample_assessment`` with a
scripted fake engine (never a real model) against the real golden-case
sample (participant 404: GT PHQ-8 = 0 on every item — see
Plan_Improve.md Sec 0/6) to prove the retrieval -> grounded interview ->
scorer -> updater wiring executes end to end and produces the expected
result schema, including the new evidence/faithfulness diagnostics.

This does not (and cannot, without a real model) assert anything about
generation quality — that is what a real batch run on ``train``/``dev``
(Plan_Improve.md Sec 6 Phase 1 steps 6-7) is for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_qwen_sample.py"
GOLDEN_SAMPLE = PROJECT_ROOT / "data" / "processed_daic_woz" / "dev" / "404.json"


def _install_stub_module(name: str, attrs: dict[str, Any]) -> None:
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(stub, attr_name, value)
    sys.modules[name] = stub


def _load_run_qwen_sample() -> Any:
    """Import scripts/run_qwen_sample.py with torch/transformers stubbed out.

    Only the names ``run_qwen_sample.py`` touches at *module import time*
    need to exist: ``torch.backends.mps.is_available`` (evaluated inside
    ``QwenInferenceEngine.__init__``, never called here because tests pass
    their own fake engine) is not needed at import time, only the two
    top-level names ``AutoModelForCausalLM``/``AutoTokenizer`` from
    ``transformers`` and the ``torch`` module object itself.
    """
    mps = types.SimpleNamespace(is_available=lambda: False)
    backends = types.SimpleNamespace(mps=mps)
    _install_stub_module(
        "torch",
        {"backends": backends, "float32": None, "bfloat16": None, "no_grad": None},
    )
    _install_stub_module(
        "transformers", {"AutoModelForCausalLM": object, "AutoTokenizer": object}
    )

    spec = importlib.util.spec_from_file_location(
        "run_qwen_sample_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEngine:
    """Deterministic stand-in for QwenInferenceEngine — never loads a model.

    Always reports "no further questioning needed" so each topic's interview
    loop runs exactly one round, keeping the test fast; always assigns a
    fixed valid score so the run completes with ``assessment_valid=True``.
    """

    def __init__(
        self,
        memory_update_score: int | None = None,
        memory_update_turn_id: str = "",
    ) -> None:
        self.truncation_count = 0
        self.calls: list[str] = []
        self._memory_update_score = memory_update_score
        self._memory_update_turn_id = memory_update_turn_id

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2560,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append(system_prompt[:40])
        if "clinical assessment evaluator" in system_prompt:
            return "0"  # necessity: always sufficient after one round
        if "psychological scale scorer" in system_prompt:
            return (
                '{"score": 1, "summary": "test scorer reply", '
                '"evidence_turn_ids": []}'
            )
        if "diagnostic updater" in system_prompt:
            if self._memory_update_score is None:
                return '{"summary": "no change needed", "updated_scores": {}}'
            payload = {
                "summary": "adjusted",
                "updated_scores": {
                    "Loss of Interest": {
                        "score": self._memory_update_score,
                        "reason": "test",
                        "evidence_turn_ids": [self._memory_update_turn_id],
                    }
                },
            }
            return json.dumps(payload)
        return "I am doing okay, thank you for asking."


class GroundedInterviewPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        if not GOLDEN_SAMPLE.is_file():
            self.skipTest(f"golden fixture not present: {GOLDEN_SAMPLE}")
        self.module = _load_run_qwen_sample()

    def test_full_pipeline_runs_and_reports_evidence_diagnostics(self) -> None:
        sample = json.loads(GOLDEN_SAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(sample["phq8_scores"]["PHQ8_Score"], 0)

        result = self.module.run_full_sample_assessment(
            sample_path=GOLDEN_SAMPLE,
            engine=FakeEngine(),
            verbose=False,
        )

        self.assertTrue(result["assessment_valid"])
        # FakeEngine always scores 1, but grounded_score() now resets any
        # nonzero score to 0 for topics with no topic-specific evidence
        # (Plan_Improve.md Sec 5.1 enforcement) — only "known"-status topics
        # keep their score.
        expected_total = result["evidence_diagnostics"]["known"]
        self.assertEqual(result["total_predicted_score"], expected_total)
        self.assertEqual(result["evidence_diagnostics"]["grounding_overrides"], 8 - expected_total)
        self.assertEqual(result["ground_truth_total"], 0)

        topics = result["topics"]
        self.assertEqual(len(topics), 8)
        self.assertEqual(topics["Sleep Problems"]["evidence_status"], "known")
        self.assertIn(topics["Sleep Problems"]["evidence_status"], ("known", "missing"))

        diagnostics = result["evidence_diagnostics"]
        self.assertEqual(diagnostics["known"] + diagnostics["missing"], 8)
        # Participant 404's Sleep Problems evidence is real (see
        # test_evidence_retrieval.py's golden case) — this pipeline-level
        # check only confirms that status made it through to the result.
        self.assertGreaterEqual(diagnostics["known"], 1)

    def test_memory_update_rejects_revision_without_evidence(self) -> None:
        result = self.module.run_full_sample_assessment(
            sample_path=GOLDEN_SAMPLE,
            engine=FakeEngine(),
            enable_memory_update=True,
            verbose=False,
        )
        # FakeEngine's default updater reply has an empty updated_scores, so
        # no topic should have been revised — nothing to reject either.
        self.assertEqual(result["updated_scores"], {})
        for info in result["topics"].values():
            self.assertNotIn("updated_score", info)

    def test_memory_update_applies_a_properly_cited_revision(self) -> None:
        from psyvec.evaluation.evidence_retrieval import (
            load_keyword_lexicon,
            load_tag_map,
            retrieve_evidence,
        )

        sample = json.loads(GOLDEN_SAMPLE.read_text(encoding="utf-8"))
        scale_file = PROJECT_ROOT / "configs" / "scales" / "PHQ-8.json"
        topics = list(json.loads(scale_file.read_text(encoding="utf-8")).keys())
        tag_map = load_tag_map(
            PROJECT_ROOT / "configs" / "scales" / "topic_tag_map.json",
            valid_topics=topics,
        )
        lexicon = load_keyword_lexicon(
            PROJECT_ROOT / "configs" / "scales" / "topic_keywords.json",
            valid_topics=topics,
        )
        bundles = retrieve_evidence(sample["real_interview"], topics, tag_map, lexicon)
        loss_of_interest_bundle = bundles["Loss of Interest"]
        self.assertEqual(
            loss_of_interest_bundle.status,
            "known",
            "test fixture assumption broken: retune this test's citation if "
            "retrieval no longer finds evidence for participant 404's Loss "
            "of Interest",
        )
        real_turn_id = loss_of_interest_bundle.snippets[0].turn_id

        fake_engine = FakeEngine(
            memory_update_score=3, memory_update_turn_id=real_turn_id
        )
        result = self.module.run_full_sample_assessment(
            sample_path=GOLDEN_SAMPLE,
            engine=fake_engine,
            enable_memory_update=True,
            verbose=False,
        )
        self.assertIn("Loss of Interest", result["updated_scores"])
        self.assertEqual(result["topics"]["Loss of Interest"]["updated_score"], 3)

    def test_memory_update_rejects_citation_borrowed_from_another_topic(self) -> None:
        """A cited turn id that is real but was only ever shown as evidence
        for a *different* topic must not be enough to raise this topic's
        score — regression for the production bug where "Low Self-Worth"
        was repeatedly scored 3/3 by citing turn ids that were only ever
        retrieved for an unrelated topic.
        """
        from psyvec.evaluation.evidence_retrieval import (
            load_keyword_lexicon,
            load_tag_map,
            retrieve_evidence,
        )

        sample = json.loads(GOLDEN_SAMPLE.read_text(encoding="utf-8"))
        scale_file = PROJECT_ROOT / "configs" / "scales" / "PHQ-8.json"
        topics = list(json.loads(scale_file.read_text(encoding="utf-8")).keys())
        tag_map = load_tag_map(
            PROJECT_ROOT / "configs" / "scales" / "topic_tag_map.json",
            valid_topics=topics,
        )
        lexicon = load_keyword_lexicon(
            PROJECT_ROOT / "configs" / "scales" / "topic_keywords.json",
            valid_topics=topics,
        )
        bundles = retrieve_evidence(sample["real_interview"], topics, tag_map, lexicon)
        sleep_bundle = bundles["Sleep Problems"]
        self.assertEqual(
            sleep_bundle.status,
            "known",
            "test fixture assumption broken: retune this test's citation if "
            "retrieval no longer finds evidence for participant 404's Sleep "
            "Problems",
        )
        # A turn id that IS real evidence, but only for "Sleep Problems" —
        # FakeEngine always proposes its revision for "Loss of Interest", so
        # citing this borrowed id must not be enough to raise that score.
        borrowed_turn_id = sleep_bundle.snippets[0].turn_id

        fake_engine = FakeEngine(
            memory_update_score=3, memory_update_turn_id=borrowed_turn_id
        )
        result = self.module.run_full_sample_assessment(
            sample_path=GOLDEN_SAMPLE,
            engine=fake_engine,
            enable_memory_update=True,
            verbose=False,
        )
        self.assertNotIn("Loss of Interest", result["updated_scores"])
        self.assertNotIn("updated_score", result["topics"]["Loss of Interest"])


if __name__ == "__main__":
    unittest.main()
