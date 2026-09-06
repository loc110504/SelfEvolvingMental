"""Wiring check for the simple direct/cot/fewshot PHQ-8 baselines.

``scripts/run_baseline_eval.py`` transitively imports ``scripts/run_qwen_sample.py``,
which hard-imports ``torch``/``transformers`` at module level — this repo's
test environment does not have those. As in
``tests/integration/test_grounded_interview_pipeline.py``, those two modules
are stubbed just enough to import, and a scripted fake engine (never a real
model) drives the pipeline against real dev/train data to prove the wiring
(prompt building, few-shot example selection, response parsing, metrics)
works end to end.
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

BASELINE_SCRIPT = PROJECT_ROOT / "scripts" / "run_baseline_eval.py"
DEV_DIR = PROJECT_ROOT / "data" / "processed_daic_woz" / "dev"
TRAIN_DIR = PROJECT_ROOT / "data" / "processed_daic_woz" / "train"


def _install_stub_module(name: str, attrs: dict[str, Any]) -> None:
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(stub, attr_name, value)
    sys.modules[name] = stub


def _load_run_baseline_eval() -> Any:
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
        "run_baseline_eval_under_test", BASELINE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEngine:
    """Always returns a fixed, valid score set — never loads a model."""

    def __init__(self, model_name: str = "fake-model") -> None:
        self.model_name = model_name
        self.truncation_count = 0
        self.prompts: list[str] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2560,
        temperature: float = 0.2,
    ) -> str:
        self.prompts.append(user_prompt)
        payload = {
            "summary": "fake reply",
            "updated_scores": {
                "Loss of Interest": {"score": 1, "reason": "x"},
                "Depressed Mood": {"score": 1, "reason": "x"},
                "Sleep Problems": {"score": 1, "reason": "x"},
                "Fatigue or Low Energy": {"score": 1, "reason": "x"},
                "Appetite or Weight Changes": {"score": 0, "reason": "x"},
                "Low Self-Worth": {"score": 0, "reason": "x"},
                "Concentration Difficulties": {"score": 0, "reason": "x"},
                "Psychomotor Changes": {"score": 0, "reason": "x"},
            },
        }
        return "Some reasoning text.\n" + json.dumps(payload)


class BaselinePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        if not DEV_DIR.is_dir() or not TRAIN_DIR.is_dir():
            self.skipTest("processed DAIC-WOZ data not present")
        self.module = _load_run_baseline_eval()
        self.topics = list(
            json.loads(self.module.DEFAULT_SCALE_FILE.read_text(encoding="utf-8")).keys()
        )
        standards = json.loads(
            self.module.DEFAULT_STANDARDS_FILE.read_text(encoding="utf-8")
        )
        self.scoring_standards = standards["PHQ-8"]
        self.sample_path = sorted(DEV_DIR.glob("*.json"))[0]

    def test_direct_baseline_produces_valid_scored_result(self) -> None:
        result = self.module.run_baseline_sample(
            self.sample_path,
            "direct",
            FakeEngine(),
            self.topics,
            self.scoring_standards,
            fewshot_block="",
            max_chars=3000,
        )
        self.assertTrue(result["assessment_valid"])
        self.assertEqual(result["total_predicted_score"], 4)  # 1+1+1+1+0+0+0+0
        self.assertEqual(len(result["item_scores"]), 8)
        self.assertIn(result["predicted_binary"], (0, 1))

    def test_cot_baseline_still_parses_despite_leading_reasoning_text(self) -> None:
        result = self.module.run_baseline_sample(
            self.sample_path,
            "cot",
            FakeEngine(),
            self.topics,
            self.scoring_standards,
            fewshot_block="",
            max_chars=3000,
        )
        self.assertTrue(result["assessment_valid"])

    def test_select_fewshot_examples_only_uses_train_split(self) -> None:
        paths = self.module.select_fewshot_examples(TRAIN_DIR, k=5)
        self.assertEqual(len(paths), 5)
        for path in paths:
            self.assertEqual(path.parent, TRAIN_DIR)
        # Deterministic: re-running picks the same examples.
        self.assertEqual(paths, self.module.select_fewshot_examples(TRAIN_DIR, k=5))

    def test_fewshot_baseline_includes_examples_in_the_prompt(self) -> None:
        example_paths = self.module.select_fewshot_examples(TRAIN_DIR, k=5)
        fewshot_block = self.module.build_fewshot_block(
            example_paths, self.topics, max_chars=500
        )
        engine = FakeEngine()
        result = self.module.run_baseline_sample(
            self.sample_path,
            "fewshot",
            engine,
            self.topics,
            self.scoring_standards,
            fewshot_block,
            max_chars=3000,
        )
        self.assertTrue(result["assessment_valid"])
        self.assertIn("Output:", engine.prompts[-1])
        self.assertIn("example", engine.prompts[-1])

    def test_compute_per_item_metrics_is_none_without_item_ground_truth(self) -> None:
        records = [
            {
                "item_scores": {"Loss of Interest": 1},
                "ground_truth_items": {},
            }
        ]
        self.assertIsNone(
            self.module.compute_per_item_metrics(records, ["Loss of Interest"])
        )

    def test_compute_per_item_metrics_computes_mae_and_accuracy(self) -> None:
        records = [
            {
                "item_scores": {"Loss of Interest": 2},
                "ground_truth_items": {"PHQ8_NoInterest": 1},
            },
            {
                "item_scores": {"Loss of Interest": 1},
                "ground_truth_items": {"PHQ8_NoInterest": 1},
            },
        ]
        metrics = self.module.compute_per_item_metrics(records, ["Loss of Interest"])
        assert metrics is not None
        self.assertAlmostEqual(metrics["Loss of Interest"]["mae"], 0.5)
        self.assertAlmostEqual(metrics["Loss of Interest"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
