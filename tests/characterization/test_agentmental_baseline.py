from __future__ import annotations

import ast
import json
import logging
import re
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from verify_agentmental_baseline import DEFAULT_MANIFEST, verify  # noqa: E402


def load_upstream_functions(*function_names: str) -> dict[str, object]:
    """Compile selected upstream functions without importing its dependencies."""
    source_path = PROJECT_ROOT / "AgentMental" / "src" / "utils.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    if len(selected) != len(function_names):
        raise AssertionError("not all requested upstream functions were found")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {
        "json": json,
        "logger": logging.getLogger("agentmental-characterization"),
        "re": re,
        "dialog_print": lambda *_args, **_kwargs: None,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


class BaselineIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (PROJECT_ROOT / "AgentMental").is_dir():
            raise unittest.SkipTest("AgentMental baseline directory not present")

    def test_pinned_upstream_and_static_smoke_checks(self) -> None:
        checks = verify(DEFAULT_MANIFEST)
        self.assertGreaterEqual(len(checks), 8)

    def test_simulator_scoring_rubric_is_dead_computation(self) -> None:
        checks = verify(DEFAULT_MANIFEST)
        self.assertIn(
            "simulator scoring rubric is computed but absent from prompts", checks
        )

    def test_simulator_transcript_reaches_system_prompt(self) -> None:
        checks = verify(DEFAULT_MANIFEST)
        self.assertIn(
            "simulator transcript is consumed and reaches the system prompt", checks
        )


class UpstreamBehaviorCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (PROJECT_ROOT / "AgentMental").is_dir():
            raise unittest.SkipTest("AgentMental baseline directory not present")
        cls.fixture = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))[
            "characterization"
        ]
        cls.functions = load_upstream_functions(
            "extract_score",
            "extract_score_and_summary",
            "is_necessary",
        )

    def test_necessity_stop_matrix(self) -> None:
        is_necessary = self.functions["is_necessary"]
        actual = {
            key: is_necessary(*(int(value) for value in key.split(":")))
            for key in self.fixture["necessity_continue_matrix"]
        }
        self.assertEqual(actual, self.fixture["necessity_continue_matrix"])

    def test_extract_score_uses_first_integer_and_silently_defaults_to_zero(self) -> None:
        extract_score = self.functions["extract_score"]
        self.assertEqual(extract_score("necessity=2, confidence=90"), 2)
        self.assertEqual(extract_score("not a score"), 0)

    def test_malformed_or_out_of_range_item_score_defaults_to_zero(self) -> None:
        extract = self.functions["extract_score_and_summary"]
        fallback = self.fixture["invalid_score_fallback"]
        self.assertEqual(extract("not-json", "PHQ-8"), (fallback, ""))
        self.assertEqual(
            extract('{"score": 9, "summary": "out of range"}', "PHQ-8"),
            (fallback, "out of range"),
        )


if __name__ == "__main__":
    unittest.main()
