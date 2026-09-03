"""Strict parsing of reasoning-model responses (no silent score guessing)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.response_parsing import (  # noqa: E402
    extract_json_objects,
    parse_necessity_score,
    parse_score_and_summary,
    parse_summary_and_updated_scores,
    strip_reasoning,
)

TOPICS = ["Loss of Interest", "Depressed Mood"]

# Verbatim shape of the Qwen3.5-9B output that produced 184 identical "1" scores.
TRUNCATED_THINKING = (
    "Thinking Process:\n\n"
    "1.  **Analyze the Request:**\n"
    "    *   **Role:** I am acting as a professional psychological scale scorer.\n"
    "    *   **Task:** Score a specific topic based on the dialogue history and"
)


class StripReasoningTest(unittest.TestCase):
    def test_removes_closed_think_tags(self) -> None:
        raw = '<think>weighing 2 vs 3</think>\n{"score": 2}'
        self.assertEqual(strip_reasoning(raw), '{"score": 2}')

    def test_drops_unclosed_think_tag_remainder(self) -> None:
        self.assertEqual(strip_reasoning("<think>still thinking about 1."), "")

    def test_truncated_thinking_heading_yields_empty(self) -> None:
        self.assertEqual(strip_reasoning(TRUNCATED_THINKING), "")

    def test_keeps_text_after_answer_marker(self) -> None:
        raw = 'Thinking Process:\n1. consider 0\nFinal Answer: {"score": 3}'
        self.assertEqual(strip_reasoning(raw), '{"score": 3}')

    def test_plain_answer_passes_through(self) -> None:
        cleaned = strip_reasoning("  Age: 30, Gender: male  ")
        self.assertEqual(cleaned, "Age: 30, Gender: male")


class ExtractJsonObjectsTest(unittest.TestCase):
    def test_returns_objects_in_order(self) -> None:
        text = '{"score": 0} noise {"score": 3}'
        self.assertEqual(extract_json_objects(text), ['{"score": 0}', '{"score": 3}'])

    def test_ignores_braces_inside_strings(self) -> None:
        text = '{"summary": "used } and { in prose", "score": 1}'
        self.assertEqual(extract_json_objects(text), [text])

    def test_handles_nested_objects(self) -> None:
        text = '{"updated_scores": {"A": {"score": 2}}}'
        self.assertEqual(extract_json_objects(text), [text])


class ParseScoreAndSummaryTest(unittest.TestCase):
    def test_parses_plain_json(self) -> None:
        result = parse_score_and_summary('{"score": 2, "summary": "several days"}')
        self.assertTrue(result.ok)
        self.assertEqual(result.score, 2)
        self.assertEqual(result.summary, "several days")

    def test_prefers_last_json_object(self) -> None:
        raw = (
            '<think>The schema is {"score": 0, "summary": "x"} — but the patient '
            'reports daily symptoms.</think>\n'
            '{"score": 3, "summary": "nearly every day"}'
        )
        result = parse_score_and_summary(raw)
        self.assertEqual(result.score, 3)
        self.assertEqual(result.summary, "nearly every day")

    def test_truncated_thinking_fails_instead_of_returning_one(self) -> None:
        result = parse_score_and_summary(TRUNCATED_THINKING)
        self.assertFalse(result.ok)
        self.assertIsNone(result.score)
        self.assertIn("truncated", result.failure_reason)
        self.assertEqual(result.raw_text, TRUNCATED_THINKING)

    def test_numbered_prose_without_score_field_fails(self) -> None:
        result = parse_score_and_summary(
            "1. The patient mentions low mood.\n2. Duration is unclear."
        )
        self.assertFalse(result.ok)
        self.assertIsNone(result.score)

    def test_bare_number_is_accepted(self) -> None:
        result = parse_score_and_summary("2")
        self.assertTrue(result.ok)
        self.assertEqual(result.score, 2)

    def test_loose_key_value_is_accepted(self) -> None:
        result = parse_score_and_summary('score: 3, summary: "daily"')
        self.assertTrue(result.ok)
        self.assertEqual(result.score, 3)

    def test_out_of_range_score_is_clamped(self) -> None:
        self.assertEqual(parse_score_and_summary('{"score": 9}').score, 3)
        self.assertEqual(parse_score_and_summary('{"score": -4}').score, 0)

    def test_prose_framed_json_is_still_parsed(self) -> None:
        raw = (
            "Reasoning: the patient reports daily symptoms.\n"
            '{"score": 3, "summary": "daily"}'
        )
        result = parse_score_and_summary(raw)
        self.assertTrue(result.ok)
        self.assertEqual(result.score, 3)

    def test_echoed_schema_inside_truncated_thinking_still_fails(self) -> None:
        raw = (
            "Thinking Process:\n\n1. The requested output is "
            '{"score": <0, 1, 2, or 3>, "summary": "<one sentence basis>"}\n'
            "2. Now weighing the evidence"
        )
        result = parse_score_and_summary(raw)
        self.assertFalse(result.ok)
        self.assertIsNone(result.score)

    def test_empty_response_fails(self) -> None:
        result = parse_score_and_summary("   ")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_reason, "empty response")


class ParseNecessityScoreTest(unittest.TestCase):
    def test_bare_digit(self) -> None:
        result = parse_necessity_score("2")
        self.assertTrue(result.ok)
        self.assertEqual(result.score, 2)

    def test_thinking_prefix_fails(self) -> None:
        result = parse_necessity_score(TRUNCATED_THINKING)
        self.assertFalse(result.ok)
        self.assertIsNone(result.score)

    def test_labelled_value(self) -> None:
        result = parse_necessity_score("Necessity: 1")
        self.assertTrue(result.ok)
        self.assertEqual(result.score, 1)


class ParseSummaryAndUpdatedScoresTest(unittest.TestCase):
    def test_parses_fenced_json(self) -> None:
        raw = (
            "```json\n"
            '{"summary": "moderate", "updated_scores": '
            '{"Loss of Interest": {"score": 2, "reason": "daily"}}}\n'
            "```"
        )
        result = parse_summary_and_updated_scores(raw, TOPICS)
        self.assertTrue(result.ok)
        self.assertEqual(result.summary, "moderate")
        self.assertEqual(result.updated_scores["Loss of Interest"]["score"], 2)

    def test_unknown_topics_are_dropped(self) -> None:
        raw = '{"summary": "s", "updated_scores": {"Not A Topic": {"score": 1}}}'
        result = parse_summary_and_updated_scores(raw, TOPICS)
        self.assertTrue(result.ok)
        self.assertEqual(result.updated_scores, {})

    def test_truncated_thinking_fails(self) -> None:
        result = parse_summary_and_updated_scores(TRUNCATED_THINKING, TOPICS)
        self.assertFalse(result.ok)
        self.assertEqual(result.updated_scores, {})
        self.assertEqual(result.summary, "")


if __name__ == "__main__":
    unittest.main()
