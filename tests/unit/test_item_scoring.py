"""Abstention must survive parsing as a distinct outcome from a zero score."""

from __future__ import annotations

import unittest

from psyvec.evaluation.item_scoring import (
    ITEM_MANIFESTATIONS,
    build_item_score_prompt,
    build_query_expansion_prompt,
    build_sufficiency_prompt,
    parse_item_score,
    parse_query_expansion,
    parse_sufficiency,
    quote_is_grounded,
)

RUBRIC = {
    "0": "not at all",
    "1": "several days",
    "2": "more than half",
    "3": "nearly every day",
}


class ParseItemScoreTest(unittest.TestCase):
    def test_abstention_is_parsed_successfully_with_no_score(self) -> None:
        parsed = parse_item_score(
            '{"sufficient": false, "score": null, "reasoning": "never raised"}'
        )
        self.assertTrue(parsed.ok)
        self.assertFalse(parsed.sufficient)
        self.assertIsNone(parsed.score)

    def test_a_real_score_is_kept(self) -> None:
        parsed = parse_item_score(
            '{"sufficient": true, "score": 2, "quote": "i barely sleep",'
            ' "turn_ids": ["turn-3"]}'
        )
        self.assertEqual(parsed.score, 2)
        self.assertEqual(parsed.turn_ids, ("turn-3",))

    def test_sufficient_without_a_usable_score_degrades_to_abstention(self) -> None:
        # Never invent a number to satisfy a self-contradictory reply.
        parsed = parse_item_score('{"sufficient": true, "score": null}')
        self.assertFalse(parsed.sufficient)
        self.assertIsNone(parsed.score)

    def test_score_zero_is_not_confused_with_abstention(self) -> None:
        parsed = parse_item_score(
            '{"sufficient": true, "score": 0, "quote": "no not really"}'
        )
        self.assertTrue(parsed.sufficient)
        self.assertEqual(parsed.score, 0)

    def test_the_last_object_wins_over_an_echoed_schema(self) -> None:
        parsed = parse_item_score(
            'Schema: {"sufficient": true, "score": 0}\n'
            'Answer: {"sufficient": true, "score": 3}'
        )
        self.assertEqual(parsed.score, 3)

    def test_out_of_range_scores_are_clamped_to_the_rubric(self) -> None:
        self.assertEqual(parse_item_score('{"sufficient": true, "score": 9}').score, 3)

    def test_unparseable_reply_is_a_failure_not_a_silent_zero(self) -> None:
        parsed = parse_item_score("I cannot answer that.")
        self.assertFalse(parsed.ok)
        self.assertIsNone(parsed.score)


class ParseHelpersTest(unittest.TestCase):
    def test_sufficiency_verdict_and_missing_note(self) -> None:
        verdict, missing = parse_sufficiency(
            '{"sufficient": false, "missing": "how often"}'
        )
        self.assertFalse(verdict)
        self.assertEqual(missing, "how often")

    def test_unparseable_sufficiency_is_none_not_false(self) -> None:
        # None means "undecided, keep looking"; False is a real verdict.
        self.assertIsNone(parse_sufficiency("no json here")[0])

    def test_query_expansion_deduplicates_case_insensitively(self) -> None:
        queries = parse_query_expansion(
            '{"queries": ["cant sleep", "CANT SLEEP", "up all night"]}'
        )
        self.assertEqual(queries, ("cant sleep", "up all night"))


class QuoteGroundingTest(unittest.TestCase):
    EVIDENCE = (
        "[turn-9] Participant: yeah i just lay there for hours staring at "
        "the ceiling"
    )

    def test_verbatim_quote_is_grounded(self) -> None:
        self.assertTrue(quote_is_grounded("i just lay there for hours", self.EVIDENCE))

    def test_fabricated_denial_is_not_grounded(self) -> None:
        # This is the exact v2 failure: a confident denial with no source.
        self.assertFalse(
            quote_is_grounded("i have not felt depressed at all", self.EVIDENCE)
        )

    def test_empty_quote_is_not_grounded(self) -> None:
        self.assertFalse(quote_is_grounded("   ", self.EVIDENCE))


class PresenceTest(unittest.TestCase):
    """`present` gates the score, so "unclear" can never become a zero."""

    def test_unclear_presence_abstains_even_with_a_number_attached(self) -> None:
        parsed = parse_item_score(
            '{"present": "unclear", "sufficient": true, "score": 0,'
            ' "frequency_basis": "none"}'
        )
        self.assertFalse(parsed.sufficient)
        self.assertIsNone(parsed.score)

    def test_an_explicit_denial_is_a_real_zero_not_an_abstention(self) -> None:
        parsed = parse_item_score(
            '{"present": "no", "sufficient": true, "score": 0,'
            ' "frequency_basis": "none", "quote": "no i sleep fine"}'
        )
        self.assertTrue(parsed.sufficient)
        self.assertEqual(parsed.score, 0)

    def test_frequency_basis_is_preserved_for_diagnostics(self) -> None:
        parsed = parse_item_score(
            '{"present": "yes", "sufficient": true, "score": 2,'
            ' "frequency_basis": "ongoing_state"}'
        )
        self.assertEqual(parsed.frequency_basis, "ongoing_state")

    def test_a_missing_presence_field_is_inferred_from_the_score(self) -> None:
        parsed = parse_item_score('{"sufficient": true, "score": 2}')
        self.assertEqual(parsed.present, "yes")
        self.assertEqual(parsed.score, 2)


class PromptTest(unittest.TestCase):
    def test_score_prompt_forbids_reading_silence_as_denial(self) -> None:
        prompt = build_item_score_prompt("Sleep Problems", RUBRIC, "no evidence")
        self.assertIn("An unasked question is not a denial", prompt)
        self.assertIn("Do NOT guess 0", prompt)

    def test_score_prompt_rules_out_under_scoring_an_ongoing_state(self) -> None:
        """The rule that recovers the severity range v2 collapsed."""
        prompt = build_item_score_prompt("Sleep Problems", RUBRIC, "x")
        self.assertIn("ongoing current state is present more than half", prompt)

    def test_score_prompt_states_the_two_week_window(self) -> None:
        prompt = build_item_score_prompt("Sleep Problems", RUBRIC, "x")
        self.assertIn("last two weeks", prompt)

    def test_score_prompt_includes_the_items_manifestations(self) -> None:
        prompt = build_item_score_prompt("Psychomotor Changes", RUBRIC, "x")
        self.assertIn(ITEM_MANIFESTATIONS["Psychomotor Changes"], prompt)

    def test_evidence_precedes_the_decision_procedure(self) -> None:
        # The procedure sits last so it is the freshest context at generation.
        prompt = build_item_score_prompt("Sleep Problems", RUBRIC, "MARKER-TEXT")
        self.assertLess(prompt.index("MARKER-TEXT"), prompt.index("## How to decide"))

    def test_score_prompt_carries_validated_lessons_when_supplied(self) -> None:
        prompt = build_item_score_prompt(
            "Sleep Problems", RUBRIC, "x", lessons=("probe frequency first",)
        )
        self.assertIn("probe frequency first", prompt)

    def test_no_lesson_block_when_there_are_no_lessons(self) -> None:
        prompt = build_item_score_prompt("Sleep Problems", RUBRIC, "x")
        self.assertNotIn("Validated rating guidance", prompt)

    def test_expansion_prompt_lists_what_was_already_tried(self) -> None:
        prompt = build_query_expansion_prompt(
            "Sleep Problems", RUBRIC, "how often", ("insomnia",)
        )
        self.assertIn("insomnia", prompt)
        self.assertIn("do not repeat", prompt.lower())

    def test_expansion_prompt_demands_spoken_not_clinical_vocabulary(self) -> None:
        prompt = build_query_expansion_prompt("Sleep Problems", RUBRIC, "", ())
        self.assertIn("no clinical jargon", prompt)

    def test_sufficiency_prompt_treats_a_denial_as_sufficient(self) -> None:
        prompt = build_sufficiency_prompt("Sleep Problems", RUBRIC, "x")
        self.assertIn("a clear denial counts as sufficient", prompt)


if __name__ == "__main__":
    unittest.main()
