"""Retrieval must preserve question context and refuse incidental matches."""

from __future__ import annotations

import unittest

from psyvec.evaluation.evidence_tiers import (
    TieredEvidence,
    build_paired_turns,
    format_tiered_evidence,
    merge_retrieved,
    overall_tier,
    search_paired_turns,
    tokenize,
)

TRANSCRIPT = [
    {"roleName": "Ellie", "content": "hi i'm ellie thanks for coming in today"},
    {"roleName": "Participant", "content": "hi"},
    {
        "roleName": "Ellie",
        "content": "easy_sleep (how easy is it for you to get a good night's sleep)",
    },
    {
        "roleName": "Participant",
        "content": "not easy at all i just lay there for hours",
    },
    {"roleName": "Participant", "content": "maybe three hours a night if i'm lucky"},
    {"roleName": "Ellie", "content": "what do you do for fun"},
    {"roleName": "Participant", "content": "i love hiking and seeing my friends"},
]


class BuildPairedTurnsTest(unittest.TestCase):
    def test_pairs_each_participant_turn_with_its_question(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        self.assertEqual(
            [turn.turn_id for turn in turns], ["turn-1", "turn-3", "turn-6"]
        )
        self.assertEqual(
            turns[1].question, "how easy is it for you to get a good night's sleep"
        )

    def test_strips_the_ellie_protocol_tag_from_the_question(self) -> None:
        self.assertEqual(build_paired_turns(TRANSCRIPT)[1].tag, "easy_sleep")

    def test_a_split_answer_is_folded_back_into_one_turn(self) -> None:
        # DAIC-WOZ segments one spoken answer across several turns; scored
        # apart, the fragments are filler and the disclosure loses its context.
        answer = build_paired_turns(TRANSCRIPT)[1]
        self.assertIn("lay there for hours", answer.answer)
        self.assertIn("three hours a night", answer.answer)
        self.assertEqual(answer.span_ids, ("turn-3", "turn-4"))
        self.assertEqual(answer.citable_ids, ("turn-3", "turn-4"))

    def test_merging_can_be_switched_off_for_an_ablation(self) -> None:
        turns = build_paired_turns(TRANSCRIPT, merge_consecutive=False)
        self.assertEqual(
            [turn.turn_id for turn in turns],
            ["turn-1", "turn-3", "turn-4", "turn-6"],
        )

    def test_a_new_question_closes_the_previous_run(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        self.assertNotIn("hiking", turns[1].answer)


class SearchTest(unittest.TestCase):
    def test_finds_the_sleep_disclosure_through_the_question_context(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        hits = search_paired_turns(turns, ["good night's sleep"], top_k=3)
        self.assertIn("turn-3", {hit.turn.turn_id for hit in hits})

    def test_min_score_rejects_an_incidental_single_token_match(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        self.assertEqual(search_paired_turns(turns, ["hiking"], min_score=99.0), ())

    def test_excluded_ids_are_never_returned(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        hits = search_paired_turns(
            turns, ["lay there for hours"], exclude_turn_ids=frozenset({"turn-3"})
        )
        self.assertNotIn("turn-3", {hit.turn.turn_id for hit in hits})

    def test_search_stamps_the_relation_it_was_given(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        hits = search_paired_turns(turns, ["hiking friends"], relation="reverse")
        self.assertTrue(all(hit.relation == "reverse" for hit in hits))

    def test_empty_query_set_returns_nothing_rather_than_everything(self) -> None:
        self.assertEqual(search_paired_turns(build_paired_turns(TRANSCRIPT), []), ())

    def test_tokenize_drops_interview_filler(self) -> None:
        self.assertNotIn("yeah", tokenize("yeah um i know like the thing"))


class MergeAndTierTest(unittest.TestCase):
    def test_merge_deduplicates_and_keeps_the_first_group_first(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        strong = search_paired_turns(turns, ["lay there for hours"], tier="direct")
        weak = search_paired_turns(turns, ["lay there for hours"], tier="expanded")
        merged = merge_retrieved(strong, weak, limit=10)
        self.assertEqual(len(merged), len(strong))
        self.assertEqual(merged[0].tier, "direct")

    def test_overall_tier_reports_the_strongest_present(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        mixed = merge_retrieved(
            search_paired_turns(turns, ["hiking friends"], tier="contextual"),
            search_paired_turns(turns, ["lay there for hours"], tier="direct"),
            limit=10,
        )
        self.assertEqual(overall_tier(mixed), "direct")

    def test_empty_retrieval_is_absent(self) -> None:
        self.assertEqual(overall_tier(()), "absent")


class FormatTest(unittest.TestCase):
    def test_absent_evidence_says_nobody_asked_not_silence(self) -> None:
        rendered = format_tiered_evidence(
            TieredEvidence(topic="Psychomotor Changes", tier="absent", turns=())
        )
        # The v2 pipeline let "no evidence" reach the scorer as an empty
        # context, which it read as a denial. The distinction must be explicit.
        self.assertIn("neither reported nor denied", rendered)

    def test_rendered_evidence_keeps_turn_ids_and_the_question(self) -> None:
        turns = build_paired_turns(TRANSCRIPT)
        hits = search_paired_turns(turns, ["good night's sleep"], top_k=1)
        rendered = format_tiered_evidence(
            TieredEvidence(topic="Sleep Problems", tier="direct", turns=hits)
        )
        self.assertIn("[turn-3]", rendered)
        self.assertIn("Interviewer:", rendered)

    def test_reverse_polarity_evidence_is_labelled_as_such(self) -> None:
        """A proud answer must not read to the scorer as symptom evidence."""
        turns = build_paired_turns(TRANSCRIPT)
        hits = search_paired_turns(
            turns, ["hiking friends"], top_k=1, relation="reverse"
        )
        rendered = format_tiered_evidence(
            TieredEvidence(topic="Low Self-Worth", tier="lexical", turns=hits)
        )
        self.assertIn("REVERSE POLARITY", rendered)
        self.assertIn("evidence AGAINST", rendered)


if __name__ == "__main__":
    unittest.main()
