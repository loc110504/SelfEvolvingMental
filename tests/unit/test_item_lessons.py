"""A lesson may only be promoted on evidence from participants it did not come from."""

from __future__ import annotations

import json
import unittest

from psyvec.evolution.item_lessons import (
    ErrorPattern,
    ReplayOutcome,
    ScoringState,
    build_distillation_prompt,
    candidate_entry,
    mine_error_patterns,
    parse_lesson,
    score_gain,
    validate_lesson,
)


def make_state(
    participant: str,
    *,
    topic: str = "Sleep Problems",
    tier: str = "lexical",
    predicted: float = 0.0,
    reference: float = 2.0,
    abstained: bool = False,
) -> ScoringState:
    return ScoringState(
        participant_id=participant,
        topic=topic,
        item_key="PHQ8_Sleep",
        evidence_tier=tier,
        evidence_text="[turn-1] Participant: i barely sleep",
        predicted=predicted,
        reference=reference,
        abstained=abstained,
        reasoning="no clear frequency stated",
    )


class MiningTest(unittest.TestCase):
    def test_a_systematically_underscored_cell_is_mined(self) -> None:
        states = [make_state(str(index)) for index in range(10)]
        patterns = mine_error_patterns(states, min_support=8)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].direction, "under")
        self.assertAlmostEqual(patterns[0].mean_signed_error, -2.0)

    def test_a_cell_below_minimum_support_is_skipped(self) -> None:
        states = [make_state(str(index)) for index in range(4)]
        self.assertEqual(mine_error_patterns(states, min_support=8), ())

    def test_scattered_error_is_not_mistaken_for_a_correctable_pattern(self) -> None:
        """Equal over- and under-scoring is difficulty, not a procedural fault."""
        states = [
            make_state(str(index), predicted=3.0 if index % 2 else 0.0, reference=1.5)
            for index in range(10)
        ]
        self.assertEqual(mine_error_patterns(states, min_support=8), ())

    def test_an_accurate_cell_is_skipped(self) -> None:
        states = [make_state(str(i), predicted=2.0, reference=2.0) for i in range(10)]
        self.assertEqual(mine_error_patterns(states, min_support=8), ())

    def test_cells_are_ranked_by_total_error_not_per_case_error(self) -> None:
        states = [make_state(str(i)) for i in range(20)] + [
            make_state(
                f"x{i}", topic="Psychomotor Changes", predicted=0.0, reference=3.0
            )
            for i in range(8)
        ]
        patterns = mine_error_patterns(states, min_support=8)
        self.assertEqual(patterns[0].topic, "Sleep Problems")

    def test_state_ids_are_stable_and_distinct_per_participant(self) -> None:
        self.assertEqual(make_state("1").state_id, make_state("1").state_id)
        self.assertNotEqual(make_state("1").state_id, make_state("2").state_id)


PATTERN = ErrorPattern(
    topic="Sleep Problems",
    evidence_tier="lexical",
    support=10,
    mean_signed_error=-2.0,
    mean_absolute_error=2.0,
    abstention_rate=0.4,
    source_state_ids=("state-a", "state-b"),
)


class DistillationTest(unittest.TestCase):
    def test_prompt_names_the_direction_of_the_error(self) -> None:
        prompt = build_distillation_prompt(PATTERN, [make_state("1")])
        self.assertIn("TOO LOW", prompt)

    def test_prompt_withholds_the_transcript_and_the_answer(self) -> None:
        prompt = build_distillation_prompt(PATTERN, [make_state("1")])
        self.assertNotIn("i barely sleep", prompt)

    def test_a_well_formed_rule_parses_into_the_paper_lesson_shape(self) -> None:
        lesson = parse_lesson(
            json.dumps(
                {
                    "trigger": "the symptom is described as ongoing with no day count",
                    "do": "rate it as present more than half the days",
                    "avoid": "defaulting to the lowest nonzero level",
                    "criterion": "the frequency basis recorded is ongoing_state",
                    "confidence": 0.7,
                }
            ),
            PATTERN,
        )
        assert lesson is not None
        self.assertEqual(lesson.role, "scorer")
        self.assertEqual(lesson.scope, "Sleep Problems|lexical")
        self.assertEqual(lesson.source_contrast_ids, PATTERN.source_state_ids)

    def test_a_rule_naming_its_supervision_is_rejected(self) -> None:
        """A lesson that quotes the reference would leak labels into inference."""
        self.assertIsNone(
            parse_lesson(
                json.dumps(
                    {
                        "trigger": "always",
                        "do": "match the ground truth score",
                        "avoid": "guessing",
                        "criterion": "it matches",
                    }
                ),
                PATTERN,
            )
        )

    def test_an_incomplete_rule_is_rejected(self) -> None:
        self.assertIsNone(
            parse_lesson(json.dumps({"trigger": "x", "do": "", "avoid": "y",
                                     "criterion": "z"}), PATTERN)
        )

    def test_unparseable_output_is_rejected(self) -> None:
        self.assertIsNone(parse_lesson("sorry, I cannot", PATTERN))

    def test_a_distilled_lesson_starts_as_an_unvalidated_candidate(self) -> None:
        lesson = parse_lesson(
            json.dumps({"trigger": "a", "do": "b", "avoid": "c", "criterion": "d"}),
            PATTERN,
        )
        assert lesson is not None
        self.assertEqual(candidate_entry(lesson).lifecycle, "candidate")


class ScoreGainTest(unittest.TestCase):
    def test_moving_toward_the_reference_is_a_positive_gain(self) -> None:
        self.assertGreater(score_gain(control=0.0, treatment=2.0, reference=2.0), 0)

    def test_moving_away_is_negative(self) -> None:
        self.assertLess(score_gain(control=2.0, treatment=0.0, reference=2.0), 0)

    def test_gain_is_clipped_to_the_unit_interval(self) -> None:
        self.assertLessEqual(score_gain(0.0, 3.0, 3.0), 1.0)
        self.assertGreaterEqual(score_gain(3.0, 0.0, 3.0), -1.0)


def outcome(
    state_id: str, control: float, treatment: float, **kw: object
) -> ReplayOutcome:
    return ReplayOutcome(
        state_id=state_id, participant_id=state_id, reference=2.0,
        control_value=control, treatment_value=treatment, **kw,  # type: ignore[arg-type]
    )


class TransferGateTest(unittest.TestCase):
    HELPFUL = [outcome(f"s{i}", 0.0, 2.0) for i in range(8)]

    def test_a_lesson_that_helps_on_held_out_states_is_promoted(self) -> None:
        verdict = validate_lesson(self.HELPFUL, excluded_state_ids=frozenset())
        self.assertTrue(verdict.promoted)
        self.assertEqual(verdict.support, 8)

    def test_source_states_are_excluded_from_their_own_verification(self) -> None:
        verdict = validate_lesson(
            self.HELPFUL, excluded_state_ids=frozenset(f"s{i}" for i in range(8))
        )
        self.assertFalse(verdict.promoted)
        self.assertIn("no_source_excluded_states", verdict.reasons)

    def test_too_few_held_out_states_blocks_promotion(self) -> None:
        verdict = validate_lesson(self.HELPFUL[:3], excluded_state_ids=frozenset())
        self.assertFalse(verdict.promoted)
        self.assertIn("insufficient_support", verdict.reasons)

    def test_one_large_win_among_losses_is_not_transfer(self) -> None:
        outcomes = [outcome("s0", 0.0, 2.0)] + [
            outcome(f"s{i}", 2.0, 1.9) for i in range(1, 8)
        ]
        verdict = validate_lesson(outcomes, excluded_state_ids=frozenset())
        self.assertFalse(verdict.promoted)
        self.assertIn("too_few_states_improved", verdict.reasons)

    def test_a_lesson_that_buys_accuracy_with_fabrication_is_blocked(self) -> None:
        outcomes = [
            outcome(f"s{i}", 0.0, 2.0, treatment_grounded=i != 0) for i in range(8)
        ]
        verdict = validate_lesson(outcomes, excluded_state_ids=frozenset())
        self.assertFalse(verdict.promoted)
        self.assertIn("safety_regression", verdict.reasons)
        self.assertEqual(verdict.safety_regressions, 1)

    def test_a_neutral_lesson_fails_the_utility_margin(self) -> None:
        outcomes = [outcome(f"s{i}", 2.0, 2.0) for i in range(8)]
        verdict = validate_lesson(outcomes, excluded_state_ids=frozenset())
        self.assertFalse(verdict.promoted)
        self.assertIn("utility_margin_below_threshold", verdict.reasons)


if __name__ == "__main__":
    unittest.main()
