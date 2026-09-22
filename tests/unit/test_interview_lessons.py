"""Acquisition-strategy lessons: mining, label-leak rejection, transfer gate."""

from __future__ import annotations

import json
import unittest

from psyvec.evolution.interview_lessons import (
    AcquisitionPattern,
    AcquisitionReplayOutcome,
    AcquisitionState,
    acquisition_value,
    build_acquisition_distillation_prompt,
    efficiency_gain,
    mine_low_yield_patterns,
    parse_acquisition_lesson,
    validate_acquisition_lesson,
)


def make_state(
    participant: str,
    *,
    topic: str = "Sleep Problems",
    tier_before: str = "absent",
    resolved_before: bool = False,
    tier_after: str = "absent",
    resolved_after: bool = False,
    round_index: int = 1,
) -> AcquisitionState:
    return AcquisitionState(
        participant_id=participant, topic=topic, tier_before=tier_before,  # type: ignore[arg-type]
        resolved_before=resolved_before, tier_after=tier_after,  # type: ignore[arg-type]
        resolved_after=resolved_after, queries_tried=("did not sleep well",),
        round_index=round_index,
    )


class MiningTest(unittest.TestCase):
    def test_a_systematically_low_yield_cell_is_mined(self) -> None:
        states = [make_state(str(i)) for i in range(10)]  # tier never improves
        patterns = mine_low_yield_patterns(states, min_support=8)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].yield_rate, 0.0)

    def test_a_high_yield_cell_is_skipped(self) -> None:
        states = [
            make_state(str(i), tier_after="lexical", resolved_after=True)
            for i in range(10)
        ]
        self.assertEqual(mine_low_yield_patterns(states, min_support=8), ())

    def test_a_cell_below_minimum_support_is_skipped(self) -> None:
        states = [make_state(str(i)) for i in range(4)]
        self.assertEqual(mine_low_yield_patterns(states, min_support=8), ())

    def test_rounds_are_ranked_by_total_waste(self) -> None:
        wasted_many = [make_state(str(i)) for i in range(20)]
        wasted_few = [
            make_state(f"x{i}", topic="Psychomotor Changes") for i in range(8)
        ]
        patterns = mine_low_yield_patterns(wasted_many + wasted_few, min_support=8)
        self.assertEqual(patterns[0].topic, "Sleep Problems")

    def test_state_ids_are_unique_per_round_even_at_the_same_tier(self) -> None:
        """Two failed rounds on the same stubborn topic must not collide."""
        first = make_state("1", round_index=1)
        second = make_state("1", round_index=2)
        self.assertNotEqual(first.state_id, second.state_id)

    def test_improved_is_true_when_the_tier_rises(self) -> None:
        improved = make_state("1", tier_after="lexical")
        self.assertTrue(improved.improved)

    def test_improved_is_true_when_resolved_without_a_tier_change(self) -> None:
        resolved = make_state("1", tier_after="absent", resolved_after=True)
        self.assertTrue(resolved.improved)


class DistillationPromptTest(unittest.TestCase):
    def test_prompt_never_leaks_a_specific_outcome(self) -> None:
        pattern = AcquisitionPattern(
            topic="Sleep Problems", tier_before="absent", support=10,
            yield_rate=0.1, source_state_ids=("a", "b"),
            sample_queries=("how do you sleep",),
        )
        prompt = build_acquisition_distillation_prompt(pattern)
        self.assertIn("Sleep Problems", prompt)
        self.assertIn("how do you sleep", prompt)
        self.assertNotIn("PHQ", prompt)


class ParseAcquisitionLessonTest(unittest.TestCase):
    pattern = AcquisitionPattern(
        topic="Sleep Problems", tier_before="absent", support=10, yield_rate=0.1,
        source_state_ids=("a",),
    )

    def test_a_well_formed_lesson_is_parsed(self) -> None:
        raw = json.dumps({
            "trigger": "sleep is absent and only sleep-specific phrases were tried",
            "do": "try phrases about tiredness or energy instead",
            "avoid": "repeating the same sleep-quality phrasing",
            "criterion": "a new turn is found",
            "confidence": 0.7,
        })
        lesson = parse_acquisition_lesson(raw, self.pattern)
        self.assertIsNotNone(lesson)
        assert lesson is not None
        self.assertEqual(lesson.role, "interviewer")
        self.assertEqual(lesson.scope, "Sleep Problems|absent")

    def test_a_lesson_naming_the_ground_truth_is_rejected(self) -> None:
        raw = json.dumps({
            "trigger": "x", "do": "y",
            "avoid": "ignoring the ground truth score", "criterion": "z",
        })
        self.assertIsNone(parse_acquisition_lesson(raw, self.pattern))

    def test_an_incomplete_lesson_is_rejected(self) -> None:
        raw = json.dumps({"trigger": "x", "do": "", "avoid": "y", "criterion": "z"})
        self.assertIsNone(parse_acquisition_lesson(raw, self.pattern))


class EfficiencyGainTest(unittest.TestCase):
    def test_gain_is_positive_when_treatment_beats_control(self) -> None:
        control = acquisition_value("absent", resolved=False)
        treatment = acquisition_value("lexical", resolved=False)
        self.assertGreater(efficiency_gain(control, treatment), 0.0)

    def test_gain_is_zero_for_identical_arms(self) -> None:
        value = acquisition_value("absent", resolved=False)
        self.assertEqual(efficiency_gain(value, value), 0.0)

    def test_gain_is_clipped_to_unit_interval(self) -> None:
        self.assertLessEqual(efficiency_gain(0.0, 100.0), 1.0)
        self.assertGreaterEqual(efficiency_gain(100.0, 0.0), -1.0)


class ValidateAcquisitionLessonTest(unittest.TestCase):
    def test_a_lesson_only_verified_on_its_own_source_is_not_promoted(self) -> None:
        outcomes = [
            AcquisitionReplayOutcome(
                state_id="s1", participant_id="1",
                control_value=0.0, treatment_value=1.0,
            )
        ]
        verdict = validate_acquisition_lesson(
            outcomes, excluded_state_ids=frozenset({"s1"})
        )
        self.assertFalse(verdict.promoted)
        self.assertIn("no_source_excluded_states", verdict.reasons)

    def test_a_lesson_with_strong_held_out_transfer_is_promoted(self) -> None:
        outcomes = [
            AcquisitionReplayOutcome(
                state_id=f"s{i}", participant_id=str(i),
                control_value=acquisition_value("absent", resolved=False),
                treatment_value=acquisition_value("lexical", resolved=False),
            )
            for i in range(8)
        ]
        verdict = validate_acquisition_lesson(
            outcomes, excluded_state_ids=frozenset(), min_support=6,
            min_mean_gain=0.02, min_improved_fraction=0.55,
        )
        self.assertTrue(verdict.promoted)

    def test_insufficient_support_blocks_promotion(self) -> None:
        outcomes = [
            AcquisitionReplayOutcome(
                state_id="s1", participant_id="1",
                control_value=acquisition_value("absent", resolved=False),
                treatment_value=acquisition_value("lexical", resolved=False),
            )
        ]
        verdict = validate_acquisition_lesson(
            outcomes, excluded_state_ids=frozenset(), min_support=6,
        )
        self.assertFalse(verdict.promoted)
        self.assertIn("insufficient_support", verdict.reasons)


if __name__ == "__main__":
    unittest.main()
