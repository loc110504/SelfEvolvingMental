"""Phase 8 P8-4: matched ablation budgets and seed robustness, no verdicts."""

from __future__ import annotations

import unittest

from psyvec.research.budget import (
    AblationArm,
    ArmOutcome,
    BudgetError,
    assert_budget_respected,
    assert_matched_budget,
    measure_ablation_effects,
    measure_seed_robustness,
)

SEEDS = (11, 12, 13)


def make_arm(arm_id: str, ablated: str | None, **overrides: object) -> AblationArm:
    fields: dict[str, object] = {
        "arm_id": arm_id,
        "ablated_component": ablated,
        "question_budget": 8,
        "model_call_budget": 16,
        "seeds": SEEDS,
    }
    fields.update(overrides)
    return AblationArm(**fields)  # type: ignore[arg-type]


def outcomes(arm_id: str, values: tuple[float, ...]) -> tuple[ArmOutcome, ...]:
    return tuple(
        ArmOutcome(
            arm_id=arm_id,
            seed=seed,
            value=value,
            questions_asked=8,
            model_calls=16,
        )
        for seed, value in zip(SEEDS, values, strict=True)
    )


class MatchedBudgetTests(unittest.TestCase):
    def test_the_reference_arm_is_the_unablated_one(self) -> None:
        arms = (make_arm("full", None), make_arm("no-lessons", "lesson_memory"))

        self.assertEqual(assert_matched_budget(arms).arm_id, "full")

    def test_two_reference_arms_are_refused(self) -> None:
        with self.assertRaises(BudgetError):
            assert_matched_budget((make_arm("a", None), make_arm("b", None)))

    def test_a_differing_budget_is_a_hard_failure(self) -> None:
        arms = (
            make_arm("full", None),
            make_arm("no-lessons", "lesson_memory", model_call_budget=32),
        )

        with self.assertRaises(BudgetError) as caught:
            assert_matched_budget(arms)
        self.assertIn("model_call_budget", str(caught.exception))

    def test_a_differing_seed_set_is_a_hard_failure(self) -> None:
        arms = (
            make_arm("full", None),
            make_arm("no-lessons", "lesson_memory", seeds=(11, 12, 99)),
        )

        with self.assertRaises(BudgetError):
            assert_matched_budget(arms)

    def test_an_arm_needs_a_positive_budget_and_unique_seeds(self) -> None:
        with self.assertRaises(BudgetError):
            make_arm("full", None, question_budget=0)
        with self.assertRaises(BudgetError):
            make_arm("full", None, seeds=(1, 1))

    def test_overspending_an_arms_budget_is_refused(self) -> None:
        arm = make_arm("full", None)
        overspent = (
            ArmOutcome(
                arm_id="full",
                seed=11,
                value=1.0,
                questions_asked=9,
                model_calls=16,
            ),
        )

        with self.assertRaises(BudgetError) as caught:
            assert_budget_respected(arm, overspent)
        self.assertIn("question budget", str(caught.exception))


class RobustnessTests(unittest.TestCase):
    def test_seed_spread_is_reported_across_every_seed(self) -> None:
        arm = make_arm("full", None)

        robustness = measure_seed_robustness(arm, outcomes("full", (3.0, 4.0, 5.0)))

        self.assertEqual(robustness.seed_count, 3)
        self.assertEqual(robustness.mean, 4.0)
        self.assertEqual(robustness.minimum, 3.0)
        self.assertEqual(robustness.maximum, 5.0)
        self.assertEqual(robustness.spread, 2.0)

    def test_a_missing_seed_is_refused(self) -> None:
        arm = make_arm("full", None)
        partial = outcomes("full", (3.0, 4.0, 5.0))[:2]

        with self.assertRaises(BudgetError) as caught:
            measure_seed_robustness(arm, partial)
        self.assertIn("missing seeds", str(caught.exception))


class AblationEffectTests(unittest.TestCase):
    def test_each_ablation_is_measured_against_the_reference_per_seed(self) -> None:
        arms = (
            make_arm("full", None),
            make_arm("no-lessons", "lesson_memory"),
        )
        measured = outcomes("full", (2.0, 3.0, 4.0)) + outcomes(
            "no-lessons", (2.5, 4.0, 4.5)
        )

        effects = measure_ablation_effects(arms, measured)

        self.assertEqual(len(effects), 1)
        effect = effects[0]
        self.assertEqual(effect.ablated_component, "lesson_memory")
        self.assertEqual(effect.reference_arm_id, "full")
        self.assertEqual(
            effect.per_seed_differences, ((11, 0.5), (12, 1.0), (13, 0.5))
        )
        self.assertAlmostEqual(effect.mean_difference, 2.0 / 3.0)

    def test_an_outcome_for_an_unknown_arm_is_refused(self) -> None:
        arms = (make_arm("full", None), make_arm("no-lessons", "lesson_memory"))
        measured = (
            outcomes("full", (2.0, 3.0, 4.0))
            + outcomes("no-lessons", (2.0, 3.0, 4.0))
            + outcomes("ghost", (1.0, 1.0, 1.0))
        )

        with self.assertRaises(BudgetError):
            measure_ablation_effects(arms, measured)


if __name__ == "__main__":
    unittest.main()
