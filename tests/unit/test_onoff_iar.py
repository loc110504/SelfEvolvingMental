from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation import (  # noqa: E402
    OnOffArm,
    OnOffResult,
    internalization_verdict,
    internalized_action_rate,
)
from psyvec.policy import PromotionError  # noqa: E402


def arm(quality: float, state_ids: tuple[str, ...] = ("s1",)) -> OnOffArm:
    return OnOffArm(quality=quality, state_ids=state_ids, seeds=(7,))


class OnOffIarTests(unittest.TestCase):
    def test_mismatched_held_out_state_ids_or_seeds_are_refused(self) -> None:
        with self.assertRaises(PromotionError):
            OnOffResult(arm(1.0), arm(1.0, ("s2",)), arm(1.0), arm(1.0))
        with self.assertRaises(PromotionError):
            OnOffResult(
                arm(1.0), arm(1.0), arm(1.0), OnOffArm(1.0, ("s1",), (8,))
            )

    def test_on_and_off_gains_are_distinct(self) -> None:
        result = OnOffResult(arm(1.0), arm(2.0), arm(4.0), arm(2.5))

        self.assertEqual(result.decomposition.combined_on_gain, 3.0)
        self.assertEqual(result.decomposition.parametric_off_gain, 0.5)

    def test_iar_uses_each_states_own_positive_actions(self) -> None:
        iar = internalized_action_rate(
            ("s1", "s2"), {"s1": "ask", "s2": "ask"},
            {"s1": {"ask"}, "s2": {"reflect"}},
        )

        self.assertEqual(iar, 0.5)

    def test_zero_triggered_states_refuses_iar(self) -> None:
        with self.assertRaises(PromotionError):
            internalized_action_rate((), {}, {})

    def test_internalization_requires_both_improvements(self) -> None:
        refused = internalization_verdict(
            initial_off_quality=1.0,
            evolved_off_quality=2.0,
            initial_iar=0.5,
            evolved_iar=0.5,
        )
        accepted = internalization_verdict(
            initial_off_quality=1.0,
            evolved_off_quality=2.0,
            initial_iar=0.5,
            evolved_iar=0.6,
        )

        self.assertFalse(refused.internalized)
        self.assertEqual(refused.reasons, ("iar_not_improved",))
        self.assertTrue(accepted.internalized)
        self.assertEqual(accepted.reasons, ())


if __name__ == "__main__":
    unittest.main()
