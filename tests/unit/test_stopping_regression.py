from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.assessment import conduct_topic_interview, should_continue  # noqa: E402
from psyvec.state import (  # noqa: E402
    AssessmentState,
    DecisionEventEngine,
    ProvenanceRef,
    TopicState,
)


def state() -> AssessmentState:
    return AssessmentState(
        case_id="case-1",
        scale_version="PHQ-8",
        current_topic_id="sleep",
        topic_states=(TopicState(topic_id="sleep", rubric_version="v1"),),
    )


def engine() -> DecisionEventEngine:
    return DecisionEventEngine(
        ProvenanceRef("run", "sha", "config", "v1", "baseline", "offline")
    )


class StoppingRegressionTests(unittest.TestCase):
    def test_policy_matches_upstream_matrix(self) -> None:
        self.assertFalse(should_continue(0, 1))
        self.assertTrue(should_continue(1, 1))
        self.assertFalse(should_continue(1, 2))
        self.assertTrue(should_continue(2, 1))
        self.assertTrue(should_continue(2, 3))

    def test_necessity_two_asks_three_questions_and_emits_three_events(self) -> None:
        decision_engine = engine()
        after = conduct_topic_interview(state(), decision_engine, 2, lambda: "Q")

        self.assertEqual(len(after.dialogue_turn_ids), 3)
        self.assertEqual(len(decision_engine.events), 3)
        self.assertEqual(
            [event.decision_index for event in decision_engine.events], [0, 1, 2]
        )

    def test_necessity_one_stops_after_two_questions(self) -> None:
        after = conduct_topic_interview(state(), engine(), 1, lambda: "Q")

        self.assertEqual(len(after.dialogue_turn_ids), 2)

    def test_necessity_zero_stops_after_one_question(self) -> None:
        after = conduct_topic_interview(state(), engine(), 0, lambda: "Q")

        self.assertEqual(len(after.dialogue_turn_ids), 1)


if __name__ == "__main__":
    unittest.main()
