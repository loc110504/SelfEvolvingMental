"""The scheduler prioritizes unresolved, high-need topics under a shared budget."""

from __future__ import annotations

import unittest

from psyvec.interview.utility import (
    TopicAcquisitionState,
    experience_match,
    next_action,
    redundancy,
    topic_need,
)


def state(
    topic: str, tier: str = "absent", rounds_spent: int = 1,
    sufficient: bool | None = None, max_rounds_per_topic: int = 3,
) -> TopicAcquisitionState:
    return TopicAcquisitionState(
        topic=topic, tier=tier, rounds_spent=rounds_spent,  # type: ignore[arg-type]
        sufficient=sufficient, max_rounds_per_topic=max_rounds_per_topic,
    )


class NeedTest(unittest.TestCase):
    def test_absent_tier_has_maximum_need(self) -> None:
        self.assertEqual(topic_need(state("A", tier="absent")), 1.0)

    def test_direct_tier_has_zero_need(self) -> None:
        self.assertEqual(topic_need(state("A", tier="direct")), 0.0)

    def test_sufficient_is_zero_need_regardless_of_tier(self) -> None:
        self.assertEqual(topic_need(state("A", tier="absent", sufficient=True)), 0.0)

    def test_need_decreases_monotonically_with_tier_strength(self) -> None:
        tiers = ("absent", "contextual", "expanded", "lexical", "direct")
        needs = [topic_need(state("A", tier=t)) for t in tiers]
        self.assertEqual(needs, sorted(needs, reverse=True))


class RedundancyTest(unittest.TestCase):
    def test_redundancy_rises_with_rounds_spent(self) -> None:
        low = redundancy(state("A", rounds_spent=1, max_rounds_per_topic=4))
        high = redundancy(state("A", rounds_spent=3, max_rounds_per_topic=4))
        self.assertLess(low, high)

    def test_redundancy_is_capped_at_one(self) -> None:
        capped = state("A", rounds_spent=10, max_rounds_per_topic=3)
        self.assertEqual(redundancy(capped), 1.0)


class ExperienceMatchTest(unittest.TestCase):
    def test_matches_only_the_exact_scope(self) -> None:
        scopes = frozenset({"Sleep Problems|absent"})
        self.assertEqual(experience_match("Sleep Problems", "absent", scopes), 1.0)
        self.assertEqual(experience_match("Sleep Problems", "lexical", scopes), 0.0)
        self.assertEqual(experience_match("Depressed Mood", "absent", scopes), 0.0)


class ResolvedTest(unittest.TestCase):
    def test_sufficient_topic_is_resolved(self) -> None:
        resolved = state("A", sufficient=True, rounds_spent=1, max_rounds_per_topic=3)
        self.assertTrue(resolved.resolved)

    def test_topic_at_round_cap_is_resolved_even_if_not_sufficient(self) -> None:
        at_cap = state("A", sufficient=False, rounds_spent=3, max_rounds_per_topic=3)
        self.assertTrue(at_cap.resolved)

    def test_unjudged_topic_below_cap_is_not_resolved(self) -> None:
        below_cap = state("A", sufficient=None, rounds_spent=1, max_rounds_per_topic=3)
        self.assertFalse(below_cap.resolved)


class NextActionTest(unittest.TestCase):
    def test_picks_the_topic_with_the_weakest_evidence(self) -> None:
        states = [
            state("Sleep Problems", tier="direct", sufficient=True),
            state("Depressed Mood", tier="absent", sufficient=None),
        ]
        self.assertEqual(next_action(states), "Depressed Mood")

    def test_returns_none_when_every_topic_is_resolved(self) -> None:
        states = [state("A", tier="direct", sufficient=True)]
        self.assertIsNone(next_action(states))

    def test_resolved_topics_are_never_picked_even_with_a_matching_lesson(self) -> None:
        states = [
            state("A", tier="direct", sufficient=True),
            state("B", tier="absent", sufficient=None),
        ]
        # A matching lesson for the resolved topic must not resurrect it as a
        # candidate — protocol legality (resolved) always wins over utility.
        scopes = frozenset({"A|direct"})
        self.assertEqual(next_action(states, lesson_scopes=scopes), "B")

    def test_ties_keep_the_earlier_topic_in_input_order(self) -> None:
        states = [state("First", tier="absent"), state("Second", tier="absent")]
        self.assertEqual(next_action(states), "First")

    def test_a_matching_lesson_can_break_a_near_tie(self) -> None:
        states = [
            state("A", tier="expanded", rounds_spent=1, max_rounds_per_topic=4),
            state("B", tier="expanded", rounds_spent=1, max_rounds_per_topic=4),
        ]
        scopes = frozenset({"B|expanded"})
        self.assertEqual(next_action(states, lesson_scopes=scopes), "B")

    def test_a_budget_of_zero_candidates_is_handled(self) -> None:
        self.assertIsNone(next_action([]))


if __name__ == "__main__":
    unittest.main()
