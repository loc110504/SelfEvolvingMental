"""The shipped evidence map must load, cover every topic, and keep polarity."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from psyvec.evaluation.evidence_map import (
    CROSS_CUTTING,
    load_evidence_map,
    load_topic_queries,
    retrieve_topic_evidence,
)
from psyvec.evaluation.evidence_tiers import build_paired_turns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCALES = PROJECT_ROOT / "configs" / "scales"
TOPICS = list(json.loads((SCALES / "PHQ-8.json").read_text(encoding="utf-8")))

TRANSCRIPT = [
    {"roleName": "Ellie", "content": "easy_sleep (how easy is it to sleep)"},
    {"roleName": "Participant", "content": "i lay awake for hours every night"},
    {"roleName": "Ellie", "content": "Ellie17Dec2012_08 (what are you most proud of)"},
    {"roleName": "Participant", "content": "my kids and finishing my degree"},
    {
        "roleName": "Ellie",
        "content": "depression_diagnosed (diagnosed with depression)",
    },
    {"roleName": "Participant", "content": "yes about three years ago"},
    {"roleName": "Ellie", "content": "where are you from"},
    {"roleName": "Participant", "content": "los angeles"},
]


class LoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence_map = load_evidence_map(
            SCALES / "topic_evidence_map.json", valid_topics=TOPICS
        )

    def test_the_shipped_map_loads(self) -> None:
        self.assertGreater(len(self.evidence_map.edges), 27)

    def test_a_tag_may_serve_several_topics(self) -> None:
        # "what are you like when you don't sleep well" speaks to both.
        topics = {edge.topic for edge in self.evidence_map.edges["sleep_affects"]}
        self.assertEqual(topics, {"Fatigue or Low Energy", "Sleep Problems"})

    def test_a_pride_question_is_mapped_as_reverse_polarity(self) -> None:
        """Retrieving this as evidence FOR low self-worth is the v2 bug."""
        edges = self.evidence_map.edges_for_topic(
            "Ellie17Dec2012_08", "Low Self-Worth"
        )
        self.assertEqual([edge.relation for edge in edges], ["reverse"])

    def test_diagnosis_history_is_cross_cutting_not_an_item(self) -> None:
        edges = self.evidence_map.context_edges("depression_diagnosed")
        self.assertTrue(edges)
        self.assertEqual(edges[0].topic, CROSS_CUTTING)

    def test_an_unknown_topic_is_rejected_rather_than_dropped(self) -> None:
        with self.assertRaises(ValueError):
            load_evidence_map(
                SCALES / "topic_evidence_map.json", valid_topics=["Sleep Problems"]
            )

    def test_shipped_queries_cover_every_topic(self) -> None:
        queries = load_topic_queries(
            SCALES / "topic_queries.json", valid_topics=TOPICS
        )
        self.assertEqual(set(queries), set(TOPICS))
        self.assertTrue(all(len(entries) >= 8 for entries in queries.values()))

    def test_a_query_file_missing_a_topic_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_topic_queries(
                SCALES / "topic_queries.json", valid_topics=[*TOPICS, "Invented Topic"]
            )


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence_map = load_evidence_map(
            SCALES / "topic_evidence_map.json", valid_topics=TOPICS
        )
        self.queries = load_topic_queries(
            SCALES / "topic_queries.json", valid_topics=TOPICS
        )
        self.turns = build_paired_turns(TRANSCRIPT)

    def _retrieve(self, topic: str) -> object:
        return retrieve_topic_evidence(
            self.turns, topic, self.evidence_map, self.queries[topic]
        )

    def test_a_direct_probe_yields_the_direct_tier(self) -> None:
        evidence = self._retrieve("Sleep Problems")
        self.assertEqual(evidence.tier, "direct")  # type: ignore[attr-defined]

    def test_reverse_polarity_survives_into_the_retrieved_turn(self) -> None:
        evidence = self._retrieve("Low Self-Worth")
        relations = {item.relation for item in evidence.turns}  # type: ignore[attr-defined]
        self.assertIn("reverse", relations)

    def test_background_is_used_only_when_nothing_specific_was_found(self) -> None:
        specific = self._retrieve("Sleep Problems")
        fallback = self._retrieve("Appetite or Weight Changes")
        self.assertNotIn(
            "context", {item.relation for item in specific.turns}  # type: ignore[attr-defined]
        )
        self.assertEqual(fallback.tier, "contextual")  # type: ignore[attr-defined]

    def test_an_uncovered_topic_with_no_background_is_absent(self) -> None:
        turns = build_paired_turns(TRANSCRIPT[-2:])
        evidence = retrieve_topic_evidence(
            turns,
            "Psychomotor Changes",
            self.evidence_map,
            self.queries["Psychomotor Changes"],
        )
        self.assertEqual(evidence.tier, "absent")
        self.assertEqual(evidence.turns, ())


if __name__ == "__main__":
    unittest.main()
