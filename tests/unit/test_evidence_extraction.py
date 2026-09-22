"""Evidence Agent parsing: grounded claims survive, fabricated citations don't."""

from __future__ import annotations

import json
import unittest

from psyvec.evaluation.evidence_extraction import (
    build_evidence_extraction_prompt,
    parse_evidence_extraction,
)


class BuildPromptTest(unittest.TestCase):
    def test_prompt_names_the_topic_and_carries_the_excerpts(self) -> None:
        prompt = build_evidence_extraction_prompt("Sleep Problems", "EVIDENCE: ...")
        self.assertIn("Sleep Problems", prompt)
        self.assertIn("EVIDENCE: ...", prompt)
        self.assertIn("REVERSE POLARITY", prompt)


class ParseEvidenceExtractionTest(unittest.TestCase):
    def test_a_grounded_claim_is_kept(self) -> None:
        raw = json.dumps({
            "claims": [
                {
                    "turn_ids": ["turn-1"], "quote": "i wake up most nights",
                    "polarity": "supports", "frequency_basis": "ongoing_state",
                    "confidence": 0.9,
                }
            ]
        })
        records = parse_evidence_extraction(
            raw, participant_id="999", topic="Sleep Problems", round_index=0,
            turn_relations={"turn-1": "probe"},
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].polarity, "supports")
        self.assertEqual(records[0].relation, "probe")
        self.assertEqual(records[0].round_index, 0)

    def test_a_claim_citing_an_unoffered_turn_id_is_dropped(self) -> None:
        """Anti-fabrication: a turn id never offered this round cannot be cited."""
        raw = json.dumps({
            "claims": [
                {
                    "turn_ids": ["turn-99"], "quote": "invented",
                    "polarity": "supports", "frequency_basis": "ongoing_state",
                    "confidence": 0.9,
                }
            ]
        })
        records = parse_evidence_extraction(
            raw, participant_id="999", topic="Sleep Problems", round_index=0,
            turn_relations={"turn-1": "probe"},
        )
        self.assertEqual(records, ())

    def test_a_claim_missing_polarity_or_quote_is_dropped(self) -> None:
        raw = json.dumps({"claims": [{"turn_ids": ["turn-1"], "quote": ""}]})
        records = parse_evidence_extraction(
            raw, participant_id="999", topic="Sleep Problems", round_index=0,
            turn_relations={"turn-1": "probe"},
        )
        self.assertEqual(records, ())

    def test_no_json_object_yields_no_records(self) -> None:
        records = parse_evidence_extraction(
            "not json at all", participant_id="999", topic="Sleep Problems",
            round_index=0, turn_relations={"turn-1": "probe"},
        )
        self.assertEqual(records, ())

    def test_relation_is_the_strongest_among_cited_turns(self) -> None:
        raw = json.dumps({
            "claims": [
                {
                    "turn_ids": ["turn-1", "turn-2"], "quote": "a b",
                    "polarity": "supports", "frequency_basis": "ongoing_state",
                    "confidence": 0.5,
                }
            ]
        })
        records = parse_evidence_extraction(
            raw, participant_id="999", topic="Sleep Problems", round_index=0,
            turn_relations={"turn-1": "context", "turn-2": "probe"},
        )
        self.assertEqual(records[0].relation, "probe")

    def test_confidence_is_clamped_to_unit_interval(self) -> None:
        raw = json.dumps({
            "claims": [
                {
                    "turn_ids": ["turn-1"], "quote": "x", "polarity": "supports",
                    "frequency_basis": "none", "confidence": 5.0,
                }
            ]
        })
        records = parse_evidence_extraction(
            raw, participant_id="999", topic="Sleep Problems", round_index=0,
            turn_relations={"turn-1": "probe"},
        )
        self.assertEqual(records[0].confidence, 1.0)

    def test_two_disagreeing_claims_both_survive(self) -> None:
        """The whole point: don't merge disagreement into one paraphrase."""
        raw = json.dumps({
            "claims": [
                {"turn_ids": ["turn-1"], "quote": "i sleep fine",
                 "polarity": "refutes", "frequency_basis": "none",
                 "confidence": 0.7},
                {"turn_ids": ["turn-2"], "quote": "i wake up a lot",
                 "polarity": "supports", "frequency_basis": "ongoing_state",
                 "confidence": 0.7},
            ]
        })
        records = parse_evidence_extraction(
            raw, participant_id="999", topic="Sleep Problems", round_index=0,
            turn_relations={"turn-1": "probe", "turn-2": "probe"},
        )
        self.assertEqual(len(records), 2)
        polarities = {record.polarity for record in records}
        self.assertEqual(polarities, {"refutes", "supports"})


if __name__ == "__main__":
    unittest.main()
