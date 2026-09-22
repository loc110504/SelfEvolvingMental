"""Patient Evidence Memory never overwrites; contradictions are flagged."""

from __future__ import annotations

import unittest

from psyvec.memory.patient_evidence import (
    EvidenceRecord,
    PatientEvidenceMemory,
    add_evidence,
    detect_contradictions,
    make_record_id,
)


def make_record(
    topic: str = "Sleep Problems",
    *,
    polarity: str = "supports",
    frequency_basis: str = "ongoing_state",
    round_index: int = 0,
    turn_ids: tuple[str, ...] = ("turn-1",),
    quote: str = "i wake up most nights",
) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=make_record_id("999", topic, turn_ids, quote),
        topic=topic,
        turn_ids=turn_ids,
        quote=quote,
        polarity=polarity,  # type: ignore[arg-type]
        frequency_basis=frequency_basis,  # type: ignore[arg-type]
        relation="probe",
        confidence=0.8,
        round_index=round_index,
    )


class AddEvidenceTest(unittest.TestCase):
    def test_records_accumulate_rather_than_overwrite(self) -> None:
        memory = PatientEvidenceMemory(participant_id="999")
        denial = make_record(
            quote="i sleep fine", polarity="refutes", frequency_basis="none"
        )
        memory = add_evidence(memory, [denial])
        memory = add_evidence(memory, [make_record(quote="i wake up most nights")])
        self.assertEqual(len(memory.records), 2)
        quotes = {record.quote for record in memory.records}
        self.assertEqual(quotes, {"i sleep fine", "i wake up most nights"})

    def test_a_duplicate_record_id_is_not_appended_twice(self) -> None:
        memory = PatientEvidenceMemory(participant_id="999")
        record = make_record()
        memory = add_evidence(memory, [record])
        memory = add_evidence(memory, [record])
        self.assertEqual(len(memory.records), 1)

    def test_empty_batch_is_a_no_op(self) -> None:
        memory = PatientEvidenceMemory(participant_id="999")
        self.assertIs(add_evidence(memory, []), memory)


class ContradictionDetectionTest(unittest.TestCase):
    def test_the_sleep_example_is_flagged(self) -> None:
        """'I sleep fine' at turn 7, then 'I wake up most nights' at turn 14."""
        early_denial = make_record(
            quote="i sleep fine", polarity="refutes", frequency_basis="none",
            round_index=0, turn_ids=("turn-7",),
        )
        later_report = make_record(
            quote="i wake up most nights", polarity="supports",
            frequency_basis="ongoing_state", round_index=1, turn_ids=("turn-14",),
        )
        edges = detect_contradictions([early_denial, later_report])
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge.topic, "Sleep Problems")
        self.assertEqual(edge.earlier_record_id, early_denial.record_id)
        both_ids = (early_denial.record_id, later_report.record_id)
        self.assertIn(edge.record_id_a, both_ids)
        self.assertIn(edge.record_id_b, both_ids)

    def test_refinement_is_not_a_contradiction(self) -> None:
        """A single episode and an ongoing state of the same polarity refine."""
        single = make_record(
            quote="one bad night last week", frequency_basis="single_episode"
        )
        ongoing = make_record(
            quote="actually most nights now", frequency_basis="ongoing_state"
        )
        self.assertEqual(detect_contradictions([single, ongoing]), ())

    def test_two_uncertain_records_never_conflict(self) -> None:
        first = make_record(quote="a", polarity="uncertain", frequency_basis="none")
        second = make_record(quote="b", polarity="refutes", frequency_basis="none")
        self.assertEqual(detect_contradictions([first, second]), ())

    def test_different_topics_are_never_compared(self) -> None:
        sleep_denial = make_record(
            topic="Sleep Problems", quote="a", polarity="refutes",
            frequency_basis="none",
        )
        mood_report = make_record(
            topic="Depressed Mood", quote="b", polarity="supports",
            frequency_basis="ongoing_state",
        )
        self.assertEqual(detect_contradictions([sleep_denial, mood_report]), ())

    def test_a_denial_before_a_denial_is_not_a_contradiction(self) -> None:
        first = make_record(quote="a", polarity="refutes", frequency_basis="none")
        second = make_record(quote="b", polarity="refutes", frequency_basis="none")
        self.assertEqual(detect_contradictions([first, second]), ())

    def test_add_evidence_recomputes_contradictions_on_new_records(self) -> None:
        memory = PatientEvidenceMemory(participant_id="999")
        denial = make_record(
            quote="i sleep fine", polarity="refutes", frequency_basis="none",
            round_index=0, turn_ids=("turn-7",),
        )
        memory = add_evidence(memory, [denial])
        self.assertEqual(memory.contradictions, ())
        report = make_record(
            quote="i wake up most nights", round_index=1, turn_ids=("turn-14",)
        )
        memory = add_evidence(memory, [report])
        self.assertTrue(memory.has_contradiction("Sleep Problems"))
        self.assertEqual(len(memory.contradictions_for("Sleep Problems")), 1)


class RecordIdTest(unittest.TestCase):
    def test_record_id_is_deterministic(self) -> None:
        first = make_record_id("999", "Sleep Problems", ("turn-1",), "i sleep fine")
        second = make_record_id("999", "Sleep Problems", ("turn-1",), "i sleep fine")
        self.assertEqual(first, second)

    def test_record_id_differs_on_quote(self) -> None:
        first = make_record_id("999", "Sleep Problems", ("turn-1",), "i sleep fine")
        second = make_record_id(
            "999", "Sleep Problems", ("turn-1",), "i wake up a lot"
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
