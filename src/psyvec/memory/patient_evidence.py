"""Provenance-preserving, contradiction-aware per-participant evidence memory.

Every other piece of evidence handling in this project (``evaluation.
evidence_tiers``, ``evaluation.item_scoring``) works with evidence as a
rendered text block handed fresh to the scorer each time: correct for scoring,
but nothing survives between retrieval rounds except a flat string. Two things
the adaptive-interviewing method needs are not representable in that string:

* **Provenance.** Which atomic claim came from which turn, with what polarity
  and frequency basis, independent of the prose it was rendered into.
* **Contradiction, not overwrite.** AgentMental's tree memory overwrites a
  topic's ``summary`` field on new evidence (see
  ``docs/implementation/00_UPSTREAM_AGENTMENTAL_AUDIT.md``); a later "I
  actually wake up most nights" can silently replace an earlier "I sleep
  fine" with no trace that the two were ever in tension. This module never
  removes or edits a record — a conflicting claim becomes a new record plus an
  explicit :class:`ContradictionEdge`, so the tension itself is queryable and
  explainable rather than resolved by whichever call ran last.

Contradiction detection is a deterministic comparison of ``polarity`` and
``frequency_basis`` across a topic's records, not a model call: two records
for the same topic are in tension only when one flatly denies the symptom
(``refutes``) and another later record actually reports it present (any
non-``uncertain`` polarity of ``supports`` with a frequency basis that means
the symptom occurred). Same-polarity records that merely differ in how much of
the symptom they describe (a single recalled episode versus an ongoing state)
are evidence refinement, not contradiction, and are left alone.

Pure dataclasses and pure functions; no model calls, stdlib only, matching the
rest of ``psyvec.evaluation``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from psyvec.state.contracts import canonical_hash

__all__ = [
    "ContradictionEdge",
    "EvidenceRecord",
    "FrequencyBasis",
    "PatientEvidenceMemory",
    "Polarity",
    "add_evidence",
    "detect_contradictions",
    "make_record_id",
]

Polarity = Literal["supports", "refutes", "uncertain"]
FrequencyBasis = Literal[
    "stated_days", "frequency_word", "ongoing_state", "single_episode", "none"
]
Relation = Literal["probe", "reverse", "context"]

#: Frequency bases that mean "the symptom did occur", used to tell a genuine
#: presence report apart from a denial or an unsettled claim.
_PRESENT_FREQUENCY_BASES = frozenset(
    {"stated_days", "frequency_word", "ongoing_state", "single_episode"}
)


def make_record_id(
    participant_id: str, topic: str, turn_ids: Sequence[str], quote: str
) -> str:
    """Content-addressed record id.

    Deterministic on ``(participant, topic, turn_ids, quote)`` rather than
    random, unlike AgentMental's ``uuid4()`` statement nodes — the same
    evidence extracted twice (e.g. a topic revisited by the scheduler) yields
    the same id, so :func:`add_evidence` can de-duplicate by id instead of by
    string equality.
    """
    return canonical_hash(
        {
            "participant": participant_id,
            "topic": topic,
            "turn_ids": list(turn_ids),
            "quote": quote,
        }
    )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One atomic, independently-sourced claim about one PHQ-8 topic.

    ``round_index`` is the acquisition round the record was extracted in
    (0 = seed retrieval, 1+ = scheduler-driven expansion rounds), which is
    what lets :func:`detect_contradictions` report which claim came later.
    """

    record_id: str
    topic: str
    turn_ids: tuple[str, ...]
    quote: str
    polarity: Polarity
    frequency_basis: FrequencyBasis
    relation: Relation
    confidence: float
    round_index: int


@dataclass(frozen=True, slots=True)
class ContradictionEdge:
    """A flagged tension between two records of the same topic.

    Neither record is removed or altered; the tension is additional
    information, not a resolution. ``earlier_record_id`` is set only when the
    two records have distinct round indices, so a caller can render "you
    said X earlier, then Y later" rather than an unordered pair.
    """

    topic: str
    record_id_a: str
    record_id_b: str
    reason: str
    earlier_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class PatientEvidenceMemory:
    """Everything acquired for one participant so far. Append-only."""

    participant_id: str
    records: tuple[EvidenceRecord, ...] = ()
    contradictions: tuple[ContradictionEdge, ...] = ()

    def records_for(self, topic: str) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.topic == topic)

    def contradictions_for(self, topic: str) -> tuple[ContradictionEdge, ...]:
        return tuple(edge for edge in self.contradictions if edge.topic == topic)

    def has_contradiction(self, topic: str) -> bool:
        return any(edge.topic == topic for edge in self.contradictions)


def _reports_presence(record: EvidenceRecord) -> bool:
    """Whether this record, alone, asserts the symptom actually occurred."""
    if record.polarity != "supports":
        return False
    return record.frequency_basis in _PRESENT_FREQUENCY_BASES


def _conflict_reason(a: EvidenceRecord, b: EvidenceRecord) -> str | None:
    """A short reason string if ``a`` and ``b`` assert incompatible things."""
    if a.polarity == "uncertain" or b.polarity == "uncertain":
        return None
    if a.polarity == b.polarity:
        # Same polarity: differing detail (a single episode versus an ongoing
        # state) is refinement, not contradiction.
        return None
    if a.polarity == "refutes" and _reports_presence(b):
        return "a flat denial stands against a later report that it occurred"
    if b.polarity == "refutes" and _reports_presence(a):
        return "a flat denial stands against a later report that it occurred"
    return None


def detect_contradictions(
    records: Sequence[EvidenceRecord],
) -> tuple[ContradictionEdge, ...]:
    """Pairwise, deterministic, same-topic contradiction detection.

    Quadratic in a topic's record count, which stays single digits per
    participant (records accumulate one small batch per acquisition round),
    so there is no need for anything cleverer here.
    """
    by_topic: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        by_topic.setdefault(record.topic, []).append(record)

    edges: list[ContradictionEdge] = []
    for topic, group in by_topic.items():
        ordered = sorted(
            group, key=lambda record: (record.round_index, record.record_id)
        )
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                reason = _conflict_reason(first, second)
                if reason is None:
                    continue
                earlier = (
                    first.record_id
                    if first.round_index < second.round_index
                    else None
                )
                edges.append(
                    ContradictionEdge(
                        topic=topic,
                        record_id_a=first.record_id,
                        record_id_b=second.record_id,
                        reason=reason,
                        earlier_record_id=earlier,
                    )
                )
    return tuple(edges)


def add_evidence(
    memory: PatientEvidenceMemory, new_records: Sequence[EvidenceRecord]
) -> PatientEvidenceMemory:
    """Append new records and recompute contradictions. Never overwrites.

    A record whose id already exists (the same claim re-extracted, e.g. a
    topic the scheduler happens to revisit) is dropped as a duplicate rather
    than appended twice; every other record is kept, however much it
    conflicts with what is already there.
    """
    if not new_records:
        return memory
    existing_ids = {record.record_id for record in memory.records}
    deduped = tuple(
        record for record in new_records if record.record_id not in existing_ids
    )
    if not deduped:
        return memory
    all_records = memory.records + deduped
    return PatientEvidenceMemory(
        participant_id=memory.participant_id,
        records=all_records,
        contradictions=detect_contradictions(all_records),
    )
