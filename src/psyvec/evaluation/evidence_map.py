"""Tag-anchored evidence retrieval driven by ``topic_evidence_map.json``.

``psyvec.evaluation.evidence_retrieval`` mapped 27 DAIC-WOZ interviewer tags to
a PHQ-8 topic and reported a binary ``known``/``missing`` status. Mining every
tagged turn in the training split found 156 distinct tags, and the 129 unmapped
ones carry most of the coverage for the items that failed worst: with the
richer map, tag-anchored coverage for Psychomotor Changes rises from 7% to 62%
of training participants and Concentration Difficulties from 40% to 58%.

Two things this module adds beyond a bigger lookup table:

**Polarity.** A tag records how its question relates to the symptom. "What are
you most proud of" and "what are your best qualities" are the *reverse* pole of
low self-worth; retrieving them as evidence *for* it, as the flat map did, is
the mechanism behind that item's negative correlation with its reference
(r = -0.13 on train despite 60% coverage). ``reverse`` turns reach the scorer
labelled as such.

**One tag, several topics.** "What are you like when you don't sleep well"
speaks to fatigue and to sleep; "have you noticed any changes in your behavior
or thoughts lately" speaks to psychomotor change and to concentration. A flat
``tag -> topic`` dict cannot say that.

The map is built from the training split only and its key tags appear in 62%
(train), 77% (dev) and 62% (test) of participants, with 5-6 unseen tags per
held-out split — so it transfers rather than memorizing.

No model calls; stdlib only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from psyvec.evaluation.evidence_tiers import (
    EvidenceRelation,
    PairedTurn,
    RetrievedTurn,
    TieredEvidence,
    merge_retrieved,
    overall_tier,
    search_paired_turns,
)

__all__ = [
    "CROSS_CUTTING",
    "EvidenceMap",
    "TagRelation",
    "load_evidence_map",
    "load_topic_queries",
    "retrieve_topic_evidence",
]

#: Sentinel topic for questions probing mood/history rather than one symptom.
CROSS_CUTTING = "CROSS_CUTTING"

_VALID_RELATIONS: frozenset[str] = frozenset({"probe", "reverse", "context"})
_VALID_STRENGTHS: frozenset[str] = frozenset({"direct", "indirect"})


@dataclass(frozen=True, slots=True)
class TagRelation:
    """One (tag -> topic) edge with its polarity and directness."""

    topic: str
    relation: EvidenceRelation
    strength: str

    @property
    def tier(self) -> str:
        """Retrieval tier this edge produces: a direct probe outranks the rest."""
        return "direct" if self.strength == "direct" else "lexical"


@dataclass(frozen=True, slots=True)
class EvidenceMap:
    """Interviewer-tag lookup with polarity, loaded from a versioned config."""

    edges: Mapping[str, tuple[TagRelation, ...]]

    def edges_for_topic(self, tag: str, topic: str) -> tuple[TagRelation, ...]:
        return tuple(edge for edge in self.edges.get(tag, ()) if edge.topic == topic)

    def context_edges(self, tag: str) -> tuple[TagRelation, ...]:
        return tuple(
            edge for edge in self.edges.get(tag, ()) if edge.topic == CROSS_CUTTING
        )


def load_evidence_map(path: Path, *, valid_topics: Sequence[str]) -> EvidenceMap:
    """Load the map, rejecting any unknown topic, relation or strength.

    Fails closed on a typo rather than silently dropping an edge, matching the
    strict-parsing convention in :mod:`psyvec.evaluation.response_parsing`: a
    silently dropped edge would show up only as unexplained coverage loss.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(valid_topics) | {CROSS_CUTTING}
    edges: dict[str, tuple[TagRelation, ...]] = {}
    for tag, entries in raw.get("tags", {}).items():
        if not isinstance(entries, list):
            raise ValueError(f"topic_evidence_map tag {tag!r} must map to a list")
        built: list[TagRelation] = []
        for entry in entries:
            topic = str(entry.get("topic", ""))
            relation = str(entry.get("relation", ""))
            strength = str(entry.get("strength", ""))
            if topic not in allowed:
                raise ValueError(f"tag {tag!r} maps to unknown topic {topic!r}")
            if relation not in _VALID_RELATIONS:
                raise ValueError(f"tag {tag!r} has unknown relation {relation!r}")
            if strength not in _VALID_STRENGTHS:
                raise ValueError(f"tag {tag!r} has unknown strength {strength!r}")
            built.append(
                TagRelation(
                    topic=topic,
                    relation=relation,  # type: ignore[arg-type]
                    strength=strength,
                )
            )
        edges[tag] = tuple(built)
    if not edges:
        raise ValueError(f"{path} contains no tag mappings")
    return EvidenceMap(edges=edges)


def load_topic_queries(
    path: Path, *, valid_topics: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Load per-topic seed retrieval queries, rejecting unknown topics."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(valid_topics)
    queries: dict[str, tuple[str, ...]] = {}
    for topic, entries in raw.items():
        if topic == "_meta":
            continue
        if topic not in allowed:
            raise ValueError(f"topic_queries has unknown topic {topic!r}")
        queries[topic] = tuple(str(entry) for entry in entries)
    missing = allowed - set(queries)
    if missing:
        raise ValueError(f"topic_queries is missing topics: {sorted(missing)}")
    return queries


def _tag_anchored(
    turns: Sequence[PairedTurn], topic: str, evidence_map: EvidenceMap
) -> tuple[list[RetrievedTurn], list[RetrievedTurn]]:
    """Split tag-anchored turns into ``(topic_specific, cross_cutting)``."""
    specific: list[RetrievedTurn] = []
    context: list[RetrievedTurn] = []
    for turn in turns:
        if turn.tag is None:
            continue
        for edge in evidence_map.edges_for_topic(turn.tag, topic):
            specific.append(
                RetrievedTurn(
                    turn=turn,
                    # A direct probe ranks above an indirect one, and a probe
                    # above a reverse-polarity turn of the same directness.
                    score=3.0 if edge.strength == "direct" else 2.0,
                    tier="direct" if edge.strength == "direct" else "lexical",
                    relation=edge.relation,
                )
            )
        for _edge in evidence_map.context_edges(turn.tag):
            context.append(
                RetrievedTurn(
                    turn=turn, score=0.5, tier="contextual", relation="context"
                )
            )
    specific.sort(key=lambda item: (-item.score, item.turn.turn_id))
    return specific, context


def retrieve_topic_evidence(
    turns: Sequence[PairedTurn],
    topic: str,
    evidence_map: EvidenceMap,
    seed_queries: Sequence[str],
    *,
    limit: int = 8,
    context_limit: int = 3,
) -> TieredEvidence:
    """First-pass evidence for one topic: tag anchors, then lexical, then context.

    Context turns are appended only when nothing topic-specific was found, so a
    participant with real disclosures is not diluted by generic mood history —
    but a participant with none still reaches the scorer with something to read
    and an explicit label saying it is background, not a symptom report.
    """
    anchored, context = _tag_anchored(turns, topic, evidence_map)
    lexical = search_paired_turns(
        turns,
        seed_queries,
        top_k=limit,
        exclude_turn_ids=frozenset(item.turn.turn_id for item in anchored),
        tier="lexical",
        relation="probe",
    )
    specific = merge_retrieved(anchored, lexical, limit=limit)
    if specific:
        selected = specific
    else:
        seen: set[str] = set()
        deduped: list[RetrievedTurn] = []
        for item in context:
            if item.turn.turn_id in seen:
                continue
            seen.add(item.turn.turn_id)
            deduped.append(item)
        selected = tuple(deduped[:context_limit])

    return TieredEvidence(
        topic=topic, tier=overall_tier(selected), turns=selected, queries_issued=()
    )
