"""The Evidence Agent: turns rendered excerpts into atomic evidence records.

Every other role in :mod:`psyvec.evaluation.item_scoring` reads a topic's
excerpts and answers one question about the whole of them (sufficient? what
score?). This role reads the same excerpts and instead pulls out every
distinct, independently-sourced claim they contain, so the acquisition loop
can build :class:`psyvec.memory.patient_evidence.PatientEvidenceMemory`
incrementally, and so contradictory claims are preserved as separate records
rather than smoothed into one paraphrase before anything downstream sees them.

The prompt asks for *every* claim relevant to the topic, explicitly including
ones that disagree with each other, because the failure mode this role exists
to avoid is exactly the one AgentMental's memory update has: given "I sleep
fine" and, ten turns later, "actually I wake up most nights", a summarizer
asked for one paraphrase will average them into "some sleep difficulty" and
neither claim survives to be checked against the other. Asked for a list, a
model producing two records with opposite polarity is doing the right thing,
not failing to converge on one answer.

Reverse-polarity excerpts (the question asked about the opposite of the
symptom) are pre-labelled in the excerpt text by
:func:`psyvec.evaluation.evidence_tiers.format_tiered_evidence`; the
extraction prompt repeats the inversion rule explicitly, because "polarity"
here means the symptom's polarity, not the literal sentiment of the reply.

Prompt construction and parsing only — no model calls, stdlib only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from psyvec.evaluation.item_scoring import ITEM_MANIFESTATIONS
from psyvec.evaluation.response_parsing import extract_json_objects, strip_reasoning
from psyvec.memory.patient_evidence import (
    EvidenceRecord,
    FrequencyBasis,
    Polarity,
    Relation,
    make_record_id,
)

__all__ = [
    "EVIDENCE_AGENT_SYSTEM_PROMPT",
    "EVIDENCE_EXTRACTION_JSON_SCHEMA",
    "ExtractedClaim",
    "build_evidence_extraction_prompt",
    "parse_evidence_extraction",
]

EVIDENCE_AGENT_SYSTEM_PROMPT = (
    "You are a clinical evidence extractor. From interview excerpts about one "
    "PHQ-8 topic, you pull out the individual, independent claims they "
    "contain about that topic. You never merge separate turns into one "
    "summary, you never invent anything the excerpts do not say, and you "
    "never resolve disagreement between two turns yourself — if the "
    "excerpts point in different directions, report every direction as its "
    "own claim and let the rating step decide what to do with the "
    "disagreement. Output valid JSON only."
)

_POLARITY_VALUES = ("supports", "refutes", "uncertain")
_FREQUENCY_VALUES = (
    "stated_days", "frequency_word", "ongoing_state", "single_episode", "none",
)

EVIDENCE_EXTRACTION_JSON_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_ids": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                    "polarity": {"type": "string", "enum": list(_POLARITY_VALUES)},
                    "frequency_basis": {
                        "type": "string", "enum": list(_FREQUENCY_VALUES),
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": [
                    "turn_ids", "quote", "polarity", "frequency_basis", "confidence",
                ],
            },
        },
    },
    "required": ["claims"],
}


def build_evidence_extraction_prompt(topic: str, evidence_text: str) -> str:
    """Ask for every atomic claim these excerpts carry about ``topic``."""
    manifestations = ITEM_MANIFESTATIONS.get(topic, "")
    return f"""## Topic
{topic}

What this topic covers in ordinary speech:
{manifestations}

## Interview excerpts
{evidence_text}

## Task
List every distinct claim in these excerpts that bears on this topic. A claim
is one turn's worth of information, not the whole excerpt block summarized.

For each claim set:
  - "turn_ids": the id(s) in [brackets] the claim came from.
  - "quote": the verbatim span you are citing.
  - "polarity": "supports" if it reports the symptom occurring, "refutes" if
    it denies the symptom, "uncertain" if it does not settle presence either
    way. IMPORTANT: for excerpts marked "REVERSE POLARITY" (asking about the
    OPPOSITE of this symptom), judge the SYMPTOM's polarity, not the literal
    answer — a specific, engaged answer there is "refutes", a flat or empty
    one is "supports".
  - "frequency_basis": "stated_days" (explicit count/fraction of days),
    "frequency_word" ("always"/"most days"/"sometimes"/"rarely"),
    "ongoing_state" (described as how things are now, no count),
    "single_episode" (one recalled occasion, not a pattern), or "none" (no
    frequency information, e.g. a denial or an unrelated background remark).
  - "confidence": how directly this claim addresses the topic, 0.0 to 1.0 —
    a direct probe answer is high, background/context is low.

If nothing in the excerpts bears on this topic, return an empty list.
Never invent a claim, a turn id, or a quote the excerpts do not contain.

## Output
{{"claims": [{{"turn_ids": ["turn-12"], "quote": "...", "polarity": "supports",
  "frequency_basis": "ongoing_state", "confidence": 0.8}}]}}"""


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    """One parsed claim, before it is resolved against known turn citations."""

    turn_ids: tuple[str, ...]
    quote: str
    polarity: Polarity
    frequency_basis: FrequencyBasis
    confidence: float


def _last_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = strip_reasoning(raw_text) or raw_text
    for span in reversed(extract_json_objects(cleaned)):
        try:
            loaded = json.loads(span)
        except (TypeError, ValueError):
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _as_polarity(value: object) -> Polarity | None:
    if isinstance(value, str) and value.strip().lower() in _POLARITY_VALUES:
        return value.strip().lower()  # type: ignore[return-value]
    return None


def _as_frequency_basis(value: object) -> FrequencyBasis:
    if isinstance(value, str) and value.strip().lower() in _FREQUENCY_VALUES:
        return value.strip().lower()  # type: ignore[return-value]
    return "none"


def _as_confidence(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.5


def _parse_claims(raw_text: str) -> tuple[ExtractedClaim, ...]:
    payload = _last_object(raw_text)
    if payload is None:
        return ()
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return ()
    claims: list[ExtractedClaim] = []
    for entry in raw_claims:
        if not isinstance(entry, dict):
            continue
        polarity = _as_polarity(entry.get("polarity"))
        quote = str(entry.get("quote") or "").strip()
        if polarity is None or not quote:
            continue
        raw_ids = entry.get("turn_ids")
        turn_ids = (
            tuple(str(item) for item in raw_ids if isinstance(item, (str, int)))
            if isinstance(raw_ids, list)
            else ()
        )
        claims.append(
            ExtractedClaim(
                turn_ids=turn_ids,
                quote=quote,
                polarity=polarity,
                frequency_basis=_as_frequency_basis(entry.get("frequency_basis")),
                confidence=_as_confidence(entry.get("confidence")),
            )
        )
    return tuple(claims)


def parse_evidence_extraction(
    raw_text: str,
    *,
    participant_id: str,
    topic: str,
    round_index: int,
    turn_relations: Mapping[str, Relation],
) -> tuple[EvidenceRecord, ...]:
    """Turn a raw extraction reply into grounded :class:`EvidenceRecord`\\ s.

    ``turn_relations`` maps every turn id that was actually offered this round
    to its retrieval ``relation`` (probe/reverse/context); a claim citing a
    turn id outside that map is dropped rather than trusted; this is the same
    anti-fabrication discipline as
    :func:`psyvec.evaluation.item_scoring.quote_is_grounded`, applied at
    extraction time instead of scoring time. A claim's relation is the
    strongest relation among its cited turns (probe > reverse > context),
    matching how ``format_tiered_evidence`` orders excerpt blocks.
    """
    _RELATION_RANK: Mapping[Relation, int] = {"probe": 2, "reverse": 1, "context": 0}
    records: list[EvidenceRecord] = []
    for claim in _parse_claims(raw_text):
        known_ids = tuple(
            turn_id for turn_id in claim.turn_ids if turn_id in turn_relations
        )
        if not known_ids:
            continue
        relation = max(
            (turn_relations[turn_id] for turn_id in known_ids),
            key=lambda rel: _RELATION_RANK[rel],
        )
        records.append(
            EvidenceRecord(
                record_id=make_record_id(participant_id, topic, known_ids, claim.quote),
                topic=topic,
                turn_ids=known_ids,
                quote=claim.quote,
                polarity=claim.polarity,
                frequency_basis=claim.frequency_basis,
                relation=relation,
                confidence=claim.confidence,
                round_index=round_index,
            )
        )
    return tuple(records)
