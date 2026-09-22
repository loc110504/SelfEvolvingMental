"""Query-driven, tiered evidence retrieval for the v3 assessment loop.

``psyvec.evaluation.evidence_retrieval`` retrieves once, with a fixed tag map
and keyword lexicon, and reports a binary ``known``/``missing`` status. The
v2 evaluation showed two consequences of that design:

* recall is the binding constraint — 4 of the 8 PHQ-8 topics have no DAIC-WOZ
  tag anchor at all, so ``Psychomotor Changes`` retrieved nothing for 93% of
  train participants and ``Appetite or Weight Changes`` for 77%; and
* ``missing`` was then consumed downstream as "symptom absent", which is a
  different claim entirely (on the missing subset the ground-truth item score
  is nonzero 55-78% of the time).

This module addresses the first point and makes the second expressible. It
provides:

1. **Paired turns.** The retrieval unit is the interviewer question together
   with the participant's reply, not the reply alone. "yeah about three hours"
   is uninterpretable on its own and invisible to a keyword scan; paired with
   "how many hours of sleep do you usually get" it is decisive evidence.
2. **Query-driven search** over every paired turn, scored with a BM25-style
   idf-weighted overlap. This is what lets a sufficiency loop issue *new*
   queries for a topic whose first-pass retrieval came up empty, instead of
   giving up after one fixed lexicon lookup.
3. **A graded tier** (``direct``/``lexical``/``expanded``/``contextual``/
   ``absent``) instead of a binary status, so a scorer can be told how much
   to trust what it was handed and an evaluation can report coverage by tier.

Nothing here calls a model. Stdlib only, matching the rest of ``psyvec``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "EvidenceRelation",
    "EvidenceTier",
    "PairedTurn",
    "RetrievedTurn",
    "TieredEvidence",
    "build_paired_turns",
    "format_tiered_evidence",
    "search_paired_turns",
    "tokenize",
]

#: How much of a topic's evidence the retriever was actually able to anchor.
#:
#: ``direct``      at least one turn elicited by an interviewer question that
#:                 maps to this exact PHQ-8 topic (tag anchor).
#: ``lexical``     matched only by the static topic keyword lexicon.
#: ``expanded``    matched only by a query the sufficiency loop generated.
#: ``contextual``  no topic-specific match; only general mood/history turns.
#: ``absent``      nothing at all was retrieved.
EvidenceTier = Literal["direct", "lexical", "expanded", "contextual", "absent"]

#: How the question that elicited a turn relates to the symptom being scored.
#:
#: ``probe``    asks about the symptom; a congruent answer supports a nonzero score.
#: ``reverse``  asks about the opposite pole (pride, strengths, recent enjoyment);
#:              a rich specific answer is evidence AGAINST, an empty one is FOR.
#: ``context``  general mood, diagnosis or treatment history; never sufficient alone.
EvidenceRelation = Literal["probe", "reverse", "context"]

#: Ordered weakest-to-strongest, so callers can compare tiers.
TIER_ORDER: tuple[EvidenceTier, ...] = (
    "absent",
    "contextual",
    "expanded",
    "lexical",
    "direct",
)

_TAG_RE = re.compile(r"^([A-Za-z0-9_]+)\s*\((.*)\)$", re.DOTALL)
_TOKEN_RE = re.compile(r"[a-z']+")

#: Words too common in interview speech to carry retrieval signal. Kept small
#: and explicit rather than pulling in an NLP dependency.
_STOPWORDS = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "do",
        "does", "did", "for", "from", "had", "has", "have", "he", "her", "him",
        "his", "how", "i", "if", "in", "is", "it", "its", "me", "my", "no",
        "not", "of", "on", "or", "our", "so", "than", "that", "the", "their",
        "them", "then", "there", "these", "they", "this", "to", "too", "us",
        "was", "we", "were", "what", "when", "where", "which", "who", "will",
        "with", "would", "you", "your", "yeah", "um", "uh", "mm", "hmm", "okay",
        "ok", "like", "just", "really", "know", "think", "thing", "things",
        "kind", "sort", "get", "got", "go", "going",
    )
)


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase alphabetic tokens with interview filler words removed."""
    return tuple(
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    )


@dataclass(frozen=True, slots=True)
class PairedTurn:
    """One interviewer prompt with the participant reply it elicited.

    ``turn_id`` names the *participant* turn, keeping citation ids compatible
    with :mod:`psyvec.evaluation.evidence_retrieval` so evidence from the
    two retrievers can be merged and cited in one namespace.
    """

    turn_id: str
    question: str
    answer: str
    tag: str | None
    span_ids: tuple[str, ...] = ()

    @property
    def searchable_text(self) -> str:
        return f"{self.question} {self.answer}"

    @property
    def citable_ids(self) -> tuple[str, ...]:
        """Every original transcript id folded into this turn."""
        return self.span_ids or (self.turn_id,)


@dataclass(frozen=True, slots=True)
class RetrievedTurn:
    """A paired turn selected for one topic, with why it was selected.

    ``relation`` records how the eliciting question relates to the symptom.
    Dropping it is what let v2 hand a scorer a proud, fluent answer to "what
    are you most proud of" as evidence *for* low self-worth, which is the
    mechanism behind that item's negative correlation with its reference.
    """

    turn: PairedTurn
    score: float
    tier: EvidenceTier
    relation: EvidenceRelation = "probe"
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TieredEvidence:
    """Everything retrieved for one topic, with its overall tier."""

    topic: str
    tier: EvidenceTier
    turns: tuple[RetrievedTurn, ...]
    queries_issued: tuple[str, ...] = ()

    @property
    def turn_ids(self) -> frozenset[str]:
        return frozenset(item.turn.turn_id for item in self.turns)

    @property
    def has_topic_specific_evidence(self) -> bool:
        """Whether any turn matched *this* topic rather than mood generally."""
        return any(item.tier != "contextual" for item in self.turns)


def _strip_tag(text: str) -> tuple[str, str | None]:
    """Split ``tag (utterance)`` into ``(utterance, tag)``; passthrough otherwise."""
    match = _TAG_RE.match(text.strip())
    if match is None:
        return text.strip(), None
    return match.group(2).strip(), match.group(1)


def build_paired_turns(
    real_interview: Sequence[Mapping[str, str]],
    *,
    participant_role: str = "Participant",
    merge_consecutive: bool = True,
) -> tuple[PairedTurn, ...]:
    """Pair every participant turn with the interviewer turn that preceded it.

    DAIC-WOZ splits one spoken answer across however many turns the segmenter
    produced, so the transcript for "what advice would you give yourself" can
    read ``um`` / ``don't think about going to law school`` / ``uh`` /
    ``get a head start studying music``. Scored as four separate fragments,
    three are content-free filler and the fourth is missing its context; the
    lexical retriever then either drops the disclosure or promotes ``um`` as
    evidence. ``merge_consecutive`` folds a run of participant turns sharing
    one eliciting question back into the single answer it was.

    ``turn_id`` stays the id of the first turn in the run and indexes into
    ``real_interview`` exactly as in :mod:`psyvec.evaluation.evidence_retrieval`,
    so citations remain interchangeable; ``span_ids`` records the whole run for
    a caller that needs to validate a citation against any of its parts.
    """
    paired: list[PairedTurn] = []
    pending_question = ""
    pending_tag: str | None = None
    open_run: list[tuple[str, str]] = []

    def flush() -> None:
        if not open_run:
            return
        ids = tuple(turn_id for turn_id, _ in open_run)
        answer = " ".join(text for _, text in open_run).strip()
        open_run.clear()
        if not answer:
            return
        paired.append(
            PairedTurn(
                turn_id=ids[0],
                question=pending_question,
                answer=answer,
                tag=pending_tag,
                span_ids=ids,
            )
        )

    for index, raw_turn in enumerate(real_interview):
        speaker = str(raw_turn.get("roleName", ""))
        content = str(raw_turn.get("content", "")).strip()
        if speaker != participant_role:
            flush()
            pending_question, pending_tag = _strip_tag(content)
            continue
        if not content:
            continue
        open_run.append((f"turn-{index}", content))
        if not merge_consecutive:
            flush()
    flush()
    return tuple(paired)


def _document_frequencies(turns: Sequence[PairedTurn]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for turn in turns:
        for token in set(tokenize(turn.searchable_text)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def _phrase_terms(query: str) -> tuple[tuple[str, ...], str]:
    """Return ``(tokens, normalized_phrase)`` for one query string."""
    tokens = tokenize(query)
    return tokens, " ".join(query.lower().split())


def search_paired_turns(
    turns: Sequence[PairedTurn],
    queries: Iterable[str],
    *,
    top_k: int = 4,
    exclude_turn_ids: frozenset[str] = frozenset(),
    min_score: float = 0.5,
    tier: EvidenceTier = "expanded",
    relation: EvidenceRelation = "probe",
) -> tuple[RetrievedTurn, ...]:
    """Rank paired turns against free-text ``queries``.

    Scoring is idf-weighted term overlap (the BM25 saturation term is dropped
    — interview turns are short enough that term frequency past the first
    occurrence carries little extra signal), with a bonus when a whole query
    phrase appears verbatim. ``min_score`` keeps a single incidental stopword-
    adjacent match from being promoted to evidence, which is what made the v2
    keyword lexicon retrieve hobby enthusiasm as anhedonia evidence.
    """
    turn_list = list(turns)
    if not turn_list:
        return ()
    frequencies = _document_frequencies(turn_list)
    total_documents = len(turn_list)

    prepared = [_phrase_terms(query) for query in queries]
    prepared = [(tokens, phrase) for tokens, phrase in prepared if tokens]
    if not prepared:
        return ()

    scored: list[RetrievedTurn] = []
    for turn in turn_list:
        if turn.turn_id in exclude_turn_ids:
            continue
        text = turn.searchable_text.lower()
        turn_tokens = set(tokenize(text))
        if not turn_tokens:
            continue
        best = 0.0
        matched: set[str] = set()
        for tokens, phrase in prepared:
            overlap = turn_tokens & set(tokens)
            if not overlap:
                continue
            weight = sum(
                math.log(1.0 + total_documents / (1 + frequencies.get(token, 0)))
                for token in overlap
            )
            coverage = len(overlap) / len(set(tokens))
            query_score = weight * (0.5 + 0.5 * coverage)
            if phrase and phrase in text:
                query_score *= 1.75
            if query_score > best:
                best = query_score
            matched |= overlap
        if best >= min_score:
            scored.append(
                RetrievedTurn(
                    turn=turn,
                    score=best,
                    tier=tier,
                    relation=relation,
                    matched_terms=tuple(sorted(matched)),
                )
            )

    scored.sort(key=lambda item: (-item.score, item.turn.turn_id))
    return tuple(scored[:top_k])


def merge_retrieved(
    *groups: Sequence[RetrievedTurn], limit: int
) -> tuple[RetrievedTurn, ...]:
    """Concatenate retrieval groups strongest-tier-first, de-duplicating by turn id."""
    seen: set[str] = set()
    merged: list[RetrievedTurn] = []
    for group in groups:
        for item in group:
            if item.turn.turn_id in seen:
                continue
            seen.add(item.turn.turn_id)
            merged.append(item)
            if len(merged) >= limit:
                return tuple(merged)
    return tuple(merged)


def overall_tier(turns: Sequence[RetrievedTurn]) -> EvidenceTier:
    """The strongest tier present, or ``absent`` for an empty selection."""
    if not turns:
        return "absent"
    return max((item.tier for item in turns), key=TIER_ORDER.index)


def format_tiered_evidence(evidence: TieredEvidence) -> str:
    """Render retrieved turns for a prompt, grouped by how they were elicited.

    An ``absent`` topic renders as an explicit statement that the interview
    never covered it — deliberately *not* as silence, so a scorer reading this
    is told the difference between "the participant denied it" and "nobody
    asked", which is the distinction v2 collapsed into a zero.
    """
    if not evidence.turns:
        return (
            "EVIDENCE: none.\n"
            "This topic was never raised anywhere in the interview. The "
            "participant neither reported nor denied this symptom."
        )

    groups: dict[EvidenceRelation, list[RetrievedTurn]] = {}
    for item in evidence.turns:
        groups.setdefault(item.relation, []).append(item)

    blocks: list[str] = []
    ordered: tuple[tuple[EvidenceRelation, str], ...] = (
        ("probe", "EVIDENCE — the participant was asked about this symptom:"),
        (
            "reverse",
            "EVIDENCE (REVERSE POLARITY) — these questions asked about the "
            "OPPOSITE of this symptom. A specific, engaged answer is evidence "
            "AGAINST the symptom; a flat, vague or 'I can't think of anything' "
            "answer is evidence FOR it:",
        ),
        (
            "context",
            "BACKGROUND — general mood, diagnosis or treatment history. Not "
            "specific to this symptom and not sufficient on its own:",
        ),
    )
    for relation, header in ordered:
        selected = groups.get(relation)
        if not selected:
            continue
        lines = [header]
        for item in selected:
            turn = item.turn
            if turn.question:
                lines.append(f"  [{turn.turn_id}] Interviewer: {turn.question}")
            lines.append(f"  [{turn.turn_id}] Participant: {turn.answer}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
