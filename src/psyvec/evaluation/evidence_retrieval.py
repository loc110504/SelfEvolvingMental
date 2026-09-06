"""Ground the client-persona role-play in the real DAIC-WOZ transcript.

Plan_Improve.md Sec 0 identifies the root cause of the previous evaluation
pipeline's inaccuracy: the simulated client answered every PHQ-8 question
using only ``real_interview[:50]`` (almost always small talk) as background,
so its answers were unconstrained invention rather than a reflection of what
the participant actually said. Symptom-relevant disclosures typically occur
well past turn 50 (the interview averages ~266 turns).

This module retrieves, for each PHQ-8 topic, the *real* transcript passages
that back it — before any role-play happens — using two free (non-LLM)
signals over the *entire* transcript:

1. Tag-anchor match: DAIC-WOZ's virtual interviewer (Ellie) sometimes emits
   its internal protocol identifier verbatim, e.g.
   ``easy_sleep (how easy is it for you to get a good night's sleep)``. These
   tags are a fixed, reliable anchor (see ``configs/scales/topic_tag_map.json``,
   built from the train split only).
2. Keyword-lexicon fallback: 5 of the 8 topics have no direct tag anchor
   (Loss of Interest, Fatigue or Low Energy, Appetite or Weight Changes,
   Concentration Difficulties, Psychomotor Changes), so every Participant
   turn is scanned for topic keywords (``configs/scales/topic_keywords.json``).

Tags that probe general mood/diagnosis/therapy rather than one specific
symptom (``depression_diagnosed``, ``feel_down``, ``therapy_*``, ...) are
collected into a separate ``CROSS_CUTTING`` bucket. Plan_Improve.md Sec 2.4
keeps the interactive interview for every topic, including ones with no
topic-specific evidence — this bucket is what grounds the client-persona's
answer in those cases instead of leaving it to invent freely.

Nothing here calls a model: this is retrieval only, used to *build* prompts
for the role-play/scorer steps in ``scripts/run_qwen_sample.py``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "CROSS_CUTTING",
    "EvidenceBundle",
    "EvidenceSnippet",
    "EvidenceSource",
    "InterviewTurn",
    "RetrievalStatus",
    "extract_ellie_tag",
    "format_evidence_for_prompt",
    "load_keyword_lexicon",
    "load_tag_map",
    "retrieve_evidence",
]

#: Sentinel topic name for tags that probe general mood/history rather than
#: one specific PHQ-8 symptom domain (see module docstring).
CROSS_CUTTING = "CROSS_CUTTING"

EvidenceSource = Literal["tag_anchor", "keyword", "cross_cutting"]
RetrievalStatus = Literal["known", "missing"]

#: Matches a whole Ellie turn of the shape ``tag_name (utterance text)``.
_TAG_RE = re.compile(r"^([A-Za-z0-9_]+)\s*\((.*)\)$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class InterviewTurn:
    """One normalized turn of ``real_interview`` with a stable turn id."""

    turn_id: str
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class EvidenceSnippet:
    """A single retrieved turn cited as evidence for one topic."""

    turn_id: str
    speaker: str
    text: str
    source: EvidenceSource
    keyword_hits: int = 0


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Everything retrieved for one PHQ-8 topic from one participant."""

    topic: str
    status: RetrievalStatus
    snippets: tuple[EvidenceSnippet, ...]
    cross_cutting: tuple[EvidenceSnippet, ...]


def extract_ellie_tag(text: str) -> tuple[str, str] | None:
    """Return ``(tag, utterance)`` if ``text`` is a tagged Ellie line, else None."""
    match = _TAG_RE.match(text.strip())
    if match is None:
        return None
    return match.group(1), match.group(2).strip()


def load_tag_map(path: Path, *, valid_topics: Sequence[str]) -> dict[str, str]:
    """Load ``topic_tag_map.json``, rejecting any target outside ``valid_topics``.

    Fails closed rather than silently ignoring a typo'd topic name, matching
    this codebase's strict-parsing convention (see
    ``psyvec.evaluation.response_parsing``).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(valid_topics) | {CROSS_CUTTING}
    tag_map: dict[str, str] = {}
    for tag, topic in raw.items():
        if tag == "_meta":
            continue
        if topic not in allowed:
            raise ValueError(
                f"topic_tag_map.json maps tag {tag!r} to unknown topic {topic!r}"
            )
        tag_map[tag] = topic
    return tag_map


def load_keyword_lexicon(
    path: Path, *, valid_topics: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Load ``topic_keywords.json``, rejecting any topic outside ``valid_topics``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(valid_topics)
    lexicon: dict[str, tuple[str, ...]] = {}
    for topic, keywords in raw.items():
        if topic == "_meta":
            continue
        if topic not in allowed:
            raise ValueError(
                f"topic_keywords.json has unknown topic {topic!r}"
            )
        lexicon[topic] = tuple(str(kw).lower() for kw in keywords)
    return lexicon


def _normalize_turns(
    real_interview: Sequence[Mapping[str, str]],
) -> tuple[InterviewTurn, ...]:
    return tuple(
        InterviewTurn(
            turn_id=f"turn-{index}",
            speaker=str(turn.get("roleName", "")),
            text=str(turn.get("content", "")),
        )
        for index, turn in enumerate(real_interview)
    )


def _participant_reply_span(
    turns: Sequence[InterviewTurn], ellie_index: int
) -> tuple[InterviewTurn, ...]:
    """Return the contiguous run of Participant turns right after ``ellie_index``."""
    replies: list[InterviewTurn] = []
    for turn in turns[ellie_index + 1 :]:
        if turn.speaker != "Participant":
            break
        replies.append(turn)
    return tuple(replies)


def _tag_anchor_snippets(
    turns: Sequence[InterviewTurn], tag_map: Mapping[str, str]
) -> dict[str, list[EvidenceSnippet]]:
    buckets: dict[str, list[EvidenceSnippet]] = {}
    for index, turn in enumerate(turns):
        if turn.speaker != "Ellie":
            continue
        found = extract_ellie_tag(turn.text)
        if found is None:
            continue
        tag, _utterance = found
        target_topic = tag_map.get(tag)
        if target_topic is None:
            continue
        source: EvidenceSource = (
            "cross_cutting" if target_topic == CROSS_CUTTING else "tag_anchor"
        )
        for reply in _participant_reply_span(turns, index):
            buckets.setdefault(target_topic, []).append(
                EvidenceSnippet(
                    turn_id=reply.turn_id,
                    speaker=reply.speaker,
                    text=reply.text,
                    source=source,
                )
            )
    return buckets


#: Cues that locally negate or contrast away a keyword match right before it
#: (e.g. "i'd rather be happy than **sad**", "i wasn't **worthless**"). This is
#: a cheap per-occurrence window check, not full negation scope resolution —
#: it only looks a few words back from the match, so a negation elsewhere in
#: a long turn does not suppress an unrelated later disclosure.
_NEGATION_CUES = ("not", "n't", "never", "nothing", "without", "rather", "instead")
_NEGATION_WINDOW_WORDS = 4


def _is_locally_negated(
    lowered_text: str, keyword: str, window: int = _NEGATION_WINDOW_WORDS
) -> bool:
    """Whether every occurrence of ``keyword`` in ``lowered_text`` is preceded
    by a negation/contrast cue within ``window`` words.
    """
    matches = list(re.finditer(re.escape(keyword), lowered_text))
    if not matches:
        return False
    for match in matches:
        preceding_words = lowered_text[: match.start()].split()[-window:]
        if not any(
            any(cue in word for cue in _NEGATION_CUES) for word in preceding_words
        ):
            return False  # at least one occurrence is NOT negated -> a real hit
    return True


#: A DAIC-WOZ Ellie question phrased as a hypothetical ("what are you like
#: when...", "what would you...") asks the participant to describe a general
#: disposition, not to report their own state over the past two weeks. A
#: keyword match in the reply to one of these is not a self-report and must
#: not be retrieved as topic evidence.
_HYPOTHETICAL_QUESTION_RE = re.compile(
    r"\bwhat are you like\b|\bwhat would you\b|\bhow would you\b|"
    r"\bif you (?:were|didn't|weren't|would)\b|\bimagine\b",
    re.IGNORECASE,
)


def _replies_to_hypothetical_question(
    turns: Sequence[InterviewTurn], index: int
) -> bool:
    if index == 0:
        return False
    previous = turns[index - 1]
    if previous.speaker != "Ellie":
        return False
    return bool(_HYPOTHETICAL_QUESTION_RE.search(previous.text))


def _keyword_snippets(
    turns: Sequence[InterviewTurn],
    keyword_lexicon: Mapping[str, Sequence[str]],
) -> dict[str, list[EvidenceSnippet]]:
    buckets: dict[str, list[EvidenceSnippet]] = {}
    for topic, keywords in keyword_lexicon.items():
        for index, turn in enumerate(turns):
            if turn.speaker != "Participant":
                continue
            if _replies_to_hypothetical_question(turns, index):
                continue
            lowered_text = turn.text.lower()
            hits = sum(
                1
                for keyword in keywords
                if keyword in lowered_text
                and not _is_locally_negated(lowered_text, keyword)
            )
            if hits == 0:
                continue
            buckets.setdefault(topic, []).append(
                EvidenceSnippet(
                    turn_id=turn.turn_id,
                    speaker=turn.speaker,
                    text=turn.text,
                    source="keyword",
                    keyword_hits=hits,
                )
            )
    return buckets


def retrieve_evidence(
    real_interview: Sequence[Mapping[str, str]],
    topics: Sequence[str],
    tag_map: Mapping[str, str],
    keyword_lexicon: Mapping[str, Sequence[str]],
    *,
    max_specific_snippets: int = 6,
    max_cross_cutting_snippets: int = 3,
) -> dict[str, EvidenceBundle]:
    """Build one :class:`EvidenceBundle` per topic from the real transcript.

    Priority within a topic's specific snippets: tag-anchor hits first (they
    were directly elicited by a question about that exact topic), then
    keyword hits ordered by hit density then length, capped at
    ``max_specific_snippets`` (Plan_Improve.md Sec 2.3e).
    """
    turns = _normalize_turns(real_interview)
    tag_buckets = _tag_anchor_snippets(turns, tag_map)
    keyword_buckets = _keyword_snippets(turns, keyword_lexicon)

    cross_cutting_snippets = tuple(
        tag_buckets.get(CROSS_CUTTING, ())[:max_cross_cutting_snippets]
    )

    bundles: dict[str, EvidenceBundle] = {}
    for topic in topics:
        anchor_list = tag_buckets.get(topic, [])
        seen_turn_ids = {snippet.turn_id for snippet in anchor_list}
        keyword_list = [
            snippet
            for snippet in keyword_buckets.get(topic, [])
            if snippet.turn_id not in seen_turn_ids
        ]
        keyword_list.sort(
            key=lambda snippet: (-snippet.keyword_hits, -len(snippet.text))
        )

        combined = (*anchor_list, *keyword_list)[:max_specific_snippets]
        status: RetrievalStatus = "known" if combined else "missing"
        bundles[topic] = EvidenceBundle(
            topic=topic,
            status=status,
            snippets=combined,
            cross_cutting=cross_cutting_snippets,
        )
    return bundles


def format_evidence_for_prompt(bundle: EvidenceBundle) -> str:
    """Render a bundle as plain text for inclusion in an LLM prompt.

    Every line is prefixed with its ``turn_id`` so a scorer/updater step can
    cite exactly which real transcript turn a score is based on.
    """
    if bundle.status == "known":
        lines = [
            f"[{snippet.turn_id}] Participant: {snippet.text}"
            for snippet in bundle.snippets
        ]
        return "\n".join(lines)

    if bundle.cross_cutting:
        lines = [
            f"[{snippet.turn_id}] Participant: {snippet.text}"
            for snippet in bundle.cross_cutting
        ]
        return (
            "No direct disclosure found for this topic. Overall mood/history "
            "context from elsewhere in the interview:\n" + "\n".join(lines)
        )

    return (
        "No direct disclosure found for this topic, and no general mood "
        "context available either."
    )
