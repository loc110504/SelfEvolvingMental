"""Policy glue for the Grounded Interview Loop (Plan_Improve.md Sec 2.4).

Plan_Improve.md keeps the multi-round interviewer/client dialogue (so the
assessment can still probe emotional nuance like a real clinical interview),
but every decision the old ``run_qwen_sample.py`` made from nothing is now
made from the retrieved :class:`~psyvec.evaluation.evidence_retrieval.
EvidenceBundle` instead:

- the client-persona's system prompt is built from real evidence for the
  topic being asked about, not a truncated small-talk prefix;
- the necessity-check default (used only when the LLM's own necessity call
  fails to parse) depends on whether evidence exists and whether a
  frequency/duration has already been established, instead of a single
  hard-coded constant;
- the follow-up question is steered at whichever PHQ-8 rubric dimension
  (frequency over the past two weeks) is still missing, instead of an
  open-ended prompt that lets the model wander;
- the scorer is asked to cite which retrieved turn ids it used, and any
  citation outside what was actually sent is flagged rather than trusted.

None of this calls a model — these are the deterministic policy decisions
around the model calls, kept separate so they can be unit tested without one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence, Set

from psyvec.evaluation.evidence_retrieval import (
    EvidenceBundle,
    RetrievalStatus,
    format_evidence_for_prompt,
)

__all__ = [
    "allowed_citation_ids",
    "build_client_system_prompt",
    "build_followup_instruction",
    "build_scorer_prompt",
    "default_necessity",
    "extract_demographics",
    "invalid_citations",
    "low_faithfulness_flag",
    "mentions_frequency",
]

# Phrases a participant might use to convey how often/long something
# happened — this is a *gate*, not a classifier: missing a phrasing only
# costs one extra follow-up question, which is safer than stopping too early
# (Plan_Improve.md Sec 7 risk table).
_FREQUENCY_RE = re.compile(
    r"\b("
    r"never|not at all|rarely|occasionally|sometimes|often|usually|always|"
    r"constantly|every day|everyday|nearly every day|most days|"
    r"more than half|half the days|a few days|several days|couple of days|"
    r"once a week|twice a week|few times a week|"
    r"\d+\s*(?:days?|times?|nights?|weeks?)"
    r")\b",
    re.IGNORECASE,
)


def mentions_frequency(text: str) -> bool:
    """Whether ``text`` already conveys a frequency/duration over two weeks."""
    return _FREQUENCY_RE.search(text) is not None


def default_necessity(status: RetrievalStatus, has_frequency_info: bool) -> int:
    """Necessity-check fallback used only when the LLM's own call fails to parse.

    ``missing`` evidence means there is nothing further to probe from real
    transcript grounding, so the conservative default is to stop. ``known``
    evidence without an established frequency/duration is the one case where
    continuing is the safer default, because PHQ-8 scoring needs that detail.
    """
    if status == "missing":
        return 0
    return 0 if has_frequency_info else 1


_KNOWN_EVIDENCE_RULE = (
    "Answer strictly consistent with the quotes below, which are your own "
    "words from earlier in this same interview. Do not contradict them or "
    "invent new symptoms beyond what they show."
)
_MISSING_EVIDENCE_RULE = (
    "You have not explicitly discussed this topic yet in the interview. "
    "Answer briefly and conservatively, consistent with the overall mood/"
    "history context below (if any is shown). Do NOT invent specific new "
    "symptoms you have no basis for."
)


def build_client_system_prompt(topic_name: str, bundle: EvidenceBundle) -> str:
    """Ground the client-persona's system prompt in this topic's real evidence."""
    evidence_text = format_evidence_for_prompt(bundle)
    rule = _KNOWN_EVIDENCE_RULE if bundle.status == "known" else _MISSING_EVIDENCE_RULE
    return (
        f'You are acting as the client in a psychological consultation about '
        f'"{topic_name}".\n\n'
        f"Real evidence from this interview relevant to this topic:\n"
        f"{evidence_text}\n\n"
        f"{rule}\n"
        "Respond truthfully and reasonably in the first person (I, me). "
        "Keep your answer concise (under 40 words)."
    )


_FREQUENCY_FOLLOWUP_HINT = (
    "The client has not yet said how many days out of the past two weeks "
    "this occurred. Ask a short, specific clinical follow-up question that "
    "asks for that frequency (e.g. 'how many days in the last two weeks...')."
)
_GENERAL_FOLLOWUP_HINT = (
    "Ask a short clinical follow-up question to clarify the severity or "
    "impact of this on daily life over the past two weeks."
)


def build_followup_instruction(
    topic_name: str, client_reply: str, has_frequency_info: bool
) -> str:
    """Steer the follow-up question at the rubric gap PHQ-8 scoring needs."""
    hint = _GENERAL_FOLLOWUP_HINT if has_frequency_info else _FREQUENCY_FOLLOWUP_HINT
    return f"Topic: {topic_name}\nPatient response: {client_reply}\n{hint}"


def build_scorer_prompt(
    topic_name: str, history_str: str, bundle: EvidenceBundle, standard_text: str
) -> str:
    """Give the scorer both the grounded dialogue and the raw cited evidence."""
    evidence_text = format_evidence_for_prompt(bundle)
    return (
        f"Topic: {topic_name}\n\n"
        f"Interview dialogue (this session):\n{history_str}\n\n"
        "Real transcript evidence for this topic "
        f"(cite these turn ids if you use them):\n{evidence_text}\n\n"
        f"Scoring standard:\n{standard_text}\n\n"
        "Score this topic from 0 to 3 based on the standard, grounded in the "
        "evidence above. If no evidence supports a nonzero score, score 0.\n"
        'Output JSON: {"score": <0, 1, 2, or 3>, "summary": "<one sentence '
        'basis>", "evidence_turn_ids": ["turn-12", ...]}'
    )


def allowed_citation_ids(bundle: EvidenceBundle) -> frozenset[str]:
    """Turn ids the scorer/updater were actually shown for this topic."""
    return frozenset(
        snippet.turn_id for snippet in (*bundle.snippets, *bundle.cross_cutting)
    )


def invalid_citations(cited: Sequence[str], allowed: Set[str]) -> tuple[str, ...]:
    """Return the subset of ``cited`` ids that were never sent to the model."""
    return tuple(turn_id for turn_id in cited if turn_id not in allowed)


# Deliberately conservative: only age is extracted with any reliability.
# Gender is not present in the processed DAIC-WOZ sample JSON at all, and a
# regex for occupation ("i'm a ...") is too noisy to be worth the risk of
# printing something nonsensical — see Plan_Improve.md Sec 2.7, which marks
# demographics as low priority since it does not affect the PHQ-8 score.
_AGE_RE = re.compile(
    r"\b(?:i'?m|i am)\s+(\d{1,2})\b|\b(\d{1,2})\s+years?\s+old\b", re.IGNORECASE
)


def extract_demographics(real_interview: Sequence[Mapping[str, str]]) -> str:
    """Best-effort, non-LLM demographics string (age only; see module note)."""
    age: str | None = None
    for turn in real_interview:
        if turn.get("roleName") != "Participant":
            continue
        match = _AGE_RE.search(str(turn.get("content", "")))
        if match is None:
            continue
        candidate = match.group(1) or match.group(2)
        if candidate is not None and 10 <= int(candidate) <= 100:
            age = candidate
            break
    return f"Age: {age or 'undisclosed'}, Gender: undisclosed, Occupation: undisclosed"


_CONTENT_WORD_RE = re.compile(r"[a-z]{4,}")


def _content_words(text: str) -> frozenset[str]:
    return frozenset(_CONTENT_WORD_RE.findall(text.lower()))


def low_faithfulness_flag(
    dialogue_text: str, bundle: EvidenceBundle, *, min_overlap: float = 0.15
) -> bool:
    """Flag dialogue that shares almost no vocabulary with its cited evidence.

    This is the heuristic from Plan_Improve.md Sec 5.1: not an entailment
    check, just a cheap tripwire that a client-persona reply may have
    invented content unconnected to the real transcript, worth a human
    reading the ``dialogue_transcript`` for that topic. Only meaningful when
    ``bundle.status == "known"`` — with no evidence to be consistent with
    there is nothing to flag against, so this always returns ``False`` for a
    ``missing`` bundle.
    """
    if bundle.status != "known":
        return False
    dialogue_words = _content_words(dialogue_text)
    if not dialogue_words:
        return False
    evidence_words = _content_words(
        " ".join(
            snippet.text for snippet in (*bundle.snippets, *bundle.cross_cutting)
        )
    )
    if not evidence_words:
        return False
    overlap = len(dialogue_words & evidence_words) / len(dialogue_words)
    return overlap < min_overlap
