"""Evidence-direct PHQ-8 item scoring with a first-class abstention channel.

Why the scorer reads the transcript instead of a role-play
----------------------------------------------------------
The v2 pipeline scored each item from a *simulated* dialogue: retrieved
evidence went to a client-persona LLM, the persona improvised an answer, and
the scorer scored the improvisation. Two failure modes followed from that
ordering, both measured in the v2 result files:

* With nothing retrieved, the persona was told to answer "conservatively" and
  reliably produced a fabricated *denial* — for a participant whose reference
  score for depressed mood is nonzero, "I have not experienced feelings of
  being down, depressed, or hopeless over the past two weeks." The scorer
  recorded 0 with the summary "The respondent explicitly states they have not
  experienced ...". No such statement exists in that transcript.
* Even with evidence retrieved, the persona paraphrased it away, so the scorer
  scored a lossy re-encoding rather than the transcript.

Here the scorer reads retrieved excerpts directly. The interviewer and
sufficiency roles survive, but they act on *evidence acquisition* — which query
to issue next, whether what is retrieved supports a rubric decision — instead
of generating a participant's words.

Why ``sufficient: false`` exists
--------------------------------
"Nobody asked" and "the participant said no" are different claims. Collapsing
them produced v2's systematic under-scoring: on topics with no retrieved
evidence the reference item score is nonzero 55-78% of the time, yet 94-100%
of those were scored 0. An abstention here is resolved downstream by
:mod:`psyvec.evaluation.imputation`, never by defaulting to zero.

Prompt design notes
-------------------
The output schema orders ``present`` and ``frequency_basis`` *before* ``score``
on purpose: a model generates left to right, so naming the presence judgement
and the frequency evidence first makes them condition the number rather than
rationalize it afterwards. They also give the evaluation a diagnostic it
otherwise lacks — how often a score rests on a stated day count versus an
inferred ongoing state.

The frequency-inference rules are clinical, not fitted to any corpus: an
ongoing current state is by definition present more than half the days, while a
single recalled episode is not. Naming both cases is what stops the model
defaulting a clearly-present-but-unquantified symptom to the bottom of the
rubric, which is where v2 lost most of its severity range.

Prompt construction and parsing only — no model calls, stdlib only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psyvec.evaluation.response_parsing import extract_json_objects, strip_reasoning

__all__ = [
    "ITEM_MANIFESTATIONS",
    "ITEM_SCORE_JSON_SCHEMA",
    "INTERVIEWER_SYSTEM_PROMPT",
    "ItemScoreParse",
    "PROMPT_VERSION",
    "QUERY_EXPANSION_JSON_SCHEMA",
    "SCORER_SYSTEM_PROMPT",
    "SUFFICIENCY_JSON_SCHEMA",
    "SUFFICIENCY_SYSTEM_PROMPT",
    "build_item_score_prompt",
    "build_query_expansion_prompt",
    "build_sufficiency_prompt",
    "parse_item_score",
    "parse_query_expansion",
    "parse_sufficiency",
    "quote_is_grounded",
]

#: Bump whenever prompt text changes, so a result file can be traced to the
#: wording that produced it (see ``psyvec.state.contracts.ProvenanceRef``).
PROMPT_VERSION = "v3.1"


SCORER_SYSTEM_PROMPT = (
    "You are a psychometric rater assigning one PHQ-8 item score from verbatim "
    "clinical interview excerpts.\n"
    "Three rules override everything else:\n"
    "1. A topic the interview never covered is NOT a denial. Abstain instead "
    "of scoring it 0.\n"
    "2. Never attribute a statement to the participant that is not in the "
    "excerpts, and never quote words they did not say.\n"
    "3. A symptom the participant clearly describes as their current ongoing "
    "state is frequent, not occasional, even when they never count the days.\n"
    "Output valid JSON only."
)

INTERVIEWER_SYSTEM_PROMPT = (
    "You are a clinical interviewer deciding where to look next in an "
    "interview transcript. You produce short search phrases in the "
    "participant's own everyday spoken vocabulary, never clinical "
    "terminology, because you are searching what a person actually said. "
    "Output valid JSON only."
)

SUFFICIENCY_SYSTEM_PROMPT = (
    "You are a sufficiency evaluator. You judge only one thing: whether the "
    "excerpts shown contain enough about the named symptom to place it on a "
    "0-3 frequency rubric. You are not scoring. Output valid JSON only."
)


#: How each PHQ-8 item actually surfaces in unstructured interview speech.
#:
#: A rubric alone ("bothered by feeling tired or having little energy") gives a
#: model nothing to match "i can barely drag myself out of bed" against. These
#: lines are deliberately generic descriptions of the construct rather than
#: phrases from any particular corpus, so they transfer to another interview
#: dataset unchanged.
ITEM_MANIFESTATIONS: Mapping[str, str] = {
    "Loss of Interest": (
        "Reduced pleasure or engagement: dropped hobbies, withdrawing from "
        "people, 'nothing feels worth doing', going through the motions, "
        "unable to name anything recently enjoyed, activities once loved now "
        "feeling flat."
    ),
    "Depressed Mood": (
        "Sad, down, low, empty, numb, hopeless or tearful mood; describing "
        "life as bleak or pointless; saying they are 'not okay', 'in a bad "
        "place', or struggling emotionally right now."
    ),
    "Sleep Problems": (
        "Trouble falling asleep, waking during the night, waking too early, "
        "unrefreshing sleep, very short sleep, or sleeping far too much. A "
        "reported number of hours well outside a normal range counts."
    ),
    "Fatigue or Low Energy": (
        "Tiredness, exhaustion, being drained or worn out, needing effort for "
        "ordinary tasks, difficulty getting out of bed, low stamina — "
        "including tiredness described as a consequence of poor sleep."
    ),
    "Appetite or Weight Changes": (
        "Eating much less or much more than usual, loss of appetite, skipping "
        "or forgetting meals, comfort or stress eating, food losing appeal, or "
        "noticeable unintended weight loss or gain."
    ),
    "Low Self-Worth": (
        "Feeling like a failure, worthless, guilty, inadequate, a burden, or "
        "that they have let themselves or others down; harsh self-criticism; "
        "regret framed as a personal shortcoming rather than bad luck."
    ),
    "Concentration Difficulties": (
        "Trouble focusing, mind wandering, re-reading, losing the thread, "
        "forgetfulness, mental fog, or difficulty making ordinary decisions."
    ),
    "Psychomotor Changes": (
        "EITHER direction counts. Slowed: moving or speaking more slowly, "
        "everything taking longer, heaviness. Agitated: restlessness, "
        "fidgeting, pacing, being unable to sit still, being wound up, "
        "irritable or quick to anger."
    ),
}


_DECISION_PROCEDURE = """Work through these in order.

STEP 1 — Timeframe. PHQ-8 asks about the last two weeks. Treat "lately",
"recently", "these days", "right now" and the present tense as inside the
window. A clearly dated past episode ("after my divorce in 2009", "when I was
in the army") is OUTSIDE it and must not be scored, unless the participant
says it is still going on.

STEP 2 — Presence. Is the symptom present in the window?
  yes      the participant reports it, directly or by describing it.
  no       the participant denies it, or describes an unaffected state.
  unclear  the excerpts do not settle it, or are about something else.

STEP 3 — Frequency, only if presence is "yes". Set "frequency_basis" to the
strongest available and score accordingly:
  stated_days     an explicit count or fraction of days -> map onto the rubric.
  frequency_word  "always"/"constantly"/"every day" -> 3;
                  "most days"/"a lot"/"usually" -> 2;
                  "sometimes"/"a few days"/"on and off" -> 1;
                  "rarely"/"hardly ever" -> 0.
  ongoing_state   described as how things ARE for them now, with no count. An
                  ongoing current state is present more than half the days:
                  score 2, or 3 if described as constant or severe.
  single_episode  one recalled occasion, not a pattern -> 1.

STEP 4 — Abstain. If presence is "unclear", or the only excerpts are
background, or nothing was retrieved: set "sufficient": false and
"score": null. Do NOT guess 0. An unasked question is not a denial, and a
downstream step handles abstentions properly.

STEP 5 — Quote. Copy verbatim the span you relied on, or "" when abstaining.
Never write a quote the excerpts do not contain."""


ITEM_SCORE_JSON_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "timeframe_ok": {"type": "boolean"},
        "present": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "frequency_basis": {
            "type": "string",
            "enum": [
                "stated_days",
                "frequency_word",
                "ongoing_state",
                "single_episode",
                "none",
            ],
        },
        "sufficient": {"type": "boolean"},
        "score": {"type": ["integer", "null"], "minimum": 0, "maximum": 3},
        "quote": {"type": "string"},
        "turn_ids": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "present",
        "frequency_basis",
        "sufficient",
        "score",
        "quote",
        "turn_ids",
        "reasoning",
    ],
}

SUFFICIENCY_JSON_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "missing": {"type": "string"},
    },
    "required": ["sufficient", "missing"],
}

QUERY_EXPANSION_JSON_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries"],
}


def _rubric_block(standard: Mapping[str, str]) -> str:
    return "\n".join(f"  {level} = {text}" for level, text in sorted(standard.items()))


def _lesson_block(lessons: Sequence[str]) -> str:
    if not lessons:
        return ""
    body = "\n".join(f"  - {lesson}" for lesson in lessons)
    return (
        "\n## Validated rating guidance for this item\n"
        "Each line below was distilled from prior cases and kept only after it "
        "improved accuracy on participants other than the ones it came from. "
        "Apply it unless these excerpts contradict it.\n" + body + "\n"
    )


def build_item_score_prompt(
    topic: str,
    standard: Mapping[str, str],
    evidence_text: str,
    *,
    lessons: Sequence[str] = (),
) -> str:
    """Ask for one item score directly from transcript excerpts.

    Section order is deliberate: the item and rubric establish what is being
    measured, the evidence arrives before any instruction about how to use it
    (so the model reads it without a prior), and the decision procedure sits
    last, immediately before generation, where it has the most influence.
    """
    manifestations = ITEM_MANIFESTATIONS.get(topic, "")
    return f"""## Item
{topic}

What this item covers in ordinary speech:
{manifestations}

## Rubric — how many days out of the last 14
{_rubric_block(standard)}

## Interview excerpts
{evidence_text}
{_lesson_block(lessons)}
## How to decide
{_DECISION_PROCEDURE}

## Output
Return one JSON object and nothing else. Fill the fields in this order; the
earlier fields are your reasoning and must justify the score that follows.

{{"timeframe_ok": true or false,
  "present": "yes" or "no" or "unclear",
  "frequency_basis": one of "stated_days" | "frequency_word" | "ongoing_state"
                     | "single_episode" | "none",
  "sufficient": true or false,
  "score": 0 or 1 or 2 or 3, or null when abstaining,
  "quote": "<verbatim span from the excerpts, or empty>",
  "turn_ids": ["turn-12"],
  "reasoning": "<one sentence citing only what the excerpts say>"}}"""


def build_sufficiency_prompt(
    topic: str, standard: Mapping[str, str], evidence_text: str
) -> str:
    """Ask whether more evidence should be sought before scoring this item."""
    manifestations = ITEM_MANIFESTATIONS.get(topic, "")
    return f"""## Item
{topic}

What this item covers in ordinary speech:
{manifestations}

## Rubric — how many days out of the last 14
{_rubric_block(standard)}

## Interview excerpts
{evidence_text}

## Question
Do these excerpts contain enough about THIS item to place the participant on
the rubric above?

Answer false if they are about a different symptom, if they only establish
general mood or diagnosis history, if they show only the reverse pole, or if
nothing was retrieved. Answer true if they settle whether the symptom is
present in the last two weeks — a clear denial counts as sufficient.

## Output
{{"sufficient": true or false,
  "missing": "<what still needs finding, one short phrase; empty if sufficient>"}}"""


def build_query_expansion_prompt(
    topic: str,
    standard: Mapping[str, str],
    missing: str,
    already_tried: Sequence[str],
) -> str:
    """Ask the interviewer role for fresh retrieval queries over the transcript.

    This replaces v2's follow-up *question* generation. In an offline benchmark
    the participant's answers are already recorded, so the productive action is
    not to invent a question and have a persona answer it — it is to search the
    recording for what they said when this ground was actually covered.

    The phrasing constraint matters: participants say "i just lay there for
    hours", never "initial insomnia", and a query in clinical register retrieves
    nothing from spontaneous speech.
    """
    tried = "\n".join(f"  - {query}" for query in already_tried) or "  (none yet)"
    manifestations = ITEM_MANIFESTATIONS.get(topic, "")
    return f"""## Item
{topic}

What this item covers in ordinary speech:
{manifestations}

## Still missing
{missing or "any mention of this symptom at all"}

## Search phrases already tried (do not repeat these)
{tried}

## Task
Propose 6 NEW short search phrases likely to appear in a recorded interview
where someone touches on this item.

Requirements:
  - everyday spoken English, 2-6 words each, no clinical jargon;
  - include indirect and euphemistic ways people raise it, and ways they raise
    it while talking about something else (work, family, daily routine);
  - include at least one phrase for the symptom's opposite pole, since a clear
    denial is also evidence;
  - do not repeat or trivially reword anything already tried.

## Output
{{"queries": ["...", "...", "...", "...", "...", "..."]}}"""


@dataclass(frozen=True, slots=True)
class ItemScoreParse:
    """Outcome of parsing one item-scoring reply.

    ``sufficient=False`` with ``score=None`` is a valid, expected result and not
    a parse failure — it is the abstention this module exists to make
    representable.
    """

    sufficient: bool
    score: int | None
    quote: str
    turn_ids: tuple[str, ...]
    reasoning: str
    present: str
    frequency_basis: str
    ok: bool
    failure_reason: str = ""
    raw_text: str = ""


def _last_object(raw_text: str) -> dict[str, Any] | None:
    """Return the last balanced JSON object in ``raw_text``, if any parses.

    Last rather than first: a model that restates the requested schema before
    answering would otherwise have its schema echo parsed as the answer.
    """
    cleaned = strip_reasoning(raw_text) or raw_text
    for span in reversed(extract_json_objects(cleaned)):
        try:
            loaded = json.loads(span)
        except (TypeError, ValueError):
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return default


def _as_score(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if float(value) != int(value):
            return None
        return max(0, min(3, int(value)))
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match is None:
            return None
        return max(0, min(3, int(match.group())))
    return None


_PRESENCE_VALUES = frozenset({"yes", "no", "unclear"})


def parse_item_score(raw_text: str) -> ItemScoreParse:
    """Parse a scorer reply, preserving abstention as a distinct outcome."""
    payload = _last_object(raw_text)
    if payload is None:
        return ItemScoreParse(
            sufficient=False,
            score=None,
            quote="",
            turn_ids=(),
            reasoning="",
            present="unclear",
            frequency_basis="none",
            ok=False,
            failure_reason="no JSON object in reply",
            raw_text=raw_text,
        )

    score = _as_score(payload.get("score"))
    presence = str(payload.get("present", "")).strip().lower()
    if presence not in _PRESENCE_VALUES:
        presence = "yes" if score is not None else "unclear"
    sufficient = _as_bool(payload.get("sufficient"), default=score is not None)

    # A reply claiming sufficiency without a usable score is self-contradictory;
    # treat it as an abstention rather than inventing a number for it.
    if sufficient and score is None:
        sufficient = False
    # "unclear" presence is an abstention no matter what number came with it —
    # this is the guard against reasserting v2's "not discussed, therefore 0".
    if presence == "unclear":
        sufficient = False
    if not sufficient:
        score = None
    # An explicit denial is a real observation of zero, not an abstention.
    if presence == "no" and sufficient:
        score = 0

    raw_ids = payload.get("turn_ids")
    turn_ids = (
        tuple(str(item) for item in raw_ids if isinstance(item, (str, int)))
        if isinstance(raw_ids, list)
        else ()
    )
    basis = str(payload.get("frequency_basis", "")).strip().lower() or "none"
    return ItemScoreParse(
        sufficient=sufficient,
        score=score,
        quote=str(payload.get("quote") or "").strip(),
        turn_ids=turn_ids,
        reasoning=str(payload.get("reasoning") or "").strip(),
        present=presence,
        frequency_basis=basis,
        ok=True,
        raw_text=raw_text,
    )


def parse_sufficiency(raw_text: str) -> tuple[bool | None, str]:
    """Return ``(sufficient, missing_description)``; ``None`` when unparseable."""
    payload = _last_object(raw_text)
    if payload is None:
        return None, ""
    missing = str(payload.get("missing") or "").strip()
    if "sufficient" not in payload:
        return None, missing
    return _as_bool(payload.get("sufficient"), default=False), missing


def parse_query_expansion(raw_text: str, *, limit: int = 8) -> tuple[str, ...]:
    """Extract generated retrieval queries, de-duplicated and length-capped."""
    payload = _last_object(raw_text)
    if payload is None:
        return ()
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        return ()
    seen: set[str] = set()
    queries: list[str] = []
    for entry in raw_queries:
        if not isinstance(entry, str):
            continue
        cleaned = " ".join(entry.split()).strip().strip('"')
        key = cleaned.lower()
        if not cleaned or key in seen or len(cleaned) > 80:
            continue
        seen.add(key)
        queries.append(cleaned)
        if len(queries) >= limit:
            break
    return tuple(queries)


_WORD_RE = re.compile(r"[a-z0-9']+")


def quote_is_grounded(quote: str, evidence_text: str, *, min_run: int = 5) -> bool:
    """Whether ``quote`` really occurs in ``evidence_text``.

    A verbatim substring check after whitespace normalization, falling back to
    "does some run of ``min_run`` consecutive quote words appear", so a scorer
    trimming an ellipsis or a transcription artifact is not flagged. This
    replaces v2's ``low_faithfulness_flag`` vocabulary-overlap heuristic, which
    fired on unrelated wording and — worse — returned ``False`` unconditionally
    for topics with no retrieved evidence, which is exactly where the
    fabricated denials happened.
    """
    if not quote.strip():
        return False
    normalized_evidence = " ".join(evidence_text.lower().split())
    normalized_quote = " ".join(quote.lower().split())
    if normalized_quote in normalized_evidence:
        return True
    words = _WORD_RE.findall(normalized_quote)
    if len(words) < min_run:
        return False
    return any(
        " ".join(words[index : index + min_run]) in normalized_evidence
        for index in range(len(words) - min_run + 1)
    )
