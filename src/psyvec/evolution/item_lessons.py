"""Mine, distil and transfer-verify procedural lessons for the scorer role.

This is the paper's fast path (contrastive lesson distillation followed by
cross-state paired validation) reduced to the one decision the assessment
pipeline actually makes many times over: assigning a PHQ-8 item score from
retrieved evidence.

The four stages, and what each corresponds to:

``mine_error_patterns``
    Role-aware hard-state mining. A "state" here is one (participant, item)
    scoring decision, and the hard cells are the (item, evidence tier) groups
    where the scorer is systematically wrong rather than noisily wrong — a
    group with a large *signed* error is a correctable procedural fault, a
    group with a large absolute error and no signed bias is just difficulty.

``build_distillation_prompt`` / ``parse_lesson``
    Contrastive distillation into the paper's lesson object
    ``(role, trigger, do, avoid, criterion, scope)``. The distiller sees
    anonymized error cases, never a raw transcript, and the resulting lesson
    text is checked for leaked reference scores before it is stored — a lesson
    that names the answer would transfer nothing and would leak supervision
    into an inference-time prompt.

``score_gain`` / ``validate_lesson``
    Cross-state paired validation. Each candidate is replayed on participants
    *excluded from its own source set*, control versus intervention from an
    identical cached state, and promoted only if it clears all of: minimum
    support, a minimum fraction of states improved, a minimum mean utility
    margin, and no safety regression. This is the gate that makes "the lesson
    helps" a measured claim rather than an assumption.

Storage and lifecycle reuse :mod:`psyvec.lessons.core` (``Lesson``,
``LessonMemoryEntry``, ``LessonStore``, near-duplicate merging), so a promoted
lesson carries the same provenance and governance record as any other.

No model calls here: prompt construction, parsing and statistics only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psyvec.evaluation.response_parsing import extract_json_objects, strip_reasoning
from psyvec.lessons.core import Lesson, LessonMemoryEntry
from psyvec.state.contracts import canonical_hash

__all__ = [
    "ErrorPattern",
    "LESSON_JSON_SCHEMA",
    "LESSON_DISTILLER_SYSTEM_PROMPT",
    "LessonVerdict",
    "ReplayOutcome",
    "ScoringState",
    "build_distillation_prompt",
    "candidate_entry",
    "mine_error_patterns",
    "parse_lesson",
    "score_gain",
    "validate_lesson",
]

#: The role these lessons condition. Kept explicit so the lesson store can
#: route by role exactly as the paper's retrieval governance requires.
SCORER_ROLE = "scorer"

#: PHQ-8 item scores span 0-3, so a three-point swing is the maximum error.
_SCORE_RANGE = 3.0


@dataclass(frozen=True, slots=True)
class ScoringState:
    """One cached (participant, item) scoring decision from a completed run."""

    participant_id: str
    topic: str
    item_key: str
    evidence_tier: str
    evidence_text: str
    predicted: float
    reference: float
    abstained: bool
    frequency_basis: str = "none"
    reasoning: str = ""

    @property
    def signed_error(self) -> float:
        return self.predicted - self.reference

    @property
    def state_id(self) -> str:
        """Stable id for source exclusion and provenance."""
        return canonical_hash(
            {
                "participant": self.participant_id,
                "topic": self.topic,
                "tier": self.evidence_tier,
            }
        )


@dataclass(frozen=True, slots=True)
class ErrorPattern:
    """A hard (item, evidence-tier) cell worth spending a lesson on."""

    topic: str
    evidence_tier: str
    support: int
    mean_signed_error: float
    mean_absolute_error: float
    abstention_rate: float
    source_state_ids: tuple[str, ...]

    @property
    def direction(self) -> str:
        return "under" if self.mean_signed_error < 0 else "over"

    @property
    def systematic_share(self) -> float:
        """How much of the error is bias rather than noise, in ``[0, 1]``.

        A cell whose scores are wrong in both directions equally is difficult,
        not miscalibrated, and a procedural rule cannot fix it. Only cells
        dominated by signed error are worth distilling.
        """
        if self.mean_absolute_error <= 1e-9:
            return 0.0
        return abs(self.mean_signed_error) / self.mean_absolute_error


def mine_error_patterns(
    states: Sequence[ScoringState],
    *,
    min_support: int = 8,
    min_absolute_error: float = 0.5,
    min_systematic_share: float = 0.4,
) -> tuple[ErrorPattern, ...]:
    """Group scoring states into (item, tier) cells and keep the correctable ones.

    Returned strongest-first by ``|mean signed error| * support``, so a caller
    spending a fixed distillation budget spends it where the most total error
    sits rather than on the largest per-case error in a cell of two.
    """
    grouped: dict[tuple[str, str], list[ScoringState]] = {}
    for state in states:
        grouped.setdefault((state.topic, state.evidence_tier), []).append(state)

    patterns: list[ErrorPattern] = []
    for (topic, tier), members in grouped.items():
        if len(members) < min_support:
            continue
        signed = sum(state.signed_error for state in members) / len(members)
        absolute = sum(abs(state.signed_error) for state in members) / len(members)
        pattern = ErrorPattern(
            topic=topic,
            evidence_tier=tier,
            support=len(members),
            mean_signed_error=signed,
            mean_absolute_error=absolute,
            abstention_rate=sum(state.abstained for state in members) / len(members),
            source_state_ids=tuple(sorted(state.state_id for state in members)),
        )
        if pattern.mean_absolute_error < min_absolute_error:
            continue
        if pattern.systematic_share < min_systematic_share:
            continue
        patterns.append(pattern)

    patterns.sort(
        key=lambda item: -abs(item.mean_signed_error) * item.support
    )
    return tuple(patterns)


LESSON_DISTILLER_SYSTEM_PROMPT = (
    "You write procedural rating rules for a clinical rating team. A good rule "
    "names the situation it applies in, one concrete action to take, one "
    "concrete action to avoid, and the observable criterion that tells the "
    "rater it worked. It never names a target score for a specific case, never "
    "mentions a dataset, and never refers to a particular participant — it has "
    "to be usable on someone the writer has never seen. Output valid JSON only."
)


LESSON_JSON_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "trigger": {"type": "string"},
        "do": {"type": "string"},
        "avoid": {"type": "string"},
        "criterion": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["trigger", "do", "avoid", "criterion"],
}


def build_distillation_prompt(
    pattern: ErrorPattern, examples: Sequence[ScoringState], *, max_examples: int = 6
) -> str:
    """Ask for one transferable rule from a cell's mis-scored cases.

    The examples carry the rater's own stated reasoning and the direction it
    went wrong, but not the transcript and not a corrected score: the point is
    to induce *why the procedure failed*, and handing over the answers invites
    a rule that memorizes them.
    """
    direction = (
        "scored TOO LOW (missed severity that was present)"
        if pattern.mean_signed_error < 0
        else "scored TOO HIGH (read severity that was not supported)"
    )
    lines: list[str] = []
    for state in examples[:max_examples]:
        gap = state.reference - state.predicted
        lines.append(
            f"  - reasoning given: {state.reasoning or '(none recorded)'}\n"
            f"    frequency basis used: {state.frequency_basis}"
            f" | abstained: {'yes' if state.abstained else 'no'}"
            f" | direction of error: {'too low' if gap > 0 else 'too high'}"
            f" by about {abs(gap):.1f} of 3"
        )
    body = "\n".join(lines) or "  (no per-case reasoning recorded)"

    return f"""## Situation
Item being rated: {pattern.topic}
Evidence available in these cases: {pattern.evidence_tier}
Across {pattern.support} cases with this profile, raters consistently
{direction}. {pattern.abstention_rate:.0%} of them declined to rate at all.

## What the raters said at the time
{body}

## Task
Write ONE rule that would have corrected this pattern, and that would still be
correct on a different person with the same evidence profile.

Requirements:
  - "trigger" describes the situation, in terms a rater can recognize BEFORE
    knowing the answer;
  - "do" is one concrete action, not a sentiment;
  - "avoid" is the specific mistake being made now;
  - "criterion" is what the rater can observe to tell the rule applied
    correctly;
  - never name a numeric target score for a specific case, a dataset, or a
    participant.

## Output
{{"trigger": "...", "do": "...", "avoid": "...", "criterion": "...",
  "confidence": 0.0 to 1.0}}"""


#: A distilled rule must not smuggle the supervision it was induced from into
#: an inference-time prompt. Phrases naming the reference are rejected outright.
_LABEL_LEAK_RE = re.compile(
    r"\b(ground[\s-]?truth|reference score|gold (?:score|label)|true score|"
    r"phq[\s-]?8?[\s-]?(?:score|total)|correct answer|the label)\b",
    re.IGNORECASE,
)


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


def parse_lesson(raw_text: str, pattern: ErrorPattern) -> Lesson | None:
    """Parse a distilled rule, rejecting empty or supervision-leaking text."""
    payload = _last_object(raw_text)
    if payload is None:
        return None
    fields = {
        key: " ".join(str(payload.get(key, "")).split())
        for key in ("trigger", "do", "avoid", "criterion")
    }
    if any(not value for value in fields.values()):
        return None
    if any(_LABEL_LEAK_RE.search(value) for value in fields.values()):
        return None

    scope = f"{pattern.topic}|{pattern.evidence_tier}"
    confidence = payload.get("confidence", 0.0)
    return Lesson(
        lesson_id=canonical_hash(
            {"role": SCORER_ROLE, "scope": scope, **fields}
        ),
        role=SCORER_ROLE,
        trigger=fields["trigger"],
        do=fields["do"],
        avoid=fields["avoid"],
        criterion=fields["criterion"],
        scope=scope,
        candidate_confidence=(
            float(confidence) if isinstance(confidence, (int, float)) else 0.0
        ),
        source_contrast_ids=pattern.source_state_ids,
    )


def candidate_entry(lesson: Lesson) -> LessonMemoryEntry:
    """Wrap a freshly distilled lesson as an unvalidated candidate."""
    return LessonMemoryEntry(lesson=lesson, governance_history=("distilled",))


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """One paired control/intervention replay from an identical cached state.

    ``control_value`` and ``treatment_value`` are the item scores *after*
    abstention resolution, so a lesson that converts an abstention into a
    correct rating is credited and one that converts it into a wrong rating is
    penalized. ``treatment_grounded`` carries the safety signal: a lesson that
    buys accuracy with fabricated quotes must not pass.
    """

    state_id: str
    participant_id: str
    reference: float
    control_value: float
    treatment_value: float
    treatment_grounded: bool = True
    control_grounded: bool = True

    @property
    def gain(self) -> float:
        return score_gain(self.control_value, self.treatment_value, self.reference)


def score_gain(control: float, treatment: float, reference: float) -> float:
    """Normalized item-score utility gain, clipped to ``[-1, 1]``.

    Follows the paper's ``G_score = clip((|y_control - y| - |y_treatment - y|)
    / 3, -1, 1)``: positive when the intervention moved the estimate toward the
    reference, and bounded so one catastrophic case cannot dominate a mean.
    """
    delta = (abs(control - reference) - abs(treatment - reference)) / _SCORE_RANGE
    return max(-1.0, min(1.0, delta))


@dataclass(frozen=True, slots=True)
class LessonVerdict:
    """Whether a candidate cleared the transfer gate, and on what evidence."""

    promoted: bool
    support: int
    mean_gain: float
    improved_fraction: float
    safety_regressions: int
    reasons: tuple[str, ...] = ()


def validate_lesson(
    outcomes: Sequence[ReplayOutcome],
    *,
    excluded_state_ids: frozenset[str],
    min_support: int = 6,
    min_mean_gain: float = 0.02,
    min_improved_fraction: float = 0.55,
) -> LessonVerdict:
    """Decide promotion from source-excluded paired replays.

    Every threshold is a caller-supplied configuration rather than a constant
    here, because they are tuned on training participants and must be reported
    alongside any promotion claim. The gate deliberately requires *both* a mean
    margin and a majority of states improved: a single large win on one
    participant is not transfer.
    """
    eligible = tuple(
        outcome for outcome in outcomes if outcome.state_id not in excluded_state_ids
    )
    reasons: list[str] = []
    if not eligible:
        return LessonVerdict(False, 0, 0.0, 0.0, 0, ("no_source_excluded_states",))

    gains = [outcome.gain for outcome in eligible]
    mean_gain = sum(gains) / len(gains)
    improved = sum(1 for gain in gains if gain > 0) / len(gains)
    regressions = sum(
        1
        for outcome in eligible
        if outcome.control_grounded and not outcome.treatment_grounded
    )

    if len(eligible) < min_support:
        reasons.append("insufficient_support")
    if mean_gain < min_mean_gain:
        reasons.append("utility_margin_below_threshold")
    if improved < min_improved_fraction:
        reasons.append("too_few_states_improved")
    if regressions:
        reasons.append("safety_regression")

    return LessonVerdict(
        promoted=not reasons,
        support=len(eligible),
        mean_gain=mean_gain,
        improved_fraction=improved,
        safety_regressions=regressions,
        reasons=tuple(reasons),
    )
