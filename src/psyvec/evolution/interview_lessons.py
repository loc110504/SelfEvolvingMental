"""Mine, distil and transfer-verify acquisition-strategy lessons.

``psyvec.evolution.item_lessons`` evolves how the fixed rubric is *applied* to
ambiguous evidence — its lessons live in the scorer's prompt. This module
evolves a different decision: at the query-expansion step
(``psyvec.evaluation.item_scoring.build_query_expansion_prompt``), what kind
of search phrase is worth trying next, given a topic's evidence tier stayed
put through a round of searching. It is the second of the method's two
*bounded* self-evolution channels — both mine → distil → cross-participant
held-out verify → promote/retire, and both leave the fixed PHQ-8 rubric,
scoring thresholds and retrieval mechanics untouched; only the phrasing
guidance handed to the interviewer role at a given ``(topic, evidence tier)``
state can change.

The four stages mirror ``item_lessons`` exactly, adapted to a different
outcome variable:

``mine_low_yield_patterns``
    A "state" here is one (participant, topic, tier-before-a-round) cell. A
    cell is worth spending a lesson on when a round spent there
    *systematically* failed to raise the evidence tier or resolve
    sufficiency — a high, well-supported failure rate, not one unlucky case.

``build_distillation_prompt`` / ``parse_lesson``
    The distiller sees only the topic, the tier the searches were starting
    from, and the search phrases already tried and their (lack of) effect —
    never a transcript, never a participant, never a score — and proposes one
    transferable rule about what to try that those searches did not cover.
    Reuses the same label-leak rejection as ``item_lessons``.

``efficiency_gain`` / ``validate_lesson``
    Cross-state paired validation exactly like ``item_lessons``: each
    candidate is replayed on participants excluded from its own source set,
    control is the cached original queries, intervention is the same state
    with the candidate lesson appended to the expansion prompt, and promotion
    requires minimum support, a minimum mean utility margin and a minimum
    fraction of states improved.

Storage and lifecycle reuse :mod:`psyvec.lessons.core` exactly as
``item_lessons`` does, so a promoted acquisition lesson carries the same
provenance and governance record as a promoted scoring lesson.

No model calls here: prompt construction, parsing and statistics only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psyvec.evaluation.evidence_tiers import TIER_ORDER, EvidenceTier
from psyvec.evaluation.item_scoring import ITEM_MANIFESTATIONS
from psyvec.evaluation.response_parsing import extract_json_objects, strip_reasoning
from psyvec.evolution.item_lessons import candidate_entry
from psyvec.lessons.core import Lesson
from psyvec.state.contracts import canonical_hash

__all__ = [
    "ACQUISITION_LESSON_JSON_SCHEMA",
    "ACQUISITION_LESSON_DISTILLER_SYSTEM_PROMPT",
    "INTERVIEWER_ROLE",
    "AcquisitionPattern",
    "AcquisitionReplayOutcome",
    "AcquisitionState",
    "AcquisitionVerdict",
    "acquisition_value",
    "build_acquisition_distillation_prompt",
    "candidate_entry",
    "efficiency_gain",
    "mine_low_yield_patterns",
    "parse_acquisition_lesson",
    "validate_acquisition_lesson",
]

#: The role these lessons condition: the query-expansion ("interviewer") step.
INTERVIEWER_ROLE = "interviewer"

#: Highest possible ``acquisition_value``: strongest tier rank (4, 0-indexed)
#: plus the resolved bonus (1), used to normalize ``efficiency_gain`` into
#: [-1, 1] the same way ``item_lessons.score_gain`` normalizes by the PHQ-8
#: item range.
_MAX_ACQUISITION_VALUE = (len(TIER_ORDER) - 1) + 1.0


@dataclass(frozen=True, slots=True)
class AcquisitionState:
    """One cached (participant, topic) acquisition-round decision.

    ``tier_before``/``resolved_before`` describe the topic immediately before
    the round being mined; ``tier_after``/``resolved_after`` describe it
    immediately after. ``queries_tried`` is what the interviewer role
    actually searched for in that round. ``round_index`` is the round's
    position within its own (participant, topic) acquisition sequence
    (matching ``psyvec.memory.patient_evidence.EvidenceRecord.round_index``'s
    numbering) — unlike a scoring state, one topic can be scheduled for more
    than one round, so this is what keeps ``state_id`` unique per round rather
    than per (participant, topic, tier): two failed rounds on the same
    stubborn topic can easily share the same starting tier.
    """

    participant_id: str
    topic: str
    tier_before: EvidenceTier
    resolved_before: bool
    tier_after: EvidenceTier
    resolved_after: bool
    queries_tried: tuple[str, ...]
    round_index: int

    @property
    def improved(self) -> bool:
        if self.resolved_after and not self.resolved_before:
            return True
        return TIER_ORDER.index(self.tier_after) > TIER_ORDER.index(self.tier_before)

    @property
    def state_id(self) -> str:
        """Stable id for source exclusion, provenance and replay lookup."""
        return canonical_hash(
            {
                "participant": self.participant_id,
                "topic": self.topic,
                "round_index": self.round_index,
            }
        )


def acquisition_value(tier: EvidenceTier, resolved: bool) -> float:
    """Scalar acquisition quality: tier strength plus a resolved bonus."""
    return float(TIER_ORDER.index(tier)) + (1.0 if resolved else 0.0)


@dataclass(frozen=True, slots=True)
class AcquisitionPattern:
    """A hard (topic, tier-before) cell worth spending a lesson on."""

    topic: str
    tier_before: EvidenceTier
    support: int
    yield_rate: float
    source_state_ids: tuple[str, ...]
    sample_queries: tuple[str, ...] = ()


def mine_low_yield_patterns(
    states: Sequence[AcquisitionState],
    *,
    min_support: int = 8,
    max_yield_rate: float = 0.35,
) -> tuple[AcquisitionPattern, ...]:
    """Group acquisition rounds into (topic, tier) cells and keep the wasteful ones.

    Returned worst-first by total rounds wasted (``(1 - yield_rate) *
    support``), so a caller spending a fixed distillation budget spends it
    where the most total waste sits.
    """
    grouped: dict[tuple[str, EvidenceTier], list[AcquisitionState]] = {}
    for state in states:
        grouped.setdefault((state.topic, state.tier_before), []).append(state)

    patterns: list[AcquisitionPattern] = []
    for (topic, tier_before), members in grouped.items():
        if len(members) < min_support:
            continue
        yield_rate = sum(1 for state in members if state.improved) / len(members)
        if yield_rate > max_yield_rate:
            continue
        sample = tuple(
            query
            for state in members[:6]
            for query in state.queries_tried[:2]
        )[:8]
        patterns.append(
            AcquisitionPattern(
                topic=topic,
                tier_before=tier_before,
                support=len(members),
                yield_rate=yield_rate,
                source_state_ids=tuple(sorted(state.state_id for state in members)),
                sample_queries=sample,
            )
        )

    patterns.sort(key=lambda item: -(1.0 - item.yield_rate) * item.support)
    return tuple(patterns)


ACQUISITION_LESSON_DISTILLER_SYSTEM_PROMPT = (
    "You write procedural search-strategy rules for a clinical interview "
    "retrieval team. A good rule names the situation it applies in, one "
    "concrete kind of search phrase to try, one concrete kind to stop trying, "
    "and the observable criterion that tells the searcher it worked. It never "
    "names a participant, a dataset, or a target outcome — it has to be "
    "usable on an interview transcript the writer has never seen. Output "
    "valid JSON only."
)

ACQUISITION_LESSON_JSON_SCHEMA: Mapping[str, Any] = {
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


def build_acquisition_distillation_prompt(
    pattern: AcquisitionPattern, *, max_sample_queries: int = 8
) -> str:
    """Ask for one transferable search-strategy rule from a wasteful cell.

    Shows only the topic, the tier searches kept starting from, and the
    phrases already tried — never a transcript, a participant, or whether the
    underlying symptom was actually present. The point is to induce a
    different *kind* of search to try, not to memorize an answer.
    """
    manifestations = ITEM_MANIFESTATIONS.get(pattern.topic, "")
    tried = "\n".join(
        f"  - {query}" for query in pattern.sample_queries[:max_sample_queries]
    ) or "  (no queries recorded)"
    return f"""## Situation
Topic being searched for: {pattern.topic}
What this topic covers in ordinary speech: {manifestations}
Evidence tier before searching: {pattern.tier_before}
Across {pattern.support} cases starting from this tier, a round of searching
raised the tier or settled the topic only {pattern.yield_rate:.0%} of the time.

## Search phrases already tried in these cases (a sample)
{tried}

## Task
Write ONE rule that would find something the searches above did not, and
that would still apply to a different transcript with the same starting
tier and the same topic.

Requirements:
  - "trigger" describes the situation, in terms a searcher can recognize
    BEFORE knowing whether anything new will be found;
  - "do" is one concrete kind of search phrase to try (e.g. a different life
    domain, a different register, an indirect or euphemistic phrasing, a
    check on the symptom's opposite pole) — not a sentiment;
  - "avoid" is the specific pattern the searches above kept repeating;
  - "criterion" is what the searcher can observe to tell the rule applied
    correctly (a new turn found, the tier rising, sufficiency being reached);
  - never name a dataset, a participant, or a specific outcome/score.

## Output
{{"trigger": "...", "do": "...", "avoid": "...", "criterion": "...",
  "confidence": 0.0 to 1.0}}"""


#: A distilled rule must not smuggle target-outcome supervision into an
#: inference-time prompt, same discipline as ``item_lessons._LABEL_LEAK_RE``.
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


def parse_acquisition_lesson(
    raw_text: str, pattern: AcquisitionPattern
) -> Lesson | None:
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

    scope = f"{pattern.topic}|{pattern.tier_before}"
    confidence = payload.get("confidence", 0.0)
    return Lesson(
        lesson_id=canonical_hash({"role": INTERVIEWER_ROLE, "scope": scope, **fields}),
        role=INTERVIEWER_ROLE,
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


@dataclass(frozen=True, slots=True)
class AcquisitionReplayOutcome:
    """One paired control/intervention replay from an identical cached state.

    ``control_value``/``treatment_value`` are :func:`acquisition_value` for
    the round as it actually ran versus as it would run with the candidate
    lesson appended to the expansion prompt.
    """

    state_id: str
    participant_id: str
    control_value: float
    treatment_value: float

    @property
    def gain(self) -> float:
        return efficiency_gain(self.control_value, self.treatment_value)


def efficiency_gain(control_value: float, treatment_value: float) -> float:
    """Normalized acquisition-quality gain, clipped to ``[-1, 1]``.

    Mirrors ``item_lessons.score_gain``'s normalize-and-clip shape: positive
    when the intervention raised the tier or resolved the topic, bounded so
    one case cannot dominate a mean.
    """
    delta = (treatment_value - control_value) / _MAX_ACQUISITION_VALUE
    return max(-1.0, min(1.0, delta))


@dataclass(frozen=True, slots=True)
class AcquisitionVerdict:
    """Whether a candidate acquisition lesson cleared the transfer gate."""

    promoted: bool
    support: int
    mean_gain: float
    improved_fraction: float
    reasons: tuple[str, ...] = ()


def validate_acquisition_lesson(
    outcomes: Sequence[AcquisitionReplayOutcome],
    *,
    excluded_state_ids: frozenset[str],
    min_support: int = 6,
    min_mean_gain: float = 0.02,
    min_improved_fraction: float = 0.55,
) -> AcquisitionVerdict:
    """Decide promotion from source-excluded paired replays.

    Same shape as ``item_lessons.validate_lesson``: every threshold is
    caller-supplied and must be reported alongside any promotion claim, and
    promotion requires both a mean margin and a majority of states improved.
    """
    eligible = tuple(
        outcome for outcome in outcomes if outcome.state_id not in excluded_state_ids
    )
    reasons: list[str] = []
    if not eligible:
        return AcquisitionVerdict(False, 0, 0.0, 0.0, ("no_source_excluded_states",))

    gains = [outcome.gain for outcome in eligible]
    mean_gain = sum(gains) / len(gains)
    improved = sum(1 for gain in gains if gain > 0) / len(gains)

    if len(eligible) < min_support:
        reasons.append("insufficient_support")
    if mean_gain < min_mean_gain:
        reasons.append("utility_margin_below_threshold")
    if improved < min_improved_fraction:
        reasons.append("too_few_states_improved")

    return AcquisitionVerdict(
        promoted=not reasons,
        support=len(eligible),
        mean_gain=mean_gain,
        improved_fraction=improved,
        reasons=tuple(reasons),
    )
