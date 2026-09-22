"""Which unresolved PHQ-8 topic is worth the next acquisition round.

``scripts/run_assessment.py`` (the existing v3 pipeline) expands every topic
independently up to its own round cap — there is no cross-topic choice being
made, so nothing there is actually *adaptive* in the sense the method needs:
"should I ask more about this one topic" is a different question from "which
of the still-unresolved topics is most worth the next round I can afford."

This module answers the second question. The design document's utility
formula is

    U(q) = U_need(q) + alpha * U_experience(q) - beta * R(q) - gamma * B(q)

Burden (``B``) is deliberately not a per-candidate term here: every legal
action consumes exactly one unit of the same shared, global round budget, so
it is constant across candidates at any given decision and cannot change
which one has the highest utility — it only ever affects *when to stop*
scheduling at all. That decision belongs to the caller, which knows the
budget; this module only ever answers "which one next, among what's still
legal," so ``gamma * B(q)`` has no term here and the stopping rule lives in
the acquisition loop instead. This keeps the mechanism auditable: every
weight that exists here is actually doing something to the ranking.

Pure functions on a small state snapshot; no model calls, stdlib only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from psyvec.evaluation.evidence_tiers import TIER_ORDER, EvidenceTier

__all__ = [
    "EXPERIENCE_WEIGHT",
    "REDUNDANCY_WEIGHT",
    "TopicAcquisitionState",
    "experience_match",
    "next_action",
    "redundancy",
    "topic_need",
    "utility",
]

#: alpha — how much a matching promoted interview-strategy lesson can move the
#: ranking. Kept well below 1.0 so a lesson can break a near-tie but can never
#: make an already-well-evidenced topic outrank one nothing has been found for.
EXPERIENCE_WEIGHT = 0.25

#: beta — how strongly rounds already spent on a topic suppress it, so the
#: scheduler does not spend the whole budget re-probing one stubborn topic
#: while others sit untouched.
REDUNDANCY_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class TopicAcquisitionState:
    """One topic's acquisition status at scheduling time.

    ``sufficient`` is ``None`` until the sufficiency role has judged the
    topic at least once this cycle; treated the same as "not yet sufficient"
    for need, but does not by itself mark the topic ``resolved`` — an
    unjudged topic is still a legal action.
    """

    topic: str
    tier: EvidenceTier
    rounds_spent: int
    sufficient: bool | None
    max_rounds_per_topic: int

    @property
    def resolved(self) -> bool:
        """Whether this topic is no longer a legal scheduling action.

        Protocol legality, not utility: a resolved topic is removed from the
        candidate set entirely rather than merely ranked low, so it can never
        be picked no matter how the weights are tuned.
        """
        return self.sufficient is True or self.rounds_spent >= self.max_rounds_per_topic


def topic_need(state: TopicAcquisitionState) -> float:
    """U_need: 1.0 while nothing useful is known, decaying as the tier improves.

    ``TIER_ORDER`` runs weakest-to-strongest (absent, contextual, expanded,
    lexical, direct); need falls linearly from 1.0 at ``absent`` to 0.0 at
    the strongest tier, and drops to 0.0 outright once the sufficiency role
    has actually said so.
    """
    if state.sufficient is True:
        return 0.0
    rank = TIER_ORDER.index(state.tier)
    return max(0.0, 1.0 - rank / (len(TIER_ORDER) - 1))


def redundancy(state: TopicAcquisitionState) -> float:
    """R: how much of this topic's own round budget is already spent, in [0, 1]."""
    if state.max_rounds_per_topic <= 0:
        return 1.0
    return min(1.0, state.rounds_spent / state.max_rounds_per_topic)


def experience_match(
    topic: str, tier: EvidenceTier, lesson_scopes: frozenset[str]
) -> float:
    """U_experience: 1.0 if a promoted interview-strategy lesson covers this
    ``(topic, tier)`` state, else 0.0.

    ``lesson_scopes`` holds ``"{topic}|{tier}"`` strings, matching the scope
    convention already used by the scorer-lesson channel
    (``psyvec.evolution.item_lessons.ErrorPattern`` /
    ``interview_lessons.AcquisitionState``), so both bounded-evolution
    channels key their lessons the same way.
    """
    return 1.0 if f"{topic}|{tier}" in lesson_scopes else 0.0


def utility(state: TopicAcquisitionState, lesson_scopes: frozenset[str]) -> float:
    """U(q) for one candidate topic, with the ``gamma * B(q)`` term omitted
    (see module docstring)."""
    return (
        topic_need(state)
        + EXPERIENCE_WEIGHT * experience_match(state.topic, state.tier, lesson_scopes)
        - REDUNDANCY_WEIGHT * redundancy(state)
    )


def next_action(
    states: Sequence[TopicAcquisitionState],
    *,
    lesson_scopes: frozenset[str] = frozenset(),
) -> str | None:
    """The topic that should receive the next expansion round, or ``None``.

    Only unresolved topics are legal candidates. Ties keep the earliest topic
    in ``states`` (``max`` never replaces on an equal key), so scheduling is
    deterministic given a fixed topic order — no arbitrary tie-break needed.
    """
    candidates = [state for state in states if not state.resolved]
    if not candidates:
        return None
    best = max(candidates, key=lambda state: utility(state, lesson_scopes))
    return best.topic
