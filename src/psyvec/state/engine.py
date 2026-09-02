"""Decision event engine: every state change is a typed pre/post event."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from psyvec.state.contracts import (
    AssessmentState,
    DecisionEvent,
    DecisionRole,
    ProvenanceRef,
    RoleAction,
    RoleOutput,
    StateContractError,
)

#: Roles allowed to emit a DecisionEvent. Reporter is deliberately absent.
DECISION_ROLES: frozenset[str] = frozenset(
    {"interviewer", "evaluator", "scorer", "updater"}
)

#: Only the Updater may revise an already-proposed item score.
SCORE_REVISING_ROLES: frozenset[str] = frozenset({"updater"})

Transition = Callable[[AssessmentState], AssessmentState]


class ReporterMutationError(StateContractError):
    """Raised when a read-only reporting role attempts to change state."""


@dataclass(slots=True)
class DecisionEventEngine:
    """Apply role transitions and record a typed event for each one."""

    provenance: ProvenanceRef
    events: list[DecisionEvent] = field(default_factory=list)
    _indexes: itertools.count[int] = field(
        default_factory=lambda: itertools.count(0), init=False
    )

    def apply(
        self,
        state: AssessmentState,
        action: RoleAction,
        output: RoleOutput,
        transition: Transition,
        *,
        outcome: str = "applied",
    ) -> tuple[AssessmentState, DecisionEvent]:
        """Run one decision and return the post-state plus its event."""

        if action.role not in DECISION_ROLES:
            raise ReporterMutationError(
                f"role {action.role!r} may not emit a decision event"
            )
        if output.action_id != action.action_id:
            raise StateContractError("output does not belong to this action")

        pre_hash = state.state_hash
        post_state = transition(state)
        if post_state.revision <= state.revision:
            raise StateContractError("state revision must increase monotonically")
        self._reject_unauthorized_score_revision(state, post_state, action.role)

        decision_index = next(self._indexes)
        event = DecisionEvent(
            event_id=f"{state.case_id}-{decision_index}",
            case_id=state.case_id,
            decision_index=decision_index,
            role=action.role,
            action=action,
            output=output,
            pre_state_hash=pre_hash,
            post_state_hash=post_state.state_hash,
            state_patch=_state_patch(state, post_state),
            provenance=self.provenance,
            outcome=outcome,
        )
        self.events.append(event)
        return post_state, event

    def report(self, state: AssessmentState, renderer: Transition) -> AssessmentState:
        """Run a read-only reporting pass: no event, no state change permitted."""

        before = state.state_hash
        rendered = renderer(state)
        if rendered.state_hash != before:
            raise ReporterMutationError("reporter must not change assessment state")
        return rendered

    def _reject_unauthorized_score_revision(
        self, before: AssessmentState, after: AssessmentState, role: DecisionRole
    ) -> None:
        if role in SCORE_REVISING_ROLES:
            return
        for old_topic in before.topic_states:
            if old_topic.proposed_score is None:
                continue
            new_topic = after.topic(old_topic.topic_id)
            if new_topic.proposed_score != old_topic.proposed_score:
                raise StateContractError(
                    f"role {role!r} may not revise the score of topic "
                    f"{old_topic.topic_id!r}"
                )


def _state_patch(
    before: AssessmentState, after: AssessmentState
) -> Mapping[str, Any]:
    """Minimal patch describing what the transition changed."""

    patch: dict[str, Any] = {
        "revision": {"from": before.revision, "to": after.revision}
    }
    changed_topics = {
        after_topic.topic_id: {
            "status": after_topic.status,
            "proposed_score": after_topic.proposed_score,
            "revision": after_topic.revision,
        }
        for before_topic, after_topic in zip(
            before.topic_states, after.topic_states, strict=True
        )
        if before_topic != after_topic
    }
    if changed_topics:
        patch["topics"] = changed_topics
    if before.current_topic_id != after.current_topic_id:
        patch["current_topic_id"] = after.current_topic_id
    if before.dialogue_turn_ids != after.dialogue_turn_ids:
        patch["dialogue_turn_ids"] = list(after.dialogue_turn_ids)
    return patch
