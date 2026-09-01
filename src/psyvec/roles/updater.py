"""The sole role allowed to revise proposed topic scores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from psyvec.state import (
    AssessmentState,
    DecisionEvent,
    DecisionEventEngine,
    RoleAction,
    RoleOutput,
    StateContractError,
    replace_topic,
)


@dataclass(frozen=True, slots=True)
class ScoreRevision:
    """A new score and the acquired evidence which supports it."""

    score: int
    evidence_slot_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Updater:
    """Apply score revisions through the decision-event engine."""

    def apply(
        self,
        state: AssessmentState,
        revisions: Mapping[str, ScoreRevision],
        engine: DecisionEventEngine,
    ) -> tuple[AssessmentState, tuple[DecisionEvent, ...]]:
        """Apply each topic revision and return the resulting events."""

        current = state
        events: list[DecisionEvent] = []
        for topic_id, revision in revisions.items():
            if not revision.evidence_slot_ids:
                raise StateContractError(
                    f"topic {topic_id!r} score revision has no evidence basis"
                )
            action_id = f"update-{topic_id}-{current.revision + 1}"
            action = RoleAction(
                action_id=action_id,
                role="updater",
                kind="revise_score",
                payload={
                    "topic_id": topic_id,
                    "score": revision.score,
                    "evidence_slot_ids": revision.evidence_slot_ids,
                },
            )
            output = RoleOutput(action_id=action_id, validation_status="valid")

            def transition(
                before: AssessmentState,
                topic_id: str = topic_id,
                score_revision: ScoreRevision = revision,
            ) -> AssessmentState:
                topic = before.topic(topic_id)
                revised_topic = replace(
                    topic,
                    proposed_score=score_revision.score,
                    score_basis_slot_ids=score_revision.evidence_slot_ids,
                    revision=topic.revision + 1,
                )
                return replace_topic(before, revised_topic)

            current, event = engine.apply(current, action, output, transition)
            events.append(event)
        return current, tuple(events)
