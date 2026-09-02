"""Baseline AgentMental topic stopping behavior through typed decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from psyvec.state import (
    AssessmentState,
    DecisionEventEngine,
    RoleAction,
    RoleOutput,
)

MAX_QUESTIONS_PER_TOPIC = 3


def should_continue(necessity_score: int, asked_questions: int) -> bool:
    """Match AgentMental's necessity decision after a question is asked."""

    return necessity_score == 2 or (
        necessity_score == 1 and asked_questions < 2
    )


def conduct_topic_interview(
    state: AssessmentState,
    engine: DecisionEventEngine,
    necessity_score: int,
    question_factory: Callable[[], str],
) -> AssessmentState:
    """Ask one topic's questions, recording each as a typed decision event."""

    asked_questions = 0
    while asked_questions < MAX_QUESTIONS_PER_TOPIC:
        turn_id = f"{state.current_topic_id or 'topic'}-{asked_questions + 1}"
        action = RoleAction(
            action_id=turn_id,
            role="interviewer",
            kind="ask_question",
            payload={"question": question_factory()},
        )
        output = RoleOutput(action_id=turn_id, validation_status="valid")

        def append_turn(
            current: AssessmentState, turn_id: str = turn_id
        ) -> AssessmentState:
            return replace(
                current,
                dialogue_turn_ids=current.dialogue_turn_ids + (turn_id,),
                revision=current.revision + 1,
            )

        state, _ = engine.apply(
            state,
            action,
            output,
            append_turn,
        )
        asked_questions += 1
        if not should_continue(necessity_score, asked_questions):
            break
    return state
