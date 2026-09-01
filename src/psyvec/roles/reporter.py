"""Read-only assessment reporting."""

from __future__ import annotations

from dataclasses import dataclass

from psyvec.state import AssessmentState, DecisionEventEngine
from psyvec.state.contracts import reject_label_keys


@dataclass(frozen=True, slots=True)
class TopicReport:
    """The score and evidence cited for one topic."""

    topic_id: str
    final_score: int | None
    status: str
    evidence_slot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        reject_label_keys(
            {
                "topic_id": self.topic_id,
                "final_score": self.final_score,
                "status": self.status,
                "evidence_slot_ids": self.evidence_slot_ids,
            },
            where="topic report",
        )


@dataclass(frozen=True, slots=True)
class AssessmentReport:
    """A label-free report rendered from a single assessment state."""

    topics: tuple[TopicReport, ...]
    total: int
    state_hash: str

    def __post_init__(self) -> None:
        reject_label_keys(
            {
                "topics": self.topics,
                "total": self.total,
                "state_hash": self.state_hash,
            },
            where="assessment report",
        )


@dataclass(frozen=True, slots=True)
class Reporter:
    """Render a report only through the engine's no-mutation gate."""

    def render(
        self, state: AssessmentState, engine: DecisionEventEngine
    ) -> AssessmentReport:
        """Produce the report without emitting a decision event."""

        reports: list[AssessmentReport] = []

        def renderer(current: AssessmentState) -> AssessmentState:
            reports.append(self._build_report(current))
            return self._render_state(current)

        engine.report(state, renderer)
        return reports[0]

    def _render_state(self, state: AssessmentState) -> AssessmentState:
        """Return state unchanged; retained as the read-only engine callback."""

        return state

    def _build_report(self, state: AssessmentState) -> AssessmentReport:
        topics = tuple(
            TopicReport(
                topic_id=topic.topic_id,
                final_score=topic.proposed_score,
                status=topic.status,
                evidence_slot_ids=topic.score_basis_slot_ids,
            )
            for topic in state.topic_states
        )
        return AssessmentReport(
            topics=topics,
            total=sum(topic.final_score or 0 for topic in topics),
            state_hash=state.state_hash,
        )
