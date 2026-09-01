from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.roles import Reporter, ScoreRevision, Updater  # noqa: E402
from psyvec.state import (  # noqa: E402
    AssessmentState,
    DecisionEventEngine,
    EvidenceSlot,
    ProvenanceRef,
    ReporterMutationError,
    StateContractError,
    TopicState,
)

PROVENANCE = ProvenanceRef(
    run_id="run-1",
    code_sha="sha",
    config_hash="config",
    prompt_version="agentmental-v1",
    policy_id="policy-0",
    model_id="qwen2.5-72b",
)


def state() -> AssessmentState:
    slot = EvidenceSlot(
        slot_id="sleep-frequency",
        kind="frequency",
        status="known",
        value="most days",
        source_turn_ids=("turn-1",),
    )
    return AssessmentState(
        case_id="case-1",
        scale_version="PHQ-8",
        topic_states=(
            TopicState(
                topic_id="sleep",
                rubric_version="v1",
                evidence_slots=(slot,),
                proposed_score=1,
                score_basis_slot_ids=("sleep-frequency",),
            ),
        ),
    )


class RoleTests(unittest.TestCase):
    def test_updater_revision_emits_pre_and_post_event(self) -> None:
        before = state()
        after, events = Updater().apply(
            before,
            {"sleep": ScoreRevision(3, ("sleep-frequency",))},
            DecisionEventEngine(PROVENANCE),
        )

        self.assertEqual(events[0].pre_state_hash, before.state_hash)
        self.assertEqual(events[0].post_state_hash, after.state_hash)
        self.assertEqual(events[0].role, "updater")

    def test_updater_rejects_revision_without_evidence(self) -> None:
        with self.assertRaisesRegex(StateContractError, "no evidence basis"):
            Updater().apply(
                state(),
                {"sleep": ScoreRevision(3, ())},
                DecisionEventEngine(PROVENANCE),
            )

    def test_reporter_cites_scores_without_events(self) -> None:
        engine = DecisionEventEngine(PROVENANCE)
        report = Reporter().render(state(), engine)

        self.assertEqual(report.topics[0].evidence_slot_ids, ("sleep-frequency",))
        self.assertEqual(engine.events, [])

    def test_reporter_mutation_is_rejected(self) -> None:
        class MutatingReporter(Reporter):
            def _render_state(self, current: AssessmentState) -> AssessmentState:
                return replace(current, revision=current.revision + 1)

        with self.assertRaises(ReporterMutationError):
            MutatingReporter().render(state(), DecisionEventEngine(PROVENANCE))


if __name__ == "__main__":
    unittest.main()
