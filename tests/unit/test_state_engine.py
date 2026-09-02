import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.state import (  # noqa: E402
    AssessmentState,
    DecisionEventEngine,
    EvidenceSlot,
    LabelLeakageError,
    ProvenanceRef,
    ReporterMutationError,
    RoleAction,
    RoleOutput,
    StateContractError,
    TopicState,
    replace_topic,
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
                topic_id="sleep", rubric_version="v1", evidence_slots=(slot,)
            ),
        ),
    )


class StateEngineTests(unittest.TestCase):
    def test_decision_records_typed_pre_and_post_state_with_provenance(self) -> None:
        engine = DecisionEventEngine(provenance=PROVENANCE)
        before = state()
        action = RoleAction(action_id="a1", role="scorer", kind="propose_score")
        output = RoleOutput(action_id="a1", validation_status="valid")

        def transition(current: AssessmentState) -> AssessmentState:
            topic = replace(
                current.topic("sleep"),
                proposed_score=2,
                score_basis_slot_ids=("sleep-frequency",),
                status="completed",
                revision=1,
            )
            return replace_topic(current, topic)

        after, event = engine.apply(before, action, output, transition)

        self.assertEqual(event.pre_state_hash, before.state_hash)
        self.assertEqual(event.post_state_hash, after.state_hash)
        self.assertEqual(event.provenance, PROVENANCE)
        self.assertEqual(event.decision_index, 0)
        self.assertGreater(after.revision, before.revision)
        self.assertEqual(event.state_patch["topics"]["sleep"]["proposed_score"], 2)

    def test_non_monotonic_revision_is_rejected(self) -> None:
        engine = DecisionEventEngine(provenance=PROVENANCE)
        action = RoleAction(action_id="a1", role="interviewer", kind="ask")
        output = RoleOutput(action_id="a1", validation_status="valid")

        with self.assertRaisesRegex(StateContractError, "monotonic"):
            engine.apply(state(), action, output, lambda current: current)

    def test_reporter_cannot_emit_an_event_or_change_state(self) -> None:
        engine = DecisionEventEngine(provenance=PROVENANCE)
        action = RoleAction(action_id="a1", role="reporter", kind="render")  # type: ignore[arg-type]
        output = RoleOutput(action_id="a1", validation_status="valid")

        with self.assertRaises(ReporterMutationError):
            engine.apply(state(), action, output, lambda current: current)

        with self.assertRaises(ReporterMutationError):
            engine.report(
                state(),
                lambda current: replace(current, revision=current.revision + 1),
            )
        self.assertEqual(engine.events, [])

    def test_only_the_updater_may_revise_a_proposed_score(self) -> None:
        engine = DecisionEventEngine(provenance=PROVENANCE)
        scored = replace(
            state(),
            topic_states=(
                replace(
                    state().topic("sleep"),
                    proposed_score=1,
                    score_basis_slot_ids=("sleep-frequency",),
                ),
            ),
        )

        def revise(current: AssessmentState) -> AssessmentState:
            topic = replace(current.topic("sleep"), proposed_score=3)
            return replace_topic(current, topic)

        evaluator = RoleAction(action_id="a1", role="evaluator", kind="revise")
        with self.assertRaisesRegex(StateContractError, "may not revise"):
            engine.apply(
                scored, evaluator, RoleOutput("a1", "valid"), revise
            )

        updater = RoleAction(action_id="a2", role="updater", kind="revise")
        after, event = engine.apply(
            scored, updater, RoleOutput("a2", "valid"), revise
        )
        self.assertEqual(after.topic("sleep").proposed_score, 3)
        self.assertEqual(event.role, "updater")

    def test_labels_never_enter_a_policy_visible_record(self) -> None:
        with self.assertRaises(LabelLeakageError):
            RoleAction(
                action_id="a1",
                role="scorer",
                kind="propose_score",
                payload={"scale_scores": {"sleep": 3}},
            )
        with self.assertRaises(LabelLeakageError):
            RoleOutput(
                action_id="a1",
                validation_status="valid",
                parsed_payload={"real_interview": ["transcript"]},
            )

    def test_evidence_and_score_invariants(self) -> None:
        with self.assertRaisesRegex(StateContractError, "known without a source"):
            EvidenceSlot(slot_id="s1", kind="frequency", status="known", value="often")
        with self.assertRaisesRegex(StateContractError, "no evidence basis"):
            TopicState(topic_id="sleep", rubric_version="v1", proposed_score=2)
        with self.assertRaisesRegex(StateContractError, "unknown evidence"):
            TopicState(
                topic_id="sleep",
                rubric_version="v1",
                proposed_score=2,
                score_basis_slot_ids=("nope",),
            )


if __name__ == "__main__":
    unittest.main()
