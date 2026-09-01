import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evolution import (  # noqa: E402
    BranchCandidate,
    BranchInvariantError,
    EvolutionContractError,
    ExperienceBuffer,
    ExperienceRecord,
    ParticipantProfile,
    assert_branch_invariants,
)
from psyvec.state import (  # noqa: E402
    AssessmentState,
    DecisionEventEngine,
    EvidenceSlot,
    LabelLeakageError,
    ProvenanceRef,
    RoleAction,
    RoleOutput,
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
    seed=7,
)


def scored_event():
    slot = EvidenceSlot(
        slot_id="sleep-frequency",
        kind="frequency",
        status="known",
        value="most days",
        source_turn_ids=("turn-1",),
    )
    state = AssessmentState(
        case_id="case-1",
        scale_version="PHQ-8",
        topic_states=(
            TopicState(topic_id="sleep", rubric_version="v1", evidence_slots=(slot,)),
        ),
    )
    engine = DecisionEventEngine(provenance=PROVENANCE)
    action = RoleAction(action_id="a1", role="scorer", kind="propose_score")

    def transition(current: AssessmentState) -> AssessmentState:
        topic = replace(
            current.topic("sleep"),
            proposed_score=2,
            score_basis_slot_ids=("sleep-frequency",),
            status="completed",
            revision=1,
        )
        return replace_topic(current, topic)

    _after, event = engine.apply(
        state, action, RoleOutput("a1", "valid"), transition
    )
    return state, event


def experience(event, *, split_id: str = "evolve") -> ExperienceRecord:
    return ExperienceRecord(
        experience_id="exp-1",
        split_id=split_id,
        participant_ref="participant-1",
        role=event.role,
        decision_event=event,
        offline_state_hash=event.pre_state_hash,
        outcome="applied",
        provenance=PROVENANCE,
        seed=7,
    )


def branch(**overrides) -> BranchCandidate:
    base = {
        "branch_id": "b1",
        "branch_set_id": "set-1",
        "starting_state_hash": "state-hash",
        "profile_hash": "profile-hash",
        "policy_id": "policy-0",
        "simulator_version": "sim-1",
        "role": "interviewer",
        "action_kind": "ask",
        "generation_strategy": "policy-low-temperature",
        "seed": 7,
    }
    base.update(overrides)
    return BranchCandidate(**base)


class ExperienceSpineTests(unittest.TestCase):
    def test_frozen_profile_hash_and_unknown_facts(self) -> None:
        profile = ParticipantProfile(
            profile_id="p1",
            simulator_version="sim-1",
            facts={"sleep": "poor"},
            unknown_slots=("appetite",),
        )

        self.assertEqual(profile.profile_hash, profile.profile_hash)
        self.assertTrue(profile.supports("sleep"))
        self.assertFalse(profile.supports("appetite"))

        with self.assertRaisesRegex(EvolutionContractError, "known and unknown"):
            ParticipantProfile(
                profile_id="p2",
                simulator_version="sim-1",
                facts={"sleep": "poor"},
                unknown_slots=("sleep",),
            )
        with self.assertRaises(LabelLeakageError):
            ParticipantProfile(
                profile_id="p3",
                simulator_version="sim-1",
                facts={"phq8_score": "12"},
            )

    def test_branch_set_shares_state_profile_and_seed(self) -> None:
        assert_branch_invariants([branch(), branch(branch_id="b2")])

        with self.assertRaisesRegex(BranchInvariantError, "seed"):
            assert_branch_invariants([branch(), branch(branch_id="b2", seed=8)])
        with self.assertRaisesRegex(BranchInvariantError, "starting_state_hash"):
            assert_branch_invariants(
                [branch(), branch(branch_id="b2", starting_state_hash="other")]
            )
        with self.assertRaisesRegex(BranchInvariantError, "at least one"):
            assert_branch_invariants([])

    def test_experience_must_cite_its_own_pre_state_and_role(self) -> None:
        _state, event = scored_event()

        record = experience(event)
        self.assertEqual(record.offline_state_hash, event.pre_state_hash)

        with self.assertRaisesRegex(EvolutionContractError, "own pre-state"):
            replace(record, offline_state_hash="wrong")
        with self.assertRaisesRegex(EvolutionContractError, "role disagrees"):
            replace(record, role="updater")

    def test_buffer_guards_split_and_rejects_label_canary(self) -> None:
        _state, event = scored_event()
        buffer = ExperienceBuffer(split_id="evolve")

        buffer.append(experience(event), policy_input={"topic": "sleep"})
        self.assertEqual(len(buffer.export()), 1)

        with self.assertRaisesRegex(EvolutionContractError, "belongs to split"):
            buffer.append(experience(event, split_id="final-test"))
        with self.assertRaises(LabelLeakageError):
            buffer.append(experience(event), policy_input={"scale_scores": {}})
        with self.assertRaisesRegex(EvolutionContractError, "holds split"):
            buffer.export(split_id="final-test")

    def test_disabled_buffer_collects_nothing_for_rollback(self) -> None:
        _state, event = scored_event()
        buffer = ExperienceBuffer(split_id="evolve", enabled=False)

        buffer.append(experience(event))

        self.assertEqual(buffer.export(), ())


if __name__ == "__main__":
    unittest.main()
