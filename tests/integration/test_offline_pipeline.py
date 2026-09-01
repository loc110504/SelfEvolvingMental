"""Offline composition test: every layer wired together with no live model.

This is the Doc 06 integration level, run entirely against injected fakes: one
topic interviewed, scored, updated and reported; the decision turned into offline
experience, mined, branched, contrasted, distilled into a retrievable lesson, and
finally promoted through the policy registry with a reproducibility manifest.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.assessment.stopping import conduct_topic_interview  # noqa: E402
from psyvec.evolution import (  # noqa: E402
    BranchResult,
    ExperienceBuffer,
    ExperienceRecord,
    ParticipantProfile,
    UtilityBreakdown,
    UtilityWeights,
    VerifiedContrast,
    build_branch_set,
    build_preference_pair,
    mine_hard_states,
    verify_local,
)
from psyvec.lessons import (  # noqa: E402
    LessonStore,
    ValidationResult,
    distill,
    validate,
)
from psyvec.policy import (  # noqa: E402
    PolicyManifest,
    PolicyRegistry,
    RoleRegressionMetric,
    evaluate_paired,
)
from psyvec.research import (  # noqa: E402
    RunManifest,
    SplitManifest,
    SplitRegistry,
    split_hash,
)
from psyvec.roles.reporter import Reporter  # noqa: E402
from psyvec.roles.updater import ScoreRevision, Updater  # noqa: E402
from psyvec.state import (  # noqa: E402
    AssessmentState,
    DecisionEventEngine,
    EvidenceSlot,
    ProvenanceRef,
    RoleAction,
    RoleOutput,
    TopicState,
    replace_topic,
)

ROLES = ("interviewer", "evaluator", "scorer", "updater")
SEEDS = (11, 12)
SEED = 11
PROVENANCE = ProvenanceRef(
    run_id="run-1",
    code_sha="code-sha",
    config_hash="config-hash",
    prompt_version="agentmental-v1",
    policy_id="policy-0",
    model_id="qwen2.5-72b",
    seed=SEED,
)
PROFILE = ParticipantProfile(
    profile_id="profile-1",
    simulator_version="sim-1",
    facts={"sleep": "poor for two weeks"},
    unknown_slots=("appetite",),
)


def starting_state() -> AssessmentState:
    slot = EvidenceSlot(
        slot_id="sleep-frequency",
        kind="frequency",
        status="known",
        value="most nights",
        source_turn_ids=("turn-0",),
    )
    return AssessmentState(
        case_id="case-1",
        scale_version="PHQ-8",
        topic_states=(
            TopicState(topic_id="sleep", rubric_version="v1", evidence_slots=(slot,)),
        ),
        current_topic_id="sleep",
        dialogue_turn_ids=("turn-0",),
    )


class OfflinePipelineTests(unittest.TestCase):
    def test_one_topic_flows_from_split_to_promoted_policy(self) -> None:
        # 1. Frozen, authorized evolve split; final-test stays untouched.
        splits = SplitRegistry()
        evolve = SplitManifest(
            split_id="evolve-1",
            kind="evolve",
            participant_ids=("participant-1",),
            authorized=True,
        )
        final_test = SplitManifest(
            split_id="final-1",
            kind="final-test",
            participant_ids=("participant-9",),
            authorized=True,
        )
        splits.register(evolve)
        splits.register(final_test)
        splits.authorize_use("evolve-1", "collection", actor="pipeline")

        # 2. Interview one topic under the characterized stopping matrix.
        engine = DecisionEventEngine(provenance=PROVENANCE)
        state = conduct_topic_interview(
            starting_state(), engine, 1, lambda: "How has your sleep been?"
        )
        self.assertEqual(len(engine.events), 2)
        interview_event = engine.events[0]

        # 3. Scorer proposes, Updater revises, Reporter only reads.
        scorer_action = RoleAction(
            action_id="score-sleep", role="scorer", kind="propose_score"
        )
        state, _score_event = engine.apply(
            state,
            scorer_action,
            RoleOutput("score-sleep", "valid"),
            lambda current: replace_topic(
                current,
                replace(
                    current.topic("sleep"),
                    proposed_score=1,
                    score_basis_slot_ids=("sleep-frequency",),
                    status="completed",
                    revision=1,
                ),
            ),
        )
        state, update_events = Updater().apply(
            state,
            {"sleep": ScoreRevision(score=2, evidence_slot_ids=("sleep-frequency",))},
            engine,
        )
        self.assertEqual(state.topic("sleep").proposed_score, 2)
        self.assertEqual(update_events[0].role, "updater")

        events_before_report = len(engine.events)
        report = Reporter().render(state, engine)
        self.assertEqual(len(engine.events), events_before_report)
        self.assertEqual(report.total, 2)
        self.assertEqual(report.topics[0].evidence_slot_ids, ("sleep-frequency",))

        # 4. One decision becomes offline experience in its own split.
        buffer = ExperienceBuffer(split_id="evolve-1")
        experience = ExperienceRecord(
            experience_id="exp-1",
            split_id="evolve-1",
            participant_ref="participant-1",
            role=interview_event.role,
            decision_event=interview_event,
            offline_state_hash=interview_event.pre_state_hash,
            outcome="applied",
            provenance=PROVENANCE,
            seed=SEED,
        )
        buffer.append(experience, policy_input={"topic": "sleep"})
        self.assertEqual(len(buffer.export()), 1)

        # 5. Mine a hard state, then branch it under one frozen profile and seed.
        hard_states = mine_hard_states(
            buffer.export(),
            threshold=0.5,
            signal_extractor=lambda _record: (0.9, 0.8, 0.7),
        )
        self.assertTrue(hard_states[0].selected)
        branches = build_branch_set(
            hard_states[0],
            experience,
            profile=PROFILE,
            policy_id="policy-0",
            simulator_version=PROFILE.simulator_version,
            seed=SEED,
            strategies=("policy-low-temperature", "exploratory"),
        )
        self.assertEqual(len(branches), 2)
        self.assertEqual({branch.seed for branch in branches}, {SEED})

        # 6. Replay outcomes, verify one local contrast, build the preference pair.
        results = [
            BranchResult(
                branch=branches[0],
                response="asks about the two-week window",
                asserted_fact_keys=("sleep",),
                utility=_utility(3.0),
            ),
            BranchResult(
                branch=branches[1],
                response="repeats the previous question",
                asserted_fact_keys=("sleep",),
                utility=_utility(0.5),
            ),
        ]
        contrast = verify_local(
            results,
            role="interviewer",
            weights=UtilityWeights(config_version="dev-1"),
            pair_margin=1.0,
            contrast_id="contrast-1",
            profile=PROFILE,
        )
        assert isinstance(contrast, VerifiedContrast)
        self.assertEqual(contrast.profile_hash, PROFILE.profile_hash)
        self.assertEqual(contrast.seed, SEED)

        pair = build_preference_pair(
            contrast,
            pair_id="pair-1",
            chosen=results[0].response,
            rejected=results[1].response,
            accepted_policy_id="policy-0",
            split_id="evolve-1",
            label_supervision_used=False,
        )
        self.assertEqual(pair.contrast_id, contrast.contrast_id)

        # 7. Fast path: an unvalidated lesson is unreachable; a promoted one is not.
        store = LessonStore()
        candidate = distill(
            [contrast],
            trigger="sleep evidence is thin",
            do="ask for the duration window",
            avoid="repeating the previous question",
            criterion="a duration slot becomes known",
            scope="sleep",
            lesson_id="lesson-1",
        )
        store.append(candidate)
        self.assertEqual(store.retrieve("interviewer", "sleep", limit=5), ())

        validated = validate(
            candidate,
            [
                ValidationResult(
                    result_id="v1", source_contrast_id="other-contrast", supports=True
                )
            ],
            excluded_source_ids=set(candidate.lesson.source_contrast_ids),
            minimum_support=1,
        )
        self.assertEqual(validated.lifecycle, "validated")
        store.append(validated)
        store.promote("lesson-1")
        retrieved = store.retrieve("interviewer", "sleep", limit=5)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].lesson.lesson_id, "lesson-1")

        # 8. Promotion: paired regression, atomic swap, then pointer rollback.
        parent = PolicyManifest(
            policy_id="policy-0",
            base_model_hash="base-hash",
            role_adapters={role: f"{role}-v1" for role in ROLES},
        )
        candidate_policy = PolicyManifest(
            policy_id="policy-1",
            base_model_hash="base-hash",
            role_adapters={
                **{role: f"{role}-v1" for role in ROLES},
                "interviewer": "interviewer-v2",
            },
            parent_policy_id="policy-0",
        )
        registry = PolicyRegistry(accepted_policy_id="policy-0")
        registry.register_candidate(parent)
        registry.register_candidate(candidate_policy)

        regression = evaluate_paired(
            candidate_policy_id="policy-1",
            baseline_policy_id="policy-0",
            candidate_seeds=SEEDS,
            baseline_seeds=SEEDS,
            candidate_participants=("participant-1",),
            baseline_participants=("participant-1",),
            role_metrics=tuple(
                RoleRegressionMetric(
                    role=role,
                    updated=role == "interviewer",
                    quality_delta=0.3 if role == "interviewer" else 0.0,
                    safety_regression=False,
                    non_inferiority_met=True,
                )
                for role in ROLES
            ),
        )
        decision = registry.promote(regression, expected_accepted_policy_id="policy-0")
        self.assertTrue(decision.promoted)
        self.assertEqual(registry.accepted_policy_id, "policy-1")
        self.assertEqual(registry.rollback(), "policy-0")

        # 9. The whole run is reproducible from one manifest, and the final-test
        #    split was never used.
        manifest = RunManifest(
            run_id="run-1",
            code_sha="code-sha",
            upstream_sha="0e2fc8ff",
            split_ids=("evolve-1",),
            split_hash=split_hash([evolve]),
            dataset_authorized=True,
            prompt_version="agentmental-v1",
            simulator_version=PROFILE.simulator_version,
            safety_version="safety-1",
            base_model_hash="base-hash",
            accepted_policy_id=registry.accepted_policy_id,
            lesson_memory_version="lessons-1",
            seeds=SEEDS,
            backend_version="fake-backend",
            adapter_hashes=candidate_policy.role_adapters,
        )
        self.assertTrue(manifest.cache_reusable_from(manifest))
        self.assertEqual(splits.final_test_access_log, [])


def _utility(total: float) -> UtilityBreakdown:
    return UtilityBreakdown(
        role="interviewer",
        task_gain=total,
        evidence_gain=0.0,
        costs=0.0,
        safety_admissible=True,
        label_supervision_used=False,
        config_version="dev-1",
    )


if __name__ == "__main__":
    unittest.main()
