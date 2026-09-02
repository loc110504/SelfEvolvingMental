import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evolution import (  # noqa: E402
    BranchCandidate,
    BranchResult,
    ContrastRejection,
    EvolutionContractError,
    ParticipantProfile,
    UtilityBreakdown,
    UtilityWeights,
    VerifiedContrast,
    build_preference_pair,
    validate_branch_result,
    verify_local,
)

WEIGHTS = UtilityWeights(config_version="dev-1")
PROFILE = ParticipantProfile(
    profile_id="p1",
    simulator_version="sim-1",
    facts={"sleep": "poor"},
    unknown_slots=("appetite",),
)


def utility(total: float, *, safe: bool = True, labels: bool = False):
    return UtilityBreakdown(
        role="interviewer",
        task_gain=total,
        evidence_gain=0.0,
        costs=0.0,
        safety_admissible=safe,
        label_supervision_used=labels,
        config_version="dev-1",
    )


def branch(branch_id: str) -> BranchCandidate:
    return BranchCandidate(
        branch_id=branch_id,
        branch_set_id="set-1",
        starting_state_hash="state-hash",
        profile_hash=PROFILE.profile_hash,
        policy_id="policy-0",
        simulator_version="sim-1",
        role="interviewer",
        action_kind="ask",
        generation_strategy="policy-low-temperature",
        seed=7,
    )


def result(branch_id: str, total: float, **overrides) -> BranchResult:
    values = {
        "branch": branch(branch_id),
        "response": f"response-{branch_id}",
        "utility": utility(total),
    }
    values.update(overrides)
    return BranchResult(**values)


class ContrastTests(unittest.TestCase):
    def test_unsupported_simulator_fact_is_invalid_with_reasons(self) -> None:
        checked = validate_branch_result(
            result("b1", 1.0, asserted_fact_keys=("appetite",)), PROFILE
        )

        self.assertEqual(checked.validity, "invalid")
        self.assertIn("unsupported_fact:appetite", checked.invalid_reasons)

        supported = validate_branch_result(
            result("b2", 1.0, asserted_fact_keys=("sleep",)), PROFILE
        )
        self.assertEqual(supported.validity, "valid")

    def test_unsafe_branch_utility_fails_closed(self) -> None:
        with self.assertRaisesRegex(EvolutionContractError, "unsafe"):
            utility(5.0, safe=False).score(WEIGHTS)

    def test_one_full_contrast_traces_state_profile_action_utility_and_seed(
        self,
    ) -> None:
        contrast = verify_local(
            [result("b1", 3.0), result("b2", 0.5)],
            role="interviewer",
            weights=WEIGHTS,
            pair_margin=1.0,
            contrast_id="c1",
            profile=PROFILE,
        )

        assert isinstance(contrast, VerifiedContrast)
        self.assertEqual(contrast.chosen_branch_id, "b1")
        self.assertEqual(contrast.rejected_branch_id, "b2")
        self.assertEqual(contrast.starting_state_hash, "state-hash")
        self.assertEqual(contrast.profile_hash, PROFILE.profile_hash)
        self.assertEqual(contrast.seed, 7)
        self.assertAlmostEqual(contrast.delta_utility, 2.5)
        self.assertTrue(contrast.checks["nonduplicate"])

        pair = build_preference_pair(
            contrast,
            pair_id="pair-1",
            chosen="response-b1",
            rejected="response-b2",
            accepted_policy_id="policy-0",
            split_id="evolve",
            label_supervision_used=False,
        )
        self.assertEqual(pair.contrast_id, "c1")
        self.assertEqual(pair.role, "interviewer")

    def test_tie_below_margin_and_too_few_valid_branches_are_rejected(self) -> None:
        tie = verify_local(
            [result("b1", 1.0), result("b2", 0.9)],
            role="interviewer",
            weights=WEIGHTS,
            pair_margin=1.0,
            contrast_id="c2",
            profile=PROFILE,
        )
        assert isinstance(tie, ContrastRejection)
        self.assertEqual(tie.reasons, ("tie_below_pair_margin",))

        too_few = verify_local(
            [result("b1", 3.0), result("b2", 0.5, asserted_fact_keys=("appetite",))],
            role="interviewer",
            weights=WEIGHTS,
            pair_margin=1.0,
            contrast_id="c3",
            profile=PROFILE,
        )
        assert isinstance(too_few, ContrastRejection)
        self.assertEqual(too_few.reasons, ("insufficient_valid_branches",))

    def test_rejected_contrast_cannot_become_a_preference_pair(self) -> None:
        rejection = ContrastRejection(role="interviewer", reasons=("x",))
        contrast = VerifiedContrast(
            contrast_id="c4",
            starting_state_hash="state-hash",
            profile_hash=PROFILE.profile_hash,
            policy_id="policy-0",
            seed=7,
            role="interviewer",
            chosen_branch_id="b1",
            rejected_branch_id="b2",
            delta_utility=2.0,
            checks={"nonduplicate": False},
            attribution="interviewer",
            status="rejected",
        )

        self.assertEqual(rejection.status, "rejected")
        with self.assertRaisesRegex(EvolutionContractError, "rejected contrast"):
            build_preference_pair(
                contrast,
                pair_id="p",
                chosen="a",
                rejected="b",
                accepted_policy_id="policy-0",
                split_id="evolve",
                label_supervision_used=False,
            )


if __name__ == "__main__":
    unittest.main()
