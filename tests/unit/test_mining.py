import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from unit.test_experience_spine import experience, scored_event  # noqa: E402

from psyvec.evolution import (  # noqa: E402
    BranchInvariantError,
    ParticipantProfile,
    assert_branch_invariants,
    build_branch_set,
    mine_hard_states,
)


class MiningTests(unittest.TestCase):
    def setUp(self) -> None:
        _state, event = scored_event()
        self.experience = experience(event)

    def test_above_threshold_is_selected_and_below_is_retained(self) -> None:
        other = replace(self.experience, experience_id="exp-2")
        records = mine_hard_states(
            (self.experience, other),
            threshold=0.5,
            signal_extractor=lambda item: (0.9, 0.6, 0.3)
            if item.experience_id == "exp-1"
            else (0.2, 0.1, 0.0),
        )

        self.assertTrue(records[0].selected)
        self.assertFalse(records[1].selected)
        self.assertEqual(records[1].experience_id, "exp-2")

    def test_threshold_is_stored_and_changes_selection(self) -> None:
        low, = mine_hard_states(
            (self.experience,),
            threshold=0.4,
            signal_extractor=lambda _: (0.6, 0.6, 0.6),
        )
        high, = mine_hard_states(
            (self.experience,),
            threshold=0.7,
            signal_extractor=lambda _: (0.6, 0.6, 0.6),
        )

        self.assertTrue(low.selected)
        self.assertFalse(high.selected)
        self.assertEqual((low.threshold_config, high.threshold_config), (0.4, 0.7))

    def test_branch_set_shares_frozen_context_for_each_strategy(self) -> None:
        hard_state, = mine_hard_states(
            (self.experience,),
            threshold=0.5,
            signal_extractor=lambda _: (0.9, 0.4, 0.2),
        )
        profile = ParticipantProfile(profile_id="p1", simulator_version="sim-1")
        branches = build_branch_set(
            hard_state,
            self.experience,
            profile=profile,
            policy_id="policy-1",
            simulator_version="sim-1",
            seed=7,
            strategies=("policy", "exploratory", "lesson"),
        )

        self.assertEqual(len(branches), 3)
        self.assertEqual(
            {
                (
                    branch.starting_state_hash,
                    branch.profile_hash,
                    branch.policy_id,
                    branch.simulator_version,
                    branch.seed,
                )
                for branch in branches
            },
            {
                (
                    self.experience.offline_state_hash,
                    profile.profile_hash,
                    "policy-1",
                    "sim-1",
                    7,
                )
            },
        )

    def test_mismatched_profile_hash_or_seed_raises_existing_error(self) -> None:
        hard_state, = mine_hard_states(
            (self.experience,),
            threshold=0.5,
            signal_extractor=lambda _: (0.9, 0.4, 0.2),
        )
        profile = ParticipantProfile(profile_id="p1", simulator_version="sim-1")
        branches = build_branch_set(
            hard_state, self.experience, profile=profile, policy_id="policy-1",
            simulator_version="sim-1", seed=7, strategies=("policy", "exploratory"),
        )

        with self.assertRaises(BranchInvariantError):
            assert_branch_invariants(
                (branches[0], replace(branches[1], profile_hash="other"))
            )
        with self.assertRaises(BranchInvariantError):
            assert_branch_invariants((branches[0], replace(branches[1], seed=8)))


if __name__ == "__main__":
    unittest.main()
