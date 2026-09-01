import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.policy import (  # noqa: E402
    PolicyManifest,
    PolicyRaceError,
    PolicyRegistry,
    PromotionError,
    RoleRegressionMetric,
    evaluate_paired,
)

ROLES = ("interviewer", "evaluator", "scorer", "updater")
SEEDS = (1, 2, 3)
PARTICIPANTS = ("participant-1", "participant-2")


def manifest(policy_id: str) -> PolicyManifest:
    return PolicyManifest(
        policy_id=policy_id,
        base_model_hash="base-hash",
        role_adapters={role: f"{role}-adapter" for role in ROLES},
        parent_policy_id="policy-0",
    )


def metrics(*, updated_role: str = "scorer", **overrides) -> tuple:
    built = []
    for role in ROLES:
        updated = role == updated_role
        built.append(
            RoleRegressionMetric(
                role=role,
                updated=updated,
                quality_delta=overrides.get("quality_delta", 0.4) if updated else 0.0,
                safety_regression=(
                    overrides.get("safety_regression", False) if updated else False
                ),
                non_inferiority_met=(
                    True if updated else overrides.get("non_inferiority_met", True)
                ),
            )
        )
    return tuple(built)


def registry() -> PolicyRegistry:
    store = PolicyRegistry(accepted_policy_id="policy-0")
    store.register_candidate(manifest("policy-0"))
    store.register_candidate(manifest("policy-1"))
    return store


def result(**overrides):
    return evaluate_paired(
        candidate_policy_id="policy-1",
        baseline_policy_id="policy-0",
        candidate_seeds=SEEDS,
        baseline_seeds=SEEDS,
        candidate_participants=PARTICIPANTS,
        baseline_participants=PARTICIPANTS,
        role_metrics=metrics(**overrides),
    )


class PolicyRegistryTests(unittest.TestCase):
    def test_passing_candidate_promotes_atomically_with_audit(self) -> None:
        store = registry()

        decision = store.promote(result(), expected_accepted_policy_id="policy-0")

        self.assertTrue(decision.promoted)
        self.assertEqual(decision.reasons, ())
        self.assertEqual(decision.previous_policy_id, "policy-0")
        self.assertEqual(store.accepted_policy_id, "policy-1")
        self.assertEqual(len(store.audit), 1)

    def test_failed_candidate_cannot_promote_but_is_audited(self) -> None:
        store = registry()

        decision = store.promote(
            result(safety_regression=True), expected_accepted_policy_id="policy-0"
        )

        self.assertFalse(decision.promoted)
        self.assertIn("safety_regression:scorer", decision.reasons)
        self.assertEqual(store.accepted_policy_id, "policy-0")
        self.assertEqual(len(store.audit), 1)

    def test_concurrent_promotion_loses_the_compare_and_swap(self) -> None:
        store = registry()
        store.promote(result(), expected_accepted_policy_id="policy-0")

        with self.assertRaises(PolicyRaceError):
            store.promote(result(), expected_accepted_policy_id="policy-0")
        self.assertEqual(store.accepted_policy_id, "policy-1")

    def test_rollback_restores_the_previous_pointer_and_keeps_artifacts(self) -> None:
        store = registry()
        store.promote(result(), expected_accepted_policy_id="policy-0")

        self.assertEqual(store.rollback(), "policy-0")
        self.assertIn("policy-1", store.manifests)
        with self.assertRaisesRegex(PromotionError, "no previous accepted policy"):
            store.rollback()

    def test_unpaired_regression_and_role_matrix_are_refused(self) -> None:
        with self.assertRaisesRegex(PromotionError, "identical seeds"):
            evaluate_paired(
                candidate_policy_id="policy-1",
                baseline_policy_id="policy-0",
                candidate_seeds=(1, 2),
                baseline_seeds=SEEDS,
                candidate_participants=PARTICIPANTS,
                baseline_participants=PARTICIPANTS,
                role_metrics=metrics(),
            )

        store = registry()
        decision = store.promote(
            result(non_inferiority_met=False), expected_accepted_policy_id="policy-0"
        )
        self.assertFalse(decision.promoted)
        self.assertIn("non_inferiority_failed:interviewer", decision.reasons)

        store2 = registry()
        stale = store2.promote(
            result(quality_delta=0.0), expected_accepted_policy_id="policy-0"
        )
        self.assertIn("no_quality_gain:scorer", stale.reasons)


if __name__ == "__main__":
    unittest.main()
