import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evolution import PreferencePair  # noqa: E402
from psyvec.policy import PolicyManifest  # noqa: E402
from psyvec.training import build_role_partitions, verify_parameter_diff  # noqa: E402


def pair(
    pair_id: str,
    *,
    role: str = "scorer",
    split_id: str = "evolve",
    contrast_id: str | None = None,
    chosen: str = "chosen",
    rejected: str = "rejected",
) -> PreferencePair:
    return PreferencePair(
        pair_id=pair_id,
        role=role,
        chosen=chosen,
        rejected=rejected,
        contrast_id=contrast_id or f"contrast-{pair_id}",
        accepted_policy_id="policy-0",
        split_id=split_id,
        label_supervision_used=False,
    )


def manifest(**changes: str) -> PolicyManifest:
    adapters = {"interviewer": "interviewer-0", "scorer": "scorer-0"}
    adapters.update(
        {role: adapter for role, adapter in changes.items() if role != "base"}
    )
    return PolicyManifest(
        policy_id="policy-1",
        base_model_hash=changes.get("base", "base-0"),
        role_adapters=adapters,
    )


class TrainingPrepTests(unittest.TestCase):
    def test_valid_single_role_partition_reports_role_and_count(self) -> None:
        result = build_role_partitions([pair("one"), pair("two")], minimum_role_pairs=2)

        self.assertTrue(result.passed)
        self.assertEqual(result.partitions[0].role, "scorer")
        self.assertEqual(result.partitions[0].pair_count, 2)

    def test_mixed_split_ids_are_refused(self) -> None:
        result = build_role_partitions(
            [pair("one", split_id="evolve"), pair("two", split_id="development")],
            minimum_role_pairs=1,
        )

        self.assertEqual(result.partitions, ())
        self.assertIn("mixed_split_ids", result.refusals[0].reasons)

    def test_duplicate_contrast_and_duplicate_text_are_refused(self) -> None:
        result = build_role_partitions(
            [
                pair("one", contrast_id="contrast"),
                pair("two", contrast_id="contrast"),
                pair("three", chosen="same", rejected="same"),
            ],
            minimum_role_pairs=1,
        )

        reasons = result.refusals[0].reasons
        self.assertIn("duplicate_contrast_id:contrast", reasons)
        self.assertIn("chosen_equals_rejected:three", reasons)

    def test_supplied_minimum_controls_refusal(self) -> None:
        pairs = [pair("one"), pair("two")]

        refused = build_role_partitions(pairs, minimum_role_pairs=3)
        accepted = build_role_partitions(pairs, minimum_role_pairs=2)

        self.assertIn("below_minimum_role_pairs", refused.refusals[0].reasons)
        self.assertTrue(accepted.passed)

    def test_parameter_diff_verification(self) -> None:
        parent = manifest()

        self.assertTrue(
            verify_parameter_diff(
                parent, manifest(scorer="scorer-1"), updated_role="scorer"
            ).passed
        )
        self.assertIn(
            "unexpected_adapter_change:interviewer",
            verify_parameter_diff(
                parent,
                manifest(scorer="scorer-1", interviewer="interviewer-1"),
                updated_role="scorer",
            ).reasons,
        )
        self.assertIn(
            "updated_adapter_unchanged:scorer",
            verify_parameter_diff(parent, manifest(), updated_role="scorer").reasons,
        )
        self.assertIn(
            "base_model_hash_changed",
            verify_parameter_diff(
                parent,
                manifest(scorer="scorer-1", base="base-1"),
                updated_role="scorer",
            ).reasons,
        )


if __name__ == "__main__":
    unittest.main()
