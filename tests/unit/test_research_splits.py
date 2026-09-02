import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.research import (  # noqa: E402
    ForbiddenSplitUseError,
    RunManifest,
    SplitManifest,
    SplitProtocolError,
    SplitRegistry,
    UnauthorizedSplitError,
    split_hash,
)


def manifest(
    split_id: str, kind: str, participants: tuple[str, ...], *, authorized: bool = True
) -> SplitManifest:
    return SplitManifest(
        split_id=split_id,
        kind=kind,  # type: ignore[arg-type]
        participant_ids=participants,
        authorized=authorized,
    )


def run_manifest(**overrides) -> RunManifest:
    values = {
        "run_id": "run-1",
        "code_sha": "code",
        "upstream_sha": "0e2fc8ff",
        "split_ids": ("evolve-1",),
        "split_hash": "split-hash",
        "dataset_authorized": True,
        "prompt_version": "agentmental-v1",
        "simulator_version": "sim-1",
        "safety_version": "safety-1",
        "base_model_hash": "base",
        "accepted_policy_id": "policy-0",
        "lesson_memory_version": "lessons-1",
        "seeds": (1, 2),
        "backend_version": "backend-1",
    }
    values.update(overrides)
    return RunManifest(**values)


class ResearchSplitTests(unittest.TestCase):
    def test_participants_must_be_disjoint_across_splits(self) -> None:
        registry = SplitRegistry()
        registry.register(manifest("evolve-1", "evolve", ("p1", "p2")))

        with self.assertRaisesRegex(SplitProtocolError, "shares participants"):
            registry.register(manifest("final-1", "final-test", ("p2", "p3")))
        with self.assertRaisesRegex(SplitProtocolError, "repeats a participant"):
            manifest("dup", "evolve", ("p9", "p9"))

    def test_permitted_use_matrix_is_enforced_per_partition(self) -> None:
        registry = SplitRegistry()
        registry.register(manifest("evolve-1", "evolve", ("p1",)))
        registry.register(manifest("final-1", "final-test", ("p2",)))

        registry.authorize_use("evolve-1", "pairs", actor="miner")

        with self.assertRaisesRegex(ForbiddenSplitUseError, "may not be used"):
            registry.authorize_use("evolve-1", "final_claim", actor="report")
        for forbidden in ("promotion", "retrieval_index", "collection"):
            with self.assertRaises(ForbiddenSplitUseError):
                registry.authorize_use("final-1", forbidden, actor="trainer")

    def test_unauthorized_split_is_refused_for_every_use(self) -> None:
        registry = SplitRegistry()
        registry.register(
            manifest("evolve-1", "evolve", ("p1",), authorized=False)
        )

        with self.assertRaisesRegex(UnauthorizedSplitError, "no recorded data"):
            registry.authorize_use("evolve-1", "collection", actor="collector")
        with self.assertRaisesRegex(SplitProtocolError, "unknown split"):
            registry.authorize_use("missing", "collection", actor="collector")

    def test_final_test_access_is_logged(self) -> None:
        registry = SplitRegistry()
        registry.register(manifest("evolve-1", "evolve", ("p1",)))
        registry.register(manifest("final-1", "final-test", ("p2",)))

        registry.authorize_use("evolve-1", "collection", actor="collector")
        self.assertEqual(registry.final_test_access_log, [])

        registry.authorize_use("final-1", "final_claim", actor="final-runner")
        self.assertEqual(
            registry.final_test_access_log, [("final-runner", "final_claim")]
        )

    def test_run_manifest_hash_gates_cache_reuse(self) -> None:
        first = run_manifest()
        same = run_manifest(run_id="run-2")
        different = run_manifest(seeds=(1, 3))

        self.assertEqual(first.manifest_hash, same.manifest_hash)
        self.assertTrue(first.cache_reusable_from(same))
        self.assertFalse(first.cache_reusable_from(different))

        with self.assertRaisesRegex(UnauthorizedSplitError, "no dataset authorization"):
            replace(first, dataset_authorized=False)

    def test_split_hash_is_order_independent(self) -> None:
        a = manifest("evolve-1", "evolve", ("p1",))
        b = manifest("final-1", "final-test", ("p2",))

        self.assertEqual(split_hash([a, b]), split_hash([b, a]))


if __name__ == "__main__":
    unittest.main()
