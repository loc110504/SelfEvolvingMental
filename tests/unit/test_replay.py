"""Phase 4 P4-3: deterministic replay and Doc 06 session isolation."""

from __future__ import annotations

import unittest

from psyvec.evolution.replay import (
    ReplayDeterminismError,
    ReplayEngine,
    ReplayIsolationError,
    ReplayRequest,
    ReplaySession,
    assert_isolated,
    determinism_report,
    replay_manifest,
)


def make_request(*, seed: int = 7, state_hash: str = "state-a") -> ReplayRequest:
    return ReplayRequest(
        state_hash=state_hash,
        profile_hash="profile-a",
        policy_id="policy-1",
        simulator_version="sim-1",
        decoding_hash="decoding-a",
        seed=seed,
    )


class DeterministicSimulator:
    """Answers only from the request, so equal requests give equal answers."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: ReplayRequest) -> dict[str, object]:
        self.calls += 1
        return {"answer": f"{request.state_hash}:{request.seed}"}


class DriftingSimulator:
    """Leaks call order into its answer, the failure replay is meant to catch."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: ReplayRequest) -> dict[str, object]:
        self.calls += 1
        return {"answer": f"{request.state_hash}:{self.calls}"}


class ReplayDeterminismTests(unittest.TestCase):
    def test_same_state_profile_and_seed_replay_to_the_same_result(self) -> None:
        engine = ReplayEngine(simulator=DeterministicSimulator())
        request = make_request()

        first = engine.run(request, ReplaySession("s1", "fresh"))
        second = engine.run(request, ReplaySession("s2", "explicit_reset"))

        self.assertEqual(first.result_hash, second.result_hash)
        self.assertEqual(first.replay_key, second.replay_key)
        engine.assert_deterministic()
        self.assertTrue(engine.report().deterministic)

    def test_a_different_seed_is_a_different_replay_key(self) -> None:
        self.assertNotEqual(
            make_request(seed=7).replay_key, make_request(seed=8).replay_key
        )

    def test_a_drifting_simulator_is_reported_and_raises(self) -> None:
        engine = ReplayEngine(simulator=DriftingSimulator())
        request = make_request()
        engine.run(request, ReplaySession("s1", "fresh"))
        engine.run(request, ReplaySession("s2", "fresh"))

        report = engine.report()

        self.assertFalse(report.deterministic)
        self.assertEqual(report.replay_count, 2)
        self.assertEqual(report.key_count, 1)
        self.assertEqual(report.mismatched_keys, (request.replay_key,))
        with self.assertRaises(ReplayDeterminismError):
            engine.assert_deterministic()
        with self.assertRaises(ReplayDeterminismError):
            replay_manifest(engine.observations)

    def test_manifest_maps_each_replay_key_to_one_result(self) -> None:
        engine = ReplayEngine(simulator=DeterministicSimulator())
        first = make_request(seed=1)
        second = make_request(seed=2)
        engine.run(first, ReplaySession("s1", "fresh"))
        engine.run(first, ReplaySession("s2", "fresh"))
        engine.run(second, ReplaySession("s3", "fresh"))

        manifest = replay_manifest(engine.observations)

        self.assertEqual(set(manifest), {first.replay_key, second.replay_key})

    def test_empty_observations_report_as_deterministic(self) -> None:
        report = determinism_report(())

        self.assertTrue(report.deterministic)
        self.assertEqual(report.replay_count, 0)


class SessionIsolationTests(unittest.TestCase):
    def test_fresh_and_explicitly_reset_sessions_are_isolated(self) -> None:
        assert_isolated(ReplaySession("s1", "fresh"))
        assert_isolated(ReplaySession("s2", "explicit_reset"))

    def test_a_clear_memory_prompt_does_not_establish_isolation(self) -> None:
        session = ReplaySession("s3", "clear_memory_prompt")

        self.assertFalse(session.isolated)
        with self.assertRaises(ReplayIsolationError) as caught:
            assert_isolated(session)
        self.assertIn("clear memory", str(caught.exception))

    def test_a_reused_session_is_refused(self) -> None:
        with self.assertRaises(ReplayIsolationError):
            assert_isolated(ReplaySession("s4", "reused"))

    def test_the_engine_refuses_to_run_in_an_unisolated_session(self) -> None:
        simulator = DeterministicSimulator()
        engine = ReplayEngine(simulator=simulator)

        with self.assertRaises(ReplayIsolationError):
            engine.run(make_request(), ReplaySession("s5", "clear_memory_prompt"))

        self.assertEqual(simulator.calls, 0)
        self.assertEqual(engine.observations, [])


if __name__ == "__main__":
    unittest.main()
