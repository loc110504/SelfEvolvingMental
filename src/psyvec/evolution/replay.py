"""Phase 4 P4-3 offline replay determinism and session isolation.

Doc 06 requires two things this module proves against an injected simulator:
same state plus same frozen profile plus same seed yields the same result, and a
"clear memory" prompt does **not** establish isolation — each replay needs a
fresh or explicitly reset session.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from psyvec.evolution.experience import EvolutionContractError
from psyvec.state.contracts import canonical_hash

IsolationKind = Literal["fresh", "explicit_reset", "clear_memory_prompt", "reused"]

#: The only two session kinds Doc 06 accepts as isolated.
ISOLATED_SESSION_KINDS: frozenset[str] = frozenset({"fresh", "explicit_reset"})


class ReplayIsolationError(EvolutionContractError):
    """Raised when a replay would run in a session that is not isolated."""


class ReplayDeterminismError(EvolutionContractError):
    """Raised when one replay key produced more than one result."""


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """Everything a replay must fix before the simulator may be called."""

    state_hash: str
    profile_hash: str
    policy_id: str
    simulator_version: str
    decoding_hash: str
    seed: int

    @property
    def replay_key(self) -> str:
        """The identity two replays must share to be required to agree."""

        return canonical_hash(
            {
                "decoding_hash": self.decoding_hash,
                "policy_id": self.policy_id,
                "profile_hash": self.profile_hash,
                "seed": self.seed,
                "simulator_version": self.simulator_version,
                "state_hash": self.state_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class ReplaySession:
    """One simulator session and how it claims to have been isolated."""

    session_id: str
    isolation: IsolationKind

    @property
    def isolated(self) -> bool:
        return self.isolation in ISOLATED_SESSION_KINDS


def assert_isolated(session: ReplaySession) -> None:
    """Fail closed for a reused session or a "clear memory" prompt.

    The prompt case is called out separately because it is the upstream habit
    Doc 06 explicitly refuses to accept as isolation.
    """

    if session.isolated:
        return
    if session.isolation == "clear_memory_prompt":
        raise ReplayIsolationError(
            f"session {session.session_id!r} relies on a 'clear memory' prompt, "
            "which does not establish isolation: use a fresh or explicitly reset "
            "session"
        )
    raise ReplayIsolationError(
        f"session {session.session_id!r} is not isolated: {session.isolation}"
    )


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """One recorded replay outcome, hashed so transcripts never need storing."""

    request: ReplayRequest
    session_id: str
    result_hash: str

    @property
    def replay_key(self) -> str:
        return self.request.replay_key


@dataclass(frozen=True, slots=True)
class DeterminismReport:
    """Measured agreement across replays, one entry per replay key."""

    replay_count: int
    key_count: int
    mismatched_keys: tuple[str, ...]

    @property
    def deterministic(self) -> bool:
        return not self.mismatched_keys


@dataclass(slots=True)
class ReplayEngine:
    """Run an injected simulator under fixed conditions and record agreement.

    ``simulator`` receives the request and returns the result to hash. It is
    never a live model here: Phase 0's endpoint approval is still blocking.
    """

    simulator: Callable[[ReplayRequest], object]
    observations: list[ReplayObservation] = field(default_factory=list)

    def run(self, request: ReplayRequest, session: ReplaySession) -> ReplayObservation:
        """Replay once in an isolated session, recording the hashed result."""

        assert_isolated(session)
        observation = ReplayObservation(
            request=request,
            session_id=session.session_id,
            result_hash=canonical_hash(self.simulator(request)),
        )
        self.observations.append(observation)
        return observation

    def report(self) -> DeterminismReport:
        """Report which replay keys disagreed, without raising."""

        return determinism_report(self.observations)

    def assert_deterministic(self) -> None:
        """Fail when any replay key produced more than one distinct result."""

        report = self.report()
        if not report.deterministic:
            raise ReplayDeterminismError(
                "replays disagreed for keys: " + ", ".join(report.mismatched_keys)
            )


def determinism_report(
    observations: Sequence[ReplayObservation],
) -> DeterminismReport:
    """Group observations by replay key and report every disagreeing key."""

    by_key: dict[str, set[str]] = {}
    for observation in observations:
        by_key.setdefault(observation.replay_key, set()).add(observation.result_hash)
    return DeterminismReport(
        replay_count=len(observations),
        key_count=len(by_key),
        mismatched_keys=tuple(
            sorted(key for key, hashes in by_key.items() if len(hashes) > 1)
        ),
    )


def replay_manifest(observations: Sequence[ReplayObservation]) -> Mapping[str, str]:
    """Return the replay-key to result-hash map, refusing a disagreeing key."""

    manifest: dict[str, str] = {}
    for observation in observations:
        existing = manifest.setdefault(observation.replay_key, observation.result_hash)
        if existing != observation.result_hash:
            raise ReplayDeterminismError(
                f"replay key {observation.replay_key} has two results"
            )
    return manifest
