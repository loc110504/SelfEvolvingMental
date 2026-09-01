"""Phase 4 offline experience spine: buffer, frozen profile, branch invariants.

Only what the Phase 4 gate needs: an accepted contrast must be traceable to a
state, a frozen profile, an action, a utility and a seed. Utility scoring,
hard-state mining and contrast verification live outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from psyvec.state.contracts import (
    DecisionEvent,
    ProvenanceRef,
    canonical_hash,
    reject_label_keys,
)


class EvolutionContractError(ValueError):
    """Raised when an offline evolution record violates an invariant."""


class BranchInvariantError(EvolutionContractError):
    """Raised when branches in one set do not share their frozen context."""


@dataclass(frozen=True, slots=True)
class ParticipantProfile:
    """A frozen simulated-participant profile. Immutable for a whole branch set."""

    profile_id: str
    simulator_version: str
    facts: Mapping[str, str] = field(default_factory=dict)
    unknown_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reject_label_keys(self.facts, where=f"profile {self.profile_id!r} facts")
        overlap = sorted(set(self.facts) & set(self.unknown_slots))
        if overlap:
            raise EvolutionContractError(
                f"profile {self.profile_id!r} declares slots both known and unknown: "
                f"{', '.join(overlap)}"
            )
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))

    @property
    def profile_hash(self) -> str:
        return canonical_hash(
            {
                "facts": dict(self.facts),
                "profile_id": self.profile_id,
                "simulator_version": self.simulator_version,
                "unknown_slots": list(self.unknown_slots),
            }
        )

    def supports(self, fact_key: str) -> bool:
        """Whether the simulator may assert this fact without inventing it."""

        return fact_key in self.facts


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    """One offline decision, tagged with the split it may be used in."""

    experience_id: str
    split_id: str
    participant_ref: str
    role: str
    decision_event: DecisionEvent
    offline_state_hash: str
    outcome: str
    provenance: ProvenanceRef
    seed: int | None = None
    utility: float | None = None

    def __post_init__(self) -> None:
        if self.offline_state_hash != self.decision_event.pre_state_hash:
            raise EvolutionContractError(
                f"experience {self.experience_id!r} does not cite its own pre-state"
            )
        if self.role != self.decision_event.role:
            raise EvolutionContractError(
                f"experience {self.experience_id!r} role disagrees with its event"
            )


@dataclass(slots=True)
class ExperienceBuffer:
    """Append-only buffer with a split guard and a label canary."""

    split_id: str
    enabled: bool = True
    records: list[ExperienceRecord] = field(default_factory=list)

    def append(
        self, record: ExperienceRecord, *, policy_input: Mapping[str, Any] | None = None
    ) -> None:
        """Record one experience, or drop it when offline collection is disabled."""

        if not self.enabled:
            return
        if record.split_id != self.split_id:
            raise EvolutionContractError(
                f"experience {record.experience_id!r} belongs to split "
                f"{record.split_id!r}, not {self.split_id!r}"
            )
        if policy_input is not None:
            reject_label_keys(
                policy_input, where=f"policy input for {record.experience_id!r}"
            )
        self.records.append(record)

    def export(self, *, split_id: str | None = None) -> tuple[ExperienceRecord, ...]:
        """Export the buffered experiences for one split only."""

        wanted = split_id or self.split_id
        if wanted != self.split_id:
            raise EvolutionContractError(
                f"buffer holds split {self.split_id!r}, not {wanted!r}"
            )
        return tuple(self.records)


@dataclass(frozen=True, slots=True)
class BranchCandidate:
    """One alternative action explored from a shared starting state."""

    branch_id: str
    branch_set_id: str
    starting_state_hash: str
    profile_hash: str
    policy_id: str
    simulator_version: str
    role: str
    action_kind: str
    generation_strategy: str
    seed: int


def assert_branch_invariants(branches: Sequence[BranchCandidate]) -> None:
    """Every branch in a set shares state, profile, policy, simulator and seed.

    The shared seed is what makes two branch outcomes comparable; a differing one
    is a hard failure rather than a tolerated difference.
    """

    if not branches:
        raise BranchInvariantError("a branch set needs at least one branch")
    first = branches[0]
    for branch in branches[1:]:
        if branch.branch_set_id != first.branch_set_id:
            raise BranchInvariantError("branches belong to different branch sets")
        for name in (
            "starting_state_hash",
            "profile_hash",
            "policy_id",
            "simulator_version",
            "seed",
        ):
            if getattr(branch, name) != getattr(first, name):
                raise BranchInvariantError(
                    f"branch {branch.branch_id!r} differs in {name}"
                )
