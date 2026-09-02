"""Hard-state scoring and configured branch-set construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from psyvec.evolution.experience import (
    BranchCandidate,
    BranchInvariantError,
    ExperienceRecord,
    ParticipantProfile,
    assert_branch_invariants,
)

SignalExtractor = Callable[[ExperienceRecord], tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class HardStateRecord:
    experience_id: str
    role: str
    missing_signal: float
    uncertainty_signal: float
    coordination_signal: float
    normalized_score: float
    threshold_config: float
    selected: bool
    rationale: str


def mine_hard_states(
    experiences: Sequence[ExperienceRecord],
    *,
    threshold: float,
    signal_extractor: SignalExtractor,
) -> tuple[HardStateRecord, ...]:
    """Score every experience, retaining both selected and unselected records."""

    records: list[HardStateRecord] = []
    for experience in experiences:
        missing, uncertainty, coordination = signal_extractor(experience)
        signals = (missing, uncertainty, coordination)
        normalized_score = sum(signals) / len(signals)
        dominant = (
            "missing"
            if missing == max(signals)
            else "uncertainty"
            if uncertainty == max(signals)
            else "coordination"
        )
        records.append(
            HardStateRecord(
                experience_id=experience.experience_id,
                role=experience.role,
                missing_signal=missing,
                uncertainty_signal=uncertainty,
                coordination_signal=coordination,
                normalized_score=normalized_score,
                threshold_config=threshold,
                selected=normalized_score >= threshold,
                rationale=f"{dominant}_signal_dominates",
            )
        )
    return tuple(records)


def build_branch_set(
    hard_state: HardStateRecord,
    experience: ExperienceRecord,
    *,
    profile: ParticipantProfile,
    policy_id: str,
    simulator_version: str,
    seed: int,
    strategies: Sequence[str],
) -> tuple[BranchCandidate, ...]:
    """Build configured alternatives from one state and verify their shared context."""

    if hard_state.experience_id != experience.experience_id:
        raise BranchInvariantError("hard state does not match its experience")
    if hard_state.role != experience.role:
        raise BranchInvariantError("hard state role does not match its experience")
    if simulator_version != profile.simulator_version:
        raise BranchInvariantError("simulator version does not match frozen profile")

    branch_set_id = f"{hard_state.experience_id}-branches"
    branches = tuple(
        BranchCandidate(
            branch_id=f"{branch_set_id}-{index}",
            branch_set_id=branch_set_id,
            starting_state_hash=experience.offline_state_hash,
            profile_hash=profile.profile_hash,
            policy_id=policy_id,
            simulator_version=simulator_version,
            role=hard_state.role,
            action_kind=experience.decision_event.action.kind,
            generation_strategy=strategy,
            seed=seed,
        )
        for index, strategy in enumerate(strategies)
    )
    assert_branch_invariants(branches)
    return branches
