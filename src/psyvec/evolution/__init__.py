"""Offline experience collection and branch bookkeeping."""

from __future__ import annotations

from psyvec.evolution.contrast import (
    BranchResult,
    ContrastRejection,
    PreferencePair,
    UtilityBreakdown,
    UtilityWeights,
    VerifiedContrast,
    build_preference_pair,
    validate_branch_result,
    verify_local,
)
from psyvec.evolution.experience import (
    BranchCandidate,
    BranchInvariantError,
    EvolutionContractError,
    ExperienceBuffer,
    ExperienceRecord,
    ParticipantProfile,
    assert_branch_invariants,
)
from psyvec.evolution.mining import (
    HardStateRecord,
    SignalExtractor,
    build_branch_set,
    mine_hard_states,
)

__all__ = [
    "BranchCandidate",
    "BranchResult",
    "BranchInvariantError",
    "EvolutionContractError",
    "ExperienceBuffer",
    "ExperienceRecord",
    "HardStateRecord",
    "ContrastRejection",
    "ParticipantProfile",
    "PreferencePair",
    "UtilityBreakdown",
    "UtilityWeights",
    "VerifiedContrast",
    "assert_branch_invariants",
    "build_branch_set",
    "build_preference_pair",
    "mine_hard_states",
    "SignalExtractor",
    "validate_branch_result",
    "verify_local",
]
