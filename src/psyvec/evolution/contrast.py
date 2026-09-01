"""Role utility, branch validity, and local contrast verification.

Utility is only evaluated for safe branches, and a contrast is only accepted when
it is traceable to a starting state, a frozen profile, an action, a utility and a
seed. Every rejection carries its reasons instead of being regenerated away.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from psyvec.evolution.experience import (
    BranchCandidate,
    EvolutionContractError,
    ParticipantProfile,
    assert_branch_invariants,
)


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    """Role-conditioned weights for ``U = w_task*G_task + w_ev*G_ev - w_cost*C``."""

    task: float = 1.0
    evidence: float = 1.0
    cost: float = 1.0
    config_version: str = "development-unfrozen"


@dataclass(frozen=True, slots=True)
class UtilityBreakdown:
    role: str
    task_gain: float
    evidence_gain: float
    costs: float
    safety_admissible: bool
    label_supervision_used: bool
    config_version: str
    normalized_total: float | None = None

    def score(self, weights: UtilityWeights) -> float:
        """Total utility. Undefined for an unsafe branch, so it fails closed."""

        if not self.safety_admissible:
            raise EvolutionContractError(
                f"utility is undefined for an unsafe {self.role!r} branch"
            )
        return (
            weights.task * self.task_gain
            + weights.evidence * self.evidence_gain
            - weights.cost * self.costs
        )


@dataclass(frozen=True, slots=True)
class BranchResult:
    """The replayed outcome of one branch, including invalid ones."""

    branch: BranchCandidate
    response: str
    asserted_fact_keys: tuple[str, ...] = ()
    safe: bool = True
    state_transition_valid: bool = True
    utility: UtilityBreakdown | None = None
    validity: str = "unchecked"
    invalid_reasons: tuple[str, ...] = ()


def validate_branch_result(
    result: BranchResult, profile: ParticipantProfile
) -> BranchResult:
    """Mark a replayed branch valid or invalid, always recording the reasons."""

    reasons: list[str] = []
    if not result.safe:
        reasons.append("unsafe_output")
    if not result.state_transition_valid:
        reasons.append("invalid_state_transition")
    if result.branch.profile_hash != profile.profile_hash:
        reasons.append("inconsistent_profile")
    if result.branch.simulator_version != profile.simulator_version:
        reasons.append("simulator_version_mismatch")
    unsupported = sorted(
        key for key in result.asserted_fact_keys if not profile.supports(key)
    )
    reasons.extend(f"unsupported_fact:{key}" for key in unsupported)
    return BranchResult(
        branch=result.branch,
        response=result.response,
        asserted_fact_keys=result.asserted_fact_keys,
        safe=result.safe,
        state_transition_valid=result.state_transition_valid,
        utility=result.utility,
        validity="valid" if not reasons else "invalid",
        invalid_reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class VerifiedContrast:
    contrast_id: str
    starting_state_hash: str
    profile_hash: str
    policy_id: str
    seed: int
    role: str
    chosen_branch_id: str
    rejected_branch_id: str
    delta_utility: float
    checks: Mapping[str, bool]
    attribution: str
    status: str = "accepted"


@dataclass(frozen=True, slots=True)
class ContrastRejection:
    role: str
    status: str = "rejected"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """The only artifact the slow path may train on."""

    pair_id: str
    role: str
    chosen: str
    rejected: str
    contrast_id: str
    accepted_policy_id: str
    split_id: str
    label_supervision_used: bool


def verify_local(
    results: Sequence[BranchResult],
    *,
    role: str,
    weights: UtilityWeights,
    pair_margin: float,
    contrast_id: str,
    profile: ParticipantProfile,
) -> VerifiedContrast | ContrastRejection:
    """Accept one contrast only when a safe, valid, attributed margin exists."""

    checked = [validate_branch_result(result, profile) for result in results]
    assert_branch_invariants([result.branch for result in checked])

    admissible = [
        result
        for result in checked
        if result.validity == "valid"
        and result.utility is not None
        and result.utility.safety_admissible
        and result.branch.role == role
    ]
    if len(admissible) < 2:
        return ContrastRejection(role=role, reasons=("insufficient_valid_branches",))

    ranked = sorted(
        admissible,
        key=lambda result: _score(result, weights),
        reverse=True,
    )
    chosen, rejected = ranked[0], ranked[-1]
    delta = _score(chosen, weights) - _score(rejected, weights)
    if delta < pair_margin:
        return ContrastRejection(role=role, reasons=("tie_below_pair_margin",))

    return VerifiedContrast(
        contrast_id=contrast_id,
        starting_state_hash=chosen.branch.starting_state_hash,
        profile_hash=chosen.branch.profile_hash,
        policy_id=chosen.branch.policy_id,
        seed=chosen.branch.seed,
        role=role,
        chosen_branch_id=chosen.branch.branch_id,
        rejected_branch_id=rejected.branch.branch_id,
        delta_utility=delta,
        checks={
            "margin_met": True,
            "safe": True,
            "nonduplicate": chosen.response != rejected.response,
            "attributed_role": True,
        },
        attribution=role,
    )


def build_preference_pair(
    contrast: VerifiedContrast,
    *,
    pair_id: str,
    chosen: str,
    rejected: str,
    accepted_policy_id: str,
    split_id: str,
    label_supervision_used: bool,
) -> PreferencePair:
    """Only an accepted, non-duplicate contrast may become a preference pair."""

    if contrast.status != "accepted":
        raise EvolutionContractError("a rejected contrast cannot become a pair")
    if not contrast.checks.get("nonduplicate", False):
        raise EvolutionContractError("a duplicate contrast cannot become a pair")
    return PreferencePair(
        pair_id=pair_id,
        role=contrast.role,
        chosen=chosen,
        rejected=rejected,
        contrast_id=contrast.contrast_id,
        accepted_policy_id=accepted_policy_id,
        split_id=split_id,
        label_supervision_used=label_supervision_used,
    )


def _score(result: BranchResult, weights: UtilityWeights) -> float:
    breakdown = result.utility
    if breakdown is None:
        raise EvolutionContractError(
            f"branch {result.branch.branch_id!r} has no utility breakdown"
        )
    return breakdown.score(weights)
