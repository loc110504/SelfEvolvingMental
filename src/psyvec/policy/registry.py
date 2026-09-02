"""Phase 7 promotion: paired regression, atomic pointer swap, audit trail.

A failed candidate can never become the accepted policy, promotion is a
compare-and-swap so a concurrent promotion loses instead of silently winning, and
rollback moves the pointer back without ever overwriting an artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


class PromotionError(RuntimeError):
    """Raised when a policy promotion is refused."""


class PolicyRaceError(PromotionError):
    """Raised when the accepted pointer moved under a compare-and-swap."""


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """An immutable policy: one accepted adapter per decision role."""

    policy_id: str
    base_model_hash: str
    role_adapters: Mapping[str, str]
    parent_policy_id: str | None = None

    def __post_init__(self) -> None:
        if not self.role_adapters:
            raise PromotionError(f"policy {self.policy_id!r} routes no role")
        object.__setattr__(
            self, "role_adapters", MappingProxyType(dict(self.role_adapters))
        )


@dataclass(frozen=True, slots=True)
class RoleRegressionMetric:
    role: str
    updated: bool
    quality_delta: float
    safety_regression: bool
    non_inferiority_met: bool


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """Paired evaluation of a candidate against the accepted policy."""

    candidate_policy_id: str
    baseline_policy_id: str
    seeds: tuple[int, ...]
    participants: tuple[str, ...]
    role_metrics: tuple[RoleRegressionMetric, ...]
    manifest_checks_passed: bool
    leakage_checks_passed: bool
    parameter_diff_ok: bool

    @property
    def failures(self) -> tuple[str, ...]:
        """Every reason this candidate must not be promoted."""

        reasons: list[str] = []
        if not self.manifest_checks_passed:
            reasons.append("manifest_check_failed")
        if not self.leakage_checks_passed:
            reasons.append("leakage_check_failed")
        if not self.parameter_diff_ok:
            reasons.append("parameter_diff_failed")
        for metric in self.role_metrics:
            if metric.safety_regression:
                reasons.append(f"safety_regression:{metric.role}")
            if not metric.updated and not metric.non_inferiority_met:
                reasons.append(f"non_inferiority_failed:{metric.role}")
            if metric.updated and metric.quality_delta <= 0:
                reasons.append(f"no_quality_gain:{metric.role}")
        return tuple(reasons)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class CheckpointDecision:
    candidate_policy_id: str
    promoted: bool
    reasons: tuple[str, ...]
    previous_policy_id: str | None


def evaluate_paired(
    *,
    candidate_policy_id: str,
    baseline_policy_id: str,
    candidate_seeds: Sequence[int],
    baseline_seeds: Sequence[int],
    candidate_participants: Sequence[str],
    baseline_participants: Sequence[str],
    role_metrics: Sequence[RoleRegressionMetric],
    manifest_checks_passed: bool = True,
    leakage_checks_passed: bool = True,
    parameter_diff_ok: bool = True,
) -> RegressionResult:
    """Build a regression result, refusing an unpaired comparison outright."""

    if tuple(candidate_seeds) != tuple(baseline_seeds):
        raise PromotionError("paired regression requires identical seeds on both sides")
    if tuple(candidate_participants) != tuple(baseline_participants):
        raise PromotionError(
            "paired regression requires identical participants on both sides"
        )
    if not role_metrics:
        raise PromotionError("paired regression requires at least one role metric")
    return RegressionResult(
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        seeds=tuple(candidate_seeds),
        participants=tuple(candidate_participants),
        role_metrics=tuple(role_metrics),
        manifest_checks_passed=manifest_checks_passed,
        leakage_checks_passed=leakage_checks_passed,
        parameter_diff_ok=parameter_diff_ok,
    )


@dataclass(slots=True)
class PolicyRegistry:
    """Append-only policy store with a single atomically swapped accepted pointer."""

    accepted_policy_id: str
    manifests: dict[str, PolicyManifest] = field(default_factory=dict)
    audit: list[CheckpointDecision] = field(default_factory=list)
    _pointer_history: list[str] = field(default_factory=list, init=False)

    def register_candidate(self, manifest: PolicyManifest) -> None:
        """Store a candidate manifest. Storing never makes it routable."""

        existing = self.manifests.get(manifest.policy_id)
        if existing is not None and existing != manifest:
            raise PromotionError(
                f"policy {manifest.policy_id!r} already exists and is immutable"
            )
        self.manifests[manifest.policy_id] = manifest

    def promote(
        self, result: RegressionResult, *, expected_accepted_policy_id: str
    ) -> CheckpointDecision:
        """Compare-and-swap the accepted pointer, only for a passing candidate."""

        if expected_accepted_policy_id != self.accepted_policy_id:
            raise PolicyRaceError(
                "accepted policy moved: expected "
                f"{expected_accepted_policy_id!r}, found {self.accepted_policy_id!r}"
            )
        if result.candidate_policy_id not in self.manifests:
            raise PromotionError(
                f"policy {result.candidate_policy_id!r} was never registered"
            )

        decision = CheckpointDecision(
            candidate_policy_id=result.candidate_policy_id,
            promoted=result.passed,
            reasons=result.failures,
            previous_policy_id=self.accepted_policy_id,
        )
        self.audit.append(decision)
        if decision.promoted:
            self._pointer_history.append(self.accepted_policy_id)
            self.accepted_policy_id = result.candidate_policy_id
        return decision

    def rollback(self) -> str:
        """Move the pointer back one accepted policy, keeping every artifact."""

        if not self._pointer_history:
            raise PromotionError("no previous accepted policy to roll back to")
        self.accepted_policy_id = self._pointer_history.pop()
        return self.accepted_policy_id
