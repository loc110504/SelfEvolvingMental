"""Manifest-only verification of a claimed role-adapter update."""

from __future__ import annotations

from dataclasses import dataclass

from psyvec.policy.registry import PolicyManifest


@dataclass(frozen=True, slots=True)
class ParameterDiffVerification:
    """The result of checking that only one declared adapter changed."""

    updated_role: str
    passed: bool
    reasons: tuple[str, ...]


def verify_parameter_diff(
    parent: PolicyManifest,
    candidate: PolicyManifest,
    *,
    updated_role: str,
) -> ParameterDiffVerification:
    """Verify a manifest claim without loading or comparing model weights."""

    reasons: list[str] = []
    if parent.base_model_hash != candidate.base_model_hash:
        reasons.append("base_model_hash_changed")

    parent_roles = set(parent.role_adapters)
    candidate_roles = set(candidate.role_adapters)
    if parent_roles != candidate_roles:
        reasons.append("adapter_roles_changed")

    if updated_role not in parent_roles or updated_role not in candidate_roles:
        reasons.append(f"updated_role_missing:{updated_role}")
    elif parent.role_adapters[updated_role] == candidate.role_adapters[updated_role]:
        reasons.append(f"updated_adapter_unchanged:{updated_role}")

    for role in sorted((parent_roles & candidate_roles) - {updated_role}):
        if parent.role_adapters[role] != candidate.role_adapters[role]:
            reasons.append(f"unexpected_adapter_change:{role}")

    return ParameterDiffVerification(
        updated_role=updated_role,
        passed=not reasons,
        reasons=tuple(reasons),
    )
