"""Request-scoped role adapter selection for model backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from psyvec.model.backend import ModelBackend, ModelRequest, ModelResponse

ROLE_ALIASES = (
    "psyvec-interviewer",
    "psyvec-evaluator",
    "psyvec-scorer",
    "psyvec-updater",
)


class AdapterRoutingError(RuntimeError):
    """An adapter cannot be safely selected for a request."""


class UnknownPolicyError(AdapterRoutingError):
    """The requested accepted policy is absent from the registry."""


class UnknownRoleError(AdapterRoutingError):
    """The requested role is absent from an accepted policy."""


@dataclass(frozen=True, slots=True)
class AdapterManifest:
    adapter_id: str
    role: str
    base_model_hash: str
    adapter_hash: str
    status: Literal["accepted", "candidate"]


@dataclass(frozen=True, slots=True)
class AdapterRegistry:
    """Immutable mapping from accepted policy and role to adapter manifests."""

    manifests: Mapping[tuple[str, str], AdapterManifest]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifests", MappingProxyType(dict(self.manifests)))

    def lookup(self, accepted_policy_id: str, role: str) -> AdapterManifest:
        policies = {policy_id for policy_id, _role in self.manifests}
        if accepted_policy_id not in policies:
            raise UnknownPolicyError(f"unknown accepted policy: {accepted_policy_id!r}")
        try:
            return self.manifests[(accepted_policy_id, role)]
        except KeyError as error:
            raise UnknownRoleError(
                f"unknown role {role!r} for accepted policy {accepted_policy_id!r}"
            ) from error


@dataclass(frozen=True, slots=True)
class RoleModelRouter:
    """Delegate requests through the accepted adapter for their logical role."""

    backend: ModelBackend
    registry: AdapterRegistry
    base_model_hash: str
    accepted_policy_id: str | None = None
    adapters_enabled: bool = True

    def resolve(self, accepted_policy_id: str, role: str) -> AdapterManifest:
        if role not in ROLE_ALIASES:
            raise UnknownRoleError(f"unsupported logical role: {role!r}")
        manifest = self.registry.lookup(accepted_policy_id, role)
        if manifest.role != role:
            raise AdapterRoutingError(
                f"adapter {manifest.adapter_id!r} is for role {manifest.role!r}, "
                f"not {role!r}"
            )
        if manifest.base_model_hash != self.base_model_hash:
            raise AdapterRoutingError(
                f"adapter {manifest.adapter_id!r} is incompatible with the base model"
            )
        if manifest.status != "accepted":
            raise AdapterRoutingError(
                f"adapter {manifest.adapter_id!r} is not an accepted adapter"
            )
        return manifest

    def generate(
        self, request: ModelRequest, accepted_policy_id: str | None = None
    ) -> ModelResponse:
        if not self.adapters_enabled:
            return self.backend.generate(request)
        policy_id = accepted_policy_id or self.accepted_policy_id
        if policy_id is None:
            raise UnknownPolicyError(
                "an accepted policy id is required for adapter routing"
            )
        manifest = self.resolve(policy_id, request.role)
        response = self.backend.generate(request)
        return replace(
            response,
            provenance=replace(
                response.provenance,
                adapter_id=manifest.adapter_id,
                adapter_hash=manifest.adapter_hash,
            ),
        )
