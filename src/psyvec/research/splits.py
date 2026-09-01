"""Phase 8 research hardening: frozen splits, permitted use, run manifests.

Final-test isolation is enforced by lineage and access, not convention: a split
must be authorized before any use, participants must be disjoint across splits,
and every final-test read is logged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from psyvec.state.contracts import canonical_hash

SplitKind = Literal["evolve", "development", "final-test", "external"]
SplitUse = Literal[
    "collection",
    "hard_states",
    "branches",
    "pairs",
    "lesson_candidates",
    "threshold_selection",
    "transfer_validation",
    "regression",
    "promotion",
    "retrieval_index",
    "final_claim",
]

#: Doc 06 split and leakage protocol, as an explicit allow-list per partition.
PERMITTED_USES: Mapping[SplitKind, frozenset[str]] = {
    "evolve": frozenset(
        {"collection", "hard_states", "branches", "pairs", "lesson_candidates"}
    ),
    "development": frozenset(
        {"threshold_selection", "transfer_validation", "regression", "retrieval_index"}
    ),
    "final-test": frozenset({"final_claim"}),
    "external": frozenset({"final_claim"}),
}


class SplitProtocolError(RuntimeError):
    """Raised when a split is used outside its authorized protocol."""


class UnauthorizedSplitError(SplitProtocolError):
    """Raised when a split has no recorded data authorization."""


class ForbiddenSplitUseError(SplitProtocolError):
    """Raised when a partition is used for something its protocol forbids."""


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A frozen participant partition. Reports de-identify these ids."""

    split_id: str
    kind: SplitKind
    participant_ids: tuple[str, ...]
    authorized: bool = False

    def __post_init__(self) -> None:
        if not self.participant_ids:
            raise SplitProtocolError(f"split {self.split_id!r} has no participants")
        if len(set(self.participant_ids)) != len(self.participant_ids):
            raise SplitProtocolError(
                f"split {self.split_id!r} repeats a participant id"
            )

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(
            {
                "kind": self.kind,
                "participant_ids": sorted(self.participant_ids),
                "split_id": self.split_id,
            }
        )


@dataclass(slots=True)
class SplitRegistry:
    """Holds the frozen splits and enforces disjointness and permitted use."""

    splits: dict[str, SplitManifest] = field(default_factory=dict)
    final_test_access_log: list[tuple[str, str]] = field(default_factory=list)

    def register(self, manifest: SplitManifest) -> None:
        """Freeze one split, refusing any participant already in another split."""

        if manifest.split_id in self.splits:
            raise SplitProtocolError(f"split {manifest.split_id!r} is already frozen")
        claimed = set(manifest.participant_ids)
        for existing in self.splits.values():
            overlap = sorted(claimed & set(existing.participant_ids))
            if overlap:
                raise SplitProtocolError(
                    f"split {manifest.split_id!r} shares participants with "
                    f"{existing.split_id!r}: {', '.join(overlap)}"
                )
        self.splits[manifest.split_id] = manifest

    def authorize_use(
        self, split_id: str, use: SplitUse, *, actor: str
    ) -> SplitManifest:
        """Return the split only when this use is authorized for its partition."""

        try:
            manifest = self.splits[split_id]
        except KeyError as error:
            raise SplitProtocolError(f"unknown split: {split_id!r}") from error
        if not manifest.authorized:
            raise UnauthorizedSplitError(
                f"split {split_id!r} has no recorded data authorization"
            )
        if use not in PERMITTED_USES[manifest.kind]:
            raise ForbiddenSplitUseError(
                f"{manifest.kind} split {split_id!r} may not be used for {use!r}"
            )
        if manifest.kind == "final-test":
            self.final_test_access_log.append((actor, use))
        return manifest


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Everything a rerun must match before any cached artifact is reused."""

    run_id: str
    code_sha: str
    upstream_sha: str
    split_ids: tuple[str, ...]
    split_hash: str
    dataset_authorized: bool
    prompt_version: str
    simulator_version: str
    safety_version: str
    base_model_hash: str
    accepted_policy_id: str
    lesson_memory_version: str
    seeds: tuple[int, ...]
    backend_version: str
    adapter_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_authorized:
            raise UnauthorizedSplitError(
                f"run {self.run_id!r} records no dataset authorization"
            )

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(
            {
                "accepted_policy_id": self.accepted_policy_id,
                "adapter_hashes": dict(self.adapter_hashes),
                "backend_version": self.backend_version,
                "base_model_hash": self.base_model_hash,
                "code_sha": self.code_sha,
                "lesson_memory_version": self.lesson_memory_version,
                "prompt_version": self.prompt_version,
                "safety_version": self.safety_version,
                "seeds": list(self.seeds),
                "simulator_version": self.simulator_version,
                "split_hash": self.split_hash,
                "split_ids": list(self.split_ids),
                "upstream_sha": self.upstream_sha,
            }
        )

    def cache_reusable_from(self, other: RunManifest) -> bool:
        """Cached artifacts are reusable only for an identical manifest hash."""

        return self.manifest_hash == other.manifest_hash


def split_hash(manifests: Sequence[SplitManifest]) -> str:
    """Content hash over the frozen split set, order-independent."""

    return canonical_hash(sorted(manifest.manifest_hash for manifest in manifests))
