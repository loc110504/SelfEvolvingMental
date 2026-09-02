"""Offline, role-local preference-pair partitions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from psyvec.evolution.contrast import PreferencePair


@dataclass(frozen=True, slots=True)
class RolePartition:
    """A trainable set of preference pairs for one role and one split."""

    role: str
    split_id: str
    pairs: tuple[PreferencePair, ...]

    @property
    def pair_count(self) -> int:
        """Number of pairs in this role-local partition."""

        return len(self.pairs)


@dataclass(frozen=True, slots=True)
class PartitionRefusal:
    """Reasons a role's complete partition cannot be used for training."""

    role: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RolePartitionBuild:
    """All accepted partitions and every role-level refusal."""

    partitions: tuple[RolePartition, ...]
    refusals: tuple[PartitionRefusal, ...]

    @property
    def passed(self) -> bool:
        """Whether every supplied role produced a usable partition."""

        return not self.refusals


def build_role_partitions(
    pairs: Sequence[PreferencePair], *, minimum_role_pairs: int
) -> RolePartitionBuild:
    """Group pairs by role, retaining complete refusal reasons for invalid groups."""

    if minimum_role_pairs < 0:
        raise ValueError("minimum_role_pairs must be non-negative")

    by_role: dict[str, list[PreferencePair]] = defaultdict(list)
    for pair in pairs:
        by_role[pair.role].append(pair)

    partitions: list[RolePartition] = []
    refusals: list[PartitionRefusal] = []
    for role, role_pairs in by_role.items():
        reasons = _partition_reasons(role_pairs, minimum_role_pairs)
        if reasons:
            refusals.append(PartitionRefusal(role=role, reasons=tuple(reasons)))
            continue
        partitions.append(
            RolePartition(
                role=role,
                split_id=role_pairs[0].split_id,
                pairs=tuple(role_pairs),
            )
        )
    return RolePartitionBuild(partitions=tuple(partitions), refusals=tuple(refusals))


def _partition_reasons(
    pairs: Sequence[PreferencePair], minimum_role_pairs: int
) -> list[str]:
    reasons: list[str] = []
    if len({pair.split_id for pair in pairs}) != 1:
        reasons.append("mixed_split_ids")

    seen_contrasts: set[str] = set()
    for pair in pairs:
        if pair.contrast_id in seen_contrasts:
            reasons.append(f"duplicate_contrast_id:{pair.contrast_id}")
        seen_contrasts.add(pair.contrast_id)
        if pair.chosen == pair.rejected:
            reasons.append(f"chosen_equals_rejected:{pair.pair_id}")

    if len(pairs) < minimum_role_pairs:
        reasons.append("below_minimum_role_pairs")
    return reasons
