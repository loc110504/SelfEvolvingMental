"""Offline preparation and verification for role-local adapter training."""

from __future__ import annotations

from psyvec.training.partitions import (
    PartitionRefusal,
    RolePartition,
    RolePartitionBuild,
    build_role_partitions,
)
from psyvec.training.verification import (
    ParameterDiffVerification,
    verify_parameter_diff,
)

__all__ = [
    "ParameterDiffVerification",
    "PartitionRefusal",
    "RolePartition",
    "RolePartitionBuild",
    "build_role_partitions",
    "verify_parameter_diff",
]
