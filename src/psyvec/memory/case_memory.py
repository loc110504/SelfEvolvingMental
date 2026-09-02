"""Typed, single-participant case memory and its read policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType

from psyvec.state import AssessmentState
from psyvec.state.contracts import reject_label_keys


def _is_utc(timestamp: datetime) -> bool:
    return (
        timestamp.tzinfo is not None
        and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp)
    )


class CaseMemoryAccessError(PermissionError):
    """Base error for denied CaseMemory reads."""


class CaseIdMismatchError(CaseMemoryAccessError):
    """Raised when a read requests a different participant's case."""


class UnauthorizedRoleError(CaseMemoryAccessError):
    """Raised when a role is not allowed to read the case."""


class ExpiredCaseMemoryError(CaseMemoryAccessError):
    """Raised when a CaseMemory has passed its retention deadline."""


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    """Roles permitted to read a case memory."""

    allowed_roles: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_roles", frozenset(self.allowed_roles))


@dataclass(frozen=True, slots=True)
class CaseMemory:
    """Sanitized, one-participant projection of an assessment state."""

    case_id: str
    assessment_state_ref: str
    evidence_index: Mapping[str, tuple[str, ...]]
    topic_summaries: Mapping[str, str]
    turn_refs: tuple[str, ...]
    created_at: datetime
    access_policy: AccessPolicy
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not _is_utc(self.created_at):
            raise ValueError("created_at must be UTC")
        if self.expires_at is not None and not _is_utc(self.expires_at):
            raise ValueError("expires_at must be UTC")
        object.__setattr__(
            self, "evidence_index", MappingProxyType(dict(self.evidence_index))
        )
        object.__setattr__(
            self, "topic_summaries", MappingProxyType(dict(self.topic_summaries))
        )
        object.__setattr__(self, "turn_refs", tuple(self.turn_refs))


@dataclass(frozen=True, slots=True)
class CaseMemoryAdapter:
    """Enforces the case and role boundary for CaseMemory reads."""

    def read(self, memory: CaseMemory, *, role: str, case_id: str) -> CaseMemory:
        """Return ``memory`` only when the request is within its read policy."""

        if case_id != memory.case_id:
            raise CaseIdMismatchError("requested case does not match CaseMemory")
        if role not in memory.access_policy.allowed_roles:
            raise UnauthorizedRoleError("role is not permitted to read CaseMemory")
        if (
            memory.expires_at is not None
            and datetime.now(timezone.utc) >= memory.expires_at
        ):
            raise ExpiredCaseMemoryError("CaseMemory has expired")
        return memory


@dataclass(frozen=True, slots=True)
class CaseMemoryBuilder:
    """Projects policy-visible AssessmentState fields into CaseMemory."""

    def build(
        self,
        state: AssessmentState,
        *,
        topic_summaries: Mapping[str, str],
        access_policy: AccessPolicy,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> CaseMemory:
        reject_label_keys(topic_summaries, where="case memory topic summaries")
        return CaseMemory(
            case_id=state.case_id,
            assessment_state_ref=state.state_hash,
            evidence_index={
                topic.topic_id: tuple(slot.slot_id for slot in topic.evidence_slots)
                for topic in state.topic_states
            },
            topic_summaries=topic_summaries,
            turn_refs=state.dialogue_turn_ids,
            created_at=created_at or datetime.now(timezone.utc),
            expires_at=expires_at,
            access_policy=access_policy,
        )
