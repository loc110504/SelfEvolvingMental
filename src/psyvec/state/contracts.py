"""Typed assessment state records for Phase 3.

Only the fields the Phase 3 gate needs are modelled. Records are immutable; state
changes happen through explicit versioned transitions in ``engine``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "psyvec-state-v1"

DecisionRole = Literal["interviewer", "evaluator", "scorer", "updater"]
EvidenceStatus = Literal["missing", "known", "unknown", "conflicting"]
TopicStatus = Literal["pending", "in_progress", "completed"]
ValidationStatus = Literal["valid", "invalid", "failed"]

#: Ground-truth keys that must never reach a policy-visible record.
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "label",
        "labels",
        "phq8_binary",
        "phq8_score",
        "real_interview",
        "scale_scores",
        "symptom_level",
    }
)


class StateContractError(ValueError):
    """Raised when a state record or transition violates an invariant."""


class LabelLeakageError(StateContractError):
    """Raised when a policy-visible record carries ground-truth supervision."""


def reject_label_keys(payload: Mapping[str, Any], *, where: str) -> None:
    """Fail closed when a policy-visible payload carries label-shaped keys."""

    leaked = sorted(set(payload) & FORBIDDEN_LABEL_KEYS)
    if leaked:
        raise LabelLeakageError(f"{where} must not contain labels: {', '.join(leaked)}")


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    run_id: str
    code_sha: str
    config_hash: str
    prompt_version: str
    policy_id: str
    model_id: str
    adapter_hash: str | None = None
    source_ids: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSlot:
    slot_id: str
    kind: str
    status: EvidenceStatus
    value: str | None = None
    confidence: float | None = None
    source_turn_ids: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status == "known" and not self.source_turn_ids:
            raise StateContractError(
                f"evidence slot {self.slot_id!r} is known without a source turn"
            )


@dataclass(frozen=True, slots=True)
class TopicState:
    topic_id: str
    rubric_version: str
    evidence_slots: tuple[EvidenceSlot, ...] = ()
    status: TopicStatus = "pending"
    proposed_score: int | None = None
    score_basis_slot_ids: tuple[str, ...] = ()
    revision: int = 0

    def __post_init__(self) -> None:
        slot_ids = {slot.slot_id for slot in self.evidence_slots}
        unknown = sorted(set(self.score_basis_slot_ids) - slot_ids)
        if unknown:
            raise StateContractError(
                f"topic {self.topic_id!r} scores cite unknown evidence: "
                f"{', '.join(unknown)}"
            )
        if self.proposed_score is not None and not self.score_basis_slot_ids:
            raise StateContractError(
                f"topic {self.topic_id!r} has a score with no evidence basis"
            )


@dataclass(frozen=True, slots=True)
class AssessmentState:
    case_id: str
    scale_version: str
    topic_states: tuple[TopicState, ...] = ()
    current_topic_id: str | None = None
    dialogue_turn_ids: tuple[str, ...] = ()
    safety_flags: tuple[str, ...] = ()
    policy_snapshot: str = ""
    revision: int = 0
    schema_version: str = SCHEMA_VERSION

    def topic(self, topic_id: str) -> TopicState:
        for topic in self.topic_states:
            if topic.topic_id == topic_id:
                return topic
        raise StateContractError(f"unknown topic: {topic_id!r}")

    @property
    def state_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class RoleAction:
    action_id: str
    role: DecisionRole
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reject_label_keys(self.payload, where=f"action {self.action_id!r} payload")


@dataclass(frozen=True, slots=True)
class RoleOutput:
    action_id: str
    validation_status: ValidationStatus
    parsed_payload: Mapping[str, Any] | None = None
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.parsed_payload is not None:
            reject_label_keys(
                self.parsed_payload, where=f"output {self.action_id!r} payload"
            )


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    event_id: str
    case_id: str
    decision_index: int
    role: DecisionRole
    action: RoleAction
    output: RoleOutput
    pre_state_hash: str
    post_state_hash: str
    state_patch: Mapping[str, Any]
    provenance: ProvenanceRef
    outcome: str
    schema_version: str = SCHEMA_VERSION


def canonical_hash(value: Any) -> str:
    """Hash a record deterministically for pre/post state comparison."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def replace_topic(
    state: AssessmentState, topic: TopicState, *, revision: int | None = None
) -> AssessmentState:
    """Return a new state with ``topic`` swapped in and the revision advanced."""

    topics: Sequence[TopicState] = tuple(
        topic if existing.topic_id == topic.topic_id else existing
        for existing in state.topic_states
    )
    if all(existing.topic_id != topic.topic_id for existing in state.topic_states):
        raise StateContractError(f"unknown topic: {topic.topic_id!r}")
    next_revision = state.revision + 1 if revision is None else revision
    if next_revision <= state.revision:
        raise StateContractError("state revision must increase monotonically")
    return AssessmentState(
        case_id=state.case_id,
        scale_version=state.scale_version,
        topic_states=tuple(topics),
        current_topic_id=state.current_topic_id,
        dialogue_turn_ids=state.dialogue_turn_ids,
        safety_flags=state.safety_flags,
        policy_snapshot=state.policy_snapshot,
        revision=next_revision,
    )
