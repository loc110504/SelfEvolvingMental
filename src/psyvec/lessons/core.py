"""Append-only procedural lesson memory for the Phase 5 fast path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from psyvec.evolution.contrast import ContrastRejection, VerifiedContrast
from psyvec.state.contracts import canonical_hash, reject_label_keys

Lifecycle = Literal[
    "candidate", "validated", "promoted", "quarantined", "retired", "rejected"
]


@dataclass(frozen=True, slots=True)
class Lesson:
    """A label-free procedural abstraction, never a raw transcript."""

    lesson_id: str
    role: str
    trigger: str
    do: str
    avoid: str
    criterion: str
    scope: str
    exceptions: tuple[str, ...] = ()
    candidate_confidence: float = 0.0
    source_contrast_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Lesson:
        """Build a lesson after rejecting caller-supplied label-shaped keys."""

        reject_label_keys(payload, where="lesson")
        return cls(
            lesson_id=str(payload["lesson_id"]),
            role=str(payload["role"]),
            trigger=str(payload["trigger"]),
            do=str(payload["do"]),
            avoid=str(payload["avoid"]),
            criterion=str(payload["criterion"]),
            scope=str(payload["scope"]),
            exceptions=tuple(str(value) for value in payload.get("exceptions", ())),
            candidate_confidence=float(payload.get("candidate_confidence", 0.0)),
            source_contrast_ids=tuple(
                str(value) for value in payload.get("source_contrast_ids", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class LessonMemoryEntry:
    lesson: Lesson
    lifecycle: Lifecycle = "candidate"
    validation_result_ids: tuple[str, ...] = ()
    version: int = 1
    merged_from: tuple[str, ...] = ()
    retrieval_stats: Mapping[str, int] = field(default_factory=dict)
    governance_history: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    result_id: str
    source_contrast_id: str
    supports: bool
    reasons: tuple[str, ...] = ()


def distill(
    contrasts: Sequence[VerifiedContrast | ContrastRejection],
    *,
    trigger: str,
    do: str,
    avoid: str,
    criterion: str,
    scope: str,
    lesson_id: str | None = None,
    exceptions: Sequence[str] = (),
    candidate_confidence: float = 0.0,
) -> LessonMemoryEntry:
    """Distill accepted contrasts into an unvalidated candidate lesson."""

    verified = tuple(item for item in contrasts if isinstance(item, VerifiedContrast))
    if not verified or len(verified) != len(contrasts):
        raise ValueError("only verified contrasts may produce a lesson")
    roles = {item.role for item in verified}
    if len(roles) != 1:
        raise ValueError("all contrasts for a lesson must have one role")
    source_ids = tuple(item.contrast_id for item in verified)
    generated_id = canonical_hash(
        {"role": verified[0].role, "scope": scope, "source_contrast_ids": source_ids}
    )
    lesson = Lesson(
        lesson_id=lesson_id or generated_id,
        role=verified[0].role,
        trigger=trigger,
        do=do,
        avoid=avoid,
        criterion=criterion,
        scope=scope,
        exceptions=tuple(exceptions),
        candidate_confidence=candidate_confidence,
        source_contrast_ids=source_ids,
    )
    return LessonMemoryEntry(lesson=lesson, governance_history=("distilled",))


def validate(
    entry: LessonMemoryEntry,
    results: Sequence[ValidationResult],
    *,
    excluded_source_ids: set[str] | frozenset[str],
    minimum_support: int,
) -> LessonMemoryEntry:
    """Validate using only transfer results outside the lesson's source contrasts."""

    if entry.lifecycle != "candidate":
        raise ValueError("only candidate lessons may be validated")
    eligible = tuple(
        result
        for result in results
        if result.source_contrast_id not in excluded_source_ids
    )
    supporting = tuple(result for result in eligible if result.supports)
    reasons: tuple[str, ...] = ()
    lifecycle: Lifecycle = "validated"
    if len(supporting) < minimum_support:
        lifecycle = "rejected"
        reasons = ("insufficient_non_excluded_support",)
    return replace(
        entry,
        lifecycle=lifecycle,
        version=entry.version + 1,
        validation_result_ids=tuple(result.result_id for result in eligible),
        governance_history=entry.governance_history
        + ("validated" if lifecycle == "validated" else "rejected:" + reasons[0],),
    )


class LessonStore:
    """In-memory append-only lesson versions with retrieval from current promotions."""

    def __init__(self) -> None:
        self._entries: dict[str, list[LessonMemoryEntry]] = {}

    @property
    def entries(self) -> tuple[LessonMemoryEntry, ...]:
        return tuple(entry for versions in self._entries.values() for entry in versions)

    def append(self, entry: LessonMemoryEntry) -> LessonMemoryEntry:
        versions = self._entries.setdefault(entry.lesson.lesson_id, [])
        if versions:
            expected = versions[-1].version + 1
            if entry.version != expected:
                raise ValueError("lesson versions must increase by one")
        elif entry.version != 1:
            raise ValueError("first lesson version must be one")
        versions.append(entry)
        return entry

    def promote(self, lesson_id: str) -> LessonMemoryEntry:
        return self._transition(lesson_id, "validated", "promoted")

    def quarantine(self, lesson_id: str) -> LessonMemoryEntry:
        return self._transition(lesson_id, "promoted", "quarantined")

    def retire(self, lesson_id: str) -> LessonMemoryEntry:
        return self._transition(lesson_id, "promoted", "retired")

    def retrieve(
        self, role: str, scope: str, limit: int
    ) -> tuple[LessonMemoryEntry, ...]:
        if limit <= 0:
            return ()
        return tuple(
            entry
            for versions in self._entries.values()
            for entry in (versions[-1],)
            if entry.lifecycle == "promoted"
            and entry.lesson.role == role
            and entry.lesson.scope == scope
        )[:limit]

    def _transition(
        self, lesson_id: str, expected: Lifecycle, lifecycle: Lifecycle
    ) -> LessonMemoryEntry:
        try:
            previous = self._entries[lesson_id][-1]
        except KeyError as error:
            raise ValueError(f"unknown lesson: {lesson_id}") from error
        if previous.lifecycle != expected:
            raise ValueError(f"cannot {lifecycle} a {previous.lifecycle} lesson")
        next_entry = replace(
            previous,
            lifecycle=lifecycle,
            version=previous.version + 1,
            governance_history=previous.governance_history + (lifecycle,),
        )
        return self.append(next_entry)
