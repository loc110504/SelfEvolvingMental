"""Append-only procedural lesson memory for the Phase 5 fast path."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
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


def lesson_similarity(first: Lesson, second: Lesson) -> float:
    """Token-overlap similarity of two lessons' procedural text.

    Deliberately crude: the merge threshold is caller-supplied, so this only has
    to order near-duplicates above unrelated lessons.
    """

    if first.role != second.role or first.scope != second.scope:
        return 0.0
    first_tokens = _lesson_tokens(first)
    second_tokens = _lesson_tokens(second)
    union = first_tokens | second_tokens
    if not union:
        return 0.0
    return len(first_tokens & second_tokens) / len(union)


def find_near_duplicates(
    entries: Sequence[LessonMemoryEntry], *, similarity_threshold: float
) -> tuple[tuple[str, ...], ...]:
    """Group current entries whose lessons are near-duplicates of each other.

    The first id in each group is the primary — the one a merge keeps.
    """

    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    groups: list[list[LessonMemoryEntry]] = []
    for entry in entries:
        for group in groups:
            if (
                lesson_similarity(group[0].lesson, entry.lesson)
                >= similarity_threshold
            ):
                group.append(entry)
                break
        else:
            groups.append([entry])
    return tuple(
        tuple(member.lesson.lesson_id for member in group)
        for group in groups
        if len(group) > 1
    )


def _lesson_tokens(lesson: Lesson) -> frozenset[str]:
    """Lowercased word tokens of the fields that carry the procedure."""

    text = " ".join(
        (lesson.trigger, lesson.do, lesson.avoid, lesson.criterion)
    ).lower()
    return frozenset(token for token in re.split(r"\W+", text) if token)


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

    def current(self, lesson_id: str) -> LessonMemoryEntry:
        """Return the newest version of one lesson."""

        try:
            return self._entries[lesson_id][-1]
        except KeyError as error:
            raise ValueError(f"unknown lesson: {lesson_id}") from error

    def merge(
        self, primary_id: str, duplicate_ids: Sequence[str]
    ) -> LessonMemoryEntry:
        """Fold near-duplicates into ``primary_id`` and retire them.

        The merged version records every folded id in ``merged_from`` and takes
        the union of the sources and validation results, so a merged lesson
        keeps citing everything that ever justified it. Duplicates are retired
        rather than deleted: the store is append-only.
        """

        if not duplicate_ids:
            raise ValueError("a merge needs at least one duplicate lesson")
        if primary_id in duplicate_ids:
            raise ValueError("a lesson cannot be merged into itself")
        if len(set(duplicate_ids)) != len(duplicate_ids):
            raise ValueError("duplicate lesson ids must be unique")

        primary = self.current(primary_id)
        duplicates = tuple(self.current(lesson_id) for lesson_id in duplicate_ids)
        mergeable: frozenset[Lifecycle] = frozenset({"validated", "promoted"})
        for entry in (primary, *duplicates):
            if entry.lifecycle not in mergeable:
                raise ValueError(f"cannot merge a {entry.lifecycle} lesson")
            if entry.lesson.role != primary.lesson.role:
                raise ValueError("merged lessons must share one role")
            if entry.lesson.scope != primary.lesson.scope:
                raise ValueError("merged lessons must share one scope")

        merged = replace(
            primary,
            lesson=replace(
                primary.lesson,
                source_contrast_ids=_ordered_union(
                    entry.lesson.source_contrast_ids
                    for entry in (primary, *duplicates)
                ),
            ),
            version=primary.version + 1,
            merged_from=_ordered_union(
                (
                    primary.merged_from,
                    tuple(entry.lesson.lesson_id for entry in duplicates),
                )
            ),
            validation_result_ids=_ordered_union(
                entry.validation_result_ids for entry in (primary, *duplicates)
            ),
            governance_history=primary.governance_history
            + tuple(f"merged:{entry.lesson.lesson_id}" for entry in duplicates),
        )
        self.append(merged)
        for entry in duplicates:
            self.append(
                replace(
                    entry,
                    lifecycle="retired",
                    version=entry.version + 1,
                    governance_history=entry.governance_history
                    + (f"retired:merged_into:{primary_id}",),
                )
            )
        return merged

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


def _ordered_union(groups: Iterable[Sequence[str]]) -> tuple[str, ...]:
    """Concatenate id sequences, keeping first-seen order and dropping repeats."""

    seen: dict[str, None] = {}
    for group in groups:
        for value in group:
            seen.setdefault(value, None)
    return tuple(seen)
