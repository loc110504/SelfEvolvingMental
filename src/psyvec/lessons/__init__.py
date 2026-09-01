"""Procedural lesson distillation, validation, and versioned retrieval."""

from __future__ import annotations

from psyvec.lessons.audit import (
    AuditDecision,
    PairedBenefitObservation,
    PoisoningCheck,
    TransferAuditVerdict,
    audit_negative_transfer,
    check_lesson_poisoning,
    quarantine_conflict,
)
from psyvec.lessons.core import (
    Lesson,
    LessonMemoryEntry,
    LessonStore,
    Lifecycle,
    ValidationResult,
    distill,
    find_near_duplicates,
    lesson_similarity,
    validate,
)

__all__ = [
    "Lesson",
    "LessonMemoryEntry",
    "LessonStore",
    "Lifecycle",
    "ValidationResult",
    "distill",
    "find_near_duplicates",
    "lesson_similarity",
    "validate",
    "AuditDecision",
    "PairedBenefitObservation",
    "PoisoningCheck",
    "TransferAuditVerdict",
    "audit_negative_transfer",
    "check_lesson_poisoning",
    "quarantine_conflict",
]
