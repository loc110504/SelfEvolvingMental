"""Offline audits for promoted procedural lessons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from psyvec.lessons.core import Lesson, LessonMemoryEntry, LessonStore
from psyvec.state.contracts import reject_label_keys

AuditDecision = Literal["keep", "retire", "insufficient-support"]


@dataclass(frozen=True, slots=True)
class PairedBenefitObservation:
    """A treatment-minus-control utility observation under matched conditions."""

    benefit: float


@dataclass(frozen=True, slots=True)
class TransferAuditVerdict:
    """The auditable result of monitoring a promoted lesson."""

    decision: AuditDecision
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PoisoningCheck:
    """Poisoning indicators found without retaining or changing a lesson."""

    reasons: tuple[str, ...]

    @property
    def single_source_support(self) -> bool:
        return "single_source_support" in self.reasons


def audit_negative_transfer(
    entry: LessonMemoryEntry,
    observations: Sequence[PairedBenefitObservation],
    *,
    minimum_transfer_support: int,
    transfer_win_rate: float,
) -> TransferAuditVerdict:
    """Return the monitoring decision for a promoted lesson.

    The development-selected support and win-rate settings are deliberately caller
    supplied.  Monitoring cannot decide on under-supported observations.
    """

    if entry.lifecycle != "promoted":
        raise ValueError("negative-transfer audit requires a promoted lesson")
    if len(observations) < minimum_transfer_support:
        return TransferAuditVerdict(
            "insufficient-support", ("insufficient_transfer_support",)
        )
    benefits = tuple(observation.benefit for observation in observations)
    mean_benefit = sum(benefits) / len(benefits)
    if mean_benefit <= 0:
        return TransferAuditVerdict("retire", ("non_positive_mean_paired_benefit",))
    win_rate = sum(benefit > 0 for benefit in benefits) / len(benefits)
    if win_rate < transfer_win_rate:
        return TransferAuditVerdict("retire", ("transfer_win_rate_below_requirement",))
    return TransferAuditVerdict("keep", ("positive_transfer_supported",))


def check_lesson_poisoning(
    lesson: Lesson,
    supporting_contrasts: Sequence[Mapping[str, Any]],
) -> PoisoningCheck:
    """Refuse label-shaped lesson text and identify single-source support."""

    for text in _lesson_text(lesson):
        key = text.partition(":")[0].strip()
        if ":" in text:
            reject_label_keys({key: text}, where="lesson text")
    source_ids = {
        str(contrast["source_id"])
        for contrast in supporting_contrasts
        if "source_id" in contrast
    }
    reasons = ("single_source_support",) if len(source_ids) == 1 else ()
    return PoisoningCheck(reasons)


def quarantine_conflict(
    store: LessonStore, first: LessonMemoryEntry, second: LessonMemoryEntry
) -> tuple[LessonMemoryEntry, LessonMemoryEntry]:
    """Quarantine both promoted recommendations when their shared scope conflicts."""

    if first.lifecycle != "promoted" or second.lifecycle != "promoted":
        raise ValueError("conflict resolution requires promoted lessons")
    if first.lesson.scope != second.lesson.scope:
        raise ValueError("conflicting lessons must share a scope")
    if first.lesson.do == second.lesson.do:
        raise ValueError("lessons do not have contradictory recommendations")
    return (
        store.quarantine(first.lesson.lesson_id),
        store.quarantine(second.lesson.lesson_id),
    )


def _lesson_text(lesson: Lesson) -> tuple[str, ...]:
    return (
        lesson.trigger,
        lesson.do,
        lesson.avoid,
        lesson.criterion,
        *lesson.exceptions,
    )
