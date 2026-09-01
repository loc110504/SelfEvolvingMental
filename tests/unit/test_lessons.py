import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evolution import ContrastRejection, VerifiedContrast  # noqa: E402
from psyvec.lessons import (  # noqa: E402
    Lesson,
    LessonMemoryEntry,
    LessonStore,
    ValidationResult,
    distill,
    validate,
)


def contrast(contrast_id: str = "contrast-1") -> VerifiedContrast:
    return VerifiedContrast(
        contrast_id=contrast_id,
        starting_state_hash="state",
        profile_hash="profile",
        policy_id="policy",
        seed=1,
        role="interviewer",
        chosen_branch_id="chosen",
        rejected_branch_id="rejected",
        delta_utility=1.0,
        checks={"safe": True},
        attribution="interviewer",
    )


def candidate(lesson_id: str = "lesson-1", scope: str = "intake") -> LessonMemoryEntry:
    return distill(
        [contrast()],
        lesson_id=lesson_id,
        trigger="when sleep is discussed",
        do="ask a focused follow-up",
        avoid="assume duration",
        criterion="obtain a concrete duration",
        scope=scope,
    )


class LessonTests(unittest.TestCase):
    def test_candidate_requires_validation_and_promotion_for_retrieval(self) -> None:
        store = LessonStore()
        entry = store.append(candidate())
        self.assertEqual(store.retrieve("interviewer", "intake", 1), ())

        validated = validate(
            entry,
            [ValidationResult("result-1", "other", True)],
            excluded_source_ids=set(entry.lesson.source_contrast_ids),
            minimum_support=1,
        )
        store.append(validated)
        store.promote(entry.lesson.lesson_id)
        self.assertEqual(len(store.retrieve("interviewer", "intake", 1)), 1)

    def test_validation_excludes_own_contrast_source(self) -> None:
        entry = candidate()
        validated = validate(
            entry,
            [ValidationResult("result-1", "contrast-1", True)],
            excluded_source_ids=set(entry.lesson.source_contrast_ids),
            minimum_support=1,
        )
        self.assertEqual(validated.lifecycle, "rejected")
        self.assertEqual(validated.validation_result_ids, ())

    def test_quarantine_hides_promoted_lesson_and_preserves_audit_version(self) -> None:
        store = LessonStore()
        entry = store.append(candidate())
        store.append(
            validate(
                entry,
                [ValidationResult("result-1", "other", True)],
                excluded_source_ids=set(),
                minimum_support=1,
            )
        )
        store.promote(entry.lesson.lesson_id)
        store.quarantine(entry.lesson.lesson_id)
        self.assertEqual(store.retrieve("interviewer", "intake", 1), ())
        self.assertEqual([item.version for item in store.entries], [1, 2, 3, 4])
        self.assertEqual(store.entries[-2].lifecycle, "promoted")

    def test_retrieve_is_bounded_and_filtered_by_role_and_scope(self) -> None:
        store = LessonStore()
        for lesson_id, role, scope in (
            ("one", "interviewer", "intake"),
            ("two", "interviewer", "intake"),
            ("three", "scorer", "intake"),
            ("four", "interviewer", "follow-up"),
        ):
            lesson = Lesson(
                lesson_id=lesson_id,
                role=role,
                trigger="trigger",
                do="do",
                avoid="avoid",
                criterion="criterion",
                scope=scope,
            )
            store.append(LessonMemoryEntry(lesson, lifecycle="validated"))
            store.promote(lesson_id)
        found = store.retrieve("interviewer", "intake", 1)
        self.assertEqual([entry.lesson.lesson_id for entry in found], ["one"])

    def test_contrast_rejection_cannot_be_distilled(self) -> None:
        with self.assertRaisesRegex(ValueError, "verified contrasts"):
            distill(
                [ContrastRejection(role="interviewer")],
                trigger="trigger",
                do="do",
                avoid="avoid",
                criterion="criterion",
                scope="intake",
            )


if __name__ == "__main__":
    unittest.main()
