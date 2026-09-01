import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.lessons import (  # noqa: E402
    Lesson,
    LessonMemoryEntry,
    LessonStore,
    PairedBenefitObservation,
    audit_negative_transfer,
    check_lesson_poisoning,
    quarantine_conflict,
)
from psyvec.state.contracts import LabelLeakageError  # noqa: E402


def promoted(lesson_id: str, do: str = "ask a follow-up") -> LessonMemoryEntry:
    return LessonMemoryEntry(
        Lesson(
            lesson_id, "interviewer", "when needed", do, "guess", "evidence", "intake"
        ),
        lifecycle="promoted",
    )


class LessonAuditTests(unittest.TestCase):
    def test_non_positive_retires_and_positive_keeps(self) -> None:
        entry = promoted("one")
        self.assertEqual(
            audit_negative_transfer(
                entry, [PairedBenefitObservation(0.0)], minimum_transfer_support=1,
                transfer_win_rate=1.0,
            ).decision,
            "retire",
        )
        self.assertEqual(
            audit_negative_transfer(
                entry, [PairedBenefitObservation(1.0), PairedBenefitObservation(2.0)],
                minimum_transfer_support=2, transfer_win_rate=1.0,
            ).decision,
            "keep",
        )

    def test_low_support_is_not_retired(self) -> None:
        self.assertEqual(
            audit_negative_transfer(
                promoted("one"), [PairedBenefitObservation(-1.0)],
                minimum_transfer_support=2, transfer_win_rate=1.0,
            ).decision,
            "insufficient-support",
        )

    def test_poisoning_flags_single_source_and_refuses_labels(self) -> None:
        self.assertTrue(
            check_lesson_poisoning(promoted("one").lesson, [{"source_id": "case-1"}])
            .single_source_support
        )
        poisoned = Lesson(
            "two", "interviewer", "label: hidden", "do", "avoid", "criterion", "intake"
        )
        with self.assertRaises(LabelLeakageError):
            check_lesson_poisoning(poisoned, [])

    def test_conflicts_quarantine_both_and_keep_versions(self) -> None:
        store = LessonStore()
        first, second = promoted("one", "ask"), promoted("two", "avoid")
        store.append(first)
        store.append(second)
        quarantine_conflict(store, first, second)
        self.assertEqual(store.retrieve("interviewer", "intake", 2), ())
        self.assertEqual([entry.version for entry in store.entries], [1, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
