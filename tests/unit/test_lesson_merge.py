"""Phase 5 P5-4: near-duplicate detection and lesson merge provenance."""

from __future__ import annotations

import unittest
from dataclasses import replace

from psyvec.lessons.core import (
    Lesson,
    LessonMemoryEntry,
    LessonStore,
    find_near_duplicates,
    lesson_similarity,
)


def make_lesson(
    lesson_id: str, *, do: str, scope: str = "phq8", **overrides: str
) -> Lesson:
    fields: dict[str, str] = {
        "lesson_id": lesson_id,
        "role": "updater",
        "trigger": "participant answers with a hedge",
        "do": do,
        "avoid": "assuming the hedge means a low score",
        "criterion": "a follow-up resolves the hedge",
        "scope": scope,
    }
    fields.update(overrides)
    return Lesson(**fields)


def store_with(*entries: LessonMemoryEntry) -> LessonStore:
    store = LessonStore()
    for entry in entries:
        store.append(entry)
    return store


def validated(
    lesson: Lesson, *, sources: tuple[str, ...], results: tuple[str, ...]
) -> LessonMemoryEntry:
    return LessonMemoryEntry(
        lesson=replace(lesson, source_contrast_ids=sources),
        lifecycle="validated",
        validation_result_ids=results,
        governance_history=("distilled", "validated"),
    )


class SimilarityTests(unittest.TestCase):
    def test_identical_procedural_text_is_fully_similar(self) -> None:
        first = make_lesson("a", do="ask one clarifying follow-up")
        second = make_lesson("b", do="ask one clarifying follow-up")

        self.assertEqual(lesson_similarity(first, second), 1.0)

    def test_a_different_scope_is_never_a_duplicate(self) -> None:
        first = make_lesson("a", do="ask one clarifying follow-up")
        second = make_lesson("b", do="ask one clarifying follow-up", scope="safety")

        self.assertEqual(lesson_similarity(first, second), 0.0)

    def test_unrelated_lessons_score_below_near_duplicates(self) -> None:
        base = make_lesson("a", do="ask one clarifying follow-up")
        near = make_lesson("b", do="ask a clarifying follow-up question")
        far = make_lesson(
            "c",
            do="stop the topic",
            trigger="budget exhausted",
            avoid="another probe",
            criterion="budget respected",
        )

        self.assertGreater(
            lesson_similarity(base, near), lesson_similarity(base, far)
        )


class NearDuplicateGroupingTests(unittest.TestCase):
    def test_near_duplicates_group_and_singletons_do_not(self) -> None:
        entries = (
            LessonMemoryEntry(make_lesson("a", do="ask one clarifying follow-up")),
            LessonMemoryEntry(make_lesson("b", do="ask one clarifying follow up")),
            LessonMemoryEntry(
                make_lesson(
                    "c",
                    do="stop the topic",
                    trigger="budget exhausted",
                    avoid="another probe",
                    criterion="budget respected",
                )
            ),
        )

        groups = find_near_duplicates(entries, similarity_threshold=0.7)

        self.assertEqual(groups, (("a", "b"),))

    def test_the_threshold_is_caller_supplied_and_validated(self) -> None:
        with self.assertRaises(ValueError):
            find_near_duplicates((), similarity_threshold=0.0)
        with self.assertRaises(ValueError):
            find_near_duplicates((), similarity_threshold=1.5)


class MergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.primary = validated(
            make_lesson("a", do="ask one clarifying follow-up"),
            sources=("contrast-1",),
            results=("result-1",),
        )
        self.duplicate = validated(
            make_lesson("b", do="ask one clarifying follow up"),
            sources=("contrast-2",),
            results=("result-2",),
        )
        self.store = store_with(self.primary, self.duplicate)

    def test_merge_records_its_sources_and_retires_the_duplicate(self) -> None:
        merged = self.store.merge("a", ["b"])

        self.assertEqual(merged.merged_from, ("b",))
        self.assertEqual(
            merged.lesson.source_contrast_ids, ("contrast-1", "contrast-2")
        )
        self.assertEqual(merged.validation_result_ids, ("result-1", "result-2"))
        self.assertEqual(merged.version, 2)
        self.assertEqual(merged.governance_history[-1], "merged:b")

        retired = self.store.current("b")
        self.assertEqual(retired.lifecycle, "retired")
        self.assertEqual(retired.governance_history[-1], "retired:merged_into:a")

    def test_only_the_merged_lesson_is_retrievable_after_promotion(self) -> None:
        self.store.merge("a", ["b"])
        self.store.promote("a")

        retrieved = self.store.retrieve("updater", "phq8", limit=5)

        self.assertEqual([entry.lesson.lesson_id for entry in retrieved], ["a"])

    def test_the_store_stays_append_only(self) -> None:
        before = len(self.store.entries)

        self.store.merge("a", ["b"])

        self.assertEqual(len(self.store.entries), before + 2)

    def test_a_candidate_lesson_cannot_be_merged(self) -> None:
        store = store_with(
            self.primary,
            LessonMemoryEntry(make_lesson("c", do="ask one clarifying follow-up")),
        )

        with self.assertRaises(ValueError) as caught:
            store.merge("a", ["c"])
        self.assertIn("candidate", str(caught.exception))

    def test_merging_across_roles_or_into_itself_is_refused(self) -> None:
        other_role = validated(
            make_lesson("d", do="ask one clarifying follow-up", role="reporter"),
            sources=("contrast-3",),
            results=("result-3",),
        )
        store = store_with(self.primary, other_role)

        with self.assertRaises(ValueError):
            store.merge("a", ["d"])
        with self.assertRaises(ValueError):
            store.merge("a", ["a"])
        with self.assertRaises(ValueError):
            store.merge("a", [])


if __name__ == "__main__":
    unittest.main()
