from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation import (  # noqa: E402
    ParticipantResult,
    paired_bootstrap,
    paired_permutation_test,
    total_score_mae,
)


def result(
    participant_id: str,
    predicted_total: float,
    reference_total: float = 10.0,
    question_budget: int = 5,
    failed: bool = False,
    invalid_reason: str | None = None,
) -> ParticipantResult:
    return ParticipantResult(
        participant_id=participant_id,
        predicted_total=predicted_total,
        reference_total=reference_total,
        questions_asked=3,
        question_budget=question_budget,
        failed=failed,
        invalid_reason=invalid_reason,
    )


class MetricsTests(unittest.TestCase):
    def test_clean_mae_has_no_failed_or_invalid_counts(self) -> None:
        metric = total_score_mae((result("a", 8.0), result("b", 13.0)))

        self.assertEqual(metric.mae, 2.5)
        self.assertEqual(metric.failed_count, 0)
        self.assertEqual(metric.invalid_count, 0)

    def test_mae_reports_failed_and_invalid_records(self) -> None:
        metric = total_score_mae(
            (
                result("valid", 8.0),
                result("failed", 99.0, failed=True),
                result("invalid", 99.0, invalid_reason="malformed"),
            )
        )

        self.assertEqual(metric.mae, 2.0)
        self.assertEqual(metric.failed_count, 1)
        self.assertEqual(metric.invalid_count, 1)

    def test_unmatched_question_budgets_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            total_score_mae((result("a", 9.0), result("b", 9.0, question_budget=6)))

    def test_bootstrap_is_deterministic_and_requires_pairs(self) -> None:
        first = (result("a", 12.0), result("b", 14.0), result("c", 16.0))
        second = (result("a", 11.0), result("b", 12.0), result("c", 13.0))
        bootstrap = paired_bootstrap(
            first, second, iterations=200, rng=random.Random(4), confidence_level=0.9
        )

        self.assertEqual(
            bootstrap,
            paired_bootstrap(
                first,
                second,
                iterations=200,
                rng=random.Random(4),
                confidence_level=0.9,
            ),
        )
        self.assertLessEqual(bootstrap.interval_lower, bootstrap.mean_difference)
        self.assertGreaterEqual(bootstrap.interval_upper, bootstrap.mean_difference)
        with self.assertRaises(ValueError):
            paired_bootstrap(
                first,
                (result("other", 11.0),),
                iterations=1,
                rng=random.Random(1),
                confidence_level=0.9,
            )

    def test_permutation_proportion_distinguishes_arms_deterministically(self) -> None:
        first = tuple(result(str(index), 20.0) for index in range(8))
        large_difference = tuple(result(str(index), 10.0) for index in range(8))
        no_difference = tuple(result(str(index), 20.0) for index in range(8))
        observed = paired_permutation_test(
            first, large_difference, iterations=2_000, rng=random.Random(9)
        )

        self.assertEqual(
            observed,
            paired_permutation_test(
                first, large_difference, iterations=2_000, rng=random.Random(9)
            ),
        )
        none = paired_permutation_test(
            first, no_difference, iterations=2_000, rng=random.Random(9)
        )
        self.assertLess(
            observed.proportion_at_least_as_extreme,
            none.proportion_at_least_as_extreme,
        )


if __name__ == "__main__":
    unittest.main()
