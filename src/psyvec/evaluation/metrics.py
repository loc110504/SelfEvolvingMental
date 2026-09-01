"""Participant-level evaluation metrics with paired uncertainty estimates."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParticipantResult:
    """One participant's total-score outcome at a fixed question budget."""

    participant_id: str
    predicted_total: float
    reference_total: float
    questions_asked: int
    question_budget: int
    failed: bool = False
    invalid_reason: str | None = None

    @property
    def completed_valid(self) -> bool:
        """Whether this result can contribute to a participant-level metric."""

        return not self.failed and self.invalid_reason is None


@dataclass(frozen=True, slots=True)
class TotalScoreMae:
    """MAE plus the outcomes excluded from its numerical calculation."""

    mae: float
    completed_count: int
    failed_count: int
    invalid_count: int


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Paired mean-difference estimate and caller-configured percentile interval."""

    mean_difference: float
    interval_lower: float
    interval_upper: float


@dataclass(frozen=True, slots=True)
class PairedPermutationResult:
    """Paired permutation measurement without a significance verdict."""

    observed_mean_difference: float
    proportion_at_least_as_extreme: float


def total_score_mae(results: Sequence[ParticipantResult]) -> TotalScoreMae:
    """Calculate total-score MAE, retaining failed and invalid outcome counts."""

    _matched_budget(results)
    completed = [result for result in results if result.completed_valid]
    if not completed:
        raise ValueError("total-score MAE requires at least one completed valid result")
    return TotalScoreMae(
        mae=sum(
            abs(result.predicted_total - result.reference_total)
            for result in completed
        )
        / len(completed),
        completed_count=len(completed),
        failed_count=sum(result.failed for result in results),
        invalid_count=sum(result.invalid_reason is not None for result in results),
    )


def paired_bootstrap(
    first_arm: Sequence[ParticipantResult],
    second_arm: Sequence[ParticipantResult],
    *,
    iterations: int,
    rng: random.Random,
    confidence_level: float,
) -> PairedBootstrapResult:
    """Bootstrap paired total-score-MAE differences using caller-provided randomness."""

    differences = _paired_differences(first_arm, second_arm)
    _validate_sampling(iterations, confidence_level)
    means = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(iterations)
    )
    tail = (1.0 - confidence_level) / 2.0
    return PairedBootstrapResult(
        mean_difference=sum(differences) / len(differences),
        interval_lower=_percentile(means, tail),
        interval_upper=_percentile(means, 1.0 - tail),
    )


def paired_permutation_test(
    first_arm: Sequence[ParticipantResult],
    second_arm: Sequence[ParticipantResult],
    *,
    iterations: int,
    rng: random.Random,
) -> PairedPermutationResult:
    """Measure how often random paired sign flips are at least as extreme."""

    differences = _paired_differences(first_arm, second_arm)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    observed = sum(differences) / len(differences)
    observed_magnitude = abs(observed)
    extreme = sum(
        abs(
            sum(
                difference if rng.randrange(2) else -difference
                for difference in differences
            )
            / len(differences)
        )
        >= observed_magnitude
        for _ in range(iterations)
    )
    return PairedPermutationResult(
        observed_mean_difference=observed,
        proportion_at_least_as_extreme=extreme / iterations,
    )


def _paired_differences(
    first_arm: Sequence[ParticipantResult], second_arm: Sequence[ParticipantResult]
) -> tuple[float, ...]:
    """Return first-minus-second MAE differences after enforcing a valid pairing."""

    _matched_budget(first_arm)
    _matched_budget(second_arm)
    if not first_arm or not second_arm:
        raise ValueError("paired statistics require non-empty arms")
    first_by_id = _results_by_id(first_arm)
    second_by_id = _results_by_id(second_arm)
    if first_by_id.keys() != second_by_id.keys():
        raise ValueError("paired statistics require matching participant ids")
    first_budget = first_arm[0].question_budget
    if first_budget != second_arm[0].question_budget:
        raise ValueError("paired statistics require matched question budgets")
    if any(not result.completed_valid for result in (*first_arm, *second_arm)):
        raise ValueError(
            "paired statistics require completed valid participant results"
        )
    return tuple(
        abs(
            first_by_id[participant_id].predicted_total
            - first_by_id[participant_id].reference_total
        )
        - abs(
            second_by_id[participant_id].predicted_total
            - second_by_id[participant_id].reference_total
        )
        for participant_id in first_by_id
    )


def _matched_budget(results: Sequence[ParticipantResult]) -> None:
    """Reject empty or internally mismatched question-budget result sets."""

    if not results:
        raise ValueError("metrics require at least one participant result")
    if any(result.question_budget != results[0].question_budget for result in results):
        raise ValueError("participant results have unmatched question budgets")


def _results_by_id(
    results: Sequence[ParticipantResult],
) -> dict[str, ParticipantResult]:
    """Index one arm, refusing duplicate participant outcomes."""

    indexed = {result.participant_id: result for result in results}
    if len(indexed) != len(results):
        raise ValueError("paired statistics require unique participant ids per arm")
    return indexed


def _validate_sampling(iterations: int, confidence_level: float) -> None:
    """Validate caller-owned bootstrap configuration."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")


def _percentile(values: Sequence[float], proportion: float) -> float:
    """Linearly interpolate a percentile from an already sorted sample."""

    position = proportion * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
