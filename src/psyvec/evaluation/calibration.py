"""Fit a monotone total-score calibration on the training split.

Two facts about the v2 results motivate this module.

*Variance compression.* Predicted totals had a standard deviation of 3.0
against a reference standard deviation of 6.5 on dev. The pipeline places
almost everyone in the 4-9 band regardless of severity, so it cannot separate
a moderate participant from a severe one even when it ranks them correctly.

*The endpoint rewards that compression.* DAIC-WOZ PHQ-8 totals are right-
skewed (dev mean 7.4, median 5), and mean absolute error is minimized at the
conditional median. A system that systematically under-scores therefore posts
a better MAE than a well-spread one with the same ranking quality: predicting
the constant 6 for every dev participant scores MAE 5.49, which beats all
three prompting baselines in this repository. MAE alone cannot tell a real
improvement from a shrinkage artifact.

A calibration map makes the trade-off explicit and tunable instead of
implicit. ``objective="mae"`` fits the shrinkage that the reported endpoint
actually rewards; ``objective="spread"`` matches the training-split variance,
preserving severity separation at some MAE cost. Fitting either on the
training split and reporting both is what lets a reader see which of the two a
given result is benefiting from.

Stdlib only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = ["TotalScoreCalibrator", "fit_calibrator"]

CalibrationObjective = Literal["mae", "spread", "identity"]


@dataclass(frozen=True, slots=True)
class TotalScoreCalibrator:
    """Affine, monotone map from raw predicted total to calibrated total."""

    intercept: float
    slope: float
    objective: str
    minimum: float = 0.0
    maximum: float = 24.0

    def apply(self, raw_total: float) -> float:
        return max(
            self.minimum, min(self.maximum, self.intercept + self.slope * raw_total)
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "intercept": self.intercept,
                "slope": self.slope,
                "objective": self.objective,
                "minimum": self.minimum,
                "maximum": self.maximum,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> TotalScoreCalibrator:
        payload = json.loads(text)
        return cls(
            intercept=float(payload["intercept"]),
            slope=float(payload["slope"]),
            objective=str(payload.get("objective", "unknown")),
            minimum=float(payload.get("minimum", 0.0)),
            maximum=float(payload.get("maximum", 24.0)),
        )

    @classmethod
    def load(cls, path: Path) -> TotalScoreCalibrator:
        return cls.from_json(path.read_text(encoding="utf-8"))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _population_sd(values: Sequence[float]) -> float:
    count = len(values)
    mean = sum(values) / count
    return float((sum((value - mean) ** 2 for value in values) / count) ** 0.5)


def fit_calibrator(
    predicted: Sequence[float],
    reference: Sequence[float],
    *,
    objective: CalibrationObjective = "mae",
    slope_grid: int = 401,
    max_slope: float = 4.0,
) -> TotalScoreCalibrator:
    """Fit ``calibrated = intercept + slope * predicted`` on training data.

    ``mae`` searches slopes on a grid and, for each, takes the intercept that
    minimizes absolute error exactly (the median residual) — a least-absolute-
    deviations fit without pulling in an optimizer. ``spread`` instead matches
    the reference standard deviation, which is the choice that preserves
    severity separation. ``identity`` returns a no-op, so an ablation can
    switch calibration off through the same interface.
    """
    if len(predicted) != len(reference):
        raise ValueError("calibration requires aligned predicted/reference sequences")
    if not predicted:
        raise ValueError("calibration requires at least one participant")

    if objective == "identity":
        return TotalScoreCalibrator(intercept=0.0, slope=1.0, objective="identity")

    if objective == "spread":
        predicted_sd = _population_sd(predicted)
        if predicted_sd <= 1e-9:
            return TotalScoreCalibrator(
                intercept=sum(reference) / len(reference),
                slope=0.0,
                objective="spread",
            )
        slope = _population_sd(reference) / predicted_sd
        intercept = sum(reference) / len(reference) - slope * (
            sum(predicted) / len(predicted)
        )
        return TotalScoreCalibrator(
            intercept=intercept, slope=slope, objective="spread"
        )

    if objective != "mae":
        raise ValueError(f"unknown calibration objective: {objective!r}")

    best: tuple[float, float, float] | None = None
    for step in range(slope_grid):
        slope = max_slope * step / (slope_grid - 1)
        intercept = _median(
            [ref - slope * pred for pred, ref in zip(predicted, reference, strict=True)]
        )
        error = sum(
            abs(max(0.0, min(24.0, intercept + slope * pred)) - ref)
            for pred, ref in zip(predicted, reference, strict=True)
        ) / len(predicted)
        if best is None or error < best[0]:
            best = (error, intercept, slope)

    assert best is not None
    _, intercept, slope = best
    return TotalScoreCalibrator(intercept=intercept, slope=slope, objective="mae")
