"""Resolve abstained PHQ-8 items by conditional imputation instead of zero.

The v2 pipeline enforced "no retrieved evidence implies score 0" as a hard
rule (``psyvec.evaluation.interview_policy.grounded_score``). Measured against
the DAIC-WOZ references, that rule discards 2.66 PHQ-8 points per participant
out of a mean total of 6.4 — roughly 40% of the signal — because on the
no-evidence subset the reference item score is nonzero 55-78% of the time.
The damage is concentrated exactly where it hurts most: participants with a
reference total of 15+ were under-predicted by 8.6 (dev) to 10.1 (train)
points, because severe participants have more symptoms to miss.

The fix is to treat an unassessed item as *missing data*, not as a zero
observation, and to impute it from the structure PHQ-8 items actually have:
they are positively correlated, so a participant whose assessed items are high
is unlikely to be at zero on the one item the interview never covered.

Two estimators are provided, both fitted only on training-split data:

``fit_from_references``
    Needs labels only, no pipeline run. For each item it regresses that item's
    reference score on the mean of the participant's *other* reference items.
``fit_from_run``
    Needs a completed training-split run. Regresses each item's reference
    score on the mean of the items the pipeline actually *assessed* for that
    participant, so the fit absorbs any systematic offset between what the
    scorer assigns and what the reference says. Preferred when available.

Both fall back to the item's training base rate when a participant has no
assessed item to condition on. Stdlib only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ImputationModel",
    "ItemFit",
    "fit_from_references",
    "fit_from_run",
]


@dataclass(frozen=True, slots=True)
class ItemFit:
    """Conditional-mean parameters for one PHQ-8 item.

    ``predicted = intercept + slope * conditioning_mean``, clipped to the
    rubric range. ``base_rate`` is the unconditional training mean, used when
    a participant has nothing assessed to condition on.
    """

    item_key: str
    intercept: float
    slope: float
    base_rate: float
    support: int

    def predict(self, conditioning_mean: float | None) -> float:
        if conditioning_mean is None:
            return _clip(self.base_rate)
        return _clip(self.intercept + self.slope * conditioning_mean)


@dataclass(frozen=True, slots=True)
class ImputationModel:
    """Per-item conditional-mean imputers fitted on a training split."""

    fits: Mapping[str, ItemFit]
    source: str

    def impute(
        self,
        assessed: Mapping[str, int | float],
        missing_items: Sequence[str],
    ) -> dict[str, float]:
        """Estimate each item in ``missing_items`` from the ``assessed`` ones."""
        conditioning = (
            sum(float(value) for value in assessed.values()) / len(assessed)
            if assessed
            else None
        )
        estimates: dict[str, float] = {}
        for item_key in missing_items:
            fit = self.fits.get(item_key)
            estimates[item_key] = (
                fit.predict(conditioning) if fit is not None else _clip(0.9)
            )
        return estimates

    def to_json(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "fits": {
                    key: {
                        "intercept": fit.intercept,
                        "slope": fit.slope,
                        "base_rate": fit.base_rate,
                        "support": fit.support,
                    }
                    for key, fit in self.fits.items()
                },
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> ImputationModel:
        payload = json.loads(text)
        return cls(
            source=str(payload.get("source", "unknown")),
            fits={
                key: ItemFit(
                    item_key=key,
                    intercept=float(value["intercept"]),
                    slope=float(value["slope"]),
                    base_rate=float(value["base_rate"]),
                    support=int(value["support"]),
                )
                for key, value in payload["fits"].items()
            },
        )

    @classmethod
    def load(cls, path: Path) -> ImputationModel:
        return cls.from_json(path.read_text(encoding="utf-8"))


def _clip(value: float, low: float = 0.0, high: float = 3.0) -> float:
    return max(low, min(high, value))


def _least_squares(
    pairs: Sequence[tuple[float, float]]
) -> tuple[float, float]:
    """Ordinary least squares slope/intercept for ``(x, y)`` pairs.

    Degenerate input (fewer than three points, or no spread in ``x``) returns
    a zero slope at the mean of ``y``: with nothing to condition on, the
    unconditional mean is the right estimator, not an extrapolated line.
    """
    if len(pairs) < 3:
        mean_y = sum(y for _, y in pairs) / len(pairs) if pairs else 0.0
        return mean_y, 0.0
    count = len(pairs)
    mean_x = sum(x for x, _ in pairs) / count
    mean_y = sum(y for _, y in pairs) / count
    variance = sum((x - mean_x) ** 2 for x, _ in pairs)
    if variance <= 1e-9:
        return mean_y, 0.0
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    slope = covariance / variance
    return mean_y - slope * mean_x, slope


def fit_from_references(
    reference_items: Sequence[Mapping[str, int]],
    item_keys: Sequence[str],
) -> ImputationModel:
    """Fit each item against the leave-one-out mean of the participant's others.

    Label-only: usable before any model has been run, and unaffected by which
    pipeline produced the assessed scores.
    """
    if not reference_items:
        raise ValueError("imputation fitting requires at least one participant")
    fits: dict[str, ItemFit] = {}
    for item_key in item_keys:
        pairs: list[tuple[float, float]] = []
        values: list[float] = []
        for record in reference_items:
            if item_key not in record:
                continue
            others = [
                float(value) for key, value in record.items() if key != item_key
            ]
            values.append(float(record[item_key]))
            if others:
                pairs.append((sum(others) / len(others), float(record[item_key])))
        if not values:
            continue
        intercept, slope = _least_squares(pairs)
        fits[item_key] = ItemFit(
            item_key=item_key,
            intercept=intercept,
            slope=slope,
            base_rate=sum(values) / len(values),
            support=len(values),
        )
    return ImputationModel(fits=fits, source="references")


def fit_from_run(
    assessed_by_participant: Sequence[Mapping[str, int | float]],
    reference_items: Sequence[Mapping[str, int]],
    item_keys: Sequence[str],
) -> ImputationModel:
    """Fit each item's reference score against the mean of what was assessed.

    ``assessed_by_participant[i]`` holds only the items the pipeline actually
    scored for participant ``i`` (abstentions omitted); ``reference_items[i]``
    holds that participant's reference scores. For item ``j`` the fit uses the
    participants where ``j`` was *not* assessed, which is precisely the
    population the imputer will be applied to.
    """
    if len(assessed_by_participant) != len(reference_items):
        raise ValueError("assessed and reference sequences must be aligned")
    if not reference_items:
        raise ValueError("imputation fitting requires at least one participant")

    fits: dict[str, ItemFit] = {}
    for item_key in item_keys:
        pairs: list[tuple[float, float]] = []
        values: list[float] = []
        paired = zip(assessed_by_participant, reference_items, strict=True)
        for assessed, reference in paired:
            if item_key not in reference or item_key in assessed:
                continue
            values.append(float(reference[item_key]))
            others = [float(value) for value in assessed.values()]
            if others:
                pairs.append((sum(others) / len(others), float(reference[item_key])))
        if len(values) < 3:
            # Too few abstentions on this item to fit anything trustworthy;
            # fall back to the label-only base rate over all participants.
            all_values = [
                float(reference[item_key])
                for reference in reference_items
                if item_key in reference
            ]
            if not all_values:
                continue
            fits[item_key] = ItemFit(
                item_key=item_key,
                intercept=sum(all_values) / len(all_values),
                slope=0.0,
                base_rate=sum(all_values) / len(all_values),
                support=len(all_values),
            )
            continue
        intercept, slope = _least_squares(pairs)
        fits[item_key] = ItemFit(
            item_key=item_key,
            intercept=intercept,
            slope=slope,
            base_rate=sum(values) / len(values),
            support=len(values),
        )
    return ImputationModel(fits=fits, source="run")
