"""Phase 8 P8-4 robustness and ablation budget.

Doc 06 requires ablation arms to run under a matched compute budget and requires
robustness to be reported across seeds. Like the rest of the statistics
machinery this module returns **measured quantities only** — never a verdict,
never a hard-coded threshold or alpha.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class BudgetError(ValueError):
    """Raised when an ablation set violates its matched-budget contract."""


@dataclass(frozen=True, slots=True)
class AblationArm:
    """One arm of an ablation, with the compute it is allowed to spend."""

    arm_id: str
    ablated_component: str | None
    question_budget: int
    model_call_budget: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.question_budget <= 0 or self.model_call_budget <= 0:
            raise BudgetError(f"arm {self.arm_id!r} needs a positive budget")
        if not self.seeds:
            raise BudgetError(f"arm {self.arm_id!r} needs at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise BudgetError(f"arm {self.arm_id!r} repeats a seed")

    @property
    def is_reference(self) -> bool:
        """The unablated arm every other arm is measured against."""

        return self.ablated_component is None


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """One arm's measured value at one seed, plus what it actually spent."""

    arm_id: str
    seed: int
    value: float
    questions_asked: int
    model_calls: int


@dataclass(frozen=True, slots=True)
class SeedRobustness:
    """Spread of one arm's value across its seeds. No verdict."""

    arm_id: str
    seed_count: int
    mean: float
    minimum: float
    maximum: float

    @property
    def spread(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True, slots=True)
class AblationEffect:
    """Per-seed and mean difference between one ablated arm and the reference."""

    arm_id: str
    ablated_component: str
    reference_arm_id: str
    per_seed_differences: tuple[tuple[int, float], ...]
    mean_difference: float


def assert_matched_budget(arms: Sequence[AblationArm]) -> AblationArm:
    """Check the arm set shares one budget and one seed set, and return the reference.

    A differing budget or seed set is what makes an ablation uninterpretable, so
    it is a hard failure rather than a reported difference.
    """

    if len(arms) < 2:
        raise BudgetError("an ablation needs a reference arm and at least one ablation")
    reference_arms = [arm for arm in arms if arm.is_reference]
    if len(reference_arms) != 1:
        raise BudgetError("an ablation needs exactly one unablated reference arm")
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise BudgetError("ablation arms must have unique ids")
    reference = reference_arms[0]
    for arm in arms:
        for name in ("question_budget", "model_call_budget", "seeds"):
            if getattr(arm, name) != getattr(reference, name):
                raise BudgetError(f"arm {arm.arm_id!r} differs in {name}")
    return reference


def assert_budget_respected(
    arm: AblationArm, outcomes: Sequence[ArmOutcome]
) -> None:
    """Check no outcome for ``arm`` spent more than the arm was allowed."""

    for outcome in outcomes:
        if outcome.arm_id != arm.arm_id:
            raise BudgetError(
                f"outcome for {outcome.arm_id!r} does not belong to {arm.arm_id!r}"
            )
        if outcome.seed not in arm.seeds:
            raise BudgetError(f"arm {arm.arm_id!r} has no seed {outcome.seed}")
        if outcome.questions_asked > arm.question_budget:
            raise BudgetError(
                f"arm {arm.arm_id!r} exceeded its question budget at seed "
                f"{outcome.seed}"
            )
        if outcome.model_calls > arm.model_call_budget:
            raise BudgetError(
                f"arm {arm.arm_id!r} exceeded its model-call budget at seed "
                f"{outcome.seed}"
            )


def measure_seed_robustness(
    arm: AblationArm, outcomes: Sequence[ArmOutcome]
) -> SeedRobustness:
    """Report an arm's spread across every one of its seeds."""

    assert_budget_respected(arm, outcomes)
    by_seed = _values_by_seed(arm, outcomes)
    values = tuple(by_seed[seed] for seed in arm.seeds)
    return SeedRobustness(
        arm_id=arm.arm_id,
        seed_count=len(values),
        mean=sum(values) / len(values),
        minimum=min(values),
        maximum=max(values),
    )


def measure_ablation_effects(
    arms: Sequence[AblationArm], outcomes: Sequence[ArmOutcome]
) -> tuple[AblationEffect, ...]:
    """Measure every ablated arm against the reference at each shared seed."""

    reference = assert_matched_budget(arms)
    outcomes_by_arm: dict[str, list[ArmOutcome]] = {arm.arm_id: [] for arm in arms}
    for outcome in outcomes:
        if outcome.arm_id not in outcomes_by_arm:
            raise BudgetError(f"outcome cites unknown arm {outcome.arm_id!r}")
        outcomes_by_arm[outcome.arm_id].append(outcome)
    reference_values = _values_by_seed(reference, outcomes_by_arm[reference.arm_id])
    assert_budget_respected(reference, outcomes_by_arm[reference.arm_id])

    effects: list[AblationEffect] = []
    for arm in arms:
        if arm.is_reference:
            continue
        assert_budget_respected(arm, outcomes_by_arm[arm.arm_id])
        arm_values = _values_by_seed(arm, outcomes_by_arm[arm.arm_id])
        differences = tuple(
            (seed, arm_values[seed] - reference_values[seed]) for seed in arm.seeds
        )
        assert arm.ablated_component is not None
        effects.append(
            AblationEffect(
                arm_id=arm.arm_id,
                ablated_component=arm.ablated_component,
                reference_arm_id=reference.arm_id,
                per_seed_differences=differences,
                mean_difference=sum(value for _, value in differences)
                / len(differences),
            )
        )
    return tuple(effects)


def _values_by_seed(
    arm: AblationArm, outcomes: Sequence[ArmOutcome]
) -> dict[int, float]:
    """Index one arm's outcomes, requiring exactly one result per declared seed."""

    values = {outcome.seed: outcome.value for outcome in outcomes}
    if len(values) != len(outcomes):
        raise BudgetError(f"arm {arm.arm_id!r} repeats a seed outcome")
    missing = sorted(set(arm.seeds) - set(values))
    if missing:
        raise BudgetError(
            f"arm {arm.arm_id!r} is missing seeds: "
            + ", ".join(str(seed) for seed in missing)
        )
    return values
