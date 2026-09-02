"""Paired memory ON/OFF evaluation and internalized-action measurement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass

from psyvec.policy import PromotionError


@dataclass(frozen=True, slots=True)
class OnOffArm:
    """One policy/memory-setting score on an ordered held-out sample."""

    quality: float
    state_ids: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OnOffDecomposition:
    """Separate retrieval-plus-policy and parametric policy gains."""

    combined_on_gain: float
    parametric_off_gain: float


@dataclass(frozen=True, slots=True)
class OnOffResult:
    """Four paired evaluation arms sharing identical held-out inputs."""

    initial_on: OnOffArm
    initial_off: OnOffArm
    evolved_on: OnOffArm
    evolved_off: OnOffArm

    def __post_init__(self) -> None:
        arms = (self.initial_on, self.initial_off, self.evolved_on, self.evolved_off)
        reference = arms[0]
        if any(
            arm.state_ids != reference.state_ids or arm.seeds != reference.seeds
            for arm in arms[1:]
        ):
            raise PromotionError(
                "memory ON/OFF evaluation requires identical held-out states and seeds"
            )

    @property
    def decomposition(self) -> OnOffDecomposition:
        return OnOffDecomposition(
            combined_on_gain=self.evolved_on.quality - self.initial_on.quality,
            parametric_off_gain=self.evolved_off.quality - self.initial_off.quality,
        )


def internalized_action_rate(
    triggered_state_ids: Sequence[str],
    evolved_off_actions: Mapping[str, str],
    positive_actions: Mapping[str, Set[str]],
) -> float:
    """Return the share of triggered states with a validated OFF action."""

    if not triggered_state_ids:
        raise PromotionError("IAR is undefined with zero triggered states")
    return sum(
        evolved_off_actions.get(state_id) in positive_actions.get(state_id, set())
        for state_id in triggered_state_ids
    ) / len(triggered_state_ids)


@dataclass(frozen=True, slots=True)
class InternalizationVerdict:
    """Whether both required indicators improved over their baselines."""

    internalized: bool
    reasons: tuple[str, ...]


def internalization_verdict(
    *,
    initial_off_quality: float,
    evolved_off_quality: float,
    initial_iar: float,
    evolved_iar: float,
) -> InternalizationVerdict:
    """Claim internalization only when OFF quality and IAR both improve."""

    reasons: list[str] = []
    if evolved_off_quality <= initial_off_quality:
        reasons.append("memory_off_quality_not_improved")
    if evolved_iar <= initial_iar:
        reasons.append("iar_not_improved")
    return InternalizationVerdict(not reasons, tuple(reasons))
