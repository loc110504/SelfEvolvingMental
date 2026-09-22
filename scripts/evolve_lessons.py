#!/usr/bin/env python3
"""Mine, distil and transfer-verify scorer lessons from a training-split run.

This is the one evolution round the paper describes, run end to end on the
decision the pipeline makes most often: assigning a PHQ-8 item score from
retrieved evidence.

    collect  ->  mine  ->  distil  ->  verify transfer  ->  promote

**Collect** is not done here: it is a completed training-split run of
``scripts/run_assessment.py``, whose records cache the exact evidence text each
item was scored from. Replaying from that cache is what makes the verification
step a matched-condition comparison rather than a re-run with fresh retrieval
noise, and it is what keeps it affordable.

**Mine** groups every (participant, item) decision into (item, evidence-tier)
cells and keeps only the cells whose error is *systematic* — dominated by
signed bias rather than scatter. A cell that is merely hard cannot be fixed by
a procedural rule.

**Distil** asks the model for one transferable rule per cell, in the paper's
lesson form (trigger / do / avoid / criterion). The distiller never sees a
transcript and the rule is rejected if it names the supervision it was induced
from.

**Verify transfer** replays the scorer on cells' participants that were *not*
used to distil the rule. Control is the cached original score — the accepted
policy — and the intervention is the same state with the candidate lesson in
the prompt. A candidate is promoted only if it clears minimum support, a
minimum fraction of states improved, a minimum mean utility margin, and shows
no grounding regression.

Rejected candidates are written out alongside promoted ones: the point of the
gate is that it sometimes says no, and a table showing how often is part of
the claim.

Example
-------
    python3 scripts/evolve_lessons.py \
        --train-run results/evaluations/gpt4omini_train \
        --backend openai --model gpt-4o-mini \
        --imputation-model configs/fitted/imputation_run.json \
        --out configs/fitted/lessons.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.imputation import ImputationModel  # noqa: E402
from psyvec.evaluation.item_scoring import (  # noqa: E402
    ITEM_SCORE_JSON_SCHEMA,
    PROMPT_VERSION,
    SCORER_SYSTEM_PROMPT,
    build_item_score_prompt,
    parse_item_score,
    quote_is_grounded,
)
from psyvec.evolution.item_lessons import (  # noqa: E402
    LESSON_DISTILLER_SYSTEM_PROMPT,
    LESSON_JSON_SCHEMA,
    ErrorPattern,
    ReplayOutcome,
    ScoringState,
    build_distillation_prompt,
    candidate_entry,
    mine_error_patterns,
    parse_lesson,
    validate_lesson,
)
from psyvec.lessons.core import LessonStore, ValidationResult, validate  # noqa: E402
from psyvec.model.llm_backend import (  # noqa: E402
    ChatBackend,
    GenerationRequest,
    UsageLedger,
    build_backend,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evolve")


def load_states(
    run_dir: Path, split_dir: Path
) -> tuple[list[ScoringState], dict[str, dict[str, int]]]:
    """Read a completed run into scoring states plus each participant's assessed items.

    The assessed-item map is what lets an abstention be resolved the same way
    at replay time as it was at run time, holding the rest of the participant
    fixed — the "same state" the branching comparison requires.
    """
    states: list[ScoringState] = []
    assessed_by_participant: dict[str, dict[str, int]] = {}
    for path in sorted(run_dir.glob("*_evaluation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        participant = str(record.get("participant_id"))
        sample_path = split_dir / f"{participant}.json"
        if not sample_path.is_file():
            continue
        reference_items = json.loads(sample_path.read_text(encoding="utf-8"))[
            "phq8_scores"
        ].get("items", {})
        assessed_by_participant[participant] = {
            info["item_key"]: int(info["score"])
            for info in record.get("topics", {}).values()
            if info.get("score") is not None and not info.get("abstained")
        }
        for topic, info in record.get("topics", {}).items():
            item_key = info.get("item_key", "")
            if item_key not in reference_items or "evidence_text" not in info:
                continue
            score = info.get("score")
            states.append(
                ScoringState(
                    participant_id=participant,
                    topic=topic,
                    item_key=item_key,
                    evidence_tier=str(info.get("evidence_tier", "unknown")),
                    evidence_text=str(info["evidence_text"]),
                    predicted=float(score) if score is not None else 0.0,
                    reference=float(reference_items[item_key]),
                    abstained=bool(info.get("abstained")),
                    frequency_basis=str(info.get("frequency_basis", "none")),
                    reasoning=str(info.get("summary", "")),
                )
            )
    return states, assessed_by_participant


def _resolved_value(
    score: float | None,
    item_key: str,
    assessed: dict[str, int],
    imputer: ImputationModel | None,
) -> float:
    """The item's contribution to the total, abstention included.

    Scoring the two arms on the *resolved* value is what credits a lesson that
    turns an abstention into a correct rating and penalizes one that turns it
    into a wrong rating. Without an imputer this falls back to zero, which is
    the v2 rule and is available as an ablation.
    """
    if score is not None:
        return float(score)
    if imputer is None:
        return 0.0
    others = {key: value for key, value in assessed.items() if key != item_key}
    return imputer.impute(others, [item_key])[item_key]


def replay_with_lesson(
    backend: ChatBackend,
    state: ScoringState,
    standard: dict[str, str],
    lesson_text: str,
    assessed: dict[str, int],
    imputer: ImputationModel | None,
) -> ReplayOutcome:
    """Re-score one cached state with the candidate lesson in the prompt."""
    parsed = parse_item_score(
        backend.generate(
            GenerationRequest(
                system=SCORER_SYSTEM_PROMPT,
                user=build_item_score_prompt(
                    state.topic, standard, state.evidence_text,
                    lessons=(lesson_text,),
                ),
                max_tokens=600,
                json_schema=ITEM_SCORE_JSON_SCHEMA,
            )
        )
    )
    treatment = _resolved_value(
        parsed.score if parsed.sufficient else None, state.item_key, assessed, imputer
    )
    control = _resolved_value(
        None if state.abstained else state.predicted,
        state.item_key, assessed, imputer,
    )
    return ReplayOutcome(
        state_id=state.state_id,
        participant_id=state.participant_id,
        reference=state.reference,
        control_value=control,
        treatment_value=treatment,
        treatment_grounded=(
            parsed.score is None
            or quote_is_grounded(parsed.quote, state.evidence_text)
        ),
        control_grounded=True,
    )


def render_lesson(lesson: Any) -> str:
    text = (
        f"When {lesson.trigger}, {lesson.do} Avoid: {lesson.avoid} "
        f"You will know it applied when {lesson.criterion}"
    )
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument(
        "--train-dir", type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "train",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "configs" / "fitted" / "lessons.json",
    )
    parser.add_argument(
        "--backend", choices=("openai", "ollama", "vllm", "local"), default="openai"
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--imputation-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260907)
    mining = parser.add_argument_group("mining thresholds (tuned on train)")
    mining.add_argument("--max-candidates", type=int, default=6)
    mining.add_argument("--min-cell-support", type=int, default=8)
    mining.add_argument("--min-absolute-error", type=float, default=0.5)
    mining.add_argument("--min-systematic-share", type=float, default=0.4)
    mining.add_argument("--distill-examples", type=int, default=6)
    gate = parser.add_argument_group("transfer gate (tuned on train)")
    gate.add_argument("--min-support", type=int, default=6)
    gate.add_argument("--min-mean-gain", type=float, default=0.02)
    gate.add_argument("--min-improved-fraction", type=float, default=0.55)
    gate.add_argument("--max-validation-states", type=int, default=24)
    args = parser.parse_args()

    standards: dict[str, dict[str, str]] = json.loads(
        (PROJECT_ROOT / "configs" / "scales" / "scoring_standards.json").read_text(
            encoding="utf-8"
        )
    )["PHQ-8"]
    imputer = (
        None if args.imputation_model is None
        else ImputationModel.load(args.imputation_model)
    )

    states, assessed_by_participant = load_states(args.train_run, args.train_dir)
    if not states:
        raise SystemExit(
            f"no replayable states in {args.train_run}: the run must have been "
            "produced without --no-store-evidence"
        )
    logger.info("loaded %d scoring states from %s", len(states), args.train_run)

    patterns = mine_error_patterns(
        states,
        min_support=args.min_cell_support,
        min_absolute_error=args.min_absolute_error,
        min_systematic_share=args.min_systematic_share,
    )
    logger.info("mined %d correctable cells", len(patterns))
    for pattern in patterns:
        logger.info(
            "  %-28s tier=%-10s n=%-3d signed=%+.2f abs=%.2f systematic=%.0f%%",
            pattern.topic, pattern.evidence_tier, pattern.support,
            pattern.mean_signed_error, pattern.mean_absolute_error,
            pattern.systematic_share * 100,
        )
    if not patterns:
        raise SystemExit("nothing correctable was mined; nothing to distil")

    ledger = UsageLedger()
    backend = build_backend(
        args.backend, args.model, api_key=args.api_key, base_url=args.base_url,
        ledger=ledger, device=args.device,
    )
    rng = random.Random(args.seed)
    store = LessonStore()
    promoted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for pattern in patterns[: args.max_candidates]:
        cell = [
            state for state in states
            if state.topic == pattern.topic
            and state.evidence_tier == pattern.evidence_tier
        ]
        # Split by participant so a lesson is never verified on someone whose
        # own error produced it.
        participants = sorted({state.participant_id for state in cell})
        rng.shuffle(participants)
        cut = max(1, len(participants) // 3)
        source_participants = set(participants[:cut])
        source_states = [
            s for s in cell if s.participant_id in source_participants
        ]
        holdout = [s for s in cell if s.participant_id not in source_participants]
        rng.shuffle(holdout)
        holdout = holdout[: args.max_validation_states]

        source_pattern = ErrorPattern(
            topic=pattern.topic,
            evidence_tier=pattern.evidence_tier,
            support=len(source_states),
            mean_signed_error=pattern.mean_signed_error,
            mean_absolute_error=pattern.mean_absolute_error,
            abstention_rate=pattern.abstention_rate,
            source_state_ids=tuple(sorted(s.state_id for s in source_states)),
        )
        worst = sorted(source_states, key=lambda s: -abs(s.signed_error))
        raw = backend.generate(
            GenerationRequest(
                system=LESSON_DISTILLER_SYSTEM_PROMPT,
                user=build_distillation_prompt(
                    source_pattern, worst, max_examples=args.distill_examples
                ),
                max_tokens=500,
                temperature=0.5,
                json_schema=LESSON_JSON_SCHEMA,
            )
        )
        lesson = parse_lesson(raw, source_pattern)
        if lesson is None:
            logger.warning(
                "%s/%s: distillation produced no usable rule",
                pattern.topic, pattern.evidence_tier,
            )
            rejected.append(
                {
                    "topic": pattern.topic, "tier": pattern.evidence_tier,
                    "reasons": ["distillation_failed"],
                }
            )
            continue

        text = render_lesson(lesson)
        logger.info("%s/%s candidate: %s", pattern.topic, pattern.evidence_tier, text)
        if not holdout:
            rejected.append(
                {
                    "topic": pattern.topic, "tier": pattern.evidence_tier,
                    "lesson": text, "reasons": ["no_holdout_participants"],
                }
            )
            continue

        outcomes: list[ReplayOutcome] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    replay_with_lesson, backend, state,
                    standards.get(state.topic, {}), text,
                    assessed_by_participant.get(state.participant_id, {}), imputer,
                )
                for state in holdout
            ]
            for future in as_completed(futures):
                try:
                    outcomes.append(future.result())
                except Exception as error:  # noqa: BLE001 - one replay must not stop the round
                    logger.error("replay failed: %s", error)

        verdict = validate_lesson(
            outcomes,
            excluded_state_ids=frozenset(source_pattern.source_state_ids),
            min_support=args.min_support,
            min_mean_gain=args.min_mean_gain,
            min_improved_fraction=args.min_improved_fraction,
        )
        entry = store.append(candidate_entry(lesson))
        results = [
            ValidationResult(
                result_id=f"{lesson.lesson_id[:12]}-{outcome.state_id[:12]}",
                source_contrast_id=outcome.state_id,
                supports=outcome.gain > 0,
            )
            for outcome in outcomes
        ]
        entry = store.append(
            validate(
                entry, results,
                excluded_source_ids=frozenset(source_pattern.source_state_ids),
                minimum_support=args.min_support,
            )
        )
        payload = {
            "topic": pattern.topic,
            "tier": pattern.evidence_tier,
            "lesson_id": lesson.lesson_id,
            "lesson": {
                "trigger": lesson.trigger, "do": lesson.do,
                "avoid": lesson.avoid, "criterion": lesson.criterion,
                "scope": lesson.scope, "role": lesson.role,
            },
            "rendered": text,
            "validation": {
                "support": verdict.support,
                "mean_gain": round(verdict.mean_gain, 4),
                "improved_fraction": round(verdict.improved_fraction, 4),
                "safety_regressions": verdict.safety_regressions,
                "source_participants": sorted(source_participants),
                "reasons": list(verdict.reasons),
            },
        }
        if verdict.promoted and entry.lifecycle == "validated":
            store.promote(lesson.lesson_id)
            promoted.append(payload)
            logger.info(
                "  PROMOTED  gain=%+.3f improved=%.0f%% n=%d",
                verdict.mean_gain, verdict.improved_fraction * 100, verdict.support,
            )
        else:
            rejected.append(payload)
            logger.info(
                "  rejected  gain=%+.3f improved=%.0f%% n=%d reasons=%s",
                verdict.mean_gain, verdict.improved_fraction * 100, verdict.support,
                verdict.reasons or (entry.lifecycle,),
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "train_run": str(args.train_run),
                "gate": {
                    "min_support": args.min_support,
                    "min_mean_gain": args.min_mean_gain,
                    "min_improved_fraction": args.min_improved_fraction,
                },
                "lessons": promoted,
                "rejected": rejected,
                "usage": ledger.summary(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "promoted %d of %d candidates -> %s", len(promoted),
        len(promoted) + len(rejected), args.out,
    )
    logger.info("usage: %s", json.dumps(ledger.summary()))


if __name__ == "__main__":
    main()
