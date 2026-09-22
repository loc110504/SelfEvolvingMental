#!/usr/bin/env python3
"""Mine, distil and transfer-verify acquisition-strategy lessons from a train run.

This is the second bounded self-evolution channel: where
``scripts/evolve_lessons.py`` evolves how the scorer *interprets* evidence,
this evolves what the interviewer role searches for next when a round of
searching on some topic did not move anything. Same governed shape:

    collect  ->  mine  ->  distil  ->  verify transfer  ->  promote

**Collect** is a completed train-split run of
``scripts/run_proposed_assessment.py``. Every scheduled expansion round is
cached in each topic's ``acquisition_trace`` with enough context (the
"missing" description, the queries already tried, and the turn ids already
known going into the round) to replay that exact round later without redoing
everything upstream of it — the acquisition-side analogue of how
``evolve_lessons.py`` replays scoring from cached ``evidence_text``.

**Mine** groups rounds into (topic, evidence-tier-before) cells and keeps the
ones where a round of searching *systematically* failed to raise the tier or
settle the topic — a well-supported low yield rate, not one unlucky
transcript.

**Distil** asks the model for one transferable search-strategy rule per cell,
shown only the topic, the starting tier, and a sample of phrases already
tried — never a transcript, a participant, or an outcome.

**Verify transfer** replays the query-expansion step on rounds from
participants *not* used to distil the rule: it re-issues the real search
against that participant's actual transcript, once with the candidate
lesson appended to the prompt and once without (the cached original queries),
and compares the resulting evidence tier. Promotion requires minimum support,
a minimum fraction of states improved, and a minimum mean utility margin,
exactly like the scorer-lesson channel's gate.

Example
-------
    python3 scripts/evolve_interview_lessons.py \
        --train-run results/proposed/gptoss20b_train \
        --backend ollama --model gpt-oss:20b \
        --out configs/fitted/interview_lessons.json
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

from psyvec.evaluation.evidence_tiers import (  # noqa: E402
    TIER_ORDER,
    build_paired_turns,
    overall_tier,
    search_paired_turns,
)
from psyvec.evaluation.item_scoring import (  # noqa: E402
    INTERVIEWER_SYSTEM_PROMPT,
    PROMPT_VERSION,
    QUERY_EXPANSION_JSON_SCHEMA,
    build_query_expansion_prompt,
    parse_query_expansion,
)
from psyvec.evolution.interview_lessons import (  # noqa: E402
    ACQUISITION_LESSON_DISTILLER_SYSTEM_PROMPT,
    ACQUISITION_LESSON_JSON_SCHEMA,
    AcquisitionPattern,
    AcquisitionReplayOutcome,
    AcquisitionState,
    acquisition_value,
    build_acquisition_distillation_prompt,
    candidate_entry,
    mine_low_yield_patterns,
    parse_acquisition_lesson,
    validate_acquisition_lesson,
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
logger = logging.getLogger("evolve-interview")


def load_states(
    run_dir: Path,
) -> tuple[list[AcquisitionState], dict[str, dict[str, Any]]]:
    """Read a completed train run into acquisition states plus replay context.

    ``contexts[state_id]`` carries what :func:`replay_with_lesson` needs to
    re-issue exactly that round's search: the topic, the "missing" text fed
    to the expansion prompt, the queries already tried before it, and the
    turn ids already known before it (so the replay excludes the same turns
    the original round did).
    """
    states: list[AcquisitionState] = []
    contexts: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*_evaluation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        participant = str(record.get("participant_id"))
        for topic, info in record.get("topics", {}).items():
            for entry in info.get("acquisition_trace", []):
                state = AcquisitionState(
                    participant_id=participant,
                    topic=topic,
                    tier_before=entry["tier_before"],
                    resolved_before=bool(entry["resolved_before"]),
                    tier_after=entry["tier_after"],
                    resolved_after=bool(entry["resolved_after"]),
                    queries_tried=tuple(entry.get("queries_tried", [])),
                    round_index=int(entry["round_index"]),
                )
                states.append(state)
                contexts[state.state_id] = {
                    "participant_id": participant,
                    "topic": topic,
                    "missing_before": str(entry.get("missing_before", "")),
                    "already_tried_before": tuple(
                        entry.get("already_tried_before", [])
                    ),
                    "exclude_turn_ids_before": tuple(
                        entry.get("exclude_turn_ids_before", [])
                    ),
                }
    return states, contexts


def replay_with_lesson(
    backend: ChatBackend,
    state: AcquisitionState,
    context: dict[str, Any],
    standard: dict[str, str],
    lesson_text: str,
    split_dir: Path,
    transcript_cache: dict[str, Any],
    evidence_limit: int,
) -> AcquisitionReplayOutcome:
    """Replay one cached round: cached (control) queries versus the candidate
    lesson appended to the expansion prompt (treatment), both scored by
    tier movement alone.

    The "resolved" bonus in :func:`acquisition_value` is deliberately left
    off both arms here (``resolved=False`` throughout): confirming a
    resolved-sufficiency claim on the treatment arm would need an extra
    sufficiency call the control arm's cached result never had to earn,
    which would bias the comparison against the treatment. Tier movement,
    available identically for both, is the fair signal.
    """
    participant_id = str(context["participant_id"])
    topic = str(context["topic"])
    if participant_id not in transcript_cache:
        sample = json.loads(
            (split_dir / f"{participant_id}.json").read_text(encoding="utf-8")
        )
        transcript_cache[participant_id] = build_paired_turns(
            sample.get("real_interview", []), merge_consecutive=True
        )
    paired_turns = transcript_cache[participant_id]

    prompt = build_query_expansion_prompt(
        topic, standard, str(context["missing_before"]),
        tuple(context["already_tried_before"]),
    )
    prompt += (
        "\n\n## Validated search guidance for this state\n  - " + lesson_text
    )
    new_queries = parse_query_expansion(
        backend.generate(
            GenerationRequest(
                system=INTERVIEWER_SYSTEM_PROMPT, user=prompt, max_tokens=300,
                temperature=0.4, json_schema=QUERY_EXPANSION_JSON_SCHEMA,
            )
        )
    )
    found = ()
    if new_queries:
        found = search_paired_turns(
            paired_turns, new_queries, top_k=evidence_limit,
            exclude_turn_ids=frozenset(context["exclude_turn_ids_before"]),
            tier="expanded",
        )
    found_tier = overall_tier(found) if found else state.tier_before
    treatment_tier = (
        found_tier
        if TIER_ORDER.index(found_tier) > TIER_ORDER.index(state.tier_before)
        else state.tier_before
    )
    return AcquisitionReplayOutcome(
        state_id=state.state_id,
        participant_id=state.participant_id,
        control_value=acquisition_value(state.tier_after, resolved=False),
        treatment_value=acquisition_value(treatment_tier, resolved=False),
    )


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
        "--out", type=Path,
        default=PROJECT_ROOT / "configs" / "fitted" / "interview_lessons.json",
    )
    parser.add_argument(
        "--backend", choices=("openai", "ollama", "vllm", "local"), default="openai"
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--evidence-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    mining = parser.add_argument_group("mining thresholds (tuned on train)")
    mining.add_argument("--max-candidates", type=int, default=6)
    mining.add_argument("--min-cell-support", type=int, default=8)
    mining.add_argument("--max-yield-rate", type=float, default=0.35)
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

    states, contexts = load_states(args.train_run)
    if not states:
        raise SystemExit(
            f"no replayable acquisition rounds in {args.train_run}: the run "
            "must have been produced by run_proposed_assessment.py"
        )
    logger.info("loaded %d acquisition rounds from %s", len(states), args.train_run)

    patterns = mine_low_yield_patterns(
        states, min_support=args.min_cell_support, max_yield_rate=args.max_yield_rate,
    )
    logger.info("mined %d low-yield cells", len(patterns))
    for pattern in patterns:
        logger.info(
            "  %-28s tier=%-10s n=%-3d yield=%.0f%%",
            pattern.topic, pattern.tier_before, pattern.support,
            pattern.yield_rate * 100,
        )
    if not patterns:
        raise SystemExit("nothing low-yield was mined; nothing to distil")

    ledger = UsageLedger()
    backend = build_backend(
        args.backend, args.model, api_key=args.api_key, base_url=args.base_url,
        ledger=ledger, device=args.device,
    )
    rng = random.Random(args.seed)
    store = LessonStore()
    promoted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    transcript_cache: dict[str, Any] = {}

    for pattern in patterns[: args.max_candidates]:
        cell = [
            state for state in states
            if state.topic == pattern.topic and state.tier_before == pattern.tier_before
        ]
        participants = sorted({state.participant_id for state in cell})
        rng.shuffle(participants)
        cut = max(1, len(participants) // 3)
        source_participants = set(participants[:cut])
        source_states = [s for s in cell if s.participant_id in source_participants]
        holdout = [s for s in cell if s.participant_id not in source_participants]
        rng.shuffle(holdout)
        holdout = holdout[: args.max_validation_states]

        source_pattern = AcquisitionPattern(
            topic=pattern.topic, tier_before=pattern.tier_before,
            support=len(source_states), yield_rate=pattern.yield_rate,
            source_state_ids=tuple(sorted(s.state_id for s in source_states)),
            sample_queries=pattern.sample_queries,
        )
        raw = backend.generate(
            GenerationRequest(
                system=ACQUISITION_LESSON_DISTILLER_SYSTEM_PROMPT,
                user=build_acquisition_distillation_prompt(source_pattern),
                max_tokens=500, temperature=0.5,
                json_schema=ACQUISITION_LESSON_JSON_SCHEMA,
            )
        )
        lesson = parse_acquisition_lesson(raw, source_pattern)
        if lesson is None:
            logger.warning(
                "%s/%s: distillation produced no usable rule",
                pattern.topic, pattern.tier_before,
            )
            rejected.append({
                "topic": pattern.topic, "tier": pattern.tier_before,
                "reasons": ["distillation_failed"],
            })
            continue

        text = (
            f"When {lesson.trigger}, {lesson.do} Avoid: {lesson.avoid} "
            f"You will know it applied when {lesson.criterion}"
        )
        text = " ".join(text.split())
        logger.info("%s/%s candidate: %s", pattern.topic, pattern.tier_before, text)
        if not holdout:
            rejected.append({
                "topic": pattern.topic, "tier": pattern.tier_before,
                "lesson": text, "reasons": ["no_holdout_participants"],
            })
            continue

        standard = standards.get(pattern.topic, {})
        outcomes: list[AcquisitionReplayOutcome] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    replay_with_lesson, backend, state, contexts[state.state_id],
                    standard, text, args.train_dir, transcript_cache,
                    args.evidence_limit,
                )
                for state in holdout
            ]
            for future in as_completed(futures):
                try:
                    outcomes.append(future.result())
                except Exception as error:  # noqa: BLE001 - one replay must not stop the round
                    logger.error("replay failed: %s", error)

        verdict = validate_acquisition_lesson(
            outcomes, excluded_state_ids=frozenset(source_pattern.source_state_ids),
            min_support=args.min_support, min_mean_gain=args.min_mean_gain,
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
            "tier": pattern.tier_before,
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
