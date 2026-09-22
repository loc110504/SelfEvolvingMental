#!/usr/bin/env python3
"""PsyVEC assessment over DAIC-WOZ, on OpenAI, vLLM or a local model.

Pipeline
--------
For each participant, for each PHQ-8 item:

1. **Acquire.** Tag-anchored retrieval over the full transcript
   (``configs/scales/topic_evidence_map.json``), then a lexical pass with the
   item's seed queries. Every excerpt keeps the interviewer question that
   elicited it and the polarity of that question.
2. **Judge sufficiency.** The sufficiency role decides whether what was
   retrieved can settle the rubric. If not, the interviewer role proposes new
   spoken-language search phrases and the transcript is searched again, up to
   ``--max-rounds``.
3. **Score or abstain.** The scorer reads the excerpts directly — no simulated
   participant anywhere in this path — and may return ``sufficient: false``
   rather than being forced to a zero.
4. **Resolve.** Abstained items are imputed from the items that were assessed,
   using a model fitted on the training split; the total is then optionally
   calibrated.

Ablations run through the same code path rather than a separate script, so an
ablation table compares one implementation against itself:

``--no-imputation``     abstentions contribute 0 (reproduces the v2 rule)
``--max-rounds 1``      no query expansion
``--no-polarity``       reverse-polarity excerpts are presented as ordinary
                        evidence (reproduces the v2 flat tag map)
``--no-turn-merge``     do not rejoin answers the transcript split across turns
``--lessons``/absent    with and without validated lesson memory

Examples
--------
    export OPENAI_API_KEY=sk-...
    python3 scripts/run_assessment.py --backend openai --model gpt-4o-mini \
        --data-dir data/processed_daic_woz/dev \
        --output-dir results/evaluations/gpt4omini_dev --concurrency 8

    python3 scripts/run_assessment.py --backend vllm \
        --model Qwen/Qwen2.5-14B-Instruct \
        --base-url http://localhost:8000/v1 \
        --data-dir data/processed_daic_woz/dev \
        --output-dir results/evaluations/qwen14b_dev --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.calibration import TotalScoreCalibrator  # noqa: E402
from psyvec.evaluation.evidence_map import (  # noqa: E402
    EvidenceMap,
    load_evidence_map,
    load_topic_queries,
    retrieve_topic_evidence,
)
from psyvec.evaluation.evidence_tiers import (  # noqa: E402
    PairedTurn,
    RetrievedTurn,
    TieredEvidence,
    build_paired_turns,
    format_tiered_evidence,
    merge_retrieved,
    overall_tier,
    search_paired_turns,
)
from psyvec.evaluation.imputation import ImputationModel  # noqa: E402
from psyvec.evaluation.item_scoring import (  # noqa: E402
    INTERVIEWER_SYSTEM_PROMPT,
    ITEM_SCORE_JSON_SCHEMA,
    PROMPT_VERSION,
    QUERY_EXPANSION_JSON_SCHEMA,
    SCORER_SYSTEM_PROMPT,
    SUFFICIENCY_JSON_SCHEMA,
    SUFFICIENCY_SYSTEM_PROMPT,
    build_item_score_prompt,
    build_query_expansion_prompt,
    build_sufficiency_prompt,
    parse_item_score,
    parse_query_expansion,
    parse_sufficiency,
    quote_is_grounded,
)
from psyvec.model.llm_backend import (  # noqa: E402
    ChatBackend,
    GenerationRequest,
    UsageLedger,
    build_backend,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("psyvec")

TOPIC_TO_ITEM_KEY = {
    "Loss of Interest": "PHQ8_NoInterest",
    "Depressed Mood": "PHQ8_Depressed",
    "Sleep Problems": "PHQ8_Sleep",
    "Fatigue or Low Energy": "PHQ8_Tired",
    "Appetite or Weight Changes": "PHQ8_Appetite",
    "Low Self-Worth": "PHQ8_Failure",
    "Concentration Difficulties": "PHQ8_Concentrating",
    "Psychomotor Changes": "PHQ8_Moving",
}
TIER_NAMES = ("direct", "lexical", "expanded", "contextual", "absent")

#: USD per million tokens. gpt-4o-mini list pricing when this was written;
#: always re-check before publishing a cost figure, which is why they are flags.
DEFAULT_INPUT_COST_PER_1M = 0.15
DEFAULT_OUTPUT_COST_PER_1M = 0.60


def categorize_score(total: float) -> str:
    if total <= 4:
        return "None / Minimal depression (0-4)"
    if total <= 9:
        return "Mild depression (5-9)"
    if total <= 14:
        return "Moderate depression (10-14)"
    if total <= 19:
        return "Moderately severe depression (15-19)"
    return "Severe depression (20-24)"


class PipelineConfig:
    """Everything the per-participant worker needs, resolved once per run."""

    def __init__(self, args: argparse.Namespace) -> None:
        scales = PROJECT_ROOT / "configs" / "scales"
        self.topics_dict: dict[str, list[str]] = json.loads(
            (scales / "PHQ-8.json").read_text(encoding="utf-8")
        )
        self.topics = list(self.topics_dict)
        self.standards: dict[str, dict[str, str]] = json.loads(
            (scales / "scoring_standards.json").read_text(encoding="utf-8")
        )["PHQ-8"]
        self.evidence_map: EvidenceMap = load_evidence_map(
            scales / "topic_evidence_map.json", valid_topics=self.topics
        )
        self.seed_queries = load_topic_queries(
            scales / "topic_queries.json", valid_topics=self.topics
        )
        self.imputer: ImputationModel | None = (
            None
            if args.no_imputation or args.imputation_model is None
            else ImputationModel.load(args.imputation_model)
        )
        self.calibrator: TotalScoreCalibrator | None = (
            None
            if args.calibrator is None
            else TotalScoreCalibrator.load(args.calibrator)
        )
        self.lessons: dict[str, tuple[str, ...]] = {}
        if args.lessons is not None:
            self.lessons = _load_lessons(args.lessons)
        self.max_rounds = max(1, args.max_rounds)
        self.evidence_limit = args.evidence_limit
        self.merge_turns = not args.no_turn_merge
        self.polarity = not args.no_polarity
        self.store_evidence = not args.no_store_evidence


def _load_lessons(path: Path) -> dict[str, tuple[str, ...]]:
    """Read promoted lessons as ``{topic: (rendered rule, ...)}``.

    Accepts both the plain ``{topic: [text, ...]}`` shape and the store export
    written by ``scripts/evolve_lessons.py``, which carries the full lesson
    object plus its validation record.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("lessons", raw)
    rendered: dict[str, list[str]] = {}
    for entry in entries if isinstance(entries, list) else []:
        topic = str(entry.get("topic") or entry.get("scope", "").split("|")[0])
        lesson = entry.get("lesson", entry)
        text = (
            f"When {lesson['trigger']}, {lesson['do']} "
            f"Avoid: {lesson['avoid']} You will know it applied when "
            f"{lesson['criterion']}"
        )
        rendered.setdefault(topic, []).append(" ".join(text.split()))
    if isinstance(entries, dict):
        for topic, items in entries.items():
            rendered.setdefault(topic, []).extend(str(item) for item in items)
    return {topic: tuple(items) for topic, items in rendered.items()}


def _flatten_polarity(evidence: TieredEvidence) -> TieredEvidence:
    """Present every excerpt as an ordinary probe, for the --no-polarity arm."""
    return TieredEvidence(
        topic=evidence.topic,
        tier=evidence.tier,
        turns=tuple(
            RetrievedTurn(
                turn=item.turn,
                score=item.score,
                tier=item.tier,
                relation="probe",
                matched_terms=item.matched_terms,
            )
            for item in evidence.turns
        ),
        queries_issued=evidence.queries_issued,
    )


def assess_topic(
    backend: ChatBackend,
    topic: str,
    config: PipelineConfig,
    seed: TieredEvidence,
    paired_turns: Sequence[PairedTurn],
) -> dict[str, Any]:
    """Acquire evidence for one item, then score it or abstain."""
    standard = config.standards.get(topic, {})
    turns = seed.turns
    queries_issued: list[str] = []
    verdicts: list[bool | None] = []
    rounds = 0

    while True:
        rounds += 1
        evidence = TieredEvidence(
            topic=topic,
            tier=overall_tier(turns),
            turns=turns,
            queries_issued=tuple(queries_issued),
        )
        if not config.polarity:
            evidence = _flatten_polarity(evidence)
        evidence_text = format_tiered_evidence(evidence)

        if rounds >= config.max_rounds:
            break
        verdict, missing = parse_sufficiency(
            backend.generate(
                GenerationRequest(
                    system=SUFFICIENCY_SYSTEM_PROMPT,
                    user=build_sufficiency_prompt(topic, standard, evidence_text),
                    max_tokens=200,
                    json_schema=SUFFICIENCY_JSON_SCHEMA,
                )
            )
        )
        verdicts.append(verdict)
        # An unparseable verdict means "keep looking" while rounds remain: one
        # extra retrieval call is cheap, and stopping early is what produced
        # v2's forced zeros.
        if verdict is True:
            break
        new_queries = parse_query_expansion(
            backend.generate(
                GenerationRequest(
                    system=INTERVIEWER_SYSTEM_PROMPT,
                    user=build_query_expansion_prompt(
                        topic, standard, missing, tuple(queries_issued)
                    ),
                    max_tokens=300,
                    temperature=0.4,
                    json_schema=QUERY_EXPANSION_JSON_SCHEMA,
                )
            )
        )
        if not new_queries:
            break
        queries_issued.extend(new_queries)
        found = search_paired_turns(
            paired_turns,
            new_queries,
            top_k=config.evidence_limit,
            exclude_turn_ids=frozenset(item.turn.turn_id for item in turns),
            tier="expanded",
        )
        if not found:
            break
        turns = merge_retrieved(turns, found, limit=config.evidence_limit)

    parsed = parse_item_score(
        backend.generate(
            GenerationRequest(
                system=SCORER_SYSTEM_PROMPT,
                user=build_item_score_prompt(
                    topic, standard, evidence_text,
                    lessons=config.lessons.get(topic, ()),
                ),
                max_tokens=600,
                json_schema=ITEM_SCORE_JSON_SCHEMA,
            )
        )
    )

    allowed_ids: set[str] = set()
    for item in turns:
        allowed_ids.update(item.turn.citable_ids)
    bad_citations = tuple(
        turn_id for turn_id in parsed.turn_ids if turn_id not in allowed_ids
    )
    record: dict[str, Any] = {
        "score": parsed.score,
        "abstained": not parsed.sufficient,
        "present": parsed.present,
        "frequency_basis": parsed.frequency_basis,
        "summary": parsed.reasoning,
        "quote": parsed.quote,
        "quote_grounded": quote_is_grounded(parsed.quote, evidence_text),
        "parse_failed": not parsed.ok,
        "rounds": rounds,
        "item_key": TOPIC_TO_ITEM_KEY.get(topic, ""),
        "evidence_tier": overall_tier(turns),
        "evidence_turn_ids": list(parsed.turn_ids),
        "retrieved_turn_ids": [item.turn.turn_id for item in turns],
        "citation_mismatch": bool(bad_citations),
        "queries_issued": queries_issued,
        "sufficiency_verdicts": verdicts,
    }
    if config.store_evidence:
        # Kept so scripts/evolve_lessons.py can replay this exact scoring
        # decision, control versus intervention, without re-retrieving.
        record["evidence_text"] = evidence_text
    return record



def assess_participant(
    backend: ChatBackend, sample_path: Path, config: PipelineConfig
) -> dict[str, Any]:
    started = time.time()
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    participant_id = str(sample.get("Participant_ID", sample_path.stem))
    reference = sample.get("phq8_scores", {})

    paired_turns = build_paired_turns(
        sample.get("real_interview", []), merge_consecutive=config.merge_turns
    )

    assessed: dict[str, dict[str, Any]] = {}
    for topic in config.topics:
        seed = retrieve_topic_evidence(
            paired_turns,
            topic,
            config.evidence_map,
            config.seed_queries[topic],
            limit=config.evidence_limit,
        )
        assessed[topic] = assess_topic(backend, topic, config, seed, paired_turns)

    scored_items = {
        info["item_key"]: int(info["score"])
        for info in assessed.values()
        if info["score"] is not None
    }
    abstained = [
        info["item_key"] for info in assessed.values() if info["score"] is None
    ]

    imputed: dict[str, float] = {}
    if abstained and config.imputer is not None:
        imputed = config.imputer.impute(scored_items, abstained)
        for info in assessed.values():
            if info["item_key"] in imputed:
                info["imputed_score"] = round(imputed[info["item_key"]], 3)

    raw_total = sum(scored_items.values()) + sum(imputed.values())
    calibrated = (
        config.calibrator.apply(raw_total)
        if config.calibrator is not None
        else raw_total
    )
    final_total = round(calibrated)

    diagnostics: dict[str, Any] = {
        tier: sum(1 for i in assessed.values() if i["evidence_tier"] == tier)
        for tier in TIER_NAMES
    }
    diagnostics.update(
        abstentions=len(abstained),
        citation_mismatches=sum(1 for i in assessed.values() if i["citation_mismatch"]),
        ungrounded_quotes=sum(
            1
            for i in assessed.values()
            if i["score"] is not None and not i["quote_grounded"]
        ),
        parse_failures=sum(1 for i in assessed.values() if i["parse_failed"]),
        expansion_rounds=sum(i["rounds"] - 1 for i in assessed.values()),
    )

    return {
        "participant_id": participant_id,
        "sample_path": str(sample_path),
        "model_name": backend.model,
        "pipeline": "psyvec-v3-evidence-direct",
        "prompt_version": PROMPT_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "assessed_item_count": len(scored_items),
        "abstained_item_count": len(abstained),
        "abstained_items": abstained,
        "imputation_applied": bool(imputed),
        "lessons_applied": sum(len(v) for v in config.lessons.values()),
        "raw_total_score": round(raw_total, 3),
        "calibrated_total_score": round(calibrated, 3),
        "total_predicted_score": final_total,
        "ground_truth_total": reference.get("PHQ8_Score"),
        "ground_truth_items": reference.get("items", {}),
        "predicted_category": categorize_score(final_total),
        "assessment_valid": True,
        "evidence_diagnostics": diagnostics,
        "topics": assessed,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("openai", "vllm", "local"), default="openai"
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None, help="defaults to $OPENAI_API_KEY")
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible endpoint; defaults to localhost:8000/v1 for vllm",
    )
    parser.add_argument("--device", default=None, help="local backend only")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--evidence-limit", type=int, default=8)
    parser.add_argument("--imputation-model", type=Path, default=None)
    parser.add_argument("--calibrator", type=Path, default=None)
    parser.add_argument("--lessons", type=Path, default=None)
    parser.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=None
    )
    parser.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    ablations = parser.add_argument_group("ablations")
    ablations.add_argument("--no-imputation", action="store_true")
    ablations.add_argument("--no-polarity", action="store_true")
    ablations.add_argument("--no-turn-merge", action="store_true")
    ablations.add_argument("--no-store-evidence", action="store_true")
    parser.add_argument(
        "--input-cost-per-1m", type=float, default=DEFAULT_INPUT_COST_PER_1M
    )
    parser.add_argument(
        "--output-cost-per-1m", type=float, default=DEFAULT_OUTPUT_COST_PER_1M
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PipelineConfig(args)

    if config.imputer is None:
        logger.warning(
            "no imputation model: abstained items contribute 0, which "
            "reproduces the v2 under-scoring this pipeline exists to fix"
        )

    samples = sorted(
        (path for path in args.data_dir.glob("*.json") if path.is_file()),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if args.num_samples:
        samples = samples[: args.num_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        samples = [
            path
            for path in samples
            if not (args.output_dir / f"{path.stem}_evaluation.json").is_file()
        ]
    if not samples:
        logger.info("nothing to do")
        return

    ledger = UsageLedger(
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )
    backend = build_backend(
        args.backend,
        args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        ledger=ledger,
        device=args.device,
        enable_thinking=args.enable_thinking,
    )
    logger.info(
        "assessing %d participants | backend=%s model=%s concurrency=%d "
        "max_rounds=%d imputation=%s lessons=%d",
        len(samples), args.backend, args.model, args.concurrency,
        config.max_rounds, config.imputer is not None,
        sum(len(v) for v in config.lessons.values()),
    )

    started = time.time()
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(assess_participant, backend, path, config): path
            for path in samples
        }
        for done, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - one bad sample must not stop the run
                logger.error("participant %s failed: %s", path.stem, error)
                failures.append(path.stem)
                continue
            (args.output_dir / f"{path.stem}_evaluation.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            records.append(record)
            logger.info(
                "[%d/%d] %s pred=%s ref=%s abstained=%d cost=$%.3f",
                done, len(samples), record["participant_id"],
                record["total_predicted_score"], record["ground_truth_total"],
                record["abstained_item_count"], ledger.cost,
            )

    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "pipeline": "psyvec-v3-evidence-direct",
        "prompt_version": PROMPT_VERSION,
        "backend": args.backend,
        "model_name": args.model,
        "data_dir": str(args.data_dir),
        "config": {
            "max_rounds": config.max_rounds,
            "evidence_limit": config.evidence_limit,
            "imputation_model": str(args.imputation_model or ""),
            "calibrator": str(args.calibrator or ""),
            "lessons": str(args.lessons or ""),
            "polarity": config.polarity,
            "turn_merge": config.merge_turns,
        },
        "scored_samples": len(records),
        "failed_samples": failures,
        "overall_elapsed_seconds": round(time.time() - started, 2),
        "usage": ledger.summary(),
        "records": [
            {
                key: record[key]
                for key in (
                    "participant_id", "total_predicted_score", "raw_total_score",
                    "ground_truth_total", "abstained_item_count", "elapsed_seconds",
                )
            }
            for record in records
        ],
    }
    summary_path = args.output_dir / f"batch_summary_{summary['timestamp']}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    logger.info("usage: %s", json.dumps(ledger.summary()))
    logger.info(
        "score it: python3 scripts/score_run.py %s --split %s",
        args.output_dir, args.data_dir.name,
    )


if __name__ == "__main__":
    main()
