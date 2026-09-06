#!/usr/bin/env python3
"""Simple single-call PHQ-8 baselines: Direct prompt, Chain-of-Thought, Few-shot.

These exist only as a comparison point against the Grounded Interview Loop
(Plan_Improve.md) — they are deliberately naive (one LLM call, no retrieval,
no multi-round interview) so the contrast is meaningful, not to be strong
methods in their own right:

- ``direct``:  read the transcript once, output PHQ-8 scores immediately.
- ``cot``:     same, but told to reason step by step before the final JSON.
- ``fewshot``: same as ``direct``, with 5 solved examples drawn from the
  ``train`` split (never dev/test) prepended to the prompt.

Every method uses the same PHQ-8 rubric
(``configs/scales/scoring_standards.json``) and the same output schema, and
reuses the already-tested
``psyvec.evaluation.response_parsing.parse_summary_and_updated_scores``
parser (the JSON key is named ``updated_scores`` only because that parser
expects that name — these are the model's first and only scores, not a
revision).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_batch_eval import compute_metrics  # noqa: E402
from run_qwen_sample import (  # noqa: E402
    DEFAULT_SCALE_FILE,
    DEFAULT_STANDARDS_FILE,
    TOPIC_TO_ITEM_KEY,
    QwenInferenceEngine,
    categorize_score,
)

from psyvec.evaluation.response_parsing import (  # noqa: E402
    parse_summary_and_updated_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("baseline-evaluator")

METHODS = ("direct", "cot", "fewshot")

SYSTEM_PROMPTS = {
    "direct": (
        "You are a psychological assessment assistant. Read the interview "
        "transcript and output the PHQ-8 scores directly. Do not explain "
        "your reasoning, output JSON only."
    ),
    "cot": (
        "You are a psychological assessment assistant. Think step by step "
        "about the evidence for each PHQ-8 symptom in the transcript, then "
        "give your final scores. Output your reasoning, then the final "
        "JSON on its own line at the end."
    ),
    "fewshot": (
        "You are a psychological assessment assistant. Following the solved "
        "examples below, read the interview transcript and output the "
        "PHQ-8 scores directly. Do not explain your reasoning, output JSON "
        "only."
    ),
}


def build_participant_text(
    real_interview: list[dict[str, str]], max_chars: int
) -> str:
    """Flatten the participant's own turns into plain text, hard-capped.

    Deliberately simple (and deliberately naive): a real baseline for this
    task would not do topic-aware retrieval, it would just hand the model
    the transcript up to some length limit.
    """
    text = " ".join(
        turn.get("content", "")
        for turn in real_interview
        if turn.get("roleName") == "Participant"
    )
    return text[:max_chars]


def build_rubric_text(topics: list[str], scoring_standards: dict[str, Any]) -> str:
    lines = []
    for topic in topics:
        standard = scoring_standards.get(topic, {})
        lines.append(
            f"- {topic}: 0={standard.get('0', '')} | 1={standard.get('1', '')} | "
            f"2={standard.get('2', '')} | 3={standard.get('3', '')}"
        )
    return "\n".join(lines)


def build_answer_schema(topics: list[str]) -> str:
    fields = ",\n".join(
        f'    "{topic}": {{"score": <0-3>, "reason": "<brief>"}}' for topic in topics
    )
    return (
        '{\n  "summary": "<one sentence overall impression>",\n'
        '  "updated_scores": {\n' + fields + "\n  }\n}"
    )


def select_fewshot_examples(train_dir: Path, k: int) -> list[Path]:
    """Pick ``k`` train participants spread across the PHQ-8 score range.

    Deterministic (sorted by score, evenly spaced indices) so re-running
    produces the same 5-shot prompt every time. Train only — never dev/test.
    """
    scored: list[tuple[int, Path]] = []
    for path in sorted(train_dir.glob("*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        scored.append((sample["phq8_scores"]["PHQ8_Score"], path))
    scored.sort(key=lambda pair: pair[0])
    if len(scored) <= k:
        return [path for _, path in scored]
    indices = sorted({round(i * (len(scored) - 1) / (k - 1)) for i in range(k)})
    return [scored[i][1] for i in indices]


def build_fewshot_block(
    example_paths: list[Path], topics: list[str], max_chars: int
) -> str:
    blocks = []
    for path in example_paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        text = build_participant_text(sample["real_interview"], max_chars)
        items = sample["phq8_scores"].get("items", {})
        answer = {
            "summary": "example",
            "updated_scores": {
                topic: {
                    "score": items.get(TOPIC_TO_ITEM_KEY[topic], 0),
                    "reason": "ground truth",
                }
                for topic in topics
            },
        }
        blocks.append(f"Transcript:\n{text}\n\nOutput:\n{json.dumps(answer)}")
    return "\n\n---\n\n".join(blocks) + "\n\n---\n\n"


def build_user_prompt(
    participant_text: str,
    topics: list[str],
    scoring_standards: dict[str, Any],
    fewshot_block: str,
) -> str:
    rubric_text = build_rubric_text(topics, scoring_standards)
    schema_text = build_answer_schema(topics)
    return f"""PHQ-8 scoring rubric (0-3 each symptom, over the past two weeks):
{rubric_text}

{fewshot_block}Transcript (participant's own words):
{participant_text}

Score all 8 topics from 0 to 3 based on the rubric above. Output strictly JSON:
{schema_text}"""


def run_baseline_sample(
    sample_path: Path,
    method: str,
    engine: QwenInferenceEngine,
    topics: list[str],
    scoring_standards: dict[str, Any],
    fewshot_block: str,
    max_chars: int,
) -> dict[str, Any]:
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    participant_id = sample.get("Participant_ID", "Unknown")
    ground_truth = sample.get("phq8_scores", {})
    participant_text = build_participant_text(
        sample.get("real_interview", []), max_chars
    )

    user_prompt = build_user_prompt(
        participant_text,
        topics,
        scoring_standards,
        fewshot_block if method == "fewshot" else "",
    )
    max_new_tokens = 4096 if method == "cot" else 1280
    raw = engine.generate(
        system_prompt=SYSTEM_PROMPTS[method],
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    parsed = parse_summary_and_updated_scores(raw, topics)

    item_scores: dict[str, int | None] = {}
    for topic in topics:
        info = parsed.updated_scores.get(topic)
        item_scores[topic] = info["score"] if info is not None else None

    all_scored = parsed.ok and all(
        score is not None for score in item_scores.values()
    )
    total = (
        sum(score for score in item_scores.values() if score is not None)
        if all_scored
        else None
    )
    category = (
        categorize_score(total) if all_scored and total is not None else "UNSCORED"
    )

    return {
        "participant_id": participant_id,
        "method": method,
        "model_name": engine.model_name,
        "sample_path": str(sample_path),
        "assessment_valid": all_scored,
        "parse_failure_reason": "" if parsed.ok else parsed.failure_reason,
        "item_scores": item_scores,
        "item_scores_by_key": {
            TOPIC_TO_ITEM_KEY[topic]: item_scores[topic] for topic in topics
        },
        "total_predicted_score": total,
        "predicted_category": category,
        "predicted_binary": (
            (1 if total is not None and total >= 10 else 0) if all_scored else None
        ),
        "ground_truth_total": ground_truth.get("PHQ8_Score", "N/A"),
        "ground_truth_binary": ground_truth.get("PHQ8_Binary", "N/A"),
        "ground_truth_items": ground_truth.get("items", {}),
        "raw_response": raw,
    }


def is_valid_record(record: dict[str, Any]) -> bool:
    pred = record.get("total_predicted_score")
    gt = record.get("ground_truth_total")
    return isinstance(pred, (int, float)) and isinstance(gt, (int, float))


def compute_per_item_metrics(
    records: list[dict[str, Any]], topics: list[str]
) -> dict[str, dict[str, float]] | None:
    """Per-topic MAE/accuracy — only meaningful when item-level ground truth
    exists (train/dev). The AVEC2017 test split withholds it, so this
    returns None there rather than a misleading empty/zero metric."""
    if not any(record.get("ground_truth_items") for record in records):
        return None

    per_item: dict[str, dict[str, float]] = {}
    for topic in topics:
        item_key = TOPIC_TO_ITEM_KEY[topic]
        pairs = [
            (record["item_scores"][topic], record["ground_truth_items"][item_key])
            for record in records
            if record["item_scores"].get(topic) is not None
            and item_key in record.get("ground_truth_items", {})
        ]
        if not pairs:
            continue
        mae = sum(abs(p - g) for p, g in pairs) / len(pairs)
        accuracy = sum(1 for p, g in pairs if p == g) / len(pairs)
        per_item[topic] = {
            "mae": round(mae, 4),
            "accuracy": round(accuracy, 4),
            "n": float(len(pairs)),
        }
    return per_item


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a simple single-call PHQ-8 baseline."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "dev",
    )
    parser.add_argument(
        "--fewshot-train-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "train",
        help=(
            "Source split for the 5-shot examples (method=fewshot only). "
            "Never dev/test."
        ),
    )
    parser.add_argument("--fewshot-k", type=int, default=5)
    parser.add_argument("--max-participant-chars", type=int, default=3000)
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--api-base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--token-scale", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluations" / "baselines",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    topics = list(json.loads(DEFAULT_SCALE_FILE.read_text(encoding="utf-8")).keys())
    standards_json = json.loads(DEFAULT_STANDARDS_FILE.read_text(encoding="utf-8"))
    scoring_standards = standards_json["PHQ-8"]

    fewshot_block = ""
    if args.method == "fewshot":
        if not args.fewshot_train_dir.is_dir():
            logger.error(
                "Few-shot train dir does not exist: %s", args.fewshot_train_dir
            )
            sys.exit(1)
        example_paths = select_fewshot_examples(args.fewshot_train_dir, args.fewshot_k)
        logger.info(
            "Few-shot examples (%d) drawn from train: %s",
            len(example_paths),
            [p.stem for p in example_paths],
        )
        fewshot_block = build_fewshot_block(
            example_paths, topics, args.max_participant_chars
        )

    json_files = sorted(
        (f for f in args.data_dir.glob("*.json") if f.is_file()),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if args.num_samples:
        json_files = json_files[: args.num_samples]
    if not json_files:
        logger.error("No JSON sample files found in %s", args.data_dir)
        sys.exit(1)

    method_output_dir = args.output_dir / args.method
    method_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Method: %s | Model: %s | Samples: %d",
        args.method,
        args.model_name,
        len(json_files),
    )
    engine = QwenInferenceEngine(
        model_name=args.model_name,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        token_scale=args.token_scale,
        enable_thinking=args.enable_thinking,
    )

    records: list[dict[str, Any]] = []
    start = time.time()
    for idx, sample_path in enumerate(json_files, 1):
        participant_id = sample_path.stem
        out_file = method_output_dir / f"{participant_id}_evaluation.json"
        if args.skip_existing and out_file.is_file():
            logger.info(
                "[%d/%d] Skipping existing %s", idx, len(json_files), participant_id
            )
            records.append(json.loads(out_file.read_text(encoding="utf-8")))
            continue

        logger.info(
            "[%d/%d] [%s] Participant %s",
            idx,
            len(json_files),
            args.method,
            participant_id,
        )
        try:
            result = run_baseline_sample(
                sample_path,
                args.method,
                engine,
                topics,
                scoring_standards,
                fewshot_block,
                args.max_participant_chars,
            )
        except Exception:
            logger.exception("Failed on %s", sample_path.name)
            continue
        out_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(result)
        pred, gt = result["total_predicted_score"], result["ground_truth_total"]
        logger.info("  -> Pred=%s GT=%s Valid=%s", pred, gt, result["assessment_valid"])

    elapsed = time.time() - start
    valid_records = [r for r in records if is_valid_record(r) and r["assessment_valid"]]
    metrics = compute_metrics(valid_records, cutoff=10)
    per_item_metrics = compute_per_item_metrics(valid_records, topics)

    print("\n" + "=" * 70)
    print(
        f"BASELINE '{args.method}' on {args.data_dir} — "
        f"{len(valid_records)}/{len(records)} scored"
    )
    print(
        f"  MAE={metrics['mae']}  RMSE={metrics['rmse']}  "
        f"Pearson r={metrics['pearson_r']}"
    )
    print(
        f"  Accuracy={metrics['accuracy'] * 100:.2f}%  "
        f"Precision={metrics['precision'] * 100:.2f}%  "
        f"Recall={metrics['recall'] * 100:.2f}%  F1={metrics['f1']:.4f}"
    )
    print("=" * 70)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_name = f"batch_summary_{args.method}_{args.data_dir.name}_{timestamp}.json"
    summary_path = args.output_dir / summary_name
    summary_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "method": args.method,
                "model_name": args.model_name,
                "data_dir": str(args.data_dir),
                "total_samples": len(json_files),
                "scored_samples": len(valid_records),
                "elapsed_seconds": round(elapsed, 2),
                "metrics": metrics,
                "per_item_metrics": per_item_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Summary saved to: %s", summary_path.resolve())


if __name__ == "__main__":
    main()
