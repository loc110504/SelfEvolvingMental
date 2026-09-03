#!/usr/bin/env python3
"""Run batch psychological evaluation across multiple DAIC-WOZ samples."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_qwen_sample import (  # noqa: E402
    DEFAULT_SCALE_FILE,
    DEFAULT_STANDARDS_FILE,
    QwenInferenceEngine,
    run_full_sample_assessment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("batch-evaluator")


def compute_pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient with pure Python fallback."""
    if len(x) < 2:
        return 0.0
    try:
        from scipy.stats import pearsonr

        val, _ = pearsonr(x, y)
        return float(val) if not math.isnan(val) else 0.0
    except ImportError:
        pass

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denominator = math.sqrt(var_x * var_y)
    if denominator == 0:
        return 0.0
    return float(cov / denominator)


def is_valid_record(record: dict[str, Any]) -> bool:
    """A record counts only if every topic parsed into a real score."""
    if not record.get("assessment_valid", True):
        return False
    pred = record.get("total_predicted_score")
    gt = record.get("ground_truth_total")
    return isinstance(pred, (int, float)) and isinstance(gt, (int, float))


def compute_metrics(
    records: list[dict[str, Any]],
    cutoff: int = 10,
) -> dict[str, float]:
    """Calculate MAE, RMSE, Pearson r, Accuracy, Precision, Recall, F1."""
    if not records:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "pearson_r": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    preds: list[float] = []
    gts: list[float] = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for r in records:
        pred = float(r["total_predicted_score"])
        gt = float(r["ground_truth_total"])
        preds.append(pred)
        gts.append(gt)

        pred_binary = 1 if pred >= cutoff else 0
        gt_binary = 1 if gt >= cutoff else 0

        if pred_binary == 1 and gt_binary == 1:
            tp += 1
        elif pred_binary == 1 and gt_binary == 0:
            fp += 1
        elif pred_binary == 0 and gt_binary == 0:
            tn += 1
        else:
            fn += 1

    n = len(records)
    mae = sum(abs(p - g) for p, g in zip(preds, gts, strict=False)) / n
    rmse = math.sqrt(sum((p - g) ** 2 for p, g in zip(preds, gts, strict=False)) / n)
    pearson = compute_pearson_r(preds, gts)

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(pearson, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch psychological evaluation across DAIC-WOZ samples."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "dev",
        help="Directory containing processed sample JSON files (e.g. dev or train)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible API base URL (e.g. http://localhost:8000/v1 for vLLM, http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--token-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply every max_new_tokens budget by this factor (default: 1.0; "
            "auto-set to 2.0 for detected reasoning models like Qwen3.x)."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Force chain-of-thought ON in the chat template (Qwen3-style models).",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Force chain-of-thought OFF in the chat template (Qwen3-style models).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluations" / "batch",
        help="Output directory to save per-sample JSONs and summary reports",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--enable-memory-update",
        action="store_true",
        help="Enable global Memory Update (SummaryAgent/Updater) adjusting initial topic scores based on full dialogue memory without LoRA",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples if their output JSON already exists in output-dir",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed dialogue steps for each sample",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    json_files = sorted(
        [f for f in data_dir.glob("*.json") if f.is_file()],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )

    if not json_files:
        logger.error("No JSON sample files found in %s", data_dir)
        sys.exit(1)

    if args.num_samples is not None and args.num_samples > 0:
        json_files = json_files[: args.num_samples]

    total_count = len(json_files)
    logger.info("Found %d samples to evaluate from: %s", total_count, data_dir)
    logger.info("Target Model: %s", args.model_name)
    logger.info("Output Directory: %s", args.output_dir)
    logger.info(
        "Memory Update: %s",
        "ENABLED (Memory-only ablation)" if args.enable_memory_update else "DISABLED (Zero-shot Baseline)",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Preload engine once to avoid reloading model per sample
    logger.info("Initializing Qwen Inference Engine once for batch run...")
    engine = QwenInferenceEngine(
        model_name=args.model_name,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        token_scale=args.token_scale,
        enable_thinking=args.enable_thinking,
    )

    records: list[dict[str, Any]] = []
    overall_start = time.time()

    for idx, sample_path in enumerate(json_files, 1):
        participant_id = sample_path.stem
        out_file = args.output_dir / f"{participant_id}_evaluation.json"

        if args.skip_existing and out_file.is_file():
            logger.info(
                "[%d/%d] Skipping existing Participant %s",
                idx,
                total_count,
                participant_id,
            )
            existing_result = json.loads(out_file.read_text(encoding="utf-8"))
            records.append(existing_result)
            continue

        logger.info(
            "[%d/%d] Evaluating Participant %s (%s)...",
            idx,
            total_count,
            participant_id,
            sample_path.name,
        )
        sample_start = time.time()
        try:
            result = run_full_sample_assessment(
                sample_path=sample_path,
                model_name=args.model_name,
                scale_file=DEFAULT_SCALE_FILE,
                standards_file=DEFAULT_STANDARDS_FILE,
                engine=engine,
                enable_memory_update=args.enable_memory_update,
                token_scale=args.token_scale,
                enable_thinking=args.enable_thinking,
                verbose=args.verbose,
            )
            sample_time = time.time() - sample_start
            result["elapsed_seconds"] = round(sample_time, 2)

            out_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            records.append(result)

            pred = result["total_predicted_score"]
            gt = result["ground_truth_total"]
            if not is_valid_record(result):
                logger.error(
                    "  ✗ Participant %s UNSCORED after %.2fs: %d parse failure(s). "
                    "First: %s",
                    participant_id,
                    sample_time,
                    result.get("parse_failure_count", 0),
                    (result.get("parse_failures") or [{"reason": "unknown"}])[0]["reason"],
                )
                continue
            diff = (
                abs(pred - gt)
                if isinstance(gt, (int, float)) and isinstance(pred, (int, float))
                else "N/A"
            )
            if args.enable_memory_update:
                init_s = result.get("initial_total_score", pred)
                logger.info(
                    "  ✓ Done in %.2fs | Init: %s/24 -> Updated: %s/24 | GT: %s/24 | |Diff|: %s",
                    sample_time,
                    init_s,
                    pred,
                    gt,
                    diff,
                )
            else:
                logger.info(
                    "  ✓ Done in %.2fs | Pred: %s/24 | GT: %s/24 | |Diff|: %s",
                    sample_time,
                    pred,
                    gt,
                    diff,
                )
        except Exception as e:
            logger.error("  ✗ Failed on %s: %s", sample_path.name, e, exc_info=True)

    overall_elapsed = time.time() - overall_start
    completed_count = len(records)
    logger.info("=" * 70)
    logger.info("Batch Evaluation Completed!")
    logger.info(
        "Processed: %d/%d samples in %.2f seconds (avg %.2fs/sample)",
        completed_count,
        total_count,
        overall_elapsed,
        (overall_elapsed / completed_count) if completed_count else 0.0,
    )
    logger.info("=" * 70)

    if not records:
        logger.warning("No completed records to summarize.")
        return

    valid_records = [r for r in records if is_valid_record(r)]
    invalid_records = [r for r in records if not is_valid_record(r)]

    if invalid_records:
        logger.error("=" * 70)
        logger.error(
            "%d/%d sample(s) produced NO usable score (parse failures) and are "
            "EXCLUDED from all metrics below:",
            len(invalid_records),
            len(records),
        )
        for r in invalid_records:
            failures = r.get("parse_failures") or []
            stages = ", ".join(sorted({str(f.get("stage", "?")) for f in failures}))
            logger.error(
                "  - Participant %s: %d failure(s) [%s]",
                r.get("participant_id", "N/A"),
                r.get("parse_failure_count", len(failures)),
                stages or "unknown",
            )
        logger.error("=" * 70)

    if not valid_records:
        logger.error(
            "ABORT: every sample failed to parse. No metrics computed. "
            "Check the model's output format (reasoning models need --token-scale "
            "and/or --disable-thinking)."
        )
        sys.exit(3)

    # Compute metrics over usable records only
    records = valid_records
    metrics = compute_metrics(valid_records, cutoff=10)

    # Print summary table
    if args.enable_memory_update:
        print("\n" + "=" * 95)
        print(
            f"{'Participant':<12} | {'Init PHQ-8':<10} | {'Final PHQ-8':<11} | {'GT PHQ-8':<10} | {'Abs Diff':<10} | {'Pred Class':<10} | {'GT Class':<10}"
        )
        print("-" * 95)
        for r in records:
            pid = str(r.get("participant_id", "N/A"))
            init_s = int(r.get("initial_total_score", r.get("total_predicted_score", 0)))
            pred = int(r.get("total_predicted_score", 0))
            gt = r.get("ground_truth_total", 0)
            gt_val = int(gt) if isinstance(gt, (int, float)) else 0
            diff = abs(pred - gt_val)
            pred_class = "Depressed" if pred >= 10 else "Non-Dep"
            gt_class = "Depressed" if gt_val >= 10 else "Non-Dep"
            print(
                f"{pid:<12} | {init_s:<10} | {pred:<11} | {gt_val:<10} | {diff:<10} | {pred_class:<10} | {gt_class:<10}"
            )
        print("-" * 95)
    else:
        print("\n" + "=" * 80)
        print(
            f"{'Participant':<12} | {'Pred PHQ-8':<12} | {'GT PHQ-8':<12} | {'Abs Diff':<10} | {'Pred Class':<10} | {'GT Class':<10}"
        )
        print("-" * 80)
        for r in records:
            pid = str(r.get("participant_id", "N/A"))
            pred = int(r.get("total_predicted_score", 0))
            gt = r.get("ground_truth_total", 0)
            gt_val = int(gt) if isinstance(gt, (int, float)) else 0
            diff = abs(pred - gt_val)
            pred_class = "Depressed" if pred >= 10 else "Non-Dep"
            gt_class = "Depressed" if gt_val >= 10 else "Non-Dep"
            print(
                f"{pid:<12} | {pred:<12} | {gt_val:<12} | {diff:<10} | {pred_class:<10} | {gt_class:<10}"
            )
        print("-" * 80)

    print("OVERALL METRICS SUMMARY:")
    print(
        f"  • Samples scored: {len(valid_records)}/{completed_count} "
        f"({len(invalid_records)} excluded for parse failure)"
    )
    print(f"  • MAE (Mean Absolute Error): {metrics['mae']}")
    print(f"  • RMSE (Root Mean Squared):  {metrics['rmse']}")
    print(f"  • Pearson Correlation (r):  {metrics['pearson_r']}")
    print("  • Binary Classification (Cutoff >= 10):")
    print(f"      Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"      Precision: {metrics['precision'] * 100:.2f}%")
    print(f"      Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"      F1-Score:  {metrics['f1']:.4f}")
    print(
        f"      Confusion: TP={int(metrics['tp'])}, FP={int(metrics['fp'])}, TN={int(metrics['tn'])}, FN={int(metrics['fn'])}"
    )
    print("=" * 80)

    # Save summary JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_json_path = args.output_dir / f"batch_summary_{timestamp}.json"
    summary_data = {
        "timestamp": timestamp,
        "model_name": args.model_name,
        "data_dir": str(data_dir),
        "memory_update_enabled": args.enable_memory_update,
        "total_samples": total_count,
        "completed_samples": completed_count,
        "scored_samples": len(valid_records),
        "unscored_samples": len(invalid_records),
        "unscored_participant_ids": [
            r.get("participant_id") for r in invalid_records
        ],
        "overall_elapsed_seconds": round(overall_elapsed, 2),
        "metrics": metrics,
        "records": [
            {
                "participant_id": r.get("participant_id"),
                "initial_total_score": r.get("initial_total_score"),
                "total_predicted_score": r.get("total_predicted_score"),
                "ground_truth_total": r.get("ground_truth_total"),
                "predicted_category": r.get("predicted_category"),
                "elapsed_seconds": r.get("elapsed_seconds"),
            }
            for r in records
        ],
    }
    summary_json_path.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save summary CSV
    summary_csv_path = args.output_dir / f"batch_summary_{timestamp}.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        headers = [
            "Participant_ID",
            "Init_Score" if args.enable_memory_update else "Pred_Score",
            "Final_Score" if args.enable_memory_update else "GT_Score",
        ]
        if args.enable_memory_update:
            headers.extend(["GT_Score", "Abs_Diff", "Pred_Binary", "GT_Binary"])
        else:
            headers.extend(["Abs_Diff", "Pred_Binary", "GT_Binary"])
        writer.writerow(headers)

        for r in records:
            pid = r.get("participant_id")
            pred = int(r.get("total_predicted_score", 0))
            gt = r.get("ground_truth_total", 0)
            gt_val = int(gt) if isinstance(gt, (int, float)) else 0
            diff = abs(pred - gt_val)
            if args.enable_memory_update:
                init_s = int(r.get("initial_total_score", pred))
                writer.writerow(
                    [
                        pid,
                        init_s,
                        pred,
                        gt_val,
                        diff,
                        1 if pred >= 10 else 0,
                        1 if gt_val >= 10 else 0,
                    ]
                )
            else:
                writer.writerow(
                    [
                        pid,
                        pred,
                        gt_val,
                        diff,
                        1 if pred >= 10 else 0,
                        1 if gt_val >= 10 else 0,
                    ]
                )

    logger.info("📁 Batch summary JSON saved to: %s", summary_json_path.resolve())
    logger.info("📊 Batch summary CSV saved to:  %s", summary_csv_path.resolve())


if __name__ == "__main__":
    main()
