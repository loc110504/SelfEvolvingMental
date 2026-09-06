#!/usr/bin/env python3
"""Print/save a comparison table across baseline (and proposed-method) runs.

Reads every ``batch_summary_*.json`` produced by ``run_baseline_eval.py``
(direct/cot/fewshot) under ``--baseline-dir``, plus any extra summary JSON
produced by ``run_batch_eval.py`` for the proposed Grounded Interview Loop
method (pass its path(s) via ``--extra-summary``), and prints one table with
MAE/RMSE/Pearson r/Accuracy/F1 side by side, split by data split (dev/test).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": data.get("method", "proposed"),
        "split": Path(str(data.get("data_dir", ""))).name,
        "model_name": data.get("model_name", ""),
        "scored_samples": data.get("scored_samples", 0),
        "metrics": data["metrics"],
        "per_item_metrics": data.get("per_item_metrics"),
        "source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluations" / "baselines",
        help="Directory to glob for baseline batch_summary_*.json files.",
    )
    parser.add_argument(
        "--extra-summary",
        type=Path,
        nargs="*",
        default=[],
        help="Additional batch_summary_*.json path(s), e.g. the proposed "
        "method's run_batch_eval.py output, to include in the comparison.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to also save the comparison table as CSV.",
    )
    args = parser.parse_args()

    summary_paths = (
        sorted(args.baseline_dir.glob("batch_summary_*.json"))
        if args.baseline_dir.is_dir()
        else []
    )
    summary_paths += list(args.extra_summary)
    if not summary_paths:
        print(
            f"No summary files found under {args.baseline_dir} "
            "and none given via --extra-summary."
        )
        return

    rows = [load_summary(p) for p in summary_paths]
    rows.sort(key=lambda r: (r["split"], r["label"]))

    header = (
        f"{'Split':<8} | {'Method':<10} | {'N':<5} | {'MAE':<7} | {'RMSE':<7} | "
        f"{'Pearson r':<9} | {'Accuracy':<9} | {'F1':<7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        m = row["metrics"]
        print(
            f"{row['split']:<8} | {row['label']:<10} | {row['scored_samples']:<5} | "
            f"{m['mae']:<7} | {m['rmse']:<7} | {m['pearson_r']:<9} | "
            f"{m['accuracy'] * 100:>7.2f}% | {m['f1']:<7}"
        )
    print("=" * len(header))

    for row in rows:
        if row["per_item_metrics"]:
            print(f"\nPer-item MAE/accuracy — {row['split']} / {row['label']}:")
            for topic, item_metrics in row["per_item_metrics"].items():
                acc_pct = item_metrics["accuracy"] * 100
                print(
                    f"  {topic:<30} MAE={item_metrics['mae']:<6} "
                    f"Acc={acc_pct:.1f}% (n={int(item_metrics['n'])})"
                )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "split", "method", "model_name", "n", "mae", "rmse",
                    "pearson_r", "accuracy", "precision", "recall", "f1", "source",
                ]
            )
            for row in rows:
                m = row["metrics"]
                writer.writerow(
                    [
                        row["split"],
                        row["label"],
                        row["model_name"],
                        row["scored_samples"],
                        m["mae"],
                        m["rmse"],
                        m["pearson_r"],
                        m["accuracy"],
                        m["precision"],
                        m["recall"],
                        m["f1"],
                        row["source"],
                    ]
                )
        print(f"\nSaved comparison CSV to: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
