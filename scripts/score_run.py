#!/usr/bin/env python3
"""Score a completed evaluation directory against the DAIC-WOZ references.

Reports the endpoints the v2 analysis showed are needed to tell a real
improvement from a shrinkage artifact:

* **Trivial floor.** The MAE of predicting a single constant (the training
  median). On DAIC-WOZ dev that floor is 5.49, which all three prompting
  baselines in this repository fail to beat — a total-score MAE means little
  without it alongside.
* **Severity-stratified bias.** Mean signed error within reference bands. A
  system can post a competitive MAE while under-predicting every severe
  participant by 8-10 points, and only this breakdown shows it.
* **Spread ratio.** Predicted standard deviation over reference standard
  deviation. Below ~0.6 the system is not separating severities regardless of
  what its MAE says.
* **Retrieval-vs-scorer decomposition.** An oracle that fills every
  unassessed/abstained item with its reference value; the gap between that and
  the actual result is the share of error attributable to evidence acquisition
  rather than to scoring.

Reads both the v2 (``evidence_status``) and v3 (``abstained``) record schemas.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BANDS = (
    (0, 4, "0-4   minimal"),
    (5, 9, "5-9   mild"),
    (10, 14, "10-14 moderate"),
    (15, 24, "15+   mod.severe/severe"),
)
TIER_NAMES = ("direct", "lexical", "expanded", "contextual", "absent")


def pearson(first: Sequence[float], second: Sequence[float]) -> float:
    count = len(first)
    if count < 2:
        return 0.0
    mean_a = sum(first) / count
    mean_b = sum(second) / count
    pairs = list(zip(first, second, strict=True))
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in pairs)
    denominator = (
        sum((a - mean_a) ** 2 for a in first) ** 0.5
        * sum((b - mean_b) ** 2 for b in second) ** 0.5
    )
    return numerator / denominator if denominator else 0.0


def binary_scores(
    pairs: Sequence[tuple[float, float]], threshold: int
) -> dict[str, float]:
    tp = sum(1 for p, r in pairs if p >= threshold and r >= threshold)
    fp = sum(1 for p, r in pairs if p >= threshold and r < threshold)
    fn = sum(1 for p, r in pairs if p < threshold and r >= threshold)
    tn = len(pairs) - tp - fp - fn
    denominator = 2 * tp + fp + fn
    return {
        "accuracy": (tp + tn) / len(pairs),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / denominator if denominator else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _is_assessed(info: dict[str, Any]) -> bool:
    """Whether this item was scored from evidence rather than defaulted.

    v3 records say so directly with ``abstained``. v2 records have no such
    field; there, anything whose ``evidence_status`` is not ``known`` was
    hard-zeroed by ``grounded_score`` rather than actually assessed.
    """
    if info.get("score") is None:
        return False
    if "abstained" in info:
        return not info["abstained"]
    return info.get("evidence_status", "known") == "known"


def load_records(
    results_dir: Path, split_dir: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in sorted(results_dir.glob("*_evaluation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        sample_path = split_dir / f"{record['participant_id']}.json"
        if not sample_path.is_file():
            continue
        reference = json.loads(sample_path.read_text(encoding="utf-8"))["phq8_scores"]
        records.append((record, reference))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "train",
    )
    args = parser.parse_args()

    split_dir = PROJECT_ROOT / "data" / "processed_daic_woz" / args.split
    records = load_records(args.results_dir, split_dir)
    if not records:
        raise SystemExit(f"no scorable records in {args.results_dir}")

    predicted = [float(r["total_predicted_score"]) for r, _ in records]
    reference = [float(g["PHQ8_Score"]) for _, g in records]
    pairs = list(zip(predicted, reference, strict=True))
    count = len(records)

    train_totals = [
        json.loads(path.read_text(encoding="utf-8"))["phq8_scores"]["PHQ8_Score"]
        for path in sorted(args.train_dir.glob("*.json"))
    ]
    floor = statistics.median(train_totals)
    floor_mae = sum(abs(floor - r) for r in reference) / count

    mae = sum(abs(p - r) for p, r in pairs) / count
    rmse = (sum((p - r) ** 2 for p, r in pairs) / count) ** 0.5
    reference_sd = statistics.pstdev(reference)
    predicted_sd = statistics.pstdev(predicted)
    spread = predicted_sd / reference_sd if reference_sd else 0.0
    binary = binary_scores(pairs, args.threshold)

    verdict = (
        "  ** WORSE THAN CONSTANT **"
        if mae >= floor_mae
        else f"; {(1 - mae / floor_mae) * 100:.0f}% better"
    )
    collapse = "  ** severity collapse **" if spread < 0.6 else ""

    print(f"\n{'=' * 74}")
    print(f"{args.results_dir}  split={args.split}  n={count}")
    print("=" * 74)
    print(
        f"  MAE              {mae:7.3f}     "
        f"(trivial floor: constant {floor:g} -> {floor_mae:.3f}{verdict})"
    )
    print(f"  RMSE             {rmse:7.3f}")
    print(f"  Pearson r        {pearson(predicted, reference):7.3f}")
    print(
        f"  spread ratio     {spread:7.3f}     "
        f"(pred sd {predicted_sd:.2f} / ref sd {reference_sd:.2f}){collapse}"
    )
    print(
        f"  binary @>={args.threshold:<4}    F1={binary['f1']:.3f}  "
        f"acc={binary['accuracy']:.3f}  P={binary['precision']:.3f}  "
        f"R={binary['recall']:.3f}  (tp={binary['tp']} fp={binary['fp']} "
        f"fn={binary['fn']} tn={binary['tn']})"
    )

    print("\n  severity-stratified signed bias (pred - reference):")
    for low, high, label in BANDS:
        band = [(p, r) for p, r in pairs if low <= r <= high]
        if not band:
            continue
        print(
            f"    {label:<24} n={len(band):<4} "
            f"bias={sum(p - r for p, r in band) / len(band):+7.2f}"
            f"   mean pred={sum(p for p, _ in band) / len(band):5.2f}"
            f"   mean ref={sum(r for _, r in band) / len(band):5.2f}"
        )

    assessed_total = 0
    item_total = 0
    oracle: list[float] = []
    tier_counts: dict[str, int] = {}
    for record, gold in records:
        items = gold.get("items", {})
        total = 0.0
        for info in record.get("topics", {}).values():
            item_total += 1
            if _is_assessed(info):
                total += float(info["score"])
                assessed_total += 1
            else:
                total += float(items.get(info.get("item_key", ""), 0))
            tier = info.get("evidence_tier") or info.get("evidence_status") or "unknown"
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        oracle.append(total)

    oracle_pairs = list(zip(oracle, reference, strict=True))
    oracle_mae = sum(abs(o - r) for o, r in oracle_pairs) / count
    print("\n  evidence acquisition:")
    print(
        f"    items assessed from evidence   {assessed_total}/{item_total} "
        f"({assessed_total / item_total * 100:.1f}%)"
    )
    print(f"    tiers: {json.dumps(dict(sorted(tier_counts.items())))}")
    print(
        f"    oracle-filled unassessed       MAE={oracle_mae:.3f}  "
        f"r={pearson(oracle, reference):.3f}"
        "   <- ceiling if acquisition were perfect"
    )
    print(f"    error attributable to acquisition: {mae - oracle_mae:+.3f} MAE\n")


if __name__ == "__main__":
    main()
