#!/usr/bin/env python3
"""Fit the training-split imputation model and total-score calibrator.

Both artifacts are fitted on the *training* split only and then applied
unchanged to dev/test, so they add no held-out information. They are written
as small JSON files under ``configs/fitted/`` and passed to
``scripts/run_openai_eval.py``.

Two fitting sources are supported:

``--from-references`` (default)
    Uses training-split PHQ-8 labels alone. Available before any model run,
    and independent of which backbone produced the scores.
``--from-run RESULTS_DIR``
    Uses a completed training-split run, so each item is fitted against the
    mean of the items that pipeline actually assessed. Preferred once a
    training run exists, because it absorbs the scorer's systematic offset.

The calibrator needs predictions, so ``--from-run`` is required to fit one.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.calibration import fit_calibrator  # noqa: E402
from psyvec.evaluation.imputation import (  # noqa: E402
    fit_from_references,
    fit_from_run,
)

ITEM_KEYS = (
    "PHQ8_NoInterest",
    "PHQ8_Depressed",
    "PHQ8_Sleep",
    "PHQ8_Tired",
    "PHQ8_Appetite",
    "PHQ8_Failure",
    "PHQ8_Concentrating",
    "PHQ8_Moving",
)


def load_references(split_dir: Path) -> dict[str, dict[str, int]]:
    references: dict[str, dict[str, int]] = {}
    for path in sorted(split_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = payload.get("phq8_scores", {})
        references[str(payload.get("Participant_ID", path.stem))] = {
            "__total__": scores.get("PHQ8_Score", 0),
            **{key: value for key, value in scores.get("items", {}).items()},
        }
    return references


def _assessed_items(record: dict[str, Any]) -> dict[str, int]:
    """Items this record actually assessed, across both output schemas.

    v3 records mark an unassessed item with ``abstained``; v2 records have no
    such field and instead expose ``evidence_status``, where anything other
    than ``known`` was hard-zeroed rather than assessed. Reading both lets a
    v2 training run seed the imputer before a v3 training run exists.
    """
    assessed: dict[str, int] = {}
    for info in record.get("topics", {}).values():
        item_key = info.get("item_key")
        score = info.get("score")
        if not item_key or score is None:
            continue
        if info.get("abstained"):
            continue
        legacy_status = info.get("evidence_status")
        if "abstained" not in info and legacy_status not in (None, "known"):
            continue
        assessed[item_key] = int(score)
    return assessed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir", type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "train",
    )
    parser.add_argument(
        "--from-run", type=Path, default=None,
        help="directory of *_evaluation.json from a training-split run",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "configs" / "fitted",
    )
    parser.add_argument(
        "--calibration-objective", default="mae", choices=("mae", "spread", "identity"),
    )
    args = parser.parse_args()

    references = load_references(args.train_dir)
    if not references:
        raise SystemExit(f"no training samples under {args.train_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference_items = [
        {key: value for key, value in record.items() if key != "__total__"}
        for record in references.values()
    ]

    if args.from_run is None:
        model = fit_from_references(reference_items, ITEM_KEYS)
        suffix = "references"
    else:
        aligned_assessed: list[dict[str, int]] = []
        aligned_reference: list[dict[str, int]] = []
        predicted_totals: list[float] = []
        reference_totals: list[float] = []
        for path in sorted(glob.glob(str(args.from_run / "*_evaluation.json"))):
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            participant = str(record.get("participant_id"))
            if participant not in references:
                continue
            reference = references[participant]
            aligned_assessed.append(_assessed_items(record))
            aligned_reference.append(
                {key: value for key, value in reference.items() if key != "__total__"}
            )
            predicted_totals.append(float(record.get("total_predicted_score") or 0))
            reference_totals.append(float(reference["__total__"]))
        if not aligned_assessed:
            raise SystemExit(f"no usable records under {args.from_run}")
        model = fit_from_run(aligned_assessed, aligned_reference, ITEM_KEYS)
        suffix = "run"

        calibrator = fit_calibrator(
            predicted_totals, reference_totals, objective=args.calibration_objective
        )
        calibrator_path = args.out_dir / f"calibrator_{args.calibration_objective}.json"
        calibrator_path.write_text(calibrator.to_json(), encoding="utf-8")
        print(
            f"calibrator ({args.calibration_objective}): "
            f"total = {calibrator.intercept:+.3f} {calibrator.slope:+.3f} * raw"
        )
        print(f"  -> {calibrator_path}")

    model_path = args.out_dir / f"imputation_{suffix}.json"
    model_path.write_text(model.to_json(), encoding="utf-8")
    print(f"\nimputation model (source={model.source}, n={len(reference_items)}):")
    for key in ITEM_KEYS:
        fit = model.fits.get(key)
        if fit is None:
            continue
        print(
            f"  {key:<20} = {fit.intercept:+.3f} {fit.slope:+.3f} * mean(assessed)"
            f"   base={fit.base_rate:.2f}  n={fit.support}"
        )
    print(f"  -> {model_path}")


if __name__ == "__main__":
    main()
