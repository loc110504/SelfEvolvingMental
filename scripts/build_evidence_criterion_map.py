#!/usr/bin/env python3
"""Package a finished run's Evidence-Criterion Maps into one review artifact.

``scripts/run_proposed_assessment.py`` already computes each participant's
Evidence-Criterion Map deterministically, straight from the rubric and the
evidence it scored from — see ``_build_criterion_map`` there — and writes it
into that participant's own ``*_evaluation.json`` alongside the prediction it
explains. Nothing here recomputes anything or calls a model; this only
collects what is already on disk into one file per split that is easy to
skim or hand to a reviewer, and flags the participants worth a second look
(a contradiction was found, or an item stayed unresolved).

Example
-------
    python3 scripts/build_evidence_criterion_map.py results/proposed/gptoss20b_dev \
        --out results/proposed/gptoss20b_dev/criterion_maps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def collect(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*_evaluation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        criterion_map = record.get("criterion_map")
        if criterion_map is None:
            continue
        contradicted = [
            row["criterion"] for row in criterion_map if row["contradiction"]
        ]
        unresolved = [
            row["criterion"] for row in criterion_map if row["status"] == "unresolved"
        ]
        rows.append(
            {
                "participant_id": record.get("participant_id"),
                "ground_truth_total": record.get("ground_truth_total"),
                "total_predicted_score": record.get("total_predicted_score"),
                "contradicted_criteria": contradicted,
                "unresolved_criteria": unresolved,
                "criterion_map": criterion_map,
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Evidence-Criterion Maps", ""]
    for row in rows:
        lines.append(
            f"## Participant {row['participant_id']}  "
            f"(predicted {row['total_predicted_score']}, "
            f"reference {row['ground_truth_total']})"
        )
        if row["contradicted_criteria"]:
            contradicted_text = ", ".join(row["contradicted_criteria"])
            lines.append(f"Contradictions found: {contradicted_text}")
        if row["unresolved_criteria"]:
            unresolved_text = ", ".join(row["unresolved_criteria"])
            lines.append(f"Unresolved (abstained): {unresolved_text}")
        lines.append("")
        lines.append(
            "| Criterion | Status | Score | Confidence | Contradiction | "
            "Counterfactual |"
        )
        lines.append("|---|---|---|---|---|---|")
        for entry in row["criterion_map"]:
            contradiction_text = "yes" if entry["contradiction"] else "no"
            lines.append(
                f"| {entry['criterion']} | {entry['status']} | {entry['score']} | "
                f"{entry['confidence']} | {contradiction_text} | "
                f"{entry['counterfactual']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    rows = collect(args.run_dir)
    if not rows:
        raise SystemExit(f"no criterion maps found in {args.run_dir}")

    out_path = args.out or (args.run_dir / "criterion_maps.json")
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} participants)")

    contradiction_count = sum(1 for row in rows if row["contradicted_criteria"])
    unresolved_count = sum(len(row["unresolved_criteria"]) for row in rows)
    print(
        f"  participants with a contradiction: {contradiction_count}/{len(rows)}"
    )
    print(f"  total unresolved (abstained) items: {unresolved_count}")

    if args.markdown:
        args.markdown.write_text(render_markdown(rows), encoding="utf-8")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
