#!/usr/bin/env python3
"""Print every real transcript line each topic keyword matches on a split.

Used to tune ``configs/scales/topic_keywords.json`` (Plan_Improve.md Sec 2.3b,
Sec 6 Phase 1 step 6): a keyword that mostly matches unrelated speech (e.g.
bare "down" matching "downtown", "lay down", "somewhere down the line") shows
up here as a long list of irrelevant example lines, which is the signal to
narrow it into a safer multi-word phrase or drop it.

MUST be run against the ``train`` split only when deciding lexicon changes,
per the split-isolation discipline in Plan_Improve.md Sec 5.2 — dev/test are
for evaluating the resulting lexicon, not for tuning it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.evidence_retrieval import load_keyword_lexicon  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "train",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=PROJECT_ROOT / "configs" / "scales" / "topic_keywords.json",
    )
    parser.add_argument(
        "--scale-file",
        type=Path,
        default=PROJECT_ROOT / "configs" / "scales" / "PHQ-8.json",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max example lines printed per keyword.",
    )
    args = parser.parse_args()

    topics = list(json.loads(args.scale_file.read_text(encoding="utf-8")).keys())
    lexicon = load_keyword_lexicon(args.lexicon, valid_topics=topics)

    hits: dict[str, dict[str, list[str]]] = {topic: {} for topic in lexicon}
    for sample_path in sorted(args.data_dir.glob("*.json")):
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        for turn in sample.get("real_interview", []):
            if turn.get("roleName") != "Participant":
                continue
            text = str(turn.get("content", ""))
            lowered = text.lower()
            for topic, keywords in lexicon.items():
                for keyword in keywords:
                    if keyword in lowered:
                        hits[topic].setdefault(keyword, []).append(text)

    for topic, keyword_hits in hits.items():
        print(f"===== {topic} =====")
        for keyword, examples in keyword_hits.items():
            print(f"  {keyword!r}: {len(examples)} hits")
            for example in examples[: args.max_examples]:
                print(f"      - {example}")
        print()


if __name__ == "__main__":
    main()
