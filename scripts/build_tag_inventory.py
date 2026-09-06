#!/usr/bin/env python3
"""Scan processed DAIC-WOZ samples for Ellie's `tag (utterance)` protocol markers.

DAIC-WOZ's virtual interviewer (Ellie) utterances sometimes carry the internal
dialogue-act identifier verbatim in the transcript, e.g. ``easy_sleep (how easy
is it for you to get a good night's sleep)``. These tags are a free, reliable
anchor for topic-specific retrieval (Plan_Improve.md Sec 2.3a) because they are
part of a fixed protocol, not model output.

This script only lists what exists in the data; it never guesses a topic
mapping. The mapping into `configs/scales/topic_tag_map.json` is a manual,
human-reviewed step (see Plan_Improve.md Sec 6, Phase 1, step 1) and MUST be
built from the ``train`` split only, to keep ``dev``/``test`` unseen during
design.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_TAG_RE = re.compile(r"^([A-Za-z0-9_]+)\s*\((.*)\)$", re.DOTALL)


def extract_tag(content: str) -> tuple[str, str] | None:
    """Return ``(tag, utterance)`` if ``content`` is a tagged Ellie line."""
    match = _TAG_RE.match(content.strip())
    if match is None:
        return None
    return match.group(1), match.group(2).strip()


def scan_directory(data_dir: Path) -> tuple[Counter[str], dict[str, str]]:
    """Return (tag -> occurrence count, tag -> first-seen utterance text)."""
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for sample_path in sorted(data_dir.glob("*.json")):
        sample: dict[str, Any] = json.loads(sample_path.read_text(encoding="utf-8"))
        for turn in sample.get("real_interview", []):
            if turn.get("roleName") != "Ellie":
                continue
            found = extract_tag(str(turn.get("content", "")))
            if found is None:
                continue
            tag, utterance = found
            counts[tag] += 1
            examples.setdefault(tag, utterance)
    return counts, examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "train",
        help=(
            "Processed DAIC-WOZ split to scan "
            "(default: train — see module docstring)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to write the inventory as JSON "
            "(tag -> {count, utterance})."
        ),
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: data dir does not exist: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    counts, examples = scan_directory(args.data_dir)

    print(f"Scanned: {args.data_dir}")
    print(f"Unique tags found: {len(counts)}\n")
    for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{count:4d}  {tag:<28} -> {examples[tag]}")

    if args.output is not None:
        payload = {
            tag: {"count": count, "utterance": examples[tag]}
            for tag, count in counts.items()
        }
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote inventory JSON to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
