"""Strict parsing of LLM assessment responses.

Reasoning-style models (Qwen3.x, DeepSeek-R1, ...) emit a long chain of
thought before the answer.  A lenient parser that scans the whole response
for the first small integer will happily lock onto the ``1.`` of a numbered
thinking list, which silently turns every assessment into a constant.  The
parsers here therefore strip reasoning first, require a real score field,
and report failure instead of inventing a value.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NecessityParse",
    "ScoreParse",
    "UpdaterParse",
    "drop_reasoning_tags",
    "extract_json_objects",
    "parse_necessity_score",
    "parse_score_and_summary",
    "parse_summary_and_updated_scores",
    "strip_reasoning",
]

_REASONING_TAGS = ("think", "thinking", "reasoning", "thought", "analysis")

_CLOSED_TAG_RE = re.compile(
    r"<\s*(" + "|".join(_REASONING_TAGS) + r")\s*>.*?<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_TAG_RE = re.compile(
    r"<\s*(?:" + "|".join(_REASONING_TAGS) + r")\s*>",
    re.IGNORECASE,
)

# "Thinking Process:", "**Thought Process**:", "Reasoning:" ... as a heading.
_REASONING_HEADING_RE = re.compile(
    r"^\s*[*#>\s]*"
    r"(?:thinking|thought|reasoning|chain[ -]of[ -]thought)"
    r"(?:\s+process)?"
    r"[*#\s]*:",
    re.IGNORECASE,
)

# Markers a model uses to hand over from thinking to the actual answer.
_ANSWER_MARKER_RE = re.compile(
    r"^\s*(?:[*#>\s]*)"
    r"(?:final\s+answer|final\s+response|final\s+output|answer|response|output"
    r"|result|json)"
    r"(?:[*#\s]*)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:-{3,}|={3,}|\*{3,})\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*$|^\s*```\s*$", re.MULTILINE)

_PREVIEW_CHARS = 200


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= _PREVIEW_CHARS:
        return flat
    return flat[:_PREVIEW_CHARS] + "..."


def drop_reasoning_tags(raw_text: str) -> str:
    """Remove ``<think>``-style blocks, including an unclosed trailing one."""
    text = _CLOSED_TAG_RE.sub("", raw_text)
    open_tag = _OPEN_TAG_RE.search(text)
    if open_tag is not None:
        # Unclosed reasoning tag: everything after it is an unfinished thought.
        text = text[: open_tag.start()]
    return text.strip()


def strip_reasoning(raw_text: str) -> str:
    """Return ``raw_text`` with chain-of-thought removed.

    Returns an empty string when the response is nothing but (possibly
    truncated) reasoning, so callers can treat that as a parse failure
    rather than scoring the model's scratchpad.
    """
    text = drop_reasoning_tags(raw_text)
    if not text:
        return ""

    if _REASONING_HEADING_RE.match(text):
        marker = _ANSWER_MARKER_RE.search(text)
        if marker is not None:
            return text[marker.end() :].strip()
        rule = _HORIZONTAL_RULE_RE.search(text)
        fence = _FENCE_RE.search(text)
        if rule is not None:
            return text[rule.end() :].strip()
        if fence is not None:
            return text[fence.start() :].strip()
        # Heading with no hand-over marker: reasoning only (usually truncated).
        return ""

    return text


def extract_json_objects(text: str) -> list[str]:
    """Return every balanced ``{...}`` span in ``text``, in order.

    String-aware, so braces inside JSON string values do not unbalance the
    scan.  Callers normally want the *last* object: reasoning traces often
    quote the requested output schema before emitting the real answer.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : index + 1])
                start = -1

    return spans


def _loaded_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for span in extract_json_objects(text):
        try:
            data = json.loads(span)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            objects.append(data)
    return objects


def _coerce_score(value: Any, min_score: int, max_score: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if value != int(value):
            return None
        number = int(value)
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(-?\d+)\s*", value)
        if match is None:
            return None
        number = int(match.group(1))
    else:
        return None
    return max(min_score, min(max_score, number))


@dataclass(frozen=True)
class ScoreParse:
    """Outcome of parsing a scorer response."""

    score: int | None
    summary: str
    ok: bool
    failure_reason: str = ""
    raw_text: str = ""


def parse_score_and_summary(
    raw_text: str,
    min_score: int = 0,
    max_score: int = 3,
) -> ScoreParse:
    """Extract an integer score and its one-line basis from a scorer reply.

    Never guesses.  When no explicit score field is present the result has
    ``ok is False`` and ``score is None``.
    """
    if not raw_text.strip():
        return ScoreParse(None, "", False, "empty response", raw_text)

    tag_free = drop_reasoning_tags(raw_text)
    cleaned = strip_reasoning(raw_text)

    # Prose framing may hide a real answer, so fall back to the whole tag-free
    # text: a schema echoed inside reasoning carries placeholders and will not
    # parse as JSON, so it cannot be mistaken for the answer.
    for source in (cleaned, tag_free):
        if not source:
            continue
        for data in reversed(_loaded_json_objects(source)):
            if "score" not in data:
                continue
            score = _coerce_score(data["score"], min_score, max_score)
            if score is None:
                continue
            summary = str(data.get("summary", "")).strip()
            return ScoreParse(score, summary, True, "", raw_text)
        break

    if not cleaned:
        return ScoreParse(
            None,
            "",
            False,
            "reasoning-only response (likely truncated before the answer)",
            raw_text,
        )

    key_matches = list(
        re.finditer(r'"?score"?\s*:\s*(-?\d+)', cleaned, re.IGNORECASE)
    )
    if key_matches:
        score = _coerce_score(key_matches[-1].group(1), min_score, max_score)
        if score is not None:
            summary_match = re.search(
                r'"?summary"?\s*:\s*"([^"]*)"', cleaned, re.IGNORECASE
            )
            summary = summary_match.group(1).strip() if summary_match else ""
            return ScoreParse(score, summary, True, "", raw_text)

    bare = re.fullmatch(r"[^\d-]*(-?\d+)[^\d]*", cleaned)
    if bare is not None:
        score = _coerce_score(bare.group(1), min_score, max_score)
        if score is not None:
            return ScoreParse(score, "", True, "", raw_text)

    return ScoreParse(
        None,
        "",
        False,
        f"no score field found in response: {_preview(cleaned)}",
        raw_text,
    )


@dataclass(frozen=True)
class NecessityParse:
    """Outcome of parsing a follow-up-necessity response."""

    score: int | None
    ok: bool
    failure_reason: str = ""
    raw_text: str = ""


def parse_necessity_score(raw_text: str) -> NecessityParse:
    """Extract a follow-up necessity rating (0, 1 or 2)."""
    if not raw_text.strip():
        return NecessityParse(None, False, "empty response", raw_text)

    cleaned = strip_reasoning(raw_text)
    if not cleaned:
        return NecessityParse(
            None,
            False,
            "reasoning-only response (likely truncated before the answer)",
            raw_text,
        )

    for data in reversed(_loaded_json_objects(cleaned)):
        for key in ("necessity", "score", "rating"):
            if key in data:
                score = _coerce_score(data[key], 0, 2)
                if score is not None:
                    return NecessityParse(score, True, "", raw_text)

    bare = re.fullmatch(r"[^\d-]*(-?\d+)[^\d]*", cleaned)
    if bare is not None:
        score = _coerce_score(bare.group(1), 0, 2)
        if score is not None:
            return NecessityParse(score, True, "", raw_text)

    key_matches = list(
        re.finditer(r"\b(?:necessity|score|rating)\b\D{0,10}([0-2])", cleaned, re.I)
    )
    if key_matches:
        return NecessityParse(int(key_matches[-1].group(1)), True, "", raw_text)

    return NecessityParse(
        None,
        False,
        f"no necessity rating found in response: {_preview(cleaned)}",
        raw_text,
    )


@dataclass(frozen=True)
class UpdaterParse:
    """Outcome of parsing the memory-update / summary agent response."""

    summary: str
    updated_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    ok: bool = False
    failure_reason: str = ""
    raw_text: str = ""


def _valid_updates(
    payload: Any,
    valid_topics: list[str],
) -> dict[str, dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return updates
    for topic, score_data in payload.items():
        if topic not in valid_topics or not isinstance(score_data, dict):
            continue
        score = _coerce_score(score_data.get("score"), 0, 3)
        if score is None:
            continue
        updates[topic] = {
            "score": score,
            "reason": str(score_data.get("reason", "")),
        }
    return updates


def parse_summary_and_updated_scores(
    raw_text: str,
    valid_topics: list[str],
) -> UpdaterParse:
    """Extract the overall summary and per-topic score adjustments."""
    if not raw_text.strip():
        return UpdaterParse("", {}, False, "empty response", raw_text)

    cleaned = strip_reasoning(raw_text)
    if not cleaned:
        return UpdaterParse(
            "",
            {},
            False,
            "reasoning-only response (likely truncated before the answer)",
            raw_text,
        )

    for data in reversed(_loaded_json_objects(cleaned)):
        if "summary" not in data and "updated_scores" not in data:
            continue
        summary = str(data.get("summary", "")).strip()
        updates = _valid_updates(data.get("updated_scores", {}), valid_topics)
        return UpdaterParse(summary, updates, True, "", raw_text)

    # Loosely formatted JSON: recover per-topic scores by pattern.
    recovered: dict[str, dict[str, Any]] = {}
    for topic in valid_topics:
        pattern = rf'"{re.escape(topic)}"\s*:\s*\{{[^}}]*"score"\s*:\s*([0-3])'
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            recovered[topic] = {
                "score": int(match.group(1)),
                "reason": "Extracted via regex fallback",
            }
    if recovered:
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', cleaned)
        summary = summary_match.group(1).strip() if summary_match else ""
        return UpdaterParse(summary, recovered, True, "", raw_text)

    return UpdaterParse(
        "",
        {},
        False,
        f"no JSON summary/updated_scores found in response: {_preview(cleaned)}",
        raw_text,
    )
