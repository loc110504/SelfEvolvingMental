#!/usr/bin/env python3
"""The proposed method: evidence-utility adaptive interviewing over DAIC-WOZ.

This is not an ablation or a variant of ``scripts/run_assessment.py`` — it is
a different scheduling loop built on the same retrieval, scoring, imputation
and calibration primitives. ``run_assessment.py`` expands every PHQ-8 topic
independently up to its own round cap; nothing there chooses *between*
topics. Here, after every topic gets one seed retrieval pass, a shared
expansion-round budget is spent one round at a time on whichever unresolved
topic currently has the highest acquisition utility
(:mod:`psyvec.interview.utility`) — need, minus redundancy, plus a bounded
bonus from any validated interview-strategy lesson for that state. Every
"answer" is still real retrieved transcript evidence; nothing is generated on
the participant's behalf.

Alongside the score, every retrieval round is passed through the Evidence
Agent (:mod:`psyvec.evaluation.evidence_extraction`), which pulls out atomic,
independently-sourced claims and adds them to a per-participant
:class:`psyvec.memory.patient_evidence.PatientEvidenceMemory` — appended, not
overwritten, so a later claim that contradicts an earlier one produces a
flagged :class:`~psyvec.memory.patient_evidence.ContradictionEdge` rather than
silently replacing it.

Pipeline, per participant, per PHQ-8 topic:

1. **Seed.** Tag-anchored + lexical retrieval (unchanged from ``run_
   assessment.py``). The Evidence Agent extracts atomic claims from what was
   found; the sufficiency role judges whether the topic is already settled.
2. **Schedule.** While any topic is unresolved and the global expansion
   budget remains, the scheduler picks the highest-utility topic, the
   interviewer role proposes new search phrases (with any matching validated
   interview-strategy lesson appended), retrieval expands, the Evidence Agent
   extracts claims from what is new, and sufficiency is re-judged.
3. **Score.** Once acquisition stops (every topic resolved, or the budget is
   spent), the scorer reads the accumulated excerpts for each topic directly
   — unchanged from ``run_assessment.py`` — and may abstain rather than being
   forced to a zero.
4. **Resolve.** Abstained items are imputed from what was assessed; the total
   is optionally calibrated. Both fitted on the training split only.
5. **Explain.** A deterministic Evidence-Criterion Map is built straight from
   the rubric and the evidence memory — no extra model call, no risk of an
   explanation that does not match the score it describes.

Example
-------
    export OPENAI_API_KEY=sk-...
    python3 scripts/run_proposed_assessment.py --backend openai \
        --model gpt-4o-mini --data-dir data/processed_daic_woz/dev \
        --output-dir results/proposed/gpt4omini_dev --concurrency 8

    # Ollama, gpt-oss:20b
    python3 scripts/run_proposed_assessment.py --backend ollama \
        --model gpt-oss:20b --data-dir data/processed_daic_woz/train \
        --output-dir results/proposed/gptoss20b_train --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.calibration import TotalScoreCalibrator  # noqa: E402
from psyvec.evaluation.evidence_extraction import (  # noqa: E402
    EVIDENCE_AGENT_SYSTEM_PROMPT,
    EVIDENCE_EXTRACTION_JSON_SCHEMA,
    build_evidence_extraction_prompt,
    parse_evidence_extraction,
)
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
from psyvec.interview.utility import TopicAcquisitionState, next_action  # noqa: E402
from psyvec.memory.patient_evidence import (  # noqa: E402
    PatientEvidenceMemory,
    add_evidence,
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
logger = logging.getLogger("psyvec-proposed")

#: Which reference JSON key each PHQ-8 topic's score is checked against.
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

#: USD per million tokens; a flag so a run's logged cost is never silently stale.
DEFAULT_INPUT_COST_PER_1M = 0.15
DEFAULT_OUTPUT_COST_PER_1M = 0.60

#: The day-count band each PHQ-8 rubric score names, for the deterministic
#: counterfactual — identical across every topic (see
#: configs/scales/scoring_standards.json), so this is not topic-specific.
_DAY_BANDS = {0: "0-1 day", 1: "2-6 days", 2: "7-11 days", 3: "12-14 days"}


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


def _load_scorer_lessons(path: Path) -> dict[str, tuple[str, ...]]:
    """Read promoted scorer-interpretation lessons as ``{topic: (rule, ...)}``.

    Verbatim from ``scripts/run_assessment.py``'s ``_load_lessons``: this file
    format is unchanged, so both pipelines read the same
    ``configs/fitted/lessons.json``.
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


def _load_interview_lessons(path: Path) -> dict[str, tuple[str, ...]]:
    """Read promoted interview-strategy lessons as ``{"topic|tier": (rule, ...)}``.

    Reads the output of ``scripts/evolve_interview_lessons.py``, scoped by
    ``(topic, evidence tier)`` rather than by topic alone, matching
    :func:`psyvec.interview.utility.experience_match`'s scope convention.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    rendered: dict[str, list[str]] = {}
    for entry in raw.get("lessons", []):
        topic = str(entry.get("topic", ""))
        tier = str(entry.get("tier", ""))
        rendered_text = str(entry.get("rendered", "")).strip()
        if not topic or not tier or not rendered_text:
            continue
        rendered.setdefault(f"{topic}|{tier}", []).append(rendered_text)
    return {scope: tuple(items) for scope, items in rendered.items()}


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
            None if args.imputation_model is None
            else ImputationModel.load(args.imputation_model)
        )
        self.calibrator: TotalScoreCalibrator | None = (
            None if args.calibrator is None
            else TotalScoreCalibrator.load(args.calibrator)
        )
        self.lessons: dict[str, tuple[str, ...]] = (
            {} if args.lessons is None else _load_scorer_lessons(args.lessons)
        )
        self.interview_lessons: dict[str, tuple[str, ...]] = (
            {} if args.interview_lessons is None
            else _load_interview_lessons(args.interview_lessons)
        )
        self.global_budget = max(0, args.global_budget)
        self.max_rounds_per_topic = max(1, args.max_rounds_per_topic)
        self.evidence_limit = args.evidence_limit


def _extract_claims(
    backend: ChatBackend,
    memory: PatientEvidenceMemory,
    *,
    participant_id: str,
    topic: str,
    new_turns: Sequence[RetrievedTurn],
    round_index: int,
) -> PatientEvidenceMemory:
    """Run the Evidence Agent over the turns newly retrieved this round only."""
    if not new_turns:
        return memory
    extraction_text = format_tiered_evidence(
        TieredEvidence(
            topic=topic, tier=overall_tier(new_turns), turns=tuple(new_turns),
            queries_issued=(),
        )
    )
    turn_relations = {item.turn.turn_id: item.relation for item in new_turns}
    raw = backend.generate(
        GenerationRequest(
            system=EVIDENCE_AGENT_SYSTEM_PROMPT,
            user=build_evidence_extraction_prompt(topic, extraction_text),
            max_tokens=700,
            json_schema=EVIDENCE_EXTRACTION_JSON_SCHEMA,
        )
    )
    records = parse_evidence_extraction(
        raw, participant_id=participant_id, topic=topic,
        round_index=round_index, turn_relations=turn_relations,
    )
    return add_evidence(memory, records)


def _run_seed_round(
    backend: ChatBackend,
    topic: str,
    config: PipelineConfig,
    paired_turns: Sequence[PairedTurn],
    participant_id: str,
    memory: PatientEvidenceMemory,
) -> tuple[dict[str, Any], PatientEvidenceMemory]:
    seed = retrieve_topic_evidence(
        paired_turns, topic, config.evidence_map, config.seed_queries[topic],
        limit=config.evidence_limit,
    )
    memory = _extract_claims(
        backend, memory, participant_id=participant_id, topic=topic,
        new_turns=seed.turns, round_index=0,
    )
    standard = config.standards.get(topic, {})
    evidence_text = format_tiered_evidence(seed)
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
    state: dict[str, Any] = {
        "turns": seed.turns,
        "queries_issued": [],
        "tier": overall_tier(seed.turns),
        "sufficient": verdict,
        "missing": missing or "",
        "rounds_spent": 1,
        "trace": [],
    }
    return state, memory


def _run_expansion_round(
    backend: ChatBackend,
    topic: str,
    config: PipelineConfig,
    paired_turns: Sequence[PairedTurn],
    participant_id: str,
    state: dict[str, Any],
    memory: PatientEvidenceMemory,
) -> tuple[dict[str, Any], PatientEvidenceMemory]:
    """Spend one scheduled expansion round on ``topic``, returning an updated state."""
    tier_before = state["tier"]
    resolved_before = state["sufficient"] is True
    round_index = state["rounds_spent"]  # 0 was the seed round
    # Snapshotted so scripts/evolve_interview_lessons.py can replay this exact
    # round (candidate lesson vs none) without re-running everything upstream
    # of it, the same way evolve_lessons.py replays scoring from cached
    # evidence_text instead of re-retrieving.
    missing_before = state["missing"]
    already_tried_before = tuple(state["queries_issued"])
    exclude_turn_ids_before = tuple(
        sorted(item.turn.turn_id for item in state["turns"])
    )

    standard = config.standards.get(topic, {})
    lesson_texts = config.interview_lessons.get(f"{topic}|{tier_before}", ())
    prompt = build_query_expansion_prompt(
        topic, standard, missing_before, already_tried_before
    )
    if lesson_texts:
        guidance = "\n".join(f"  - {text}" for text in lesson_texts)
        prompt += (
            "\n\n## Validated search guidance for this state\n"
            "Each line was distilled from prior cases and kept only after it "
            "improved evidence acquisition on participants other than the "
            "ones it came from. Apply it unless it plainly does not fit.\n"
            + guidance
        )
    new_queries = parse_query_expansion(
        backend.generate(
            GenerationRequest(
                system=INTERVIEWER_SYSTEM_PROMPT, user=prompt, max_tokens=300,
                temperature=0.4, json_schema=QUERY_EXPANSION_JSON_SCHEMA,
            )
        )
    )
    state = dict(state)
    state["rounds_spent"] = state["rounds_spent"] + 1

    if not new_queries:
        state["trace"] = [*state["trace"], {
            "round_index": round_index, "tier_before": tier_before,
            "resolved_before": resolved_before, "tier_after": tier_before,
            "resolved_after": resolved_before, "queries_tried": [],
            "missing_before": missing_before,
            "already_tried_before": list(already_tried_before),
            "exclude_turn_ids_before": list(exclude_turn_ids_before),
        }]
        return state, memory

    state["queries_issued"] = [*state["queries_issued"], *new_queries]
    found = search_paired_turns(
        paired_turns, new_queries, top_k=config.evidence_limit,
        exclude_turn_ids=frozenset(exclude_turn_ids_before),
        tier="expanded",
    )
    if found:
        memory = _extract_claims(
            backend, memory, participant_id=participant_id, topic=topic,
            new_turns=found, round_index=round_index,
        )
        state["turns"] = merge_retrieved(
            state["turns"], found, limit=config.evidence_limit
        )
    state["tier"] = overall_tier(state["turns"])

    if state["tier"] == tier_before:
        # Nothing new was found: the tier cannot have improved, and a fresh
        # sufficiency call would almost certainly repeat the same verdict —
        # skip it rather than spend a call confirming no change.
        verdict, missing = state["sufficient"], state["missing"]
    else:
        evidence_text = format_tiered_evidence(
            TieredEvidence(
                topic=topic, tier=state["tier"], turns=state["turns"],
                queries_issued=tuple(state["queries_issued"]),
            )
        )
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
    state["sufficient"] = verdict
    state["missing"] = missing or ""

    resolved_after = state["sufficient"] is True
    state["trace"] = [*state["trace"], {
        "round_index": round_index, "tier_before": tier_before,
        "resolved_before": resolved_before, "tier_after": state["tier"],
        "resolved_after": resolved_after, "queries_tried": list(new_queries),
        "missing_before": missing_before,
        "already_tried_before": list(already_tried_before),
        "exclude_turn_ids_before": list(exclude_turn_ids_before),
    }]
    return state, memory


def _counterfactual(score: int | None, sufficient: bool) -> str:
    """Deterministic "what evidence would change this" statement from the rubric.

    No model call: the rubric's day-count bands are the same four values for
    every PHQ-8 topic, so the adjacent-band condition is read straight off
    ``_DAY_BANDS`` rather than asked of anything that could hallucinate it.
    """
    if not sufficient or score is None:
        return (
            "no evidence was found; a credible report of how often this "
            "occurred over the last two weeks would let this item be scored "
            "instead of abstained"
        )
    parts: list[str] = []
    if score < 3:
        parts.append(
            f"evidence of {_DAY_BANDS[score + 1]} would raise the score to {score + 1}"
        )
    if score > 0:
        parts.append(
            f"evidence of only {_DAY_BANDS[score - 1]} would lower the score "
            f"to {score - 1}"
        )
    if not parts:
        return "the score is already at the rubric boundary in both directions"
    return "; ".join(parts)


def _build_criterion_map(
    assessed: dict[str, dict[str, Any]], memory: PatientEvidenceMemory
) -> list[dict[str, Any]]:
    """The Evidence-Criterion Map: one row per PHQ-8 topic, from the same
    evidence and scores already computed — nothing here can disagree with the
    prediction it explains, because it is read from it, not generated
    alongside it.
    """
    rows: list[dict[str, Any]] = []
    for topic, info in assessed.items():
        if info["abstained"]:
            status = "unresolved"
        elif info["present"] == "no":
            status = "refuted"
        else:
            status = "supported"
        records = memory.records_for(topic)
        contradictions = memory.contradictions_for(topic)
        confidence = (
            round(sum(record.confidence for record in records) / len(records), 3)
            if records else None
        )
        rows.append(
            {
                "criterion": topic,
                "status": status,
                "score": info["score"],
                "evidence_record_ids": [record.record_id for record in records],
                "quote": info["quote"],
                "confidence": confidence,
                "contradiction": bool(contradictions),
                "contradiction_details": [
                    {
                        "record_id_a": edge.record_id_a,
                        "record_id_b": edge.record_id_b,
                        "reason": edge.reason,
                    }
                    for edge in contradictions
                ],
                "counterfactual": _counterfactual(info["score"], not info["abstained"]),
            }
        )
    return rows


def assess_participant(
    backend: ChatBackend, sample_path: Path, config: PipelineConfig
) -> dict[str, Any]:
    started = time.time()
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    participant_id = str(sample.get("Participant_ID", sample_path.stem))
    reference = sample.get("phq8_scores", {})

    paired_turns = build_paired_turns(
        sample.get("real_interview", []), merge_consecutive=True
    )

    memory = PatientEvidenceMemory(participant_id=participant_id)
    topic_states: dict[str, dict[str, Any]] = {}
    for topic in config.topics:
        state, memory = _run_seed_round(
            backend, topic, config, paired_turns, participant_id, memory
        )
        topic_states[topic] = state

    lesson_scopes = frozenset(config.interview_lessons)
    budget = config.global_budget
    while budget > 0:
        candidates = [
            TopicAcquisitionState(
                topic=topic, tier=state["tier"], rounds_spent=state["rounds_spent"],
                sufficient=state["sufficient"],
                max_rounds_per_topic=config.max_rounds_per_topic,
            )
            for topic, state in topic_states.items()
        ]
        target = next_action(candidates, lesson_scopes=lesson_scopes)
        if target is None:
            break
        topic_states[target], memory = _run_expansion_round(
            backend, target, config, paired_turns, participant_id,
            topic_states[target], memory,
        )
        budget -= 1

    assessed: dict[str, dict[str, Any]] = {}
    for topic in config.topics:
        state = topic_states[topic]
        standard = config.standards.get(topic, {})
        evidence_text = format_tiered_evidence(
            TieredEvidence(
                topic=topic, tier=state["tier"], turns=state["turns"],
                queries_issued=tuple(state["queries_issued"]),
            )
        )
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
        for item in state["turns"]:
            allowed_ids.update(item.turn.citable_ids)
        bad_citations = tuple(
            turn_id for turn_id in parsed.turn_ids if turn_id not in allowed_ids
        )
        assessed[topic] = {
            "score": parsed.score,
            "abstained": not parsed.sufficient,
            "present": parsed.present,
            "frequency_basis": parsed.frequency_basis,
            "summary": parsed.reasoning,
            "quote": parsed.quote,
            "quote_grounded": quote_is_grounded(parsed.quote, evidence_text),
            "parse_failed": not parsed.ok,
            "rounds": state["rounds_spent"],
            "item_key": TOPIC_TO_ITEM_KEY.get(topic, ""),
            "evidence_tier": state["tier"],
            "evidence_turn_ids": list(parsed.turn_ids),
            "retrieved_turn_ids": [item.turn.turn_id for item in state["turns"]],
            "citation_mismatch": bool(bad_citations),
            "queries_issued": list(state["queries_issued"]),
            "evidence_text": evidence_text,
            "acquisition_trace": state["trace"],
        }

    scored_items = {
        info["item_key"]: int(info["score"])
        for info in assessed.values() if info["score"] is not None
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
        config.calibrator.apply(raw_total) if config.calibrator is not None
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
            1 for i in assessed.values()
            if i["score"] is not None and not i["quote_grounded"]
        ),
        parse_failures=sum(1 for i in assessed.values() if i["parse_failed"]),
        expansion_rounds=sum(i["rounds"] - 1 for i in assessed.values()),
        contradictions_found=len(memory.contradictions),
    )

    return {
        "participant_id": participant_id,
        "sample_path": str(sample_path),
        "model_name": backend.model,
        "pipeline": "psyvec-proposed-evidence-utility-adaptive",
        "prompt_version": PROMPT_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "assessed_item_count": len(scored_items),
        "abstained_item_count": len(abstained),
        "abstained_items": abstained,
        "imputation_applied": bool(imputed),
        "scorer_lessons_applied": sum(len(v) for v in config.lessons.values()),
        "interview_lessons_applied": sum(
            len(v) for v in config.interview_lessons.values()
        ),
        "raw_total_score": round(raw_total, 3),
        "calibrated_total_score": round(calibrated, 3),
        "total_predicted_score": final_total,
        "ground_truth_total": reference.get("PHQ8_Score"),
        "ground_truth_items": reference.get("items", {}),
        "predicted_category": categorize_score(final_total),
        "assessment_valid": True,
        "evidence_diagnostics": diagnostics,
        "topics": assessed,
        "evidence_memory": {
            "records": [asdict(record) for record in memory.records],
            "contradictions": [asdict(edge) for edge in memory.contradictions],
        },
        "criterion_map": _build_criterion_map(assessed, memory),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("openai", "ollama", "vllm", "local"), default="openai"
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None, help="defaults to $OPENAI_API_KEY")
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible endpoint; defaults per --backend (see llm_backend)",
    )
    parser.add_argument("--device", default=None, help="local backend only")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--global-budget", type=int, default=6,
        help="total extra expansion rounds shared across all 8 topics",
    )
    parser.add_argument(
        "--max-rounds-per-topic", type=int, default=3,
        help="hard cap on rounds (including the seed round) any one topic can consume",
    )
    parser.add_argument("--evidence-limit", type=int, default=8)
    parser.add_argument("--imputation-model", type=Path, default=None)
    parser.add_argument("--calibrator", type=Path, default=None)
    parser.add_argument(
        "--lessons", type=Path, default=None,
        help="promoted scorer-interpretation lessons (configs/fitted/lessons.json)",
    )
    parser.add_argument(
        "--interview-lessons", type=Path, default=None,
        help="promoted acquisition-strategy lessons "
        "(configs/fitted/interview_lessons.json)",
    )
    parser.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=None
    )
    parser.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
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
            "no imputation model: abstained items contribute 0 to the total"
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
            path for path in samples
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
        args.backend, args.model, api_key=args.api_key, base_url=args.base_url,
        ledger=ledger, device=args.device, enable_thinking=args.enable_thinking,
    )
    logger.info(
        "assessing %d participants | backend=%s model=%s concurrency=%d "
        "global_budget=%d max_rounds_per_topic=%d imputation=%s "
        "scorer_lessons=%d interview_lessons=%d",
        len(samples), args.backend, args.model, args.concurrency,
        config.global_budget, config.max_rounds_per_topic,
        config.imputer is not None,
        sum(len(v) for v in config.lessons.values()),
        sum(len(v) for v in config.interview_lessons.values()),
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
                "[%d/%d] %s pred=%s ref=%s abstained=%d contradictions=%d cost=$%.3f",
                done, len(samples), record["participant_id"],
                record["total_predicted_score"], record["ground_truth_total"],
                record["abstained_item_count"],
                record["evidence_diagnostics"]["contradictions_found"], ledger.cost,
            )

    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "pipeline": "psyvec-proposed-evidence-utility-adaptive",
        "prompt_version": PROMPT_VERSION,
        "backend": args.backend,
        "model_name": args.model,
        "data_dir": str(args.data_dir),
        "config": {
            "global_budget": config.global_budget,
            "max_rounds_per_topic": config.max_rounds_per_topic,
            "evidence_limit": config.evidence_limit,
            "imputation_model": str(args.imputation_model or ""),
            "calibrator": str(args.calibrator or ""),
            "lessons": str(args.lessons or ""),
            "interview_lessons": str(args.interview_lessons or ""),
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
