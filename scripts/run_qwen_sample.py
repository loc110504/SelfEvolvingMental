#!/usr/bin/env python3
"""Run end-to-end psychological assessment on 1 sample using lightweight Qwen2.5-0.5B-Instruct."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.response_parsing import (  # noqa: E402
    parse_necessity_score,
    parse_score_and_summary,
    parse_summary_and_updated_scores,
    strip_reasoning,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("qwen-runner")


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


def _fmt_total(total: int | None) -> str:
    return "UNSCORED (parse failure)" if total is None else f"{total}/24"


def categorize_score(total_score: int) -> str:
    if total_score <= 4:
        return "None / Minimal depression (0-4)"
    elif total_score <= 9:
        return "Mild depression (5-9)"
    elif total_score <= 14:
        return "Moderate depression (10-14)"
    elif total_score <= 19:
        return "Moderately severe depression (15-19)"
    else:
        return "Severe depression (20-24)"


REASONING_MODEL_HINTS = ("qwen3", "r1", "deepseek-r1", "qwq", "thinking", "reason")


def looks_like_reasoning_model(model_name: str) -> bool:
    """Heuristic: does this model emit chain-of-thought before answering?"""
    lowered = model_name.lower()
    return any(hint in lowered for hint in REASONING_MODEL_HINTS)


class QwenInferenceEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        token_scale: float = 1.0,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.client = None
        self.token_scale = max(1.0, float(token_scale))
        self.enable_thinking = enable_thinking
        # Set by every generate() call so callers can detect budget truncation.
        self.last_truncated = False
        self.truncation_count = 0

        if looks_like_reasoning_model(model_name):
            if enable_thinking is None:
                self.enable_thinking = False
            if token_scale <= 1.0:
                self.token_scale = 8.0
            logger.warning(
                "'%s' looks like a reasoning model: enable_thinking=%s, "
                "token_scale=%.1f (chain-of-thought would otherwise consume the "
                "whole token budget and truncate the answer).",
                model_name,
                self.enable_thinking,
                self.token_scale,
            )

        if api_base_url:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=api_base_url,
                api_key=api_key or "EMPTY",
            )
            logger.info(
                "Connected to OpenAI-compatible endpoint (%s) with model '%s'",
                api_base_url,
                model_name,
            )
        else:
            if device is None:
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            else:
                self.device = device

            logger.info("Loading %s on device '%s'...", model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float32 if self.device == "mps" else torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(
                self.device
            )
            self.model.eval()
            logger.info("Model loaded successfully!")

    def _budget(self, max_new_tokens: int) -> int:
        return max(1, int(math.ceil(max_new_tokens * self.token_scale)))

    def _record_truncation(self, truncated: bool) -> None:
        self.last_truncated = truncated
        if truncated:
            self.truncation_count += 1

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.2,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        budget = self._budget(max_new_tokens)

        if self.client is not None:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": budget,
            }
            if temperature > 0.0:
                kwargs["temperature"] = temperature
            else:
                kwargs["temperature"] = 0.0
            if self.enable_thinking is not None:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                }

            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content or ""
            self._record_truncation(getattr(choice, "finish_reason", None) == "length")
            return content.strip()

        template_kwargs: dict[str, Any] = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except TypeError:
            # Chat template does not support the thinking switch.
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=budget,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=0.9 if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[:, model_inputs.input_ids.shape[1] :]
        self._record_truncation(int(new_tokens.shape[1]) >= budget)
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return response


DEFAULT_SCALE_FILE = (
    PROJECT_ROOT / "configs" / "scales" / "PHQ-8.json"
    if (PROJECT_ROOT / "configs" / "scales" / "PHQ-8.json").is_file()
    else PROJECT_ROOT / "AgentMental" / "scales" / "PHQ-8.json"
)
DEFAULT_STANDARDS_FILE = (
    PROJECT_ROOT / "configs" / "scales" / "scoring_standards.json"
    if (PROJECT_ROOT / "configs" / "scales" / "scoring_standards.json").is_file()
    else PROJECT_ROOT / "AgentMental" / "scales" / "scoring_standards.json"
)


def run_full_sample_assessment(
    sample_path: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    scale_file: Path = DEFAULT_SCALE_FILE,
    standards_file: Path = DEFAULT_STANDARDS_FILE,
    engine: QwenInferenceEngine | None = None,
    api_base_url: str | None = None,
    api_key: str | None = None,
    enable_memory_update: bool = False,
    token_scale: float = 1.0,
    enable_thinking: bool | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    def vprint(*args: Any, **kwargs: Any) -> None:
        if verbose:
            print(*args, **kwargs)

    parse_failures: list[dict[str, str]] = []

    def record_failure(stage: str, reason: str, raw_text: str) -> None:
        """Log a parse failure loudly instead of silently defaulting."""
        parse_failures.append({"stage": stage, "reason": reason, "raw_response": raw_text})
        logger.warning("PARSE FAILURE [%s]: %s", stage, reason)

    logger.info("=" * 60)
    logger.info("Starting Full Sample Assessment with %s", model_name)
    logger.info("Sample File: %s", sample_path)
    logger.info("=" * 60)

    # 1. Load Data & Scales
    sample_data = json.loads(sample_path.read_text(encoding="utf-8"))
    participant_id = sample_data.get("Participant_ID", "Unknown")
    real_interview = sample_data.get("real_interview", [])
    ground_truth = sample_data.get("phq8_scores", {})

    topics_dict = json.loads(scale_file.read_text(encoding="utf-8"))
    scoring_standards = json.loads(standards_file.read_text(encoding="utf-8"))["PHQ-8"]

    # 2. Build Client Interview Context for Simulation
    interview_history = ""
    for turn in real_interview[:50]:  # Limit context for speed & compact prompt
        speaker = turn.get("roleName", "Speaker")
        text = turn.get("content", "")
        interview_history += f"{speaker}: {text}\n"

    client_system_prompt = f"""
    You are acting as the client in a psychological consultation.
Here is your actual background dialogue from an interview:
{interview_history}

Please respond truthfully and reasonably in the first person (I, me) as this participant.
Keep your answer concise (under 40 words)."""

    # 3. Initialize Engine
    if engine is None:
        engine = QwenInferenceEngine(
            model_name=model_name,
            api_base_url=api_base_url,
            api_key=api_key,
            token_scale=token_scale,
            enable_thinking=enable_thinking,
        )

    truncation_baseline = engine.truncation_count

    # Step 1: Basic Information Gathering
    vprint("\n" + "=" * 50)
    vprint("STEP 1: Basic Information Collection")
    vprint("=" * 50)

    initial_question = (
        "Hello, I am your dedicated psychological assistant. Before we begin the PHQ-8 assessment, "
        "could you please tell me your basic information: age, gender, and occupation?"
    )
    vprint(f"[Interviewer]: {initial_question}")

    demographics_raw = engine.generate(
        system_prompt=client_system_prompt,
        user_prompt=(
            f"The interviewer asks:\n'{initial_question}'\n"
            "Provide your age, gender, and occupation based on the context in the format 'Age: <age>, Gender: <gender>, Occupation: <occupation>'."
        ),
        max_new_tokens=40,
        temperature=0.1,
    )
    client_demographics = strip_reasoning(demographics_raw)
    if not client_demographics:
        record_failure(
            "demographics",
            "reasoning-only response (likely truncated before the answer)",
            demographics_raw,
        )
    vprint(f"[Participant {participant_id}]: {client_demographics}")

    # Step 2: Assessment across 8 PHQ-8 Topics
    vprint("\n" + "=" * 50)
    vprint("STEP 2: PHQ-8 Topic-by-Topic Assessment (8 Topics)")
    vprint("=" * 50)

    assessed_topics: dict[str, dict[str, Any]] = {}
    dialogue_transcript: list[dict[str, str]] = []

    for topic_idx, (topic_name, example_questions) in enumerate(topics_dict.items(), 1):
        vprint(f"\n>>> Topic {topic_idx}/8: [{topic_name}]")
        standard_text = json.dumps(scoring_standards.get(topic_name, {}), indent=2)

        topic_history = []
        depth = 0
        max_depth = 2

        # Initial topic question
        question = example_questions[0]
        vprint(f"  [Q1]: {question}")

        while depth < max_depth:
            # Client answers
            client_prompt = (
                f"Topic: {topic_name}\n"
                f"Question: {question}\n"
                "Answer concisely (1-2 sentences) reflecting your symptoms or state over the past two weeks."
            )
            client_reply_raw = engine.generate(
                system_prompt=client_system_prompt,
                user_prompt=client_prompt,
                max_new_tokens=50,
                temperature=0.2,
            )
            client_reply = strip_reasoning(client_reply_raw)
            if not client_reply:
                record_failure(
                    f"client_reply[{topic_name}]",
                    "reasoning-only response (likely truncated before the answer)",
                    client_reply_raw,
                )
            vprint(f"  [A{depth+1}]: {client_reply}")

            topic_history.append({"question": question, "answer": client_reply})
            dialogue_transcript.append({"role": "interviewer", "topic": topic_name, "content": question})
            dialogue_transcript.append({"role": "participant", "topic": topic_name, "content": client_reply})

            # Check necessity of follow-up question
            history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in topic_history])
            necessity_prompt = f"""Topic: {topic_name}
History:
{history_text}

Rate necessity for further questioning (0 = sufficient, 1 = somewhat necessary, 2 = very necessary).
Return only the single number 0, 1, or 2."""
            necessity_resp = engine.generate(
                system_prompt="You are a clinical assessment evaluator. Return only 0, 1, or 2.",
                user_prompt=necessity_prompt,
                max_new_tokens=5,
                temperature=0.0,
            )
            necessity_parse = parse_necessity_score(necessity_resp)
            if not necessity_parse.ok:
                record_failure(
                    f"necessity[{topic_name}]",
                    necessity_parse.failure_reason,
                    necessity_resp,
                )
            # 0 = stop asking: the conservative default when unparseable.
            necessity = necessity_parse.score if necessity_parse.score is not None else 0
            depth += 1

            if necessity == 0 or depth >= max_depth:
                break

            # Generate follow-up question
            followup_prompt = f"""Topic: {topic_name}
Patient response: {client_reply}
Ask a short clinical follow-up question to clarify the frequency or severity over the past two weeks."""
            followup_raw = engine.generate(
                system_prompt="You are an empathetic psychological interviewer. Generate a short clinical follow-up question.",
                user_prompt=followup_prompt,
                max_new_tokens=40,
                temperature=0.3,
            )
            followup_question = strip_reasoning(followup_raw)
            if not followup_question:
                record_failure(
                    f"followup_question[{topic_name}]",
                    "reasoning-only response (likely truncated before the answer)",
                    followup_raw,
                )
                break
            question = followup_question
            vprint(f"  [Q{depth+1}]: {question}")

        # Scorer evaluates this topic
        history_str = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in topic_history])
        scorer_prompt = f"""Topic: {topic_name}
Dialogue history:
{history_str}

Scoring standard:
{standard_text}

Score this topic from 0 to 3 based on the standard. Output JSON:
{{"score": <0, 1, 2, or 3>, "summary": "<one sentence basis>"}}"""

        scorer_resp = engine.generate(
            system_prompt="You are a professional psychological scale scorer. Output valid JSON only.",
            user_prompt=scorer_prompt,
            max_new_tokens=60,
            temperature=0.0,
        )
        score_parse = parse_score_and_summary(scorer_resp)
        if not score_parse.ok:
            record_failure(f"scorer[{topic_name}]", score_parse.failure_reason, scorer_resp)
            vprint(f"  --> SCORE PARSE FAILED: {score_parse.failure_reason}")
        else:
            vprint(
                f"  --> Assigned Score: {score_parse.score} | Reason: {score_parse.summary}"
            )

        assessed_topics[topic_name] = {
            "score": score_parse.score,
            "summary": score_parse.summary,
            "parse_failed": not score_parse.ok,
            "rounds": depth,
            "item_key": TOPIC_TO_ITEM_KEY.get(topic_name, ""),
        }
        if not score_parse.ok:
            assessed_topics[topic_name]["raw_response"] = scorer_resp

    scored_topics = [t for t in assessed_topics.values() if t["score"] is not None]
    all_topics_scored = len(scored_topics) == len(assessed_topics)
    initial_total_score = (
        sum(int(t["score"]) for t in scored_topics) if all_topics_scored else None
    )
    updated_scores: dict[str, dict[str, Any]] = {}
    overall_summary = ""

    # Optional Memory Update (Updater / SummaryAgent step)
    if enable_memory_update:
        vprint("\n" + "=" * 50)
        vprint("STEP 2.5: Global Memory Update (Updater / SummaryAgent)")
        vprint("=" * 50)

        history_str = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in dialogue_transcript
        )
        initial_scores_str = "\n".join(
            f"- {t}: {info['score'] if info['score'] is not None else 'UNPARSED'} pts "
            f"(Reason: {info['summary']})"
            for t, info in assessed_topics.items()
        )
        memory_prompt = f"""Full Consultation Dialogue:
{history_str}

Initial Topic Scores:
{initial_scores_str}

Analyze the complete dialogue history and memory across all topics. Make reasonable minor adjustments to the initial topic scores (0-3) if warranted by the overall clinical picture.
Output strictly in JSON format:
{{
  "summary": "<overall assessment summary>",
  "updated_scores": {{
    "<Topic Name>": {{"score": <0, 1, 2, or 3>, "reason": "<clinical adjustment basis>"}}
  }}
}}"""
        updater_resp = engine.generate(
            system_prompt=(
                "You are an expert clinical psychological supervisor and diagnostic updater. "
                "Review the full consultation history and adjust topic scores if needed. Output JSON only."
            ),
            user_prompt=memory_prompt,
            max_new_tokens=350,
            temperature=0.0,
        )
        updater_parse = parse_summary_and_updated_scores(
            updater_resp, list(topics_dict.keys())
        )
        if not updater_parse.ok:
            record_failure("memory_update", updater_parse.failure_reason, updater_resp)
            vprint(f"  [Memory Update] PARSE FAILED: {updater_parse.failure_reason}")
        overall_summary = updater_parse.summary
        updated_scores = updater_parse.updated_scores

        if updated_scores:
            vprint("  [Memory Update Adjustments]:")
            for topic, update_info in updated_scores.items():
                old_score = assessed_topics[topic]["score"]
                assessed_topics[topic]["parse_failed"] = False
                new_score = update_info["score"]
                assessed_topics[topic]["initial_score"] = old_score
                assessed_topics[topic]["updated_score"] = new_score
                assessed_topics[topic]["score"] = new_score
                assessed_topics[topic]["update_reason"] = update_info["reason"]
                vprint(
                    f"    • {topic}: {old_score} -> {new_score} "
                    f"({update_info['reason']})"
                )
        else:
            vprint("  [Memory Update]: Initial scores confirmed without changes.")

    # Step 3: Compute Summary Report
    scored_topics = [t for t in assessed_topics.values() if t["score"] is not None]
    all_topics_scored = len(scored_topics) == len(assessed_topics)
    final_total_score = (
        sum(int(t["score"]) for t in scored_topics) if all_topics_scored else None
    )
    category = categorize_score(final_total_score) if all_topics_scored else "UNSCORED"
    assessment_valid = all_topics_scored

    vprint("\n" + "=" * 50)
    vprint("STEP 3: Summary Report & Comparison")
    vprint("=" * 50)

    gt_total = ground_truth.get("PHQ8_Score", "N/A")
    gt_items = ground_truth.get("items", {})

    vprint(f"Participant ID: {participant_id}")
    vprint(f"Demographics:   {client_demographics}\n")
    if enable_memory_update:
        vprint(f"{'Topic Name':<30} | {'Init':<6} | {'Final':<6} | {'GT':<6} | Summary Basis")
        vprint("-" * 85)
        for topic_name, info in assessed_topics.items():
            item_key = info["item_key"]
            gt_item_score = gt_items.get(item_key, "-")
            init_s = info.get("initial_score", info["score"])
            final_s = info["score"]
            init_str = "FAIL" if init_s is None else str(init_s)
            final_str = "FAIL" if final_s is None else str(final_s)
            vprint(
                f"{topic_name:<30} | {init_str:<6} | {final_str:<6} | {str(gt_item_score):<6} | {info['summary']}"
            )
        vprint("-" * 85)
        vprint(f"INITIAL PHQ-8 SCORE: {_fmt_total(initial_total_score)}")
        vprint(f"FINAL PHQ-8 SCORE:   {_fmt_total(final_total_score)} (Memory Update Applied)")
    else:
        vprint(f"{'Topic Name':<30} | {'Pred Score':<10} | {'GT Score':<10} | Summary Basis")
        vprint("-" * 80)
        for topic_name, info in assessed_topics.items():
            item_key = info["item_key"]
            gt_item_score = gt_items.get(item_key, "-")
            score_str = "FAIL" if info["score"] is None else str(info["score"])
            vprint(f"{topic_name:<30} | {score_str:<10} | {str(gt_item_score):<10} | {info['summary']}")
        vprint("-" * 80)
        vprint(
            f"TOTAL PHQ-8 SCORE: Predicted = {_fmt_total(final_total_score)} "
            f"| Ground Truth = {gt_total}/24"
        )

    vprint(f"Depression Severity: {category}")
    if parse_failures:
        vprint("")
        vprint(f"!! {len(parse_failures)} PARSE FAILURE(S) — RESULT IS NOT USABLE:")
        for failure in parse_failures:
            vprint(f"   - [{failure['stage']}] {failure['reason']}")
        logger.error(
            "Participant %s: %d parse failure(s); assessment_valid=%s",
            participant_id,
            len(parse_failures),
            assessment_valid,
        )
    sample_truncations = engine.truncation_count - truncation_baseline
    if sample_truncations:
        vprint(
            f"!! {sample_truncations} generation(s) hit the token budget "
            "(raise --token-scale)."
        )
    vprint("=" * 80)

    result_data = {
        "participant_id": participant_id,
        "sample_path": str(sample_path),
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "demographics": client_demographics,
        "memory_update_enabled": enable_memory_update,
        "initial_total_score": initial_total_score,
        "total_predicted_score": final_total_score,
        "ground_truth_total": gt_total,
        "predicted_category": category,
        "assessment_valid": assessment_valid,
        "parse_failure_count": len(parse_failures),
        "parse_failures": parse_failures,
        "truncated_generations": sample_truncations,
        "overall_summary": overall_summary,
        "updated_scores": updated_scores,
        "topics": assessed_topics,
        "dialogue_transcript": dialogue_transcript,
    }
    return result_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1 full sample assessment with Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_daic_woz" / "dev" / "302.json",
        help="Path to processed DAIC-WOZ sample JSON file",
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
        "--enable-memory-update",
        action="store_true",
        help="Enable global Memory Update (SummaryAgent/Updater) to adjust initial topic scores based on full dialogue memory without LoRA",
    )
    parser.add_argument(
        "--token-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply every max_new_tokens budget by this factor. Reasoning models "
            "(Qwen3.x, R1, QwQ) need >= 8 or the chain-of-thought eats the budget and "
            "the answer is truncated. Auto-set to 8 for detected reasoning models."
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
        default=PROJECT_ROOT / "results" / "evaluations",
        help="Directory to save assessment results",
    )
    args = parser.parse_args()

    start_time = time.time()
    result = run_full_sample_assessment(
        sample_path=args.sample_path,
        model_name=args.model_name,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        enable_memory_update=args.enable_memory_update,
        token_scale=args.token_scale,
        enable_thinking=args.enable_thinking,
    )
    elapsed = time.time() - start_time

    # Save to output file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    participant_id = result["participant_id"]
    output_file = args.output_dir / f"{participant_id}_evaluation.json"
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nExecution finished in {elapsed:.2f} seconds.")
    print(f"📁 Output JSON saved to: {output_file.resolve()}")

    if not result["assessment_valid"]:
        print(
            f"\nFAILED: {result['parse_failure_count']} parse failure(s); "
            "no usable PHQ-8 total was produced. "
            "Inspect 'parse_failures[].raw_response' in the output JSON."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()

