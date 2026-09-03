#!/usr/bin/env python3
"""Run end-to-end psychological assessment on 1 sample using lightweight Qwen2.5-0.5B-Instruct."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


class QwenInferenceEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None) -> None:
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading %s on device '%s'...", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float32 if self.device == "mps" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully!")

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
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=0.9 if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[:, model_inputs.input_ids.shape[1] :]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return response


def parse_score_and_summary(raw_text: str) -> tuple[int, str]:
    """Extract integer score (0-3) and summary from LLM response."""
    # Try JSON block
    json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            score = int(data.get("score", 0))
            score = max(0, min(3, score))
            summary = str(data.get("summary", "")).strip()
            return score, summary
        except Exception:
            pass

    # Regex fallback
    score_match = re.search(r'"?score"?\s*:\s*(\d+)', raw_text, re.IGNORECASE)
    if score_match:
        score = max(0, min(3, int(score_match.group(1))))
    else:
        num_match = re.search(r"\b([0-3])\b", raw_text)
        score = int(num_match.group(1)) if num_match else 0

    return score, raw_text.strip()


def parse_necessity_score(raw_text: str) -> int:
    """Extract necessity score (0, 1, or 2)."""
    match = re.search(r"\b([0-2])\b", raw_text)
    if match:
        return int(match.group(1))
    return 0


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
    verbose: bool = True,
) -> dict[str, Any]:
    def vprint(*args: Any, **kwargs: Any) -> None:
        if verbose:
            print(*args, **kwargs)

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
        engine = QwenInferenceEngine(model_name=model_name)

    # Step 1: Basic Information Gathering
    vprint("\n" + "=" * 50)
    vprint("STEP 1: Basic Information Collection")
    vprint("=" * 50)

    initial_question = (
        "Hello, I am your dedicated psychological assistant. Before we begin the PHQ-8 assessment, "
        "could you please tell me your basic information: age, gender, and occupation?"
    )
    vprint(f"[Interviewer]: {initial_question}")

    client_demographics = engine.generate(
        system_prompt=client_system_prompt,
        user_prompt=(
            f"The interviewer asks:\n'{initial_question}'\n"
            "Provide your age, gender, and occupation based on the context in the format 'Age: <age>, Gender: <gender>, Occupation: <occupation>'."
        ),
        max_new_tokens=40,
        temperature=0.1,
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
            client_reply = engine.generate(
                system_prompt=client_system_prompt,
                user_prompt=client_prompt,
                max_new_tokens=50,
                temperature=0.2,
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
            necessity = parse_necessity_score(necessity_resp)
            depth += 1

            if necessity == 0 or depth >= max_depth:
                break

            # Generate follow-up question
            followup_prompt = f"""Topic: {topic_name}
Patient response: {client_reply}
Ask a short clinical follow-up question to clarify the frequency or severity over the past two weeks."""
            question = engine.generate(
                system_prompt="You are an empathetic psychological interviewer. Generate a short clinical follow-up question.",
                user_prompt=followup_prompt,
                max_new_tokens=40,
                temperature=0.3,
            )
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
        score, summary = parse_score_and_summary(scorer_resp)
        vprint(f"  --> Assigned Score: {score} | Reason: {summary}")

        assessed_topics[topic_name] = {
            "score": score,
            "summary": summary,
            "rounds": depth,
            "item_key": TOPIC_TO_ITEM_KEY.get(topic_name, ""),
        }

    # Step 3: Compute Summary Report
    total_score = sum(t["score"] for t in assessed_topics.values())
    category = categorize_score(total_score)

    vprint("\n" + "=" * 50)
    vprint("STEP 3: Summary Report & Comparison")
    vprint("=" * 50)

    gt_total = ground_truth.get("PHQ8_Score", "N/A")
    gt_items = ground_truth.get("items", {})

    vprint(f"Participant ID: {participant_id}")
    vprint(f"Demographics:   {client_demographics}\n")
    vprint(f"{'Topic Name':<30} | {'Pred Score':<10} | {'GT Score':<10} | Summary Basis")
    vprint("-" * 80)
    for topic_name, info in assessed_topics.items():
        item_key = info["item_key"]
        gt_item_score = gt_items.get(item_key, "-")
        vprint(f"{topic_name:<30} | {info['score']:<10} | {str(gt_item_score):<10} | {info['summary']}")

    vprint("-" * 80)
    vprint(f"TOTAL PHQ-8 SCORE: Predicted = {total_score}/24 | Ground Truth = {gt_total}/24")
    vprint(f"Depression Severity: {category}")
    vprint("=" * 80)

    result_data = {
        "participant_id": participant_id,
        "sample_path": str(sample_path),
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "demographics": client_demographics,
        "total_predicted_score": total_score,
        "ground_truth_total": gt_total,
        "predicted_category": category,
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluations",
        help="Directory to save assessment results",
    )
    args = parser.parse_args()

    start_time = time.time()
    result = run_full_sample_assessment(sample_path=args.sample_path, model_name=args.model_name)
    elapsed = time.time() - start_time

    # Save to output file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    participant_id = result["participant_id"]
    output_file = args.output_dir / f"{participant_id}_evaluation.json"
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nExecution finished in {elapsed:.2f} seconds.")
    print(f"📁 Output JSON saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()

