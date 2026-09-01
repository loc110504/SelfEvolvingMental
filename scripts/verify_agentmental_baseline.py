#!/usr/bin/env python3
"""Verify the pinned AgentMental baseline without importing or running it."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "characterization"
    / "agentmental-baseline.json"
)


class BaselineVerificationError(RuntimeError):
    """Raised when the local upstream no longer matches the approved baseline."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the approved deterministic UTF-8 JSON representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(agentmental_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(agentmental_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def assigned_expression_names(tree: ast.AST, target_name: str) -> set[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        ):
            return {
                child.id
                for child in ast.walk(node.value)
                if isinstance(child, ast.Name)
            }
    raise BaselineVerificationError(f"assignment not found: {target_name}")


def function_name_uses(tree: ast.AST, function_name: str) -> set[str]:
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise BaselineVerificationError(f"function not found: {function_name}")
    return {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def function_argument_names(tree: ast.AST, function_name: str) -> list[str]:
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise BaselineVerificationError(f"function not found: {function_name}")
    return [argument.arg for argument in function.args.args]


def verify_source_characterization(agentmental_root: Path) -> list[str]:
    checks: list[str] = []
    assessment_source = (agentmental_root / "src" / "assessment.py").read_text(
        encoding="utf-8"
    )
    assessment_tree = ast.parse(assessment_source)

    forbidden_policy_inputs = {"scale_scores", "real_interview"}
    for payload in (
        "question_payload",
        "necessity_payload",
        "scoring_payload",
        "summary_payload",
    ):
        names = assigned_expression_names(assessment_tree, payload)
        leaked = names & forbidden_policy_inputs
        if leaked:
            raise BaselineVerificationError(
                f"hidden supervision reaches {payload}: {sorted(leaked)}"
            )
    checks.append("policy payload AST contains no label/source-transcript variables")

    simulator_tree = ast.parse(
        (agentmental_root / "src" / "generate_response.py").read_text(
            encoding="utf-8"
        )
    )
    simulator_arguments = function_argument_names(
        simulator_tree, "generate_mock_response"
    )
    if "scale_scores" not in simulator_arguments:
        raise BaselineVerificationError(
            "simulator label-shaped argument changed; refresh characterization"
        )
    simulator_uses = function_name_uses(simulator_tree, "generate_mock_response")
    if "scale_scores" in simulator_uses:
        raise BaselineVerificationError(
            "simulator now consumes scale_scores; leakage review required"
        )
    checks.append("simulator label argument is present but unused")

    scoring_standard_names = assigned_expression_names(
        simulator_tree, "scoring_standard_str"
    )
    if "scoring_standard" not in scoring_standard_names:
        raise BaselineVerificationError("simulator scoring rubric assignment changed")
    for node in ast.walk(simulator_tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "prompt"
            for target in node.targets
        ):
            continue
        if "scoring_standard_str" in {
            child.id
            for child in ast.walk(node.value)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }:
            raise BaselineVerificationError(
                "simulator scoring rubric now reaches a prompt; leakage review required"
            )
    checks.append("simulator scoring rubric is computed but absent from prompts")

    if "real_interview" not in simulator_uses:
        raise BaselineVerificationError("simulator no longer consumes real_interview")
    system_prompt_names = assigned_expression_names(simulator_tree, "system_prompt")
    if "interview_history" not in system_prompt_names:
        raise BaselineVerificationError(
            "simulator transcript no longer reaches the system prompt"
        )
    checks.append("simulator transcript is consumed and reaches the system prompt")

    if "max_depth = 3" not in assessment_source:
        raise BaselineVerificationError("three-question topic cap changed")
    if "example_questions_str" not in assessment_source:
        raise BaselineVerificationError("question-example quirk changed")
    question_names = assigned_expression_names(assessment_tree, "question_payload")
    if "example_questions_str" in question_names:
        raise BaselineVerificationError(
            "example questions are now wired; refresh characterization intentionally"
        )
    checks.append("three-question cap and omitted-example quirk preserved")
    return checks


def verify(manifest_path: Path = DEFAULT_MANIFEST) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstream = manifest["upstream"]
    agentmental_root = PROJECT_ROOT / "AgentMental"
    checks: list[str] = []

    actual_commit = git(agentmental_root, "rev-parse", "HEAD")
    if actual_commit != upstream["commit"]:
        raise BaselineVerificationError(
            f"commit mismatch: expected {upstream['commit']}, got {actual_commit}"
        )
    actual_tree = git(agentmental_root, "rev-parse", "HEAD^{tree}")
    if actual_tree != upstream["tree"]:
        raise BaselineVerificationError(
            f"tree mismatch: expected {upstream['tree']}, got {actual_tree}"
        )
    if git(agentmental_root, "status", "--porcelain"):
        raise BaselineVerificationError("AgentMental worktree is not pristine")
    checks.append("pinned commit/tree and pristine worktree")

    actual_origin = git(agentmental_root, "remote", "get-url", "origin")
    if actual_origin != upstream["origin"]:
        raise BaselineVerificationError(
            f"origin mismatch: expected {upstream['origin']}, got {actual_origin}"
        )
    checks.append("read-only upstream origin identity")

    expected_files = upstream["tracked_files_sha256"]
    actual_files = set(git(agentmental_root, "ls-files").splitlines())
    if actual_files != set(expected_files):
        raise BaselineVerificationError("tracked file inventory changed")
    for relative_path, expected_hash in expected_files.items():
        actual_hash = sha256_file(agentmental_root / relative_path)
        if actual_hash != expected_hash:
            raise BaselineVerificationError(
                f"SHA-256 mismatch for {relative_path}: {actual_hash}"
            )
    checks.append(f"SHA-256 verified for {len(expected_files)} tracked files")

    config_entries = json.loads(
        (agentmental_root / "src" / "OAI_CONFIG_LIST").read_text(encoding="utf-8")
    )
    if not config_entries or any(
        entry.get("api_key") != "your_api_key_here"
        or entry.get("base_url") != "your_api_base_url_here"
        for entry in config_entries
    ):
        raise BaselineVerificationError(
            "tracked baseline config contains non-placeholder credentials"
        )
    checks.append("tracked model configuration contains placeholders only")

    phq8 = json.loads(
        (agentmental_root / "scales" / "PHQ-8.json").read_text(encoding="utf-8")
    )
    standards = json.loads(
        (agentmental_root / "scales" / "scoring_standards.json").read_text(
            encoding="utf-8"
        )
    )["PHQ-8"]
    expected_phq8 = manifest["characterization"]["phq8"]
    if len(phq8) != expected_phq8["topic_count"]:
        raise BaselineVerificationError("PHQ-8 topic count changed")
    if set(phq8) != set(standards):
        raise BaselineVerificationError("question and scoring topics differ")
    if any(
        len(examples) != expected_phq8["question_examples_per_topic"]
        for examples in phq8.values()
    ):
        raise BaselineVerificationError("PHQ-8 question-example count changed")
    checks.append("PHQ-8 question and scoring assets agree")

    checks.extend(verify_source_characterization(agentmental_root))

    canonical = canonical_json_bytes(manifest)
    checks.append(f"manifest canonical SHA-256 {hashlib.sha256(canonical).hexdigest()}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        checks = verify(args.manifest)
    except (BaselineVerificationError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    for check in checks:
        print(f"PASS: {check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
