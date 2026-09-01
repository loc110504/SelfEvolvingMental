Status: PARTIALLY COMPLETE / GATE BLOCKED

# Phase 0 baseline report

## Scope completed

- Pinned and verified the pristine AgentMental clone at commit
  `0e2fc8ff27552845ae743e3373351120a96e216b` and tree
  `10b3807b12e2284e74dab5c21435014153e6290d`.
- Recorded SHA-256 for every upstream tracked file in a deterministic JSON
  fixture.
- Added an offline smoke verifier for commit, tree, worktree cleanliness,
  origin identity, file inventory, file hashes, placeholder credentials, PHQ-8
  assets, and selected source-level invariants.
- Added characterization tests for the upstream stopping matrix and invalid
  score fallbacks without importing AutoGen or calling a model.
- Added AST-based inspection proving that current policy payload expressions do
  not reference `scale_scores` or `real_interview`.
- No AgentMental business logic was modified. No dataset was downloaded or
  opened, and no network/model request was made.

## Reproducible checks

Run from the project root:

```sh
python3 scripts/verify_agentmental_baseline.py
python3 -m unittest discover -s tests -p 'test_*.py'
git -C AgentMental status --short
```

The smoke checks require only Python's standard library and Git. The observed
development interpreter during capture was Python 3.14.6. This does not claim
that upstream dependencies support Python 3.14; the runnable legacy environment
remains to be pinned.

Current offline result: all 116 tests in the full suite pass, including 6 upstream
characterization tests; SHA-256 values for all 23 upstream tracked files are
verified; the AgentMental nested worktree remains pristine.

## Split and secret inventory

- No participant data or split manifest exists in the pinned clone. Only the
  three upstream data preparation/download scripts are present.
- No evolve/development/final-test partition is inferred or created. Dataset
  authorization remains false in the machine-readable fixture.
- `src/OAI_CONFIG_LIST` contains placeholder keys and URLs only. The verifier
  fails if those tracked placeholders are replaced by apparent deployment
  values.
- Direct-client environment variable names and insecure placeholder defaults in
  `memory.py` and `generate_response.py` are recorded upstream risks. The
  verifier never reads or prints process environment secrets.
- Root `.env`, `artifacts/`, `logs/`, `data/raw/`, and `data/processed/` are
  ignored to reduce accidental commits; this is not a substitute for an
  approved data-access policy.

## Characterized upstream quirks

These observations are regression evidence, not correctness requirements for
the new system:

1. Invalid, malformed, or missing scoring output may silently become score 0.
   PsyVEC must replace this with an explicit failed event.
2. Necessity score 2 always continues until the enclosing three-question cap;
   score 1 continues only while fewer than two questions have been asked.
3. Each PHQ-8 topic has three example questions, but the computed examples are
   omitted from `question_payload`.
4. `scoring_standard_str` is built from the topic rubric but never reaches a
   simulator prompt. A future unused-variable cleanup that wires it in would
   leak the scoring rubric into the participant simulator.
5. `scale_scores` is passed into the automated simulator interface even though
   the pinned function does not read it. The actual leakage surface is instead
   the deliberately transcript-conditioned simulator: `real_interview` is
   flattened into `interview_history` and interpolated into its system prompt.
   This is upstream's design, not a clone defect; Phase 4's profile-first
   simulator must replace that data channel and must not copy `scale_scores`.
6. SummaryAgent may revise item scores and is therefore not a read-only
   reporter. The target design must split ReportingAgent from the true Updater.
7. AutoGen roles share one global model config; memory and simulator create
   separate direct OpenAI clients.
8. Logs may contain prompts, answers, memory, model responses, and demographic
   data. New runtime logging must redact raw PII by default.
9. The README entrypoint, empty `data_dir`, placeholder paths, undeclared
   dependencies, and missing endpoint/data prevent an authorized end-to-end run.

## Gate status

Phase 0 gate: **BLOCKED**.

The static freeze/audit and offline characterization portions pass. The required
end-to-end baseline cannot yet be documented as runnable because all of the
following remain intentionally absent:

- an approved baseline model and endpoint configuration;
- a compatible, pinned legacy Python environment;
- an authorized dataset or a separately approved synthetic baseline fixture.

Do not use DAIC-WOZ merely to clear this gate. Once those inputs are approved,
capture an end-to-end trace and its content hashes, rerun this suite, and update
the gate status. Phase 1 implementation must not be declared complete before
baseline parity can be evaluated against that trace.

## Deviations

- P0-2, creation of an authorized environment, is unmet because a compatible,
  pinned legacy Python environment has not yet been approved.
- The trace-capture half of P0-3 is unmet because an approved baseline model and
  endpoint configuration and an authorized dataset or separately approved
  synthetic baseline fixture remain absent. The inability to run end to end is
  recorded as a blocker rather than replacing real evidence with a fabricated
  fixture.

## Deferred findings

The following is explicitly deferred by directive because it does not block the
next phase.

1. **Governance.** Doc 07 requires Checkpoints A/B/C to be recorded before a
   phase starts. No report records checkpoint status.

## New TBDs

- Select a supported legacy Python version and lock strategy after dependency
  compatibility is tested.
- Approve a baseline model/endpoint or a synthetic fixture protocol.
- Confirm AgentMental and any dataset license obligations before distribution or
  data ingestion.
