# Handoff — resume point for a fresh session

Written at the end of the offline implementation push. Read this first, then
[STATUS.md](STATUS.md) for the phase table, then the individual phase reports.

Everything below is verifiable in the repo. Nothing here is an approval, a result, or
evidence of a run.

## Verify you are where this document says

Run from the repository root:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'    # expect 127 tests, OK
python3 scripts/verify_agentmental_baseline.py          # expect 11 PASS lines
ruff check src tests/unit tests/integration             # expect clean
mypy --strict src/psyvec                                # expect 45 source files, no issues
git -C AgentMental status --short                        # expect empty
```

The verifier's manifest hash must still be
`f52bee3bf9721749be188793719e9969f657867823e00ccfdc1cd37e5467af09`. If it changed,
someone edited the pinned clone and the Phase 0 evidence is void — stop and investigate
before doing anything else.

## FIRST DECISION: the work is untracked

`src/`, `tests/`, `scripts/`, `configs/`, `docs/`, `pyproject.toml` and `AgentMental/` are
all **untracked** in `main` (`git status`). 127 tests' worth of implementation exists only
as working-tree files plus 17 worktree branches. Nothing was committed to `main` because
the user never asked for a commit.

Decide with the user before other work:

1. Commit the working tree to a branch off `main` (recommended), or
2. Keep it untracked and continue, accepting the loss risk.

`AgentMental/` is a nested clone with its own `.git`. Adding it to the parent repo creates
a gitlink with no `.gitmodules`, and a fresh checkout of that gitlink is an empty
directory — which breaks `scripts/verify_agentmental_baseline.py`. Options: add it to
`.gitignore` and document the clone step, or make it a real submodule pinned at
`0e2fc8ff27552845ae743e3373351120a96e216b`. Do not commit it as a bare gitlink.

## What exists

Twelve packages under `src/psyvec/`, all offline, all against injected fakes:

| Package | Role |
|---|---|
| `config.py` | layered backend config (defaults < env < manifest < overrides), secret-safe `provenance_hash` |
| `model/` | `ModelBackend` contract, legacy AutoGen bridge, OpenAI-compatible adapter, assembly root, parity traces, role adapter registry + router |
| `integration/` | runtime redirect of the three upstream call shapes behind the feature flag |
| `state/` | typed `AssessmentState` / `DecisionEvent` records and the decision event engine |
| `roles/` | true `Updater` (only writer of revised scores) and read-only `Reporter` |
| `memory/` | single-participant `CaseMemory`, access policy, adapter |
| `assessment/` | characterized topic stopping matrix driven through typed events |
| `evolution/` | frozen profile, experience buffer, branch invariants, hard-state mining, utility, validity, local contrast, preference pairs |
| `lessons/` | distill, source-excluded validation, versioned store, bounded retrieval, poisoning and negative-transfer audit |
| `training/` | role partitions and manifest parameter-diff verification (no trainer) |
| `policy/` | paired regression, compare-and-swap promotion, pointer rollback, audit |
| `evaluation/` | memory ON/OFF, IAR, participant-level MAE, seeded bootstrap and permutation |
| `research/` | frozen splits, permitted-use matrix, final-test access log, run manifest |
| `privacy/` | redaction by default, participant pseudonyms, release audit |

`tests/integration/test_offline_pipeline.py` composes every layer end to end in one test.

## Remaining work, by what blocks it

### A. Blocked on human approval or data (no code will unblock these)

1. An approved, pinned legacy Python environment. Dependency compatibility for the
   upstream `requirements.txt` was never tested; the observed development interpreter was
   Python 3.14.6, which is **not** a claim that upstream supports it.
2. An approved baseline model and endpoint, or an approved synthetic fixture protocol.
3. An authorized dataset or approved synthetic baseline fixture. **Do not use DAIC-WOZ
   merely to clear the Phase 0 gate** — the Phase 0 report says so explicitly.
4. Doc 07 Checkpoints A, B and C are still unrecorded, and Doc 07 requires them before a
   phase starts. Ask the user how they want this recorded.
5. License obligations for AgentMental and any dataset, before distribution or ingestion.

Once 1-3 exist: capture the frozen end-to-end baseline trace and its content hashes, then
Phase 1 item 4 (before/after characterization equivalence with an approved tolerance) and
the Phase 3 dual-run rollback become possible. Both are currently the only reason those
gates are not closed.

### B. Blocked on absent libraries

`torch`, `transformers`, `peft`, `trl`, `vllm` are not installed and were deliberately not
added. This blocks:

- Phase 2 P2-3 (real Transformers backend), P2-4 (vLLM spike), P2-5 (concurrency traces).
- Phase 6 P6-3 (TRL/PEFT role-DPO trainer). No adapter weights have ever existed.

Everything else in Phases 2 and 6 is done. When the libraries are approved, the router and
`verify_parameter_diff` are the seams they plug into — neither needs redesign.

### C. Implementable offline right now

- **Phase 4 P4-3 replay determinism.** A replay engine proving same state + same profile +
  same seed yields the same result, and that a "clear memory" prompt does not establish
  isolation (Doc 06 requires a fresh or explicitly reset session, verified by tests).
- **Phase 8 P8-4** robustness and ablation budget.
- **Phase 5** lesson merge for near-duplicates. `LessonMemoryEntry.merged_from` exists but
  nothing writes it.
- Wiring `psyvec.privacy` into the runtime logging path. It exists but nothing calls it, so
  upstream quirk 7 (logs may contain prompts, answers, memory and demographics) is
  addressed in principle only.

## Deferred findings

Recorded, non-blocking, listed here so none is lost. Each lives under
`## Deferred findings` in its phase report.

- **Phase 0** — governance: Checkpoints A/B/C unrecorded.
- **Phase 1** — `timeout_seconds` is read into config and never enforced by any backend;
  upstream `cache_seed: 42` is dropped and absent from `provenance_hash`;
  `ModelResponse.__post_init__` makes a legitimately empty completion indistinguishable
  from a failure; `OpenAICompatibleBackend` detects timeouts by exception class-name
  substring.
- **Phase 3** — `roles/reporter.py` calls `reject_label_keys` on a dict of hard-coded field
  names, so that check can never fire: it reads like a label canary but is dead validation;
  `DecisionEventEngine.decision_index` counts per engine instance, not per case, so two
  cases sharing one engine would interleave indexes.
- Also worth knowing: `select_backend` is no longer used by the assembly root (the flag is
  evaluated once in `build_model_backend`); it survives as the standalone rollback selector.
  And `_PROJECT_ROOT` for the default manifest is `Path(__file__).resolve().parents[2]`,
  correct for a source checkout but wrong if the package is installed to site-packages.

## Ground rules that produced this state

Keep these unless the user changes them.

- **`AgentMental/` is byte-identical to the pinned commit and must stay that way.** Phase 1
  integration was built as a runtime redirect in `src/psyvec/integration/` precisely so the
  clone's SHA-256 inventory keeps verifying. Do not "just edit" the upstream files.
- Speed-first directive in force: minimum code, minimum tests, no refactors, no
  architecture for later, defer any finding that does not break the current or next phase.
- No fabricated approvals, parity evidence, or results. A blocker is recorded as BLOCKED.
- Every threshold that Doc 04/06 marks development-selected stays caller-supplied. Nothing
  hard-codes `hard_state_threshold`, `pair_margin`, `minimum_role_pairs`,
  `transfer_win_rate`, alpha, or a significance verdict.

## Worktree branches

Seventeen branches hold the per-task commits, all merged into the working tree already.
They are kept for diff review and can be removed once the work is committed:

`psyvec-backend-assembly`, `psyvec-trace-completeness`, `phase2-router-registry`,
`phase-docs`, `p3-roles`, `p3-casememory`, `p3-stopping`, `p4-miner`, `p5-fastpath`,
`docs-phases`, `p6-preference`, `p7-onoff`, `p5-audit-docs`, `p8-privacy`, `p8-metrics`,
`p0-evidence`, `docs-final` — each under `VanwPham/`.

`git worktree list` shows their paths under `~/orca/workspaces/SelfEvolvingMental/`. Each
holds a `TASK.md` (its brief) and, where a review round happened, a `FIXES.md`. Those two
files were never committed.

## Delegation pattern that worked

Six waves, seventeen sub-agents, one worktree each, `gpt-5.6-terra` (`high` for the first
wave, `medium` after). The pattern:

1. `orca worktree create --repo id:<repoId> --name <task> --no-parent --setup skip`
2. rsync the working tree in (excluding `.git`, `AgentMental`, caches), symlink
   `AgentMental` to the main clone, commit a baseline snapshot so the worker's diff is clean
3. `orca terminal create --worktree id:<repoId>::<path> --command 'codex --model
   gpt-5.6-terra -c model_reasoning_effort="medium"'`
4. write the brief to `TASK.md`, then `orca terminal send --text "Read TASK.md ..." --enter`
5. poll the branch head in a background shell loop; review the diff; send a `FIXES.md`
   round if needed; copy files into the main tree; re-run all four gates

Give each worker a disjoint file set and say explicitly which files another agent owns —
that is what kept seventeen parallel agents from colliding.

**Gotcha:** Codex starts its MCP servers for ~30s after launch, and a `--enter` sent during
that window lands in the composer without submitting. `orca terminal wait --for tui-idle`
can also report idle during that startup. Read the terminal to confirm the prompt actually
submitted; if it is sitting in the input line, send an empty `--text "" --enter`. Two of
seventeen workers hit this.
