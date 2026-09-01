Status: OFFLINE CRITERIA MET / LIVE REPLAY BLOCKED

# Phase 4 experience report

## Implemented offline

- Added the offline experience spine in `src/psyvec/evolution/experience.py`:
  frozen `ParticipantProfile` with a deterministic `profile_hash` and separate
  known/unknown slots; `ExperienceRecord`; `ExperienceBuffer`; and
  `BranchCandidate` plus shared-context branch invariants.
- `ExperienceRecord` must cite its own event pre-state hash and agree with the
  event role. The buffer enforces its split, rejects label-shaped policy input,
  and has an `enabled=False` rollback path. Branches must share state, profile,
  policy, simulator version, and seed.
- Added utility, validation, and local contrast verification in
  `src/psyvec/evolution/contrast.py`. Unsafe-branch utility fails closed;
  validation retains every invalid reason, including `unsupported_fact:<key>`;
  local verification rejects insufficient valid branches and ties below the
  pair margin; accepted contrasts can produce preference pairs.
- One full offline contrast is covered from state, profile, action, utility,
  and seed in the unit tests.
- **P4-3 replay determinism (added later, offline).** `src/psyvec/evolution/replay.py`
  fixes a replay's identity in `ReplayRequest.replay_key` (state, profile,
  policy, simulator version, decoding and seed) and hashes each result, so two
  replays sharing a key must agree. `ReplayEngine.assert_deterministic` and
  `replay_manifest` fail on a disagreeing key; `determinism_report` reports
  without raising. Session isolation is enforced separately: only `fresh` and
  `explicit_reset` sessions may run, and a `clear_memory_prompt` session is
  refused with its own message, because Doc 06 says a "clear memory" prompt does
  not establish isolation. Covered by `tests/unit/test_replay.py`, including the
  case where the engine refuses to call the simulator at all.

## Deliberately not changed

No simulated participant was ever run because the approved endpoint and dataset
still blocking Phase 0 are needed: the replay engine proves determinism against
an injected simulator, never a live one. Ticket P4-4, the hard-state miner and
branch-generation strategies, was completed separately in
`src/psyvec/evolution/mining.py`. The `AgentMental/` baseline was not modified. No live model, dataset,
or participant simulator was used, and no training or DPO run has happened.

## Reproducible checks

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_agentmental_baseline.py
ruff check src tests/unit tests/integration
mypy --strict src/psyvec
```

Run from the repository root: **164 tests passed**; the baseline verifier
reported **11 PASS lines**; Ruff reported no issues; and strict mypy completed
successfully across **48 source files**.

## Gate status

**Phase 4: offline criteria met, live replay BLOCKED.** The spine, validity
checks, utility and local contrast verification exist; one full contrast is
traceable to state, profile, action, utility and seed; P4-4's miner and branch
generation exist; and P4-3's deterministic replay proof and session-isolation
refusal now exist too. NOT possible yet: any replay against a real simulator,
which needs the approved endpoint and dataset still blocking Phase 0.

Every gate above is proven against injected fakes and stubs, never a live
model, dataset, or participant simulator. No training and no DPO run has
happened.
