Status: PARTIALLY COMPLETE

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

## Deliberately not changed

Ticket P4-4, the hard-state miner and branch-generation strategies, is in
progress by another agent and is not finished. No simulated participant was
ever run because the approved endpoint and dataset still blocking Phase 0 are
needed. The `AgentMental/` baseline was not modified. No live model, dataset,
or participant simulator was used, and no training or DPO run has happened.

## Reproducible checks

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_agentmental_baseline.py
ruff check src tests/unit
mypy --strict src/psyvec
```

Run from the repository root: **80 tests passed**; the baseline verifier
reported **9 PASS lines**; Ruff reported no issues in **26 source files**; and
strict mypy completed successfully.

## Gate status

**Phase 4: partially complete.** The spine, validity checks, utility and local
contrast verification exist and one full contrast is traceable to state,
profile, action, utility and seed. NOT done: ticket P4-4 the hard-state miner
and branch generation strategies (another agent is implementing it in parallel
— say it is in progress, not finished). No simulated participant was ever run,
because that needs the approved endpoint and dataset still blocking Phase 0.

Every gate above is proven against injected fakes and stubs, never a live
model, dataset, or participant simulator. No training and no DPO run has
happened.
