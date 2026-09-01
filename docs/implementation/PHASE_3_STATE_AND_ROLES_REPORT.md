Status: OFFLINE CRITERIA MET / DUAL-RUN ROLLBACK NOT DEMONSTRATED

# Phase 3 state and roles report

## Implemented offline

- Added typed immutable state and event records in `src/psyvec/state/`:
  `EvidenceSlot`, `TopicState`, `AssessmentState`, `RoleAction`, `RoleOutput`,
  `DecisionEvent`, `ProvenanceRef`, `canonical_hash`, `replace_topic`, and
  `DecisionEventEngine`.
- Enforced monotonic revisions; source-turn citations for known evidence;
  existing evidence for proposed scores; updater-only revision of an existing
  proposed score; read-only reporting passes; and refusal of
  `FORBIDDEN_LABEL_KEYS` in policy-visible payloads.
- Split the mutating `Updater` from the read-only `Reporter` in
  `src/psyvec/roles/`, fixing the upstream `SummaryAgent` defect recorded as
  quirk 5 in the Phase 0 report.
- Added single-participant `CaseMemory`, `AccessPolicy`, and an adapter that
  refuses a wrong `case_id`, unauthorized role, or expired memory.
- Reproduced the characterized stopping matrix through typed decision events:
  necessity 2 continues to the three-question cap; necessity 1 continues only
  while fewer than two questions have been asked; 0 stops.

## Deliberately not changed

The `AgentMental/` baseline was not modified. No dual-run against the legacy
engine was performed because the Phase 0 baseline trace remains unavailable.
No live model, dataset, or participant simulator was used, and no training or
DPO run has happened.

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

**Phase 3: offline criteria met, dual-run rollback NOT demonstrated.** Typed
pre/post-state and provenance exist for every decision. The Doc 07 gate also
asks for a dual-run against the legacy engine, which needs the Phase 0 baseline
trace and is therefore still blocked.

Every gate above is proven against injected fakes and stubs, never a live
model, dataset, or participant simulator. No training and no DPO run has
happened.

## Deferred findings

The following are deferred by directive because they do not block the next
phase.

1. **`Reporter` label check is false assurance.**
   `src/psyvec/roles/reporter.py` calls `reject_label_keys` on a dictionary
   whose keys are hard-coded field names (`"topic_id"`, `"final_score"`,
   `"status"`, and others). Those can never be in `FORBIDDEN_LABEL_KEYS`, so
   the check can never fire. It reads like a label canary but is dead
   validation — a real canary would inspect the values or the source state, not
   literal field names.
2. **`DecisionEventEngine` numbers events per engine, not per case.**
   `decision_index` comes from a counter owned by the engine instance, so two
   cases sharing one engine would interleave their indexes. Single-case use is
   fine today.
