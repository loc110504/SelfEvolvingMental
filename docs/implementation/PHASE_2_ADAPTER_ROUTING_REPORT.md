Status: OFFLINE CRITERIA MET / INTEGRATION BLOCKED

# Phase 2 adapter routing report

## Implemented offline

- Added immutable `AdapterManifest` and `AdapterRegistry`, keyed by
  `(accepted_policy_id, role)`, plus `RoleModelRouter` in
  `src/psyvec/model/adapters.py`.
- Defined and tested the four logical aliases: `psyvec-interviewer`,
  `psyvec-evaluator`, `psyvec-scorer`, and `psyvec-updater`.
- The router rejects an unknown policy, unknown or unsupported role, a
  wrong-role manifest, base-model-hash mismatch, and a non-accepted
  `candidate` adapter. It stamps the resolved adapter id and hash into
  `ResponseProvenance`.
- `adapters_enabled=False` bypasses adapter resolution and delegates directly
  to the injected backend as the rollback path. The four-role mapping and
  rejection paths are covered offline in `tests/unit/test_role_router.py`.

## Deliberately not changed

The pinned `AgentMental/` baseline remains pristine. Ticket P2-3 (a real
Transformers backend), P2-4 (the vLLM spike), and P2-5 (concurrency traces)
are deferred and NOT done. No `torch`, `transformers`, or `vllm` was installed
or added. Routing is proven against an injected backend only; no adapter
weights were ever loaded. No live model, dataset, or participant simulator was
used, and no training or DPO run has happened.

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

**Phase 2: offline criteria met, integration BLOCKED.** The Doc 07 gate
("four requests prove the correct logical adapter; disabling the adapter
backend rolls back") is demonstrated offline. Deferred and NOT done: ticket
P2-3 a real Transformers backend, P2-4 the vLLM spike, and P2-5 concurrency
traces — no `torch`, `transformers` or `vllm` is installed and none was added.
Routing is proven against an injected backend only; no adapter weights were
ever loaded.

Every gate above is proven against injected fakes and stubs, never a live
model, dataset, or participant simulator. No training and no DPO run has
happened.
