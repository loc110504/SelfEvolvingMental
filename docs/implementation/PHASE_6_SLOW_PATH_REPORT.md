Status: BLOCKED

# Phase 6 slow-path report

## Implemented offline

- Added `src/psyvec/training/partitions.py`, which builds role-local preference-pair partitions and records refusal reasons for mixed split ids, duplicate `contrast_id`, chosen-equals-rejected pairs, and a partition below the caller-supplied `minimum_role_pairs`.
- Added `src/psyvec/training/verification.py`, whose `verify_parameter_diff` confirms that exactly one declared role adapter id changed while `base_model_hash` did not.
- The partition and manifest-difference behavior is covered by offline unit tests in `tests/unit/test_training_prep.py`.

## Deliberately not changed

The pinned `AgentMental/` baseline remains pristine. Ticket P6-3, the TRL/PEFT role-DPO trainer, is not implemented and cannot be: `torch`, `peft` and `trl` are not installed and were deliberately not added. No training has run. No DPO run has happened. No adapter weights have ever existed, been written, or been loaded. `verify_parameter_diff` checks a *claimed* manifest difference; it never compares real weights.

## Reproducible checks

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_agentmental_baseline.py
ruff check src tests/unit tests/integration
mypy --strict src/psyvec
```

Run from the repository root: **127 tests passed**; the baseline verifier reported **11 PASS lines**, including SHA-256 verification for **23 tracked upstream files**; Ruff found no issues across **69 Python files**; and strict mypy completed successfully across **45 source files**.

## Gate status

Phase 6 is **BLOCKED**. Ticket P6-3, the TRL/PEFT role-DPO trainer, is not implemented and cannot be: `torch`, `peft` and `trl` are not installed and were deliberately not added. What exists is the offline preparation and verification logic only. **No training has run. No DPO run has happened. No adapter weights have ever existed, been written, or been loaded.** `verify_parameter_diff` checks a *claimed* manifest difference; it never compares real weights. Doc 07 also makes P6 conditional on the Phase 2 routing gate, whose own P2-3 Transformers backend is blocked for the same reason.
