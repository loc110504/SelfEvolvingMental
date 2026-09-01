Status: PARTIALLY COMPLETE

# Phase 7 promotion report

## Implemented offline

- Added an immutable accepted-policy registry, paired candidate-versus-baseline
  regression records, and promotion refusal when manifest, leakage, parameter,
  safety, quality, or non-inferiority checks fail.
- Added compare-and-swap promotion, an append-only checkpoint audit trail, and
  pointer rollback that preserves registered policy artifacts.
- Added the lesson promotion monitoring audit and append-only conflict
  quarantine described in the Phase 5 report.

## Deliberately not changed

No code under the pinned `AgentMental/` baseline was changed. No candidate
adapter, live model, dataset, participant simulator, training, or DPO run has
been used. No training and no DPO run has happened, and every check is against
injected fakes rather than a live model, dataset, or participant simulator.

## Reproducible checks

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_agentmental_baseline.py
ruff check src tests/unit
mypy --strict src/psyvec
```

Run from the repository root: **98 tests passed**; the baseline verifier
reported **9 PASS lines**; Ruff reported no issues in **32 source files**; and
strict mypy completed successfully.

## Gate status

**Phase 7: partially complete.** The accepted-policy registry, paired
regression, compare-and-swap promotion, pointer rollback and the promotion
audit exist and a failed candidate cannot promote. NOT done here: P7-3 memory
ON/OFF and P7-4 IAR are being implemented by another agent in parallel — say
in progress, not finished. Phase 7's real gate additionally depends on Phase
6, whose trainer ticket P6-3 is blocked because `torch`, `peft` and `trl` are
not installed. No candidate adapter has ever existed, so no promotion has been
exercised against a real artifact.

No training and no DPO run has happened, and every check is against injected
fakes rather than a live model, dataset, or participant simulator.
