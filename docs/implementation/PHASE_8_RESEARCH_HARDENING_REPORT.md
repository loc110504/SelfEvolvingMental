Status: PARTIALLY COMPLETE / FINAL EVALUATION BLOCKED

# Phase 8 research hardening report

## Implemented offline

- Added `src/psyvec/research/splits.py`: frozen `SplitManifest` records with content hashes; a `SplitRegistry` that refuses participant overlap across splits and any unauthorized use; the Doc 06 `PERMITTED_USES` allow-list per partition; logged final-test access; and `RunManifest.manifest_hash` as the cache-reuse gate.
- Added `src/psyvec/evaluation/metrics.py`: participant-level total-score MAE retaining completed, failed, and invalid counts; matched-budget enforcement; and caller-seeded paired bootstrap and permutation measurement machinery.
- Added `src/psyvec/privacy/`: redaction by default with a recorded raw-text opt-in, stable irreversible participant pseudonyms, and a release audit that returns every finding.
- The split, metric, and privacy contracts are unit-tested offline in `tests/unit/test_research_splits.py`, `tests/unit/test_metrics.py`, and `tests/unit/test_privacy.py`.

## Deliberately not changed

The pinned `AgentMental/` baseline remains pristine. No live model, real dataset, participant simulator run, training, DPO run, or adapter weights have ever existed. The statistics machinery deliberately returns **measured statistics only and never a significance verdict**, because Doc 06 forbids inferring significance without measured participant-level outputs. No threshold or alpha is hard-coded anywhere.

## Reproducible checks

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_agentmental_baseline.py
ruff check src tests/unit tests/integration
mypy --strict src/psyvec
```

Run from the repository root: **127 tests passed**; the baseline verifier reported **11 PASS lines**, including SHA-256 verification for **23 tracked upstream files**; Ruff found no issues across **69 Python files**; and strict mypy completed successfully across **45 source files**.

## Gate status

Phase 8 is **partially complete, final evaluation BLOCKED**. The split protocol, reproducibility manifest, metric and statistics machinery, and the release privacy audit exist and are unit-tested offline. NOT done: ticket P8-4, the robustness and ablation budget. Not possible yet: the frozen-protocol final-test run itself, which needs the authorized dataset and approved endpoint still blocking Phase 0, plus preregistered endpoints and thresholds. The statistics machinery deliberately returns **measured statistics only and never a significance verdict**, because Doc 06 forbids inferring significance without measured participant-level outputs, and no threshold or alpha is hard-coded anywhere.
