Status: OFFLINE CRITERIA MET / FINAL EVALUATION BLOCKED

# Phase 8 research hardening report

## Implemented offline

- Added `src/psyvec/research/splits.py`: frozen `SplitManifest` records with content hashes; a `SplitRegistry` that refuses participant overlap across splits and any unauthorized use; the Doc 06 `PERMITTED_USES` allow-list per partition; logged final-test access; and `RunManifest.manifest_hash` as the cache-reuse gate.
- Added `src/psyvec/evaluation/metrics.py`: participant-level total-score MAE retaining completed, failed, and invalid counts; matched-budget enforcement; and caller-seeded paired bootstrap and permutation measurement machinery.
- Added `src/psyvec/privacy/`: redaction by default with a recorded raw-text opt-in, stable irreversible participant pseudonyms, and a release audit that returns every finding.
- **P8-4 robustness and ablation budget (added later, offline).** `src/psyvec/research/budget.py` adds `AblationArm` (one reference arm plus ablations, each with a question and model-call budget and a shared seed set), `assert_matched_budget` (a differing budget or seed set is a hard failure, not a reported difference), `assert_budget_respected` (no arm may overspend), `measure_seed_robustness` (mean, minimum, maximum and spread across seeds) and `measure_ablation_effects` (per-seed and mean difference against the reference). Like the rest of the statistics machinery it returns measured quantities only — no verdict, no threshold, no alpha. Covered by `tests/unit/test_ablation_budget.py`.
- **Privacy wired into the runtime path (added later, offline).** `src/psyvec/privacy/runtime.py` adds `RedactingRuntimeLog`, and `psyvec.integration.legacy_bridge` now accepts it on both call-site builders. Prompts, completions, memory and demographics reach a sink hashed, a participant id is replaced by its pseudonym before logging, and `raw_text_opt_in` records whether raw text was explicitly allowed. This turns Doc 00 quirk 7 from addressed-in-principle into an actual redacted default. Covered by `tests/unit/test_privacy_runtime.py`, which also asserts a redacted record passes `audit_release`.
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

Run from the repository root: **164 tests passed**; the baseline verifier reported **11 PASS lines**, including SHA-256 verification for **23 tracked upstream files**; Ruff found no issues; and strict mypy completed successfully across **48 source files**.

## Gate status

Phase 8 is **offline criteria met, final evaluation BLOCKED**. The split protocol, reproducibility manifest, metric and statistics machinery, and the release privacy audit exist and are unit-tested offline. P8-4, the robustness and ablation budget, is now implemented offline. Not possible yet: the frozen-protocol final-test run itself, which needs the authorized dataset and approved endpoint still blocking Phase 0, plus preregistered endpoints and thresholds. The statistics machinery deliberately returns **measured statistics only and never a significance verdict**, because Doc 06 forbids inferring significance without measured participant-level outputs, and no threshold or alpha is hard-coded anywhere.
