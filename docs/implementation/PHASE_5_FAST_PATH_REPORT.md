Status: OFFLINE CRITERIA MET

# Phase 5 fast-path report

## Implemented offline

- Added immutable paired-benefit observations and an audit verdict for promoted
  lessons. The caller supplies `minimum_transfer_support` and
  `transfer_win_rate`; insufficient support remains distinct from retirement,
  and a non-positive mean paired benefit retires the lesson.
- Added poisoning checks that reuse the state label-key rejection contract for
  label-shaped lesson text and flag contrast support that traces to one source
  id as `single_source_support`.
- Added same-scope contradiction resolution that quarantines both promoted
  lessons using `LessonStore`. Store transitions remain append-only, so the
  previous promoted versions remain auditable and retrieval excludes both.
- Added four offline unit tests for monitoring, support, poisoning, and
  conflict quarantine.
- **P5-4 merge (added later, offline).** `lesson_similarity` scores two lessons
  by token overlap of their procedural text, returning zero across roles or
  scopes; `find_near_duplicates` groups current entries above a
  **caller-supplied** similarity threshold; and `LessonStore.merge` folds
  duplicates into a primary, writing `merged_from` (previously an unused field)
  plus the union of source contrast ids and validation result ids. Duplicates
  are retired with `retired:merged_into:<primary>` rather than deleted, so the
  store stays append-only. Covered by `tests/unit/test_lesson_merge.py`.

## Deliberately not changed

No code under the pinned `AgentMental/` baseline was changed. No threshold was
hard-coded: transfer support, win rate and the merge similarity threshold
remain caller-supplied,
development-selected values. No training and no DPO run has happened, and every
check is against injected fakes rather than a live model, dataset, or
participant simulator.

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

**Phase 5: offline criteria met.** The Doc 07 gate ("an unvalidated lesson is
impossible to retrieve") holds: `retrieve` returns only `promoted` entries.
Everything is proven against constructed records — no lesson was ever distilled
from a real model run, and no retrieval ever fed a live assessment.

No training and no DPO run has happened, and every check is against injected
fakes rather than a live model, dataset, or participant simulator.
