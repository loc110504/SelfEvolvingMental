# PsyVEC implementation status

Resuming in a fresh session? Read [HANDOFF.md](HANDOFF.md) first.

| Phase | What landed | Gate state |
|---|---|---|
| 0 | Pinned baseline audit and offline characterization | BLOCKED (external) |
| 1 | Model/backend abstraction and compatibility ports | BLOCKED (external) |
| 2 | Offline role-adapter registry and routing | partially complete; P2-3/P2-4/P2-5 BLOCKED (dependency) |
| 3 | Typed decision state, separated roles, and case memory | partially complete |
| 4 | Offline experience spine, contrast verification, and replay determinism | offline criteria met |
| 5 | Lesson-memory audit, poisoning checks, and quarantine | offline criteria met |
| 6 | Role preference partitions and manifest-difference verification | BLOCKED (dependency) |
| 7 | Policy registry, regression records, and promotion audit | partially complete |
| 8 | Split protocol, metrics/statistics, manifests, ablation budget, and privacy audit | offline criteria met |

Row 3 is `partially complete` rather than `offline criteria met` because Doc 07's
Phase 3 gate also requires a dual-run rollback against the legacy engine, which
needs the still-blocked Phase 0 baseline trace. See the
[Phase 3 report](PHASE_3_STATE_AND_ROLES_REPORT.md).

## Offline work completed after the first handoff

- Phase 4 P4-3: `src/psyvec/evolution/replay.py` — deterministic replay keys and
  Doc 06 session isolation (a "clear memory" prompt is refused).
- Phase 5 P5-4: `LessonStore.merge` and `find_near_duplicates` —
  near-duplicate lessons fold into one entry that records `merged_from`.
- Phase 8 P8-4: `src/psyvec/research/budget.py` — matched ablation budgets and
  seed robustness, measurements only.
- `src/psyvec/privacy/runtime.py` is wired into
  `psyvec.integration.legacy_bridge`, so the runtime logging path redacts by
  default (Doc 00 quirk 7).

## Scope of evidence

Every gate is demonstrated against injected fakes and constructed records — no live model, no real dataset, no participant simulator run, no training, no DPO, and no adapter weights have ever existed. `AgentMental/` is byte-identical to the pinned commit and its verifier reports every check passing.

## Blockers

External blockers: an approved legacy Python environment; an approved baseline model and endpoint; an authorized dataset or an approved synthetic fixture; and the Doc 07 Checkpoints A/B/C, which are still unrecorded.

Library blockers: `torch`, `transformers`, `peft`, `trl`, and `vllm` are absent. This blocks Phase 2 P2-3/P2-4/P2-5 and Phase 6 P6-3.

## Offline integration

`tests/integration/test_offline_pipeline.py` composes every layer end to end offline: split, interview, score, update, report, experience, mining, branches, contrast, preference pair, lesson, promotion, and run manifest. This is the Doc 06 integration level, still against fakes.

## Deferred findings index

- Phase 0: [baseline report](PHASE_0_BASELINE_REPORT.md)
- Phase 1: [backend report](PHASE_1_BACKEND_REPORT.md)
- Phase 3: [state and roles report](PHASE_3_STATE_AND_ROLES_REPORT.md)
- Phase 4: [experience report](PHASE_4_EXPERIENCE_REPORT.md)
