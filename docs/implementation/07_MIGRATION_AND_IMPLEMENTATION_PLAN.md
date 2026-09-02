Status: DRAFT FOR CHECKPOINT C
Source commit: `0e2fc8ff27552845ae743e3373351120a96e216b`
Depends on: Docs 00-06
Blocks: coding authorization and implementation scheduling
Open decisions: integration layout; baseline environment; ticket ownership/estimates; all blocking TBDs from prior docs

# Migration and implementation plan

## Migration posture

**[DECISION]** Work from the pinned local upstream source; never deliver through upstream `origin`. First create characterization seams around current behavior, then migrate responsibility into a new package. Do not rewrite protocol and model/simulator/evolution simultaneously. Each phase is independently reversible to the last accepted baseline/policy/memory version.

## Current-to-target map

| Current source | Target | Treatment |
|---|---|---|
| `src/agents.py:QuestionAgent` | `agents/interviewer.py` | Extract prompt intent, typed I/O, role router |
| `NecessityAgent` + `is_necessary` | `agents/evaluator.py` + stop policy | Preserve characterized logic, make output/state typed |
| `ScoringAgent` | `agents/scorer.py` | Typed evidence/score basis and backend |
| `SummaryAgent` | `agents/reporter.py` | Retain final read-only reporting; remove state mutation authority |
| Memory holistic reassessment | `agents/updater.py` | Use as prompt evidence only; new typed patch semantics |
| `assessment.perform_assessment` | `assessment/engine.py` | Incrementally extract engine/state/events |
| `MemoryGraph` | `memory/case_memory.py` | Migrate case-local material; no cross-case reuse |
| `generate_mock_response` | profile/simulator/replay modules | Rewrite; no full-transcript runtime prompt |
| `config.py`/direct OpenAI calls | model backends/router/registry + configs | Wrap, then replace globals |
| loaders/data scripts | typed dataset/profile preparation | Refactor with split/lineage contracts |
| `result.py` | evaluation runner/metrics/audit | Rewrite as manifest-driven evaluation |
| absent | evolution/training/lesson modules | New components |

Doc 00 owns the full file-by-file reuse classification.

## Proposed target tree

```text
src/psyvec/
├── agents/{interviewer,evaluator,scorer,updater,reporter}.py
├── assessment/{engine,state,events,validation}.py
├── model/{backend,role_router,adapter_registry,legacy_autogen_backend,
│          transformers_backend,vllm_backend}.py
├── memory/{case_memory,lesson_memory,retrieval}.py
├── simulator/{profile,profile_builder,participant,replay,validity}.py
├── evolution/{buffer,hard_state,branching,utility,local_verification,
│              attribution,lesson_distillation,transfer_validation,coordinator}.py
├── training/{preference_dataset,dpo,adapters,regression_gate}.py
├── evaluation/{metrics,memory_audit,regression,runner,audit_report}.py
├── data/{contracts,splits,lineage}.py
└── config.py
configs/{base,development,training,evaluation}/
tests/{unit,integration,characterization,regression,leakage}/
scripts/{prepare_profiles,run_assessment,run_evolution,train_role,evaluate}.py
artifacts/  # ignored; manifests define layout
```

Exact package root is **[TBD]**: this repository may vendor a preserved AgentMental subtree or establish an integration package adjacent to the read-only clone. No upstream file is edited until approved.

## Dependency changes

Phase 0 first repairs/pins baseline runtime dependencies. Core adds a schema/CLI/test/packaging toolchain selected by maintainers. Phase 2 adds Transformers, PEFT, Accelerate and optional vLLM/bitsandbytes extras. Phase 6 adds TRL. Evaluation declares numpy/scipy/scikit-learn and approved statistical/report dependencies. Runtime, training, simulation, and evaluation extras remain separable with a lockfile and software bill of materials.

## Ordered phases, tests, gates, and rollback

| Phase | Changes and tickets | Required tests | Gate / rollback |
|---|---|---|---|
| 0 Freeze/audit | P0-1 pin SHA/tree/license; P0-2 create authorized environment; P0-3 capture fixtures/traces; P0-4 split/secret inventory | Baseline smoke; fixture hashes; no-label prompt inspection | End-to-end baseline documented/runnable. Rollback: pristine clone/pinned fixture |
| 1 Backend abstraction | P1-1 `ModelBackend`; P1-2 legacy AutoGen backend; P1-3 isolate memory/simulator API clients; P1-4 layered config/provenance | Characterization equivalence, timeout/parse failures, config redaction | Baseline flow goes through backend within approved tolerance. Feature flag restores legacy calls |
| 2 Role LoRA plumbing | P2-1 router/aliases; P2-2 adapter manifests/registry; P2-3 Transformers backend; P2-4 optional vLLM spike; P2-5 concurrency traces | Four-role mapping, wrong-adapter rejection, adapter bleed, backend contract | Four requests prove correct logical adapter. **No DPO.** Disable adapter backend to rollback |
| 3 Structured state + Updater | P3-1 schemas/projections; P3-2 event engine; P3-3 evidence slots; P3-4 true Updater; P3-5 read-only Reporter; P3-6 CaseMemory adapter | State properties, pre/post events, access negatives, baseline topic/stop/report regression | Every decision has typed pre/post-state/provenance. Dual-run legacy engine rollback |
| 4 Experience/replay | P4-1 buffer/export; P4-2 profile builder/freeze; P4-3 simulator/replay/validity; P4-4 miner/branches; P4-5 utility/verifier/attribution | Unknown facts, same-state hashes, common seeds, invalid branches, one full contrast, label canaries | Accepted contrast traces to state/profile/actions/utility/seed. Offline feature disabled for rollback |
| 5 Fast path | P5-1 distiller; P5-2 source-excluded validator; P5-3 lesson store/retriever; P5-4 merge/quarantine/retire; P5-5 poisoning audit | Paired control/intervention, source exclusion, lifecycle/retrieval, negative transfer | Unvalidated lesson impossible to retrieve. Restore prior immutable memory index |
| 6 Slow path | P6-1 preference builder; P6-2 role partitions; P6-3 TRL/PEFT trainer; P6-4 manifests/parameter diff; P6-5 candidate assembly | Leakage/dedup, one role smoke update, base/other-adapter hashes, restart | Only correct role adapter changes; Phase 2 gate already passed. Candidate non-routable/retain accepted |
| 7 Regression/internalization | P7-1 paired regression runner; P7-2 atomic policy registry; P7-3 ON/OFF; P7-4 IAR; P7-5 promotion audit | Failed-candidate rejection, CAS race, rollback, paired seeds, role matrix | Failed candidate cannot promote; full audit generated. Pointer rollback to prior accepted policy |
| 8 Research hardening | P8-1 frozen split/runner; P8-2 metrics/statistics; P8-3 reproducibility; P8-4 robustness/ablation budget; P8-5 release/privacy audit | Manifest replay, final-test isolation, matched compute, failed-run reporting | Checkpoint C plus frozen protocol; independent reproducibility review |

Critical dependency: `P6-*` cannot begin until Phase 2 routing is demonstrated and tested. Phase 4 depends on typed Phase 3 events. Phases 5 and 6 share only locally verified contrasts and may proceed separately after Phase 4; Phase 7 assembles their accepted outputs.

## Upstream file change plan

- Leave clone pristine in Phase 0; store audit/fixtures outside its Git history.
- Phase 1 minimally redirect `src/config.py`, `agents.py`, `assessment.py`, `memory.py`, and `generate_response.py` through compatibility ports without semantic prompt/control changes.
- Phase 3 replaces orchestration ownership in `assessment.py`; prompts migrate from `agents.py`; `memory.py` becomes a compatibility adapter; SummaryAgent becomes reporter.
- Phase 4 deprecates `generate_response.py` automated path after parity/safety tests.
- `main.py`, `data_load.py`, `utils.py`, `logging_setup.py`, data scripts, requirements, README, and `result.py` migrate only under their phase tickets and retain rollback fixtures.

## Rollback and data migration

Use feature flags for engine/backend/simulator paths, immutable versioned schema migrations, dual-read/compare before cutover, and append-only audit records. Never overwrite accepted adapters or lesson indexes. A failed online change returns to the Phase 0/previous characterized path; failed offline jobs cannot affect runtime. Migration scripts are idempotent, dry-run capable, and record input/output hashes.

## Principal risks

Clinical/simulator hallucination; label/test leakage; silent zero-score error bias; baseline not reproducible due placeholders/dependencies; model/version incompatibility; adapter bleed/cross-role regression; small/noisy preferences; lesson negative transfer; licensed-data privacy; AutoGen coupling; evaluation selection bias; resource infeasibility. Controls are owned by Docs 01-06 and enforced as phase gates.

## Coding authorization checklist

- Upstream SHA and runnable baseline pinned; reuse matrix approved.
- Architecture, typed state/access contracts, and Reporter/Updater semantics approved.
- Simulator fact/unknown/leakage rules and authorized split manifests approved.
- Role aliases/routing, backbone, adapter lifecycle, and DPO data contract approved.
- Safety and regression criteria approved; all blocking TBDs resolved.
- Checkpoints A, B, C recorded. Until then, this is a plan only.

## Source-backed facts

- **[UPSTREAM]** Exact current files and gaps come from Doc 00 at the pinned SHA.
- **[PAPER]** Evolution dependency semantics reflected in Phases 4-7 come from the supplied research specification.

## Engineering decisions

- **[DECISION]** Preserve the requested Phase 0-8 sequence; audit found no reason to reorder it.
- **[DECISION]** Feature flags, immutable artifacts, and characterization fixtures are the rollback backbone.

## TBD / unresolved

- **[TBD]** Repository integration layout, maintainers/ticket estimates, baseline endpoint/model, and dependency/lock tooling.
- **[TBD]** Every blocking decision listed in Docs 01-06 must be approved before its phase starts.
