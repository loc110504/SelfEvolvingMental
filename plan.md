Status: APPROVED BASELINE / ROADMAP

This document defines project intent, governance constraints, scope, and
high-level implementation phases.

It is not the authoritative source for detailed implementation behavior.
See [`docs/implementation/README.md`](docs/implementation/README.md).

# PsyVEC-on-AgentMental project constitution and roadmap

## Project objective

Build an offline-research implementation of PsyVEC on top of useful interactive
assessment behavior in AgentMental. Preserve a characterized baseline before
adding typed state, role-specific model routing, verified experience, lesson
memory, role-specific LoRA-DPO, regression gating, and internalization audits.

The system is controlled research support, not production clinical diagnosis,
and must not autonomously learn from live patients.

## Upstream and delivery constraints

Upstream: `https://github.com/MindIntLab-HFUT/AgentMental`

AgentMental is cloned, read, and reused locally as source material.

- No fork is required.
- Do not push to upstream or open an upstream pull request.
- Do not add or use a delivery remote for upstream.
- The clone's `origin` may remain as read-only metadata.
- Delivery occurs only in this project after explicit authorization.
- Pin the exact audited commit before implementation.

The documentation baseline audits commit
`0e2fc8ff27552845ae743e3373351120a96e216b`. A later upstream change requires a
new audit and explicit baseline decision; it must not silently replace this SHA.

## Scope guard

### In scope

- AgentMental-compatible interactive PHQ-8 baseline.
- Interviewer, Evaluator, Scorer, and true Updater decision roles.
- A separate ReportingAgent where existing final reporting remains useful.
- Typed assessment state, evidence slots, transitions, and decision events.
- Separate CaseMemory, LessonMemory, ExperienceBuffer, and policy adapters.
- Frozen participant profiles and controlled same-state replay.
- Hard-state mining, role utility, local verification, and attribution.
- Fast-path lesson distillation, transfer validation, governance, and retrieval.
- Slow-path role preference data, LoRA-DPO, and candidate adapters.
- Regression gates, Memory ON/OFF, Internalized Action Rate, and reproducibility.

### Out of scope initially

- Online or autonomous learning from live patients.
- Production clinical diagnosis or treatment decisions.
- Web UI, multi-node training, or arbitrary production LoRA upload.
- Orchestration-framework migration based only on preference.
- The full RQ1-RQ7 publication suite in early phases.
- Implementation before checkpoints and blocking decisions are approved.

## Architectural principles

### Specifications before code

Research concepts become interfaces, schemas, lifecycles, invariants, access
and failure rules, provenance, acceptance criteria, and tests before code.
Specifications distinguish:

- **[PAPER]** directly supported by the supplied PsyVEC research draft.
- **[UPSTREAM]** verified in pinned AgentMental source.
- **[DECISION]** project engineering behavior.
- **[TBD]** unresolved configuration, experiment, inspection, or approval.

Paper placeholders are not results. Development-selected values are not paper
facts and remain configurable until chosen under an approved protocol.

### Preserve baseline before extension

Understand and characterize AgentMental first. Isolate model calls and state
transitions while preserving observable control flow. Add verified evolution
only after role routing and typed state are proven.

### Separate online assessment from offline evolution

Online assessment uses an accepted policy snapshot, current-case state, and
promoted lessons. It cannot train, promote lessons, mutate adapters, or read the
ExperienceBuffer.

Offline evolution collects eligible experience, explores matched branches,
verifies contrasts, validates lessons, trains candidate adapters, and submits
artifacts to gates. It cannot publish directly to runtime.

### Shared backbone and localized adaptation

Use one compatible shared backbone with separate LoRA adapters for Interviewer,
Evaluator, Scorer, and Updater. Logical roles map through `RoleModelRouter` and
`ModelBackend`; agents do not manage PEFT or training internals.

DPO must not begin until role-specific adapter routing is demonstrated and
tested.

### Verified experience before consolidation

A raw or successful trajectory is not reusable supervision. Alternatives must
be compared from the same starting state with the same frozen participant
profile and controlled simulator conditions. Safety is a hard gate.

A locally valid action preference and a transferable lesson are different
claims. The slow path may consume an accepted local pair. The fast path may
retrieve a lesson only after source-excluded transfer validation.

### Knowledge segregation

- `CaseMemory` is current-participant state and is isolated or destroyed after
  the case according to policy.
- `LessonMemory` contains only transfer-validated procedural lessons.
- `ExperienceBuffer` is an offline audit/training store, never prompt content.
- Policy adapters contain parametrically consolidated role behavior.

These are not interchangeable forms of generic memory.

### Leakage, safety, and provenance

Ground-truth labels and hidden profile fields are unavailable to policies,
prompts, state summaries, memories, and retrieval keys. Test participants are
isolated from evolution, validation, model selection, and index building.

Every decision, contrast, lesson, training run, checkpoint, and evaluation must
trace to versioned inputs, artifacts, configuration, and seeds. Invalid or
unsafe artifacts fail closed and remain auditable.

## High-level target stack

- Existing AgentMental/AutoGen orchestration where useful in early phases.
- Framework-neutral model backend, role router, and adapter registry.
- Transformers for model representation and direct development/testing.
- PEFT for role LoRA; TRL for role DPO after routing is proven.
- Accelerate for device orchestration.
- Optional validated vLLM OpenAI-compatible multi-LoRA serving.
- Optional bitsandbytes/QLoRA resource profile.

Do not migrate to LangGraph or LangChain merely for LoRA support.

## AgentMental reuse strategy

Doc 00 governs file-level reuse. At a high level:

- Reuse validated PHQ rubrics, protocol, question, sufficiency, scoring, and
  final-report behavior.
- Refactor global agent/model coupling behind role-aware backends.
- Map current question, necessity, and scoring logic to Interviewer, Evaluator,
  and Scorer seeds.
- Preserve final summary as ReportingAgent, not PsyVEC Updater.
- Create a dedicated typed Updater for evidence/belief/state revision.
- Reuse graph memory only where appropriate for CaseMemory.
- Rewrite simulation around ProfileBuilder, FrozenParticipantProfile,
  ParticipantSimulator, and ReplayEngine.
- Add evolution, lessons, role preferences, LoRA/DPO, regression, and
  internalization as new components.

## Implementation roadmap

### Phase 0 - Freeze upstream baseline

Pin source, establish an authorized runnable setup, document flow, and capture
characterization fixtures without changing business logic.

Gate: baseline is explainable and reproducible end to end.

### Phase 1 - Model/backend abstraction

Isolate AutoGen and direct API calls behind configurable backends while
preserving baseline behavior.

Gate: baseline runs through the abstraction within approved tolerances.

### Phase 2 - Role-specific LoRA plumbing

Add role aliases, adapter registry, direct Transformers backend, and optionally
validate vLLM. Do not run DPO.

Gate: four roles trace to correct adapters without cross-role bleed.

### Phase 3 - Structured state and true Updater

Add typed state/evidence/events, atomic patches, true Updater, and read-only
Reporter.

Gate: every decision has typed pre-state, action/output, post-state, provenance.

### Phase 4 - Experience and replay core

Add ExperienceBuffer, frozen profiles, simulator/replay, HardStateMiner,
BranchGenerator, UtilityEvaluator, attribution, and LocalContrastVerifier.

Gate: each accepted contrast traces to common state/profile, actions, utility,
validity, attribution, simulator version, and seed.

### Phase 5 - Fast path

Add lesson distillation, source-excluded transfer validation, LessonMemory,
retrieval, governance, and poisoning tests.

Gate: no unvalidated, rejected, or quarantined lesson is retrievable.

### Phase 6 - Slow path

Build role-partitioned preference data and role-local TRL/PEFT DPO.

Gate: only intended role adapter changes; backbone and others remain unchanged.

### Phase 7 - Regression and internalization

Add candidate/accepted lifecycle, atomic promotion, RegressionGate, Memory
ON/OFF, and Internalized Action Rate.

Gate: failed candidates cannot become runtime policy; accepted changes have a
paired audit.

### Phase 8 - Research evaluation hardening

Freeze protocols/splits, expand metrics and robustness/ablation runners, record
compute/failures, and produce reproducible reports.

Gate: final-test access is isolated and manifest-driven runs are reproducible.

## Top-level acceptance gates

| Gate | Required evidence |
|---|---|
| Upstream audit | Pinned SHA and approved reuse matrix |
| Baseline preservation | Characterization across backend refactor |
| Role routing | Correct aliases/adapters and isolation tests |
| State contract | Typed pre/post state for every role decision |
| Replay integrity | Same state/profile/simulator conditions/seed policy |
| Local verification | Safety, validity, utility, attribution, provenance |
| Fast path | Transfer validation before retrieval |
| Slow path | Target-role-only update after Phase 2 |
| Regression | Failed candidate cannot promote; rollback proven |
| Internalization | Paired Memory ON/OFF and IAR report |
| Reproducibility | Split lineage, manifests, artifact hashes, failures |

## Human approval checkpoints

### Checkpoint A - Upstream and scope

- Upstream audit and pinned baseline.
- File reuse matrix.
- Scope, non-goals, constraints, and acceptance model.

### Checkpoint B - Architecture and contracts

- Architecture and online/offline boundaries.
- Domain/state/memory/access contracts.
- Role-aware model, LoRA, training, and serving design.

### Checkpoint C - Evolution and delivery

- Evolution semantics.
- Simulator, leakage, evaluation, and reproducibility protocol.
- Migration phases, tickets, tests, and rollback.

Coding begins only after applicable checkpoint decisions are recorded.

## Conditions required before coding

- [ ] Upstream commit and runnable baseline pinned.
- [ ] Reuse matrix approved.
- [ ] Architecture and state contracts approved.
- [ ] Simulator access/leakage rules and split manifests approved.
- [ ] LoRA routing approved.
- [ ] DPO data/reference-policy contract approved.
- [ ] Safety and regression criteria approved.
- [ ] Blocking TBDs resolved.

Experimental thresholds may remain configurable only when their development
selection lifecycle, data boundary, and freeze point are approved.

## Documentation index

- [Implementation README](docs/implementation/README.md) - navigation,
  dependencies, readiness, and checkpoints.
- [Doc 00 - Upstream audit](docs/implementation/00_UPSTREAM_AGENTMENTAL_AUDIT.md) -
  source facts, flow, risks, and reuse matrix.
- [Doc 01 - Scope and acceptance](docs/implementation/01_SCOPE_REQUIREMENTS_AND_ACCEPTANCE.md) -
  requirements, non-goals, definition of done, and gates.
- [Doc 02 - Architecture](docs/implementation/02_TARGET_SYSTEM_ARCHITECTURE.md) -
  components, boundaries, direction, and topology.
- [Doc 03 - Data contracts](docs/implementation/03_DOMAIN_STATE_MEMORY_AND_DATA_CONTRACTS.md) -
  schemas, lifecycles, access, and failures.
- [Doc 04 - Evolution algorithm](docs/implementation/04_PSYVEC_EVOLUTION_ALGORITHM_SPEC.md) -
  replay, verification, fast/slow paths, and regression.
- [Doc 05 - Model and LoRA](docs/implementation/05_MODEL_LORA_TRAINING_AND_SERVING_DESIGN.md) -
  backends, adapters, DPO, serving, and promotion.
- [Doc 06 - Simulator and evaluation](docs/implementation/06_SIMULATOR_EVALUATION_AND_REPRODUCIBILITY.md) -
  integrity, leakage, metrics, tests, and manifests.
- [Doc 07 - Migration plan](docs/implementation/07_MIGRATION_AND_IMPLEMENTATION_PLAN.md) -
  target tree, changes, phases, tickets, dependencies, and rollback.

# Source of Truth

`plan.md` defines project intent, governance constraints, high-level roadmap,
and scope.

It is NOT the authoritative source for detailed implementation behavior.

For implementation details, follow the approved documents under
`docs/implementation/`.

Precedence when documents conflict:

1. Approved ADR / explicitly approved later decision
2. Most specific approved document under `docs/implementation/`
3. `plan.md`
4. Research paper / upstream source as reference evidence

The research paper and upstream source remain evidence sources, but project
implementation behavior is governed by the approved project specifications.

## Governance of change

Changes to scope, safety/leakage boundaries, online/offline separation,
knowledge segregation, adapter localization, consolidation gates, final-test
isolation, or source precedence require explicit human approval. Update the
most specific owning document first and record migration consequences.
