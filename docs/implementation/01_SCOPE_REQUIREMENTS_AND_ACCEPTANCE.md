Status: DRAFT FOR CHECKPOINT A
Source commit: `0e2fc8ff27552845ae743e3373351120a96e216b`
Depends on: Doc 00; supplied PsyVEC paper
Blocks: Docs 02-07; implementation authorization
Open decisions: clinical safety policy; dataset permissions/splits; quantitative regression tolerances

# Scope, requirements, and acceptance

## Objective and users

**[DECISION]** Build a documentation-first, offline-research PsyVEC extension that preserves a characterized AgentMental assessment baseline while adding auditable, verified cross-case evolution. Primary users are researchers, implementers, reviewers, and experiment operators - not patients receiving autonomous diagnosis.

## In scope

- AgentMental-compatible interactive PHQ-8 baseline and typed assessment state.
- Interviewer, Evaluator, Scorer, Updater, plus separate ReportingAgent.
- CaseMemory, LessonMemory/retrieval, ExperienceBuffer, and policy adapters as distinct stores.
- FrozenParticipantProfile, deterministic same-state replay, hard-state mining, branching, role utility, local verification, attribution.
- Contrastive lesson distillation and source-excluded transfer validation.
- Shared backbone with role-specific LoRA; role-local preference datasets and DPO.
- Candidate/accepted checkpoint lifecycle, regression gate, Memory ON/OFF, IAR.
- Split-aware evaluation, leakage enforcement, provenance, reproducible manifests.

## Non-goals for the first implementation

No live-patient learning; production clinical diagnosis; web UI; multi-node training; arbitrary production LoRA upload; framework migration for preference alone; full RQ1-RQ7 publication suite in early phases; model training before Phase 6; or modification of upstream business logic during this documentation task.

## Functional requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| FR-01 | Preserve baseline topic flow, stop logic, scoring, and reporting behind stable interfaces | Characterization traces before/after Phase 1 |
| FR-02 | Route every logical role through `RoleModelRouter` | Trace shows requested role, alias, backend, adapter hash |
| FR-03 | Represent each decision as typed pre-state/action/output/post-state/provenance | Schema validation and replayable event log |
| FR-04 | Enforce role- and store-specific access | Negative access/leakage tests |
| FR-05 | Replay candidates from identical state/profile/simulator conditions | Branch-set identity/hash assertions |
| FR-06 | Admit only safe, valid, attributed local contrasts | Rejection reason and decomposed utility recorded |
| FR-07 | Retrieve only promoted, compatible lessons | Quarantined/rejected lesson poisoning test |
| FR-08 | Build role-partitioned preference data and update only that adapter | Parameter-diff test |
| FR-09 | Promote no failed candidate checkpoint | Atomic registry test and audit decision |
| FR-10 | Run Memory ON/OFF and IAR on identical held-out states/seeds | Paired report manifest |

## Non-functional requirements and invariants

- **Reproducibility:** immutable commit/config/prompt/data/profile/simulator/policy/seed identifiers; deterministic mode; cached artifacts content-addressed.
- **Auditability:** every promoted lesson, preference pair, and checkpoint traces to source events and gate decisions.
- **Privacy:** de-identification, least privilege, retention/deletion policy, no raw licensed transcript redistribution.
- **Leakage prevention:** participant-disjoint evolve/development/final-test sets; final test untouched until frozen protocol; ground truth absent from policy state/prompt/retrieval; test material absent from evolution indices.
- **Safety:** unsafe actions are inadmissible, not outweighed by utility; escalation behavior is versioned and human-approved.
- **Configurability:** backend, adapter alias, decoding, thresholds, resource profiles, and split IDs are explicit; no secret in tracked config.
- **Failure visibility:** invalid outputs fail closed for consolidation. Evaluation never silently converts infrastructure/model errors into clinical score zero.

## Phase definition of done

| Gate | Definition of done |
|---|---|
| Upstream audit | Pinned SHA, end-to-end source facts, complete reuse matrix |
| Baseline preservation | Runnable frozen fixture; Phase 1 output/control-flow equivalence within approved tolerance |
| Role routing | Four roles reach correct aliases/adapters; no DPO required |
| State contract | Every role decision validates typed pre/post state and provenance |
| Replay integrity | Same starting-state/profile/version/seed policy; invalid facts rejected |
| Local verification | Accepted contrast has safety, validity, attribution, utility, and margin evidence |
| Fast path | Only source-excluded transfer-validated lessons retrievable |
| Slow path | Only target-role adapter changes; accepted policy remains reference |
| Regression | Participant-disjoint safety/non-inferiority gate; failed candidate cannot promote |
| Internalization | Paired Memory ON/OFF and IAR report |
| Reproducibility | Independent rerun reconstructs selected artifact from manifest |

## Overall definition of done

All Phase 0-8 gates pass; the three review checkpoints are approved; blocking TBDs are resolved; security/privacy review is recorded; baseline and evolved artifacts are distinguishable; no claim relies on paper placeholders; rollback restores last accepted adapters and lesson index without loss of audit history.

## Test strategy ownership

Doc 06 owns test protocols and metrics; Doc 07 maps them to milestones. Minimum coverage includes contract/unit tests, baseline characterization, one-topic and one-assessment integration, one full contrast, one lesson validation, one role-local DPO smoke test, candidate rejection, leakage/property tests, and manifest replay.

## Source-backed facts

- **[PAPER]** Policies cannot observe ground-truth item scores; evolution is offline; local contrast and lesson transfer are separate claims/gates.
- **[UPSTREAM]** Baseline limitations and behavior are documented in Doc 00.

## Engineering decisions

- **[DECISION]** Clinical deployment and live learning remain out of scope.
- **[DECISION]** All consolidation paths fail closed and retain rejected evidence for audit.

## TBD / unresolved

- **[TBD]** Human-approved safety taxonomy, escalation cues, and redaction/retention policy.
- **[TBD]** Authorized datasets, participant splits, baseline model/endpoints, and resource budget.
- **[TBD]** Development-selected margins, non-inferiority tolerances, and implementation-ready acceptance values.
