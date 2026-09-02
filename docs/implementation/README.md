# PsyVEC-on-AgentMental implementation specifications

Status: **PHASES 0–8 OFFLINE IMPLEMENTATION RECORDED — EXTERNAL AND DEPENDENCY GATES REMAIN BLOCKED**

These documents translate the approved roadmap and the supplied PsyVEC research draft into implementation contracts. Phase 0/1 offline PsyVEC implementation has begun. AgentMental remains pristine; no training or DPO run has been performed, and no AgentMental business-logic change has been made.

The audited upstream is `https://github.com/MindIntLab-HFUT/AgentMental` at commit `0e2fc8ff27552845ae743e3373351120a96e216b` (2026-05-16). Tags mean: **[PAPER]** supplied `main-5p.pdf`; **[UPSTREAM]** audited clone; **[DECISION]** approved/proposed project behavior; **[TBD]** human or development-set decision still open.

## Reading order and ownership

1. [00 - Upstream AgentMental audit](00_UPSTREAM_AGENTMENTAL_AUDIT.md) owns verified source facts and the reuse matrix.
2. [01 - Scope, requirements, and acceptance](01_SCOPE_REQUIREMENTS_AND_ACCEPTANCE.md) owns scope and gates.
3. [02 - Target system architecture](02_TARGET_SYSTEM_ARCHITECTURE.md) owns boundaries and dependency direction.
4. [03 - Domain, state, memory, and data contracts](03_DOMAIN_STATE_MEMORY_AND_DATA_CONTRACTS.md) owns typed records and access rules.
5. [04 - PsyVEC evolution algorithm](04_PSYVEC_EVOLUTION_ALGORITHM_SPEC.md) owns evolution semantics.
6. [05 - Model, LoRA, training, and serving](05_MODEL_LORA_TRAINING_AND_SERVING_DESIGN.md) owns adapter and backend lifecycle.
7. [06 - Simulator, evaluation, and reproducibility](06_SIMULATOR_EVALUATION_AND_REPRODUCIBILITY.md) owns leakage controls and evaluation.
8. [07 - Migration and implementation plan](07_MIGRATION_AND_IMPLEMENTATION_PLAN.md) owns coding order, target tree, tickets, and rollback.
9. [Overall implementation status](STATUS.md) is the single current-status view across phases 0–8.
   Start from [HANDOFF.md](HANDOFF.md) when resuming in a fresh session: verification commands, remaining work grouped by blocker, and the ground rules in force.
10. [Phase 0 baseline report](PHASE_0_BASELINE_REPORT.md) records the baseline freeze, offline characterization, and gate status.
11. [Phase 1 backend report](PHASE_1_BACKEND_REPORT.md) records the backend abstraction implementation and integration gate status.
12. [Phase 2 adapter routing report](PHASE_2_ADAPTER_ROUTING_REPORT.md) records offline role-adapter routing and its blocked integration gate.
13. [Phase 3 state and roles report](PHASE_3_STATE_AND_ROLES_REPORT.md) records typed decision state, role separation, and the blocked dual-run gate.
14. [Phase 4 experience report](PHASE_4_EXPERIENCE_REPORT.md) records the offline experience spine and its partial-completion gate.
15. [Phase 5 fast-path report](PHASE_5_FAST_PATH_REPORT.md) records lesson-memory auditing and its offline gate.
16. [Phase 6 slow-path report](PHASE_6_SLOW_PATH_REPORT.md) records offline training preparation and its blocked trainer gate.
17. [Phase 7 promotion report](PHASE_7_PROMOTION_REPORT.md) records policy promotion controls and its partial-completion gate.
18. [Phase 8 research hardening report](PHASE_8_RESEARCH_HARDENING_REPORT.md) records offline research controls and the blocked final-evaluation gate.

Dependency direction is `00 + paper -> 01 -> 02 -> 03 -> (04, 05, 06) -> 07`. A document links to the owner instead of redefining its contracts.

## Review checkpoints

- Checkpoint A: approve 00 and 01.
- Checkpoint B: approve 02, 03, and 05.
- Checkpoint C: approve 04, 06, and 07.

## Readiness

Stage A documentation is complete enough for review, but implementation remains blocked until all three checkpoints pass and blocking TBDs are resolved. In particular, humans must approve the licensed-data/split policy, safety and escalation policy, typed state schema, simulator fact policy, base-model/deployment profile, role routing, DPO contract, and quantitative regression criteria.

## Source-backed facts

- **[PAPER]** PsyVEC is an offline simulated-participant research framework, not autonomous clinical diagnosis or live-patient learning.
- **[UPSTREAM]** The audit SHA and source evidence are recorded in Doc 00.

## Engineering decisions

- **[DECISION]** Detailed behavior is governed by these documents after approval; `plan.md` remains governance and roadmap.

## TBD / unresolved

- **[TBD]** All open decisions summarized above and enumerated per document require checkpoint review.
