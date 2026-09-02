Status: DRAFT FOR CHECKPOINT B
Source commit: `0e2fc8ff27552845ae743e3373351120a96e216b`
Depends on: Docs 00-01
Blocks: Docs 03-07; module implementation
Open decisions: process topology; first backbone/backend; AutoGen exit criteria; storage technologies

# Target system architecture

## Context and trust boundary

```text
Authorized transcript + labels -> offline profile/split preparation
                                      |
User or frozen simulator -> ONLINE AssessmentEngine -> accepted role policy
                                      |                 + promoted lessons
                                      v
                              case-local events
                                      |
                                      v (offline export, de-identified)
OFFLINE ExperienceBuffer -> verify -> lessons / preference data -> candidate adapters
                                                           -> RegressionGate -> accepted registry
```

**[DECISION]** Online assessment cannot mutate cross-case memory or adapters. Offline jobs cannot publish directly; promotion is an explicit transaction after validation/regression.

## Component topology

```text
AssessmentEngine
 ├─ Agents: Interviewer | Evaluator | Scorer | Updater | Reporter
 ├─ AssessmentState + DecisionEvent
 ├─ CaseMemory
 ├─ LessonRetriever -> read-only accepted LessonMemory
 └─ RoleModelRouter -> ModelBackend -> accepted AdapterRegistry

OfflineEvolutionCoordinator
 ├─ ExperienceBuffer -> HardStateMiner -> BranchGenerator
 ├─ ReplayEngine -> ParticipantSimulator -> FrozenParticipantProfile
 ├─ UtilityEvaluator -> LocalContrastVerifier -> AttributionRouter
 ├─ LessonDistiller -> TransferValidator -> LessonMemory governance
 ├─ PreferenceDatasetBuilder -> RoleDPOTrainer
 └─ RegressionGate -> AdapterRegistry promotion

EvaluationRunner -> Metrics + AuditReporter (reads immutable artifacts)
```

Doc 03 owns payload schemas; Doc 04 owns algorithm order; Doc 05 owns backend/training lifecycle; Doc 06 owns simulator and evaluation rules.

## Boundaries, ownership, dependency direction

| Boundary | Owns | Must not know |
|---|---|---|
| Agent | Role prompt/decision contract | PEFT/TRl internals, labels, hidden profile |
| AssessmentEngine | Role scheduling and state transitions | DPO, utility labels, simulator hidden state |
| ModelBackend | Message generation and response metadata | hard-state or lesson promotion business logic |
| Memory ports | Store-specific read/write policy | model training internals |
| Simulator | Answers from frozen facts | accepted action preference or policy labels |
| Utility/Verifier | Offline scoring and admissibility | inference prompt construction |
| Trainer | Accepted role dataset to candidate adapter | raw transcripts or test participants |
| RegressionGate | Candidate assembly/evaluation/promotion decision | training mutation |

Dependencies point inward through protocols: orchestration depends on agent/model/memory interfaces, concrete AutoGen/Transformers/vLLM/storage adapters depend on those interfaces. Offline evolution may read exported events but online code never imports evolution/training packages.

## Core interfaces

```python
class ModelBackend(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse: ...

class RoleModelRouter(Protocol):
    def resolve(self, role: DecisionRole, policy_id: str) -> ModelBinding: ...

class CaseMemoryPort(Protocol):
    def apply(self, transition: StateTransition) -> None: ...
    def view_for(self, role: DecisionRole) -> CaseMemoryView: ...

class ReplayEngine(Protocol):
    def replay(self, branch_set: BranchSet) -> list[BranchResult]: ...

class PromotionRegistry(Protocol):
    def promote(self, decision: CheckpointDecision) -> AcceptedPolicy: ...
```

## Runtime lifecycle

1. Create case with pinned policy, lesson-index, prompt, and config versions.
2. Engine requests a role; the role receives only its authorized state view and retrieved promoted lessons.
3. Router resolves logical alias and accepted adapter; backend emits content plus model provenance.
4. Validator parses role output; Updater proposes a typed state patch; engine checks invariants and commits atomically.
5. Every decision emits immutable `DecisionEvent`. Reporter reads a sanitized final state and cannot mutate it.
6. Close/isolate CaseMemory; offline export applies consent, de-identification, and split policy.

## Failure rules

- Backend timeout/invalid JSON: bounded retry under the same request ID, then explicit failed event; never fabricated score.
- Unknown alias/adapter mismatch: fail startup or request closed; no default cross-role adapter.
- State invariant failure: reject patch, preserve pre-state, record reason.
- Retrieval/store outage: configured fail-closed or memory-disabled mode, visibly tagged; never use unvalidated lessons.
- Simulator inconsistency/unsupported fact: invalidate branch, do not retry-until-success silently.
- Training failure: candidate remains non-deployable. Promotion uses atomic compare-and-swap against expected accepted version.

## Configuration and provenance

Layering: versioned safe defaults < environment/deployment profile < run manifest < explicit CLI overrides. Secrets exist only in secret providers/environment and are redacted. Startup validates the resolved config and emits its non-secret hash.

Every span/event includes run/case/decision IDs; code SHA; dataset/split; role; prompt; backend/model/adapter; lesson index and retrieved lesson IDs; decoding; state hashes; latency/tokens; simulator/profile/seed where applicable; gate outcome. Raw sensitive content is separately access-controlled and never required as an ordinary metric tag.

## Security and data boundary

Policy-visible state is a projection, not the persisted offline record. Labels and hidden profile facts are separate encrypted/access-controlled fields. ExperienceBuffer is never queried by inference. Training consumes only accepted, minimized preference records. Final-test storage is isolated and opened only by an approved frozen evaluation run.

## Technology decisions

- **[DECISION]** Retain an AutoGen compatibility adapter in Phases 1-2 because upstream relies on it; orchestration interfaces remain framework-neutral.
- **[DECISION]** Direct Transformers is the development/test backend; vLLM is a candidate OpenAI-compatible multi-LoRA serving adapter after feasibility validation.
- **[DECISION]** Training runs in a separate process/job from serving. Exact database/object store is deferred behind ports.

## Source-backed facts

- **[PAPER]** Online role decisions use shared observable case state, while offline branching/utility may access controlled hidden supervision.
- **[UPSTREAM]** Current orchestration, memory, and API coupling motivating these seams is detailed in Doc 00.

## Engineering decisions

- **[DECISION]** Online/offline import and mutation boundaries are mandatory.
- **[DECISION]** Reporting is read-only and separate from the state-updating role.

## TBD / unresolved

- **[TBD]** Deployment process topology, queue/store technologies, retention periods, and failure retry budgets.
- **[TBD]** AutoGen removal criteria and vLLM version/feature benchmark.
- **[TBD]** First supported backbone, hardware profiles, and approved safety service/policy.
