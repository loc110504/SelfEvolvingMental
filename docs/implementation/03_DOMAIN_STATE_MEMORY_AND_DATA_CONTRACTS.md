Status: DRAFT FOR CHECKPOINT B
Source commit: `0e2fc8ff27552845ae743e3373351120a96e216b`
Depends on: Docs 00-02
Blocks: Docs 04-07; schema implementation
Open decisions: serialization library; evidence ontology; retention/access roles; safety enum

# Domain, state, memory, and data contracts

## Conventions

**[DECISION]** Schemas are language-neutral records shown as compact Python-like types. All records carry `schema_version`, stable IDs, UTC timestamps where lifecycle matters, and a `ProvenanceRef`. Stored artifacts are immutable except through explicit versioned state transitions. PII and labels are separate classifications.

```python
DecisionRole = Literal["interviewer", "evaluator", "scorer", "updater"]
EvidenceStatus = Literal["missing", "known", "unknown", "conflicting"]
Lifecycle = Literal["candidate", "validated", "promoted", "quarantined", "retired", "rejected"]
ProvenanceRef = {run_id, code_sha, config_hash, prompt_version,
                 policy_id, model_id, adapter_hash?, source_ids, seed?}
```

## Assessment contracts

```python
EvidenceSlot = {slot_id, kind, status, value?, confidence?, source_turn_ids,
                time_window?, contradictions, provenance}
TopicState = {topic_id, rubric_version, evidence_slots, status,
              proposed_score?, score_basis_slot_ids, revision}
DialogueTurn = {turn_id, speaker, content_ref, sanitized_text?, created_at, provenance}
AssessmentState = {case_id, scale_version, current_topic_id?, topic_states,
                   dialogue_turn_ids, case_memory_version, safety_flags,
                   policy_snapshot, revision}
RoleAction = {action_id, role, kind, payload, rationale_ref?, requested_at}
RoleOutput = {action_id, raw_response_ref, parsed_payload?, validation_status,
              errors, model_provenance}
DecisionEvent = {event_id, case_id, decision_index, pre_state_hash, role,
                 action, output, post_state_hash, state_patch, retrieved_lesson_ids,
                 provenance, outcome}
```

Invariants: revisions increase monotonically; state patches reference their pre-state; evidence values cite acquired turns; score bases cite evidence; no unknown slot becomes known without a source; Reporter emits no DecisionEvent/state patch; labels never appear in these policy-visible records.

## Memory contracts

```python
CaseMemory = {case_id, assessment_state_ref, evidence_index, topic_summaries,
              turn_refs, created_at, expires_at?, access_policy}
Lesson = {lesson_id, role, trigger, do, avoid, criterion, scope, exceptions,
          candidate_confidence, source_contrast_ids}
LessonMemoryEntry = {lesson, lifecycle, validation_result_ids, version,
                     merged_from, retrieval_stats, governance_history}
RetrievedLesson = {lesson_id, memory_version, role, matched_scope,
                   semantic_score, validated_confidence, rank}
```

CaseMemory is one participant only. LessonMemory contains only promoted procedural abstractions, never raw cross-case transcript. Retrieved lessons are bounded and role/scope filtered. Policy adapters are model artifacts, not memory records.

## Evolution contracts

```python
ExperienceRecord = {experience_id, split_id, participant_ref, decision_event,
                    offline_state_ref, role, action, outcome_ref, utility?, provenance}
HardStateRecord = {experience_id, role, missing_signal, uncertainty_signal,
                   coordination_signal, normalized_score, threshold_config,
                   selected, rationale}
BranchCandidate = {branch_id, branch_set_id, starting_state_hash, role, action,
                   generation_strategy, policy_snapshot, provenance}
BranchResult = {branch_id, profile_hash, simulator_version, response,
                post_state_hash?, validity, invalid_reasons, utility?}
UtilityBreakdown = {role, task_gain, evidence_gain, costs, safety_admissible,
                    normalized_total?, label_supervision_used, config_version}
VerifiedContrast = {contrast_id, starting_state_hash, role, chosen_branch_id,
                    rejected_branch_id, delta_utility, checks, attribution,
                    status, provenance}
PreferencePair = {pair_id, role, policy_input, chosen, rejected,
                  contrast_id, accepted_policy_id, split_id, provenance}
TransferValidationResult = {lesson_id, source_excluded_participants,
                            paired_state_ids, control_memory_version, deltas,
                            support, win_rate, mean_gain, safety_regressions,
                            config, decision}
```

ExperienceBuffer is an audit/training store and has no inference interface. A PreferencePair is valid only from a locally accepted contrast with unique/direct attribution. Transfer validation excludes the source participant and compares current memory versus current memory plus candidate lesson.

## Simulation contracts

```python
ProfileFact = {fact_id, topic_id, slot_kind, value, source_span_refs,
               permissible_paraphrases?, confidence, hidden_from_policy}
UnknownFact = {topic_id, slot_kind, reason, permitted_response_modes}
SimulatorProvenance = {profile_builder_version, profile_hash, simulator_model,
                       prompt_version, decoding_config, seed, response_hash}
FrozenParticipantProfile = {profile_id, participant_ref, source_split,
                            facts, unknown_facts, hidden_item_scores_ref?,
                            frozen_at, builder_provenance, content_hash}
```

After freezing, facts cannot change. Unsupported questions produce an approved uncertainty/refusal behavior, never invented symptoms. Hidden score references are not facts exposed to the assessment policy.

## Training and checkpoint contracts

```python
AdapterManifest = {adapter_id, role, base_model_hash, parent_adapter_id?,
                   peft_config, artifact_hash, status, created_by_run}
TrainingRunManifest = {run_id, code_sha, role, dataset_hash, split_ids,
                       accepted_reference_policy, base_model, trainable_params,
                       training_config, seeds, hardware, software_versions,
                       output_adapter, failures}
RegressionResult = {candidate_policy, development_split, paired_run_ids,
                    quality, cost, redundancy, safety, role_matrix,
                    tolerances, passed, reasons}
CheckpointDecision = {decision_id, expected_accepted_policy, candidate_policy,
                      regression_result, decision, reviewer?, decided_at}
```

Candidate is never routable in normal assessment. Promotion is atomic; adapter/base compatibility hashes must match; training updates only the target role adapter.

## Field-level access matrix

Legend: `R` read, `W` append/propose, `H` hidden, `-` none.

| Data field | Interviewer | Evaluator | Scorer | Updater | Simulator | Offline utility | Trainer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sanitized acquired dialogue | R | R | R | R | R (branch context) | R | R (pair input only) |
| Policy-visible evidence slots | R | R | R | R/W | R | R | R (pair input) |
| Missing/unknown slots | R | R | R | R/W | R | R | R (pair input) |
| Current predicted scores/bases | R | R | R/W | R/W | - | R | R (pair input) |
| CaseMemory, current case | R | R | R | R/W | scoped R | R | - |
| Promoted role lessons | R | R | R | R | - | R | - |
| Raw cross-case transcript | - | - | - | - | - | restricted profile build only | - |
| Frozen profile visible facts | H | H | H | H | R | R | H |
| Hidden item/total labels | H | H | H | H | optional H* | R | H |
| ExperienceBuffer raw records | - | - | - | - | - | R/W | minimized accepted pairs only |
| Accepted PreferencePair | - | - | - | - | - | R/W | R |
| Test participants/artifacts | H | H | H | H | approved final run only | approved final run only | H |

`*` The simulator ordinarily does not need score labels; if a validated profile implementation uses them, that use must be separately configured and never surfaced in responses. Offline utility is the intended label consumer.

## State and lifecycle transitions

- Assessment: `created -> active -> completed | aborted`; each accepted Updater patch increments revision.
- Lesson: `candidate -> validated -> promoted`; or `candidate/validated -> rejected/quarantined`; `promoted -> merged/retired`; only promoted is retrievable.
- Adapter: `candidate -> regression_passed -> accepted`; or `candidate -> rejected`; accepted is immutable and may become the next parent/reference.
- Experience: `collected -> eligible -> branched -> locally_verified`; invalid/rejected records remain auditable but never become training/retrieval inputs.

## Validation and failure rules

Schema/version/enum/hash failures reject ingestion. Unauthorized projections fail closed and emit security audit events. Conflicting evidence remains `conflicting`, not overwritten. Orphaned provenance blocks consolidation. Deletion requests tombstone derived links and trigger index rebuild per approved retention policy; exact regulated-data behavior is TBD.

## Source-backed facts

- **[PAPER]** Observable state includes topic, dialogue, case memory, known and missing evidence; experience records state/role/action/output/next-state/utility/provenance.
- **[UPSTREAM]** Current graph fields and score visibility are described in Doc 00.

## Engineering decisions

- **[DECISION]** Typed immutable events and projections enforce access rather than relying on prompts alone.
- **[DECISION]** Stores and lifecycles above are non-interchangeable.

## TBD / unresolved

- **[TBD]** Choose Pydantic/dataclasses/JSON Schema and canonical serialization/hash format.
- **[TBD]** Approve PHQ evidence ontology, confidence semantics, safety flags, content storage/encryption, retention, and deletion.
- **[TBD]** Decide whether the simulator may ever consume hidden score labels; default recommendation is no.
