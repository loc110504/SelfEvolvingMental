Status: DRAFT FOR CHECKPOINT B
Source commit: `0e2fc8ff27552845ae743e3373351120a96e216b`
Depends on: Docs 00-03
Blocks: Phases 1-2 and 6-7; Doc 07 approval
Open decisions: backbone; target modules/rank; vLLM feasibility; hardware profiles; artifact registry

# Model, LoRA, training, and serving design

## Chosen abstraction

```text
Agent/AssessmentEngine
 -> RoleModelRouter(logical role, accepted policy)
 -> ModelBackend
 -> shared compatible backbone
    + exactly one role-specific accepted adapter
```

**[DECISION]** Logical aliases are initially `psyvec-interviewer`, `psyvec-evaluator`, `psyvec-scorer`, `psyvec-updater`. Aliases are stable API concepts; physical model/adapter paths are registry data. Reporter may use a configured base/legacy binding but is not a trainable PsyVEC decision role in the first design.

## Runtime contract

```python
ModelRequest = {request_id, role, messages, assessment_state_view,
                retrieved_lessons?, decoding_config, policy_snapshot}
ModelResponse = {content, structured_output?, finish_reason, usage, latency,
                 backend, base_model_hash, adapter_id/hash, prompt_version}

generate(request: ModelRequest) -> ModelResponse
```

Router rejects missing/wrong-role adapters and base-model incompatibility. Orchestration never calls PEFT APIs and model layer never mines hard states, calculates utility, or promotes lessons.

## Toolchain evaluation

- **[DECISION]** Transformers represents/loads the shared backbone and is the direct development/testing generation fallback.
- **[DECISION]** PEFT owns LoRA attachment, saving/loading, adapter config, and trainable-parameter assertions.
- **[DECISION]** TRL is the preferred role-DPO trainer; Accelerate owns device/precision/distributed execution.
- **[DECISION]** bitsandbytes/QLoRA is optional per hardware profile, never silently enabled.
- **[DECISION]** vLLM is a candidate OpenAI-compatible multi-LoRA serving backend, accepted only after version-specific tests for target architecture, adapter limits/switching, alias isolation, concurrency, determinism, and provenance. AutoGen can consume this endpoint through the backend adapter; no LangChain/LangGraph migration is required.

These are engineering directions, not paper claims. Exact current library APIs/versions must be pinned during implementation because upstream contains none of them.

## Loading and adapter lifecycle

Load one base model per serving process/device profile where practical. Validate tokenizer/chat template, dtype/quantization, context length, and base hash. Registry maps `(accepted_policy_id, role)` to immutable `AdapterManifest`. Adapter switch is request-scoped and concurrency-safe; response provenance confirms the resolved adapter. Warmup/health checks cover every role before traffic.

```text
artifacts/
  bases/<base-hash>/manifest.json          # reference, not necessarily copied weights
  adapters/<role>/<adapter-id>/{adapter_config.json, weights, manifest.json}
  policies/<policy-id>/policy-manifest.json
  runs/<run-id>/{training-manifest.json, metrics, logs}
```

Candidate adapter status is non-routable. Regression-passed policy assembly atomically changes the accepted registry pointer; rollback restores the previous pointer, never overwrites artifacts.

## Training contract and preference schema

```python
train_role_adapter(
    role: DecisionRole,
    accepted_policy: PolicyManifest,
    preference_dataset: Dataset[PreferencePair],
    training_config: RoleTrainingConfig
) -> CandidateAdapter
```

Serialized DPO examples contain role, policy-visible input/messages/state projection, chosen output, rejected output, contrast/provenance IDs, accepted reference policy, split ID, and optional sample weight only if approved. They contain no raw hidden profile, ground-truth label, test participant, or retrieved raw experience.

Start from accepted role adapter; use that accepted policy as DPO reference. Freeze backbone and all other adapters. Train only target adapter parameters, then assert parameter hashes/diffs. Dataset builders enforce participant/split disjointness and input-length/content policies. Low pair count returns a documented no-update.

## QLoRA and resource profiles

QLoRA may quantize the backbone for adapter training while keeping adapter artifacts compatible with the declared inference base, subject to numerical/compatibility tests. Profiles declare GPU model/count, VRAM, dtype, quantization, sequence length, batch/accumulation, optimizer, checkpointing, and expected peak memory. No fixed hardware or VRAM claim is approved yet.

## Serving and fallback

- `TransformersBackend`: single-process correctness/dev tests, explicit adapter activation, deterministic configuration where supported.
- `VLLMBackend`: candidate higher-throughput OpenAI-compatible service with role aliases and server-side LoRA mapping.
- `LegacyAutoGenBackend`: characterization bridge reproducing upstream endpoint behavior in Phases 1-2.

Memory-OFF is an AssessmentEngine request mode that omits retrieval; it does not select a different adapter or training checkpoint. Backends must make ON/OFF traces comparable.

## Training manifest and failure rules

Manifest fields are owned by Doc 03 and include code/data/prompt/base/reference/adapter hashes, PEFT/TRL/Transformers/Accelerate versions, seeds, decoding/training config, parameter counts, hardware/precision, failures, and output hash. NaN/loss failure, data leakage, hash mismatch, unexpected trainable parameter, incomplete artifact, or regression failure blocks promotion. Interrupted runs resume only from manifest-compatible checkpoints.

## Acceptance tests

1. Four aliases resolve to four correct role manifests over one shared compatible base.
2. Concurrent requests cannot bleed adapters; responses report expected hash.
3. Direct backend and serving backend satisfy the same contract/schema.
4. Role training changes only the target adapter; base and other adapters hash-identical.
5. Candidate is inaccessible until atomic promotion; rollback returns prior accepted policy.
6. Memory ON/OFF uses identical model binding and decoding.

## Source-backed facts

- **[PAPER]** Four functional policies share a frozen backbone, only role adapter parameters update, and accepted round policy is the next DPO reference.
- **[UPSTREAM]** All AutoGen roles currently share one global model config and direct API calls exist outside orchestration.

## Engineering decisions

- **[DECISION]** Use backend/router/registry ports and retain AutoGen compatibility initially.
- **[DECISION]** Demonstrated/tested role routing is a hard prerequisite for DPO.

## TBD / unresolved

- **[TBD]** Base model/license, tokenizer/template, PEFT target modules/rank, DPO beta/LR, precision, sequence limits, and minimum pairs.
- **[TBD]** vLLM version and feasibility benchmark; QLoRA need; deployment and registry technology.
