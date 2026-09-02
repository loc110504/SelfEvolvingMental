Status: IN PROGRESS / INTEGRATION GATE BLOCKED

# Phase 1 model/backend abstraction report

## Implemented offline

- Added the framework-neutral `ModelBackend`, `ModelRequest`, `ModelResponse`,
  message, provenance, and explicit failure contracts under `src/psyvec/model/`.
- Added a dependency-injected `LegacyAutoGenBackend` bridge that preserves the
  existing `makerequest(manager, proxy, recipient, payload)` call shape without
  importing AutoGen in the contract package.
- Added an `OpenAICompatibleBackend` and `LegacyModelPort` seam for the direct
  memory-extraction, holistic-reassessment, and participant-simulator call
  shapes. Client construction and credentials remain outside these backend
  objects.
- Added layered backend configuration in the required order: safe defaults,
  environment, run manifest, then explicit overrides. Secrets are absent from
  `repr`, redacted from diagnostics, and excluded from the non-secret
  provenance hash.
- Added a backend config-file manifest loader and runtime assembly root that
  builds a `ModelBackend` from a resolved `BackendConfig` with an injectable
  client factory. `openai` is not imported at module import time, and credentials
  are never sourced from the versioned manifest. The default manifest path's
  `_PROJECT_ROOT` is `Path(__file__).resolve().parents[2]`: correct for a source
  checkout, but an installed site-packages copy cannot find that default and
  fails clearly with `ConfigurationError`.
- Added the `use_backend_abstraction` feature flag so an integration can select
  the characterized legacy path during rollback. The flag is evaluated once in
  `build_model_backend`; `select_backend` is no longer used by the assembly root
  and remains the standalone rollback selector for callers that already hold both
  backends.
- Backend responses retain normalized content, usage, latency, and provenance,
  but not the raw SDK response object, reducing accidental sensitive-content
  retention.
- Added offline tests for request preservation, provenance, malformed/missing
  responses, timeout visibility, direct-client argument mapping, configuration
  precedence, validation, redaction, and rollback selection.
- Added a content-minimized trace recorder and parity comparator. Prompt and
  control-flow changes always fail comparison; any permitted response mismatch
  count must be supplied explicitly by the approved evaluation protocol.
- Parity trace entries now record latency on both success and failure plus
  response provenance. `compare_traces` separates provenance invariants (`model`,
  `prompt_version`, and adapter ids), which are hard failures, from provenance
  metadata (`backend`, `config_hash`), which is reported only; its latency budget
  caps the maximum per-entry candidate slowdown.

## Deliberately not changed

The pinned `AgentMental/` subtree remains pristine. Its source files have not
yet been redirected through the new ports because doing so would invalidate the
Phase 0 source hashes before an approved runnable baseline trace exists. No
network/model request, dataset access, or participant-data operation was made.

## Reproducible checks

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_agentmental_baseline.py
ruff check src tests/unit
mypy --strict src/psyvec
git -C AgentMental status --short
```

## Remaining Phase 1 work

1. Approve a compatible legacy environment, endpoint/model, and authorized
   dataset or approved synthetic fixture as recorded in the Phase 0 report.
2. Capture the frozen end-to-end baseline trace.
3. Redirect the assessment roles, memory calls, and simulator calls through
   the compatibility ports behind the feature flag.
4. Run before/after characterization equivalence and record the approved
   tolerance, prompt/control-flow diffs, latency, failures, and provenance.
5. Keep the gate blocked unless that evidence passes; do not infer parity from
   unit tests alone.

The remaining Phase 1 blockers are external, not code: an approved legacy
environment, an approved endpoint/model, and an authorized dataset or approved
synthetic fixture. Parity must not be inferred from unit tests.

## Deferred findings

The following are deferred by directive as non-blocking for the next phase.

1. **`timeout_seconds` is not enforced.** It is read into `BackendConfig` and
   never consumed by any backend or client. The tests prove exception mapping
   (`TimeoutError` -> `BackendTimeoutError`), not enforcement.
2. **Upstream `cache_seed: 42` is dropped.** `AgentMental/src/config.py:11`
   sets it; `BackendConfig` has no seed field and `provenance_hash` omits it.
   Upstream determinism partly came from that cache seed, and the parity
   comparator compares response hashes.
3. **`ModelResponse.__post_init__` raises `BackendResponseError` on empty
   content**, so a legitimately empty completion is indistinguishable from a
   backend failure. This is intentional under the no-silent-zero rule and is
   recorded so it is not mistaken for a bug later.
4. **`OpenAICompatibleBackend` detects timeouts by exception class-name
   substring** (`"timeout" in type(error).__name__.lower()`), which is fragile.
