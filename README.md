# SelfEvolvingMental

Offline-research implementation of PsyVEC built from the characterized
AgentMental baseline. The project is not a clinical diagnostic system and must
not learn autonomously from live patients.

Implementation follows the gated Phase 0-8 roadmap in
[`plan.md`](plan.md) and [`docs/implementation/`](docs/implementation/).

## Current status

Phase 0 static baseline freeze and offline characterization are implemented.
The end-to-end Phase 0 gate remains blocked until an approved model endpoint,
compatible legacy environment, and authorized or approved synthetic data
fixture are available. See
[`PHASE_0_BASELINE_REPORT.md`](docs/implementation/PHASE_0_BASELINE_REPORT.md).

Phase 1 has started with offline backend contracts, legacy compatibility
adapters, layered secret-safe configuration, and contract tests. Runtime
integration/parity remains blocked by the Phase 0 gate. See
[`PHASE_1_BACKEND_REPORT.md`](docs/implementation/PHASE_1_BACKEND_REPORT.md).

## Offline checks

These commands do not call a model or access participant data:

```sh
python3 scripts/verify_agentmental_baseline.py
python3 -m unittest discover -s tests -p 'test_*.py'
ruff check src tests/unit
mypy --strict src/psyvec
```
