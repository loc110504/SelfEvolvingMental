"""Content-minimized trace capture and Phase 1 parity comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter

from psyvec.model.backend import (
    BackendError,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    ResponseProvenance,
)


@dataclass(frozen=True, slots=True)
class BackendTraceEntry:
    request_id: str
    role: str
    request_hash: str
    response_hash: str | None
    outcome: str
    latency_ms: float
    failure_type: str | None = None
    provenance: ResponseProvenance | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")


@dataclass(frozen=True, slots=True)
class ParityResult:
    baseline_count: int
    candidate_count: int
    control_flow_mismatches: tuple[int, ...]
    request_mismatches: tuple[int, ...]
    response_mismatches: tuple[int, ...]
    outcome_mismatches: tuple[int, ...]
    provenance_invariant_mismatches: tuple[int, ...] = ()
    provenance_metadata_differences: tuple[int, ...] = ()
    latency_deltas_ms: tuple[float, ...] = ()
    total_latency_regression_ms: float = 0.0
    max_latency_regression_ms: float = 0.0
    mean_latency_regression_ms: float = 0.0

    def passes(
        self,
        *,
        allowed_response_mismatches: int,
        latency_budget_ms: float | None = None,
    ) -> bool:
        """Return whether required parity criteria and an optional slowdown cap pass.

        When supplied, ``latency_budget_ms`` caps the largest per-entry candidate
        slowdown. Faster candidate entries have zero regression and do not consume
        the budget.
        """

        if allowed_response_mismatches < 0:
            raise ValueError("allowed_response_mismatches cannot be negative")
        if latency_budget_ms is not None and latency_budget_ms < 0:
            raise ValueError("latency_budget_ms cannot be negative")
        return (
            self.baseline_count == self.candidate_count
            and not self.control_flow_mismatches
            and not self.request_mismatches
            and not self.outcome_mismatches
            and not self.provenance_invariant_mismatches
            and len(self.response_mismatches) <= allowed_response_mismatches
            and (
                latency_budget_ms is None
                or self.max_latency_regression_ms <= latency_budget_ms
            )
        )


class TraceRecorderBackend:
    """Wrap a backend and retain minimized trace evidence, never raw content."""

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend
        self.entries: list[BackendTraceEntry] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        request_hash = _request_hash(request)
        started = perf_counter()
        try:
            response = self._backend.generate(request)
        except BackendError as error:
            self.entries.append(
                BackendTraceEntry(
                    request_id=request.request_id,
                    role=request.role,
                    request_hash=request_hash,
                    response_hash=None,
                    outcome="failed",
                    failure_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
            raise
        self.entries.append(
            BackendTraceEntry(
                request_id=request.request_id,
                role=request.role,
                request_hash=request_hash,
                response_hash=_text_hash(response.content),
                outcome="completed",
                latency_ms=(perf_counter() - started) * 1000,
                provenance=response.provenance,
            )
        )
        return response


def compare_traces(
    baseline: list[BackendTraceEntry] | tuple[BackendTraceEntry, ...],
    candidate: list[BackendTraceEntry] | tuple[BackendTraceEntry, ...],
) -> ParityResult:
    paired_count = min(len(baseline), len(candidate))
    control_flow_mismatches = []
    request_mismatches = []
    response_mismatches = []
    outcome_mismatches = []
    provenance_invariant_mismatches = []
    provenance_metadata_differences = []
    latency_deltas_ms = []
    for index in range(paired_count):
        before = baseline[index]
        after = candidate[index]
        if (before.request_id, before.role) != (after.request_id, after.role):
            control_flow_mismatches.append(index)
        if before.request_hash != after.request_hash:
            request_mismatches.append(index)
        if (before.outcome, before.failure_type) != (
            after.outcome,
            after.failure_type,
        ):
            outcome_mismatches.append(index)
        if before.response_hash != after.response_hash:
            response_mismatches.append(index)
        if _provenance_invariants(before.provenance) != _provenance_invariants(
            after.provenance
        ):
            provenance_invariant_mismatches.append(index)
        if _provenance_metadata(before.provenance) != _provenance_metadata(
            after.provenance
        ):
            provenance_metadata_differences.append(index)
        latency_deltas_ms.append(after.latency_ms - before.latency_ms)
    latency_regressions_ms = [max(delta, 0.0) for delta in latency_deltas_ms]
    total_latency_regression_ms = sum(latency_regressions_ms)
    return ParityResult(
        baseline_count=len(baseline),
        candidate_count=len(candidate),
        control_flow_mismatches=tuple(control_flow_mismatches),
        request_mismatches=tuple(request_mismatches),
        response_mismatches=tuple(response_mismatches),
        outcome_mismatches=tuple(outcome_mismatches),
        provenance_invariant_mismatches=tuple(provenance_invariant_mismatches),
        provenance_metadata_differences=tuple(provenance_metadata_differences),
        latency_deltas_ms=tuple(latency_deltas_ms),
        total_latency_regression_ms=total_latency_regression_ms,
        max_latency_regression_ms=max(latency_regressions_ms, default=0.0),
        mean_latency_regression_ms=(
            total_latency_regression_ms / paired_count if paired_count else 0.0
        ),
    )


def _provenance_invariants(
    provenance: ResponseProvenance | None,
) -> tuple[str, str, str | None, str | None] | None:
    if provenance is None:
        return None
    return (
        provenance.model,
        provenance.prompt_version,
        provenance.adapter_id,
        provenance.adapter_hash,
    )


def _provenance_metadata(
    provenance: ResponseProvenance | None,
) -> tuple[str, str] | None:
    if provenance is None:
        return None
    return provenance.backend, provenance.config_hash


def _request_hash(request: ModelRequest) -> str:
    canonical = {
        "max_tokens": request.max_tokens,
        "messages": [
            {"content": message.content, "role": message.role}
            for message in request.messages
        ],
        "model": request.model,
        "prompt_version": request.prompt_version,
        "request_id": request.request_id,
        "role": request.role,
        "temperature": request.temperature,
    }
    encoded = json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
