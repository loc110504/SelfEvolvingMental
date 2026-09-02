from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.model import (  # noqa: E402
    BackendResponseError,
    BackendTraceEntry,
    LegacyAutoGenBackend,
    Message,
    ModelRequest,
    ModelResponse,
    ResponseProvenance,
    TraceRecorderBackend,
    compare_traces,
)


def make_request(content: str = "same payload") -> ModelRequest:
    return ModelRequest(
        request_id="request-1",
        role="interviewer",
        messages=(Message(role="user", content=content),),
        model="baseline-model",
    )


class StaticBackend:
    def __init__(self, response: ModelResponse) -> None:
        self._response = response

    def generate(self, _request: ModelRequest) -> ModelResponse:
        return self._response


def make_response(
    *,
    content: str = "same response",
    backend: str = "test-backend",
    config_hash: str = "config-hash",
    model: str = "baseline-model",
    prompt_version: str = "agentmental-v1",
    adapter_id: str | None = None,
    adapter_hash: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        request_id="request-1",
        content=content,
        finish_reason="stop",
        usage={},
        latency_ms=500.0,
        provenance=ResponseProvenance(
            backend=backend,
            model=model,
            prompt_version=prompt_version,
            config_hash=config_hash,
            adapter_id=adapter_id,
            adapter_hash=adapter_hash,
        ),
    )


class BackendParityTests(unittest.TestCase):
    def test_trace_entry_requires_latency(self) -> None:
        with self.assertRaises(TypeError):
            BackendTraceEntry(
                request_id="request-1",
                role="interviewer",
                request_hash="request-hash",
                response_hash="response-hash",
                outcome="completed",
            )

    def test_identical_paths_pass_with_zero_tolerance(self) -> None:
        baseline = TraceRecorderBackend(
            LegacyAutoGenBackend(lambda _request: "same", config_hash="same")
        )
        candidate = TraceRecorderBackend(
            LegacyAutoGenBackend(lambda _request: "same", config_hash="same")
        )

        baseline.generate(make_request())
        candidate.generate(make_request())

        result = compare_traces(baseline.entries, candidate.entries)
        self.assertTrue(result.passes(allowed_response_mismatches=0))

    def test_response_tolerance_is_explicit_not_hard_coded(self) -> None:
        baseline = TraceRecorderBackend(
            LegacyAutoGenBackend(lambda _request: "before", config_hash="same")
        )
        candidate = TraceRecorderBackend(
            LegacyAutoGenBackend(lambda _request: "after", config_hash="same")
        )
        baseline.generate(make_request())
        candidate.generate(make_request())

        result = compare_traces(baseline.entries, candidate.entries)
        self.assertFalse(result.passes(allowed_response_mismatches=0))
        self.assertTrue(result.passes(allowed_response_mismatches=1))

    def test_prompt_or_control_flow_change_cannot_be_tolerated(self) -> None:
        baseline = TraceRecorderBackend(
            LegacyAutoGenBackend(lambda _request: "same", config_hash="same")
        )
        candidate = TraceRecorderBackend(
            LegacyAutoGenBackend(lambda _request: "same", config_hash="same")
        )
        baseline.generate(make_request("before payload"))
        candidate.generate(make_request("after payload"))

        result = compare_traces(baseline.entries, candidate.entries)
        self.assertFalse(result.passes(allowed_response_mismatches=1))
        self.assertEqual(result.request_mismatches, (0,))

    def test_success_trace_records_latency_and_provenance(self) -> None:
        recorder = TraceRecorderBackend(StaticBackend(make_response(adapter_id="lora")))

        with patch("psyvec.model.parity.perf_counter", side_effect=(10.0, 10.125)):
            recorder.generate(make_request())

        entry = recorder.entries[0]
        self.assertAlmostEqual(entry.latency_ms, 125.0)
        self.assertEqual(entry.provenance, make_response(adapter_id="lora").provenance)

    def test_failure_trace_records_latency_without_raw_content(self) -> None:
        recorder = TraceRecorderBackend(
            LegacyAutoGenBackend(lambda _request: None, config_hash="hash")
        )
        with (
            patch("psyvec.model.parity.perf_counter", side_effect=(20.0, 20.075)),
            self.assertRaises(BackendResponseError),
        ):
            recorder.generate(make_request("sensitive participant text"))

        entry = recorder.entries[0]
        self.assertEqual(entry.outcome, "failed")
        self.assertEqual(entry.failure_type, "BackendResponseError")
        self.assertAlmostEqual(entry.latency_ms, 75.0)
        self.assertIsNone(entry.provenance)
        self.assertNotIn("sensitive participant text", repr(entry))

    def test_provenance_metadata_difference_is_reported_without_failing(self) -> None:
        for metadata_change in (
            {"backend": "openai-compatible"},
            {"config_hash": "candidate-config"},
        ):
            with self.subTest(metadata_change=metadata_change):
                baseline = TraceRecorderBackend(StaticBackend(make_response()))
                candidate = TraceRecorderBackend(
                    StaticBackend(make_response(**metadata_change))
                )
                baseline.generate(make_request())
                candidate.generate(make_request())

                result = compare_traces(baseline.entries, candidate.entries)

                self.assertEqual(result.provenance_metadata_differences, (0,))
                self.assertEqual(result.provenance_invariant_mismatches, ())
                self.assertTrue(result.passes(allowed_response_mismatches=0))

    def test_provenance_invariant_mismatches_fail(self) -> None:
        for differing_response in (
            make_response(model="candidate-model"),
            make_response(prompt_version="candidate-prompt-v1"),
        ):
            with self.subTest(response=differing_response):
                baseline = TraceRecorderBackend(StaticBackend(make_response()))
                candidate = TraceRecorderBackend(StaticBackend(differing_response))
                baseline.generate(make_request())
                candidate.generate(make_request())

                result = compare_traces(baseline.entries, candidate.entries)

                self.assertEqual(result.provenance_invariant_mismatches, (0,))
                self.assertFalse(result.passes(allowed_response_mismatches=0))

    def test_latency_only_difference_is_reported_without_failing(self) -> None:
        baseline = TraceRecorderBackend(StaticBackend(make_response()))
        candidate = TraceRecorderBackend(StaticBackend(make_response()))
        with patch("psyvec.model.parity.perf_counter", side_effect=(1.0, 1.010)):
            baseline.generate(make_request())
        with patch("psyvec.model.parity.perf_counter", side_effect=(2.0, 2.035)):
            candidate.generate(make_request())

        result = compare_traces(baseline.entries, candidate.entries)

        self.assertAlmostEqual(result.latency_deltas_ms[0], 25.0)
        self.assertAlmostEqual(result.total_latency_regression_ms, 25.0)
        self.assertAlmostEqual(result.max_latency_regression_ms, 25.0)
        self.assertAlmostEqual(result.mean_latency_regression_ms, 25.0)
        self.assertTrue(result.passes(allowed_response_mismatches=0))
        self.assertFalse(
            result.passes(allowed_response_mismatches=0, latency_budget_ms=20.0)
        )

    def test_faster_candidate_does_not_consume_latency_budget(self) -> None:
        baseline = TraceRecorderBackend(StaticBackend(make_response()))
        candidate = TraceRecorderBackend(StaticBackend(make_response()))
        with patch("psyvec.model.parity.perf_counter", side_effect=(1.0, 1.035)):
            baseline.generate(make_request())
        with patch("psyvec.model.parity.perf_counter", side_effect=(2.0, 2.010)):
            candidate.generate(make_request())

        result = compare_traces(baseline.entries, candidate.entries)

        self.assertAlmostEqual(result.latency_deltas_ms[0], -25.0)
        self.assertEqual(result.max_latency_regression_ms, 0.0)
        self.assertEqual(result.mean_latency_regression_ms, 0.0)
        self.assertTrue(
            result.passes(allowed_response_mismatches=0, latency_budget_ms=0.0)
        )

    def test_latency_budget_uses_maximum_regression_not_total_regression(self) -> None:
        baseline = TraceRecorderBackend(StaticBackend(make_response()))
        candidate = TraceRecorderBackend(StaticBackend(make_response()))
        for start in (1.0, 3.0):
            with patch(
                "psyvec.model.parity.perf_counter",
                side_effect=(start, start + 0.010),
            ):
                baseline.generate(make_request())
            with patch(
                "psyvec.model.parity.perf_counter",
                side_effect=(start + 1.0, start + 1.015),
            ):
                candidate.generate(make_request())

        result = compare_traces(baseline.entries, candidate.entries)

        self.assertAlmostEqual(result.total_latency_regression_ms, 10.0)
        self.assertAlmostEqual(result.max_latency_regression_ms, 5.0)
        self.assertAlmostEqual(result.mean_latency_regression_ms, 5.0)
        self.assertTrue(
            result.passes(allowed_response_mismatches=0, latency_budget_ms=5.0)
        )

    def test_trace_entries_never_store_raw_prompt_or_response_content(self) -> None:
        prompt = "sensitive participant prompt"
        response = "sensitive participant response"
        recorder = TraceRecorderBackend(StaticBackend(make_response(content=response)))

        recorder.generate(make_request(prompt))

        trace = repr(recorder.entries[0])
        self.assertNotIn(prompt, trace)
        self.assertNotIn(response, trace)


if __name__ == "__main__":
    unittest.main()
