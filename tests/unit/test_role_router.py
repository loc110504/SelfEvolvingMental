from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.model import (  # noqa: E402
    ROLE_ALIASES,
    AdapterManifest,
    AdapterRegistry,
    AdapterRoutingError,
    Message,
    ModelRequest,
    ModelResponse,
    ResponseProvenance,
    RoleModelRouter,
)


class RecordingBackend:
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            request_id=request.request_id,
            content="response",
            finish_reason="stop",
            usage={},
            latency_ms=0.0,
            provenance=ResponseProvenance(
                backend="recording",
                model=request.model,
                prompt_version=request.prompt_version,
                config_hash="config-hash",
            ),
        )


def request(role: str) -> ModelRequest:
    return ModelRequest(
        request_id=role,
        role=role,
        messages=(Message(role="user", content="payload"),),
        model="base-model",
    )


def manifest(role: str, **changes: str) -> AdapterManifest:
    values = {
        "adapter_id": f"{role}-adapter",
        "role": role,
        "base_model_hash": "base-hash",
        "adapter_hash": f"{role}-hash",
        "status": "accepted",
    }
    values.update(changes)
    return AdapterManifest(**values)  # type: ignore[arg-type]


def router(manifests: dict[tuple[str, str], AdapterManifest]) -> RoleModelRouter:
    return RoleModelRouter(
        backend=RecordingBackend(),
        registry=AdapterRegistry(manifests),
        base_model_hash="base-hash",
        accepted_policy_id="policy-1",
    )


class RoleModelRouterTests(unittest.TestCase):
    def test_four_role_mapping(self) -> None:
        actual_router = router(
            {("policy-1", role): manifest(role) for role in ROLE_ALIASES}
        )

        for role in ROLE_ALIASES:
            response = actual_router.generate(request(role))
            self.assertEqual(response.provenance.adapter_id, f"{role}-adapter")
            self.assertEqual(response.provenance.adapter_hash, f"{role}-hash")

    def test_wrong_adapter_rejection(self) -> None:
        actual_router = router(
            {("policy-1", "psyvec-scorer"): manifest("psyvec-evaluator")}
        )

        with self.assertRaises(AdapterRoutingError):
            actual_router.generate(request("psyvec-scorer"))

    def test_base_model_incompatibility_rejection(self) -> None:
        actual_router = router(
            {
                ("policy-1", "psyvec-scorer"): manifest(
                    "psyvec-scorer", base_model_hash="different-base"
                )
            }
        )

        with self.assertRaises(AdapterRoutingError):
            actual_router.generate(request("psyvec-scorer"))

    def test_candidate_adapter_is_non_routable(self) -> None:
        actual_router = router(
            {
                ("policy-1", "psyvec-scorer"): manifest(
                    "psyvec-scorer", status="candidate"
                )
            }
        )

        with self.assertRaises(AdapterRoutingError):
            actual_router.generate(request("psyvec-scorer"))

    def test_adapter_bleed(self) -> None:
        actual_router = router(
            {
                ("policy-1", "psyvec-interviewer"): manifest("psyvec-interviewer"),
                ("policy-1", "psyvec-evaluator"): manifest("psyvec-evaluator"),
            }
        )

        first = actual_router.generate(request("psyvec-interviewer"))
        second = actual_router.generate(request("psyvec-evaluator"))

        self.assertEqual(first.provenance.adapter_id, "psyvec-interviewer-adapter")
        self.assertEqual(second.provenance.adapter_id, "psyvec-evaluator-adapter")


if __name__ == "__main__":
    unittest.main()
