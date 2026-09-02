import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.integration import (  # noqa: E402
    LegacyPortInstaller,
    PatchTarget,
    build_autogen_role_handler,
    build_direct_completion,
)
from psyvec.model.backend import (  # noqa: E402
    BackendResponseError,
    ModelRequest,
    ModelResponse,
    ResponseProvenance,
)
from psyvec.model.legacy_ports import LegacyModelPort  # noqa: E402


class RecordingBackend:
    def __init__(self, content: str = "routed", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ModelResponse(
            request_id=request.request_id,
            content=self.content,
            finish_reason="stop",
            usage={},
            latency_ms=1.0,
            provenance=ResponseProvenance(
                backend="fake",
                model=request.model,
                prompt_version=request.prompt_version,
                config_hash="hash",
            ),
        )


def port(backend: RecordingBackend) -> LegacyModelPort:
    return LegacyModelPort(
        backend=backend, model="configured-model", prompt_version="agentmental-v1"
    )


class LegacyBridgeTests(unittest.TestCase):
    def test_autogen_role_handler_forwards_payload_verbatim(self) -> None:
        backend = RecordingBackend()
        scorer = SimpleNamespace(name="ScoringAgent")
        make_request = build_autogen_role_handler(
            port(backend), role_of=lambda recipient: recipient.name
        )

        payload = "Topic: sleep\nHistory:\nQ: a"
        result = make_request(object(), object(), scorer, payload)

        self.assertEqual(result, "routed")
        self.assertEqual(backend.requests[0].role, "ScoringAgent")
        self.assertEqual(backend.requests[0].messages[-1].content, payload)

    def test_direct_completion_routes_memory_and_simulator_shape(self) -> None:
        backend = RecordingBackend(content="simulated answer")
        complete = build_direct_completion(
            port(backend), role="participant-simulator", request_id_prefix="sim"
        )

        result = complete("system context", "please answer")

        self.assertEqual(result, "simulated answer")
        roles = [message.role for message in backend.requests[0].messages]
        self.assertEqual(roles, ["system", "user"])

    def test_backend_failure_returns_none_like_the_legacy_call(self) -> None:
        backend = RecordingBackend(error=BackendResponseError("bad"))
        make_request = build_autogen_role_handler(
            port(backend), role_of=lambda _recipient: "scorer"
        )

        self.assertIsNone(make_request(object(), object(), object(), "payload"))

    def test_installer_redirects_then_restores_the_original_callable(self) -> None:
        def legacy(_manager, _proxy, _recipient, _payload):
            return "legacy"

        module = SimpleNamespace(makerequest=legacy)
        installer = LegacyPortInstaller(
            targets=(
                PatchTarget(
                    module=module, attribute="makerequest", replacement=lambda *_: "new"
                ),
            )
        )

        installer.install()
        self.assertTrue(installer.installed)
        self.assertEqual(module.makerequest(1, 2, 3, "p"), "new")

        installer.restore()
        self.assertFalse(installer.installed)
        self.assertIs(module.makerequest, legacy)

    def test_feature_flag_off_leaves_the_legacy_path_untouched(self) -> None:
        def legacy(_manager, _proxy, _recipient, _payload):
            return "legacy"

        module = SimpleNamespace(makerequest=legacy)
        installer = LegacyPortInstaller(
            targets=(
                PatchTarget(
                    module=module, attribute="makerequest", replacement=lambda *_: "new"
                ),
            ),
            enabled=False,
        )

        installer.install()

        self.assertFalse(installer.installed)
        self.assertIs(module.makerequest, legacy)


if __name__ == "__main__":
    unittest.main()
