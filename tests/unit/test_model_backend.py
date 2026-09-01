from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.config import BackendConfig  # noqa: E402
from psyvec.model import (  # noqa: E402
    BackendResponseError,
    BackendTimeoutError,
    LegacyAutoGenBackend,
    LegacyModelPort,
    Message,
    ModelBackend,
    ModelRequest,
    OpenAICompatibleBackend,
    select_backend,
)


def request(*, content: str = "payload") -> ModelRequest:
    return ModelRequest(
        request_id="request-1",
        role="scorer",
        messages=(Message(role="user", content=content),),
        model="baseline-model",
        max_tokens=128,
    )


class LegacyAutoGenBackendTests(unittest.TestCase):
    def test_bridge_preserves_payload_and_recipient(self) -> None:
        calls = []
        manager = object()
        proxy = object()
        scorer = object()

        def make_request(actual_manager, actual_proxy, recipient, payload):
            calls.append((actual_manager, actual_proxy, recipient, payload))
            return '{"score": 2, "summary": "basis"}'

        backend = LegacyAutoGenBackend.from_makerequest(
            make_request=make_request,
            group_chat_manager=manager,
            user_proxy=proxy,
            role_agents={"scorer": scorer},
            config_hash="config-hash",
        )

        self.assertIsInstance(backend, ModelBackend)
        response = backend.generate(request())
        self.assertEqual(calls, [(manager, proxy, scorer, "payload")])
        self.assertEqual(response.parse_json()["score"], 2)
        self.assertEqual(response.provenance.backend, "legacy-autogen")

    def test_missing_content_is_an_explicit_failure(self) -> None:
        backend = LegacyAutoGenBackend(lambda _request: None, config_hash="hash")
        with self.assertRaises(BackendResponseError):
            backend.generate(request())

    def test_timeout_is_an_explicit_failure(self) -> None:
        def time_out(_request):
            raise TimeoutError("late")

        backend = LegacyAutoGenBackend(time_out, config_hash="hash")
        with self.assertRaises(BackendTimeoutError):
            backend.generate(request())


class FakeCompletions:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.arguments = None

    def create(self, **arguments):
        self.arguments = arguments
        if self.error:
            raise self.error
        return self.result


class OpenAICompatibleBackendTests(unittest.TestCase):
    def test_direct_client_call_uses_the_shared_contract(self) -> None:
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="extracted"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )
        completions = FakeCompletions(result=completion)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        backend = OpenAICompatibleBackend(client, config_hash="config-hash")

        response = backend.generate(request(content="extract memory"))

        self.assertEqual(response.content, "extracted")
        self.assertEqual(response.usage["total_tokens"], 7)
        self.assertEqual(
            completions.arguments,
            {
                "model": "baseline-model",
                "messages": [{"role": "user", "content": "extract memory"}],
                "temperature": 0.0,
                "max_tokens": 128,
            },
        )

    def test_invalid_response_shape_fails_closed(self) -> None:
        completions = FakeCompletions(result=SimpleNamespace(choices=[]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        backend = OpenAICompatibleBackend(client, config_hash="hash")
        with self.assertRaises(BackendResponseError):
            backend.generate(request())

    def test_timeout_fails_closed(self) -> None:
        completions = FakeCompletions(error=TimeoutError("late"))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        backend = OpenAICompatibleBackend(client, config_hash="hash")
        with self.assertRaises(BackendTimeoutError):
            backend.generate(request())

    def test_malformed_json_is_not_converted_to_a_score(self) -> None:
        response = LegacyAutoGenBackend(
            lambda _request: "not-json", config_hash="hash"
        ).generate(request())
        with self.assertRaises(BackendResponseError):
            response.parse_json()


class LegacyModelPortTests(unittest.TestCase):
    def test_memory_and_simulator_prompts_use_shared_request_contract(self) -> None:
        captured = []

        def handler(actual_request):
            captured.append(actual_request)
            return "legacy response"

        port = LegacyModelPort(
            backend=LegacyAutoGenBackend(handler, config_hash="hash"),
            model="baseline-model",
            prompt_version="memory-v1",
            max_tokens=512,
        )

        response = port.complete(
            request_id="memory-1",
            role="memory-extractor",
            system_prompt="extract only explicit facts",
            user_prompt="participant response",
        )

        self.assertEqual(response.content, "legacy response")
        self.assertEqual(captured[0].role, "memory-extractor")
        self.assertEqual(captured[0].messages[0].role, "system")
        self.assertEqual(captured[0].messages[1].content, "participant response")


class BackendSelectionTests(unittest.TestCase):
    def test_feature_flag_selects_abstraction_or_legacy_path(self) -> None:
        abstraction = LegacyAutoGenBackend(lambda _request: "new", config_hash="a")
        legacy = LegacyAutoGenBackend(lambda _request: "old", config_hash="b")

        self.assertIs(
            select_backend(
                BackendConfig(use_backend_abstraction=True),
                abstraction_backend=abstraction,
                legacy_backend=legacy,
            ),
            abstraction,
        )
        self.assertIs(
            select_backend(
                BackendConfig(use_backend_abstraction=False),
                abstraction_backend=abstraction,
                legacy_backend=legacy,
            ),
            legacy,
        )


if __name__ == "__main__":
    unittest.main()
