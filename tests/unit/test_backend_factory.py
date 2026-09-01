from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.config import BackendConfig, ConfigurationError  # noqa: E402
from psyvec.model import (  # noqa: E402
    LegacyAutoGenBackend,
    Message,
    ModelRequest,
    OpenAICompatibleBackend,
    build_model_backend,
)


def request() -> ModelRequest:
    return ModelRequest(
        request_id="factory-request",
        role="scorer",
        messages=(Message(role="user", content="score this"),),
        model="configured-model",
    )


class BackendFactoryTests(unittest.TestCase):
    def test_factory_builds_openai_backend_with_injected_client(self) -> None:
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="factory response"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_arguments: completion)
            )
        )
        captured: dict[str, str] = {}

        def client_factory(*, api_base_url: str, api_key: str):
            captured.update(
                api_base_url=api_base_url,
                api_key=api_key,
            )
            return client

        config = BackendConfig(
            backend="openai-compatible",
            api_base_url="https://offline.example/v1",
            api_key="test-only-secret",
            model="configured-model",
        )
        backend = build_model_backend(config, client_factory=client_factory)

        self.assertIsInstance(backend, OpenAICompatibleBackend)
        self.assertEqual(
            captured,
            {
                "api_base_url": "https://offline.example/v1",
                "api_key": "test-only-secret",
            },
        )
        response = backend.generate(request())
        self.assertEqual(response.content, "factory response")
        self.assertEqual(response.provenance.config_hash, config.provenance_hash)

    def test_factory_builds_legacy_autogen_backend(self) -> None:
        calls = []
        manager = object()
        proxy = object()
        scorer = object()

        def make_request(actual_manager, actual_proxy, recipient, payload):
            calls.append((actual_manager, actual_proxy, recipient, payload))
            return "legacy factory response"

        config = BackendConfig(backend="legacy-autogen")
        backend = build_model_backend(
            config,
            make_request=make_request,
            group_chat_manager=manager,
            user_proxy=proxy,
            role_agents={"scorer": scorer},
        )

        self.assertIsInstance(backend, LegacyAutoGenBackend)
        response = backend.generate(request())
        self.assertEqual(calls, [(manager, proxy, scorer, "score this")])
        self.assertEqual(response.provenance.config_hash, config.provenance_hash)

    def test_feature_flag_selects_legacy_without_openai_credentials(self) -> None:
        config = BackendConfig(
            backend="openai-compatible",
            use_backend_abstraction=False,
        )
        backend = build_model_backend(
            config,
            make_request=lambda *_arguments: "legacy response",
            role_agents={"scorer": object()},
        )

        self.assertIsInstance(backend, LegacyAutoGenBackend)

    def test_openai_compatible_backend_requires_credentials(self) -> None:
        for config in (
            BackendConfig(backend="openai-compatible", api_key="test-only-secret"),
            BackendConfig(
                backend="openai-compatible",
                api_base_url="https://offline.example/v1",
            ),
        ):
            with self.subTest(config=config.redacted_dict()), self.assertRaises(
                ConfigurationError
            ):
                build_model_backend(config)


if __name__ == "__main__":
    unittest.main()
