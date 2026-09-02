"""Runtime assembly for configured PsyVEC model backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from psyvec.config import BackendConfig, ConfigurationError
from psyvec.model.backend import ModelBackend
from psyvec.model.legacy_autogen_backend import LegacyAutoGenBackend
from psyvec.model.openai_compatible_backend import OpenAICompatibleBackend


class OpenAIClientFactory(Protocol):
    """Create an OpenAI-compatible client without making a network request."""

    def __call__(self, *, api_base_url: str, api_key: str) -> Any:
        """Return a configured client."""


MakeRequest = Callable[[Any, Any, Any, str], str | None]


def build_model_backend(
    config: BackendConfig,
    *,
    client_factory: OpenAIClientFactory | None = None,
    make_request: MakeRequest | None = None,
    group_chat_manager: Any = None,
    user_proxy: Any = None,
    role_agents: Mapping[str, Any] | None = None,
) -> ModelBackend:
    """Assemble the backend selected by ``config`` from injected dependencies.

    The factory evaluates the feature flag before construction, so a rollback
    to the legacy path does not require OpenAI-compatible credentials or its
    package.
    """

    if config.backend not in {"legacy-autogen", "openai-compatible"}:
        raise ConfigurationError(f"unsupported backend: {config.backend!r}")

    if config.use_backend_abstraction:
        return _build_configured_backend(
            config,
            client_factory=client_factory,
            make_request=make_request,
            group_chat_manager=group_chat_manager,
            user_proxy=user_proxy,
            role_agents=role_agents,
        )
    return _build_legacy_backend(
        config,
        make_request=make_request,
        group_chat_manager=group_chat_manager,
        user_proxy=user_proxy,
        role_agents=role_agents,
    )


def _build_configured_backend(
    config: BackendConfig,
    *,
    client_factory: OpenAIClientFactory | None,
    make_request: MakeRequest | None,
    group_chat_manager: Any,
    user_proxy: Any,
    role_agents: Mapping[str, Any] | None,
) -> ModelBackend:
    if config.backend == "openai-compatible":
        if not config.api_base_url:
            raise ConfigurationError(
                "openai-compatible backend requires api_base_url"
            )
        if not config.api_key:
            raise ConfigurationError("openai-compatible backend requires api_key")
        factory = client_factory or _default_openai_client_factory
        client = factory(
            api_base_url=config.api_base_url,
            api_key=config.api_key,
        )
        return OpenAICompatibleBackend(client, config_hash=config.provenance_hash)

    return _build_legacy_backend(
        config,
        make_request=make_request,
        group_chat_manager=group_chat_manager,
        user_proxy=user_proxy,
        role_agents=role_agents,
    )


def _build_legacy_backend(
    config: BackendConfig,
    *,
    make_request: MakeRequest | None,
    group_chat_manager: Any,
    user_proxy: Any,
    role_agents: Mapping[str, Any] | None,
) -> LegacyAutoGenBackend:
    if make_request is None:
        raise ConfigurationError("legacy-autogen backend requires make_request")
    if role_agents is None:
        raise ConfigurationError("legacy-autogen backend requires role_agents")
    return LegacyAutoGenBackend.from_makerequest(
        make_request=make_request,
        group_chat_manager=group_chat_manager,
        user_proxy=user_proxy,
        role_agents=role_agents,
        config_hash=config.provenance_hash,
    )


def _default_openai_client_factory(*, api_base_url: str, api_key: str) -> Any:
    """Construct the optional OpenAI client only when this backend is selected."""

    try:
        from openai import OpenAI
    except ImportError as error:
        raise ConfigurationError(
            "openai-compatible backend requires the optional openai package"
        ) from error
    return OpenAI(base_url=api_base_url, api_key=api_key)
