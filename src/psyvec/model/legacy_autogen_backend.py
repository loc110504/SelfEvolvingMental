"""AutoGen compatibility bridge without importing AutoGen at contract level."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from psyvec.model.backend import (
    BackendError,
    BackendResponseError,
    BackendTimeoutError,
    ModelRequest,
    ModelResponse,
    ResponseProvenance,
)

RequestHandler = Callable[[ModelRequest], str | None]


class LegacyAutoGenBackend:
    """Adapt the existing ``makerequest`` path to ``ModelBackend``.

    The injected handler keeps this module importable in the offline test
    environment where AutoGen is intentionally unavailable.
    """

    def __init__(self, handler: RequestHandler, *, config_hash: str) -> None:
        self._handler = handler
        self._config_hash = config_hash

    @classmethod
    def from_makerequest(
        cls,
        *,
        make_request: Callable[[Any, Any, Any, str], str | None],
        group_chat_manager: Any,
        user_proxy: Any,
        role_agents: Mapping[str, Any],
        config_hash: str,
    ) -> LegacyAutoGenBackend:
        """Build a bridge that preserves the upstream one-payload call shape."""

        def handler(request: ModelRequest) -> str | None:
            try:
                recipient = role_agents[request.role]
            except KeyError as error:
                raise BackendError(
                    f"no legacy agent configured for {request.role!r}"
                ) from error
            payload = request.messages[-1].content
            return make_request(group_chat_manager, user_proxy, recipient, payload)

        return cls(handler, config_hash=config_hash)

    def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        try:
            content = self._handler(request)
        except TimeoutError as error:
            raise BackendTimeoutError(
                f"legacy AutoGen request {request.request_id!r} timed out"
            ) from error
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(
                f"legacy AutoGen request {request.request_id!r} failed"
            ) from error
        latency_ms = (time.perf_counter() - started) * 1000
        if content is None:
            raise BackendResponseError(
                f"legacy AutoGen request {request.request_id!r} returned no content"
            )
        return ModelResponse(
            request_id=request.request_id,
            content=content,
            finish_reason=None,
            usage={},
            latency_ms=latency_ms,
            provenance=ResponseProvenance(
                backend="legacy-autogen",
                model=request.model,
                prompt_version=request.prompt_version,
                config_hash=self._config_hash,
            ),
        )
