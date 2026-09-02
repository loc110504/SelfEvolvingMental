"""Compatibility adapter for legacy direct OpenAI-style API calls."""

from __future__ import annotations

import time
from typing import Any

from psyvec.model.backend import (
    BackendError,
    BackendResponseError,
    BackendTimeoutError,
    ModelRequest,
    ModelResponse,
    ResponseProvenance,
)


class OpenAICompatibleBackend:
    """Wrap an injected OpenAI-compatible client used by memory/simulation."""

    def __init__(self, client: Any, *, config_hash: str) -> None:
        self._client = client
        self._config_hash = config_hash

    def generate(self, request: ModelRequest) -> ModelResponse:
        arguments: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            arguments["max_tokens"] = request.max_tokens

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**arguments)
        except TimeoutError as error:
            raise BackendTimeoutError(
                f"OpenAI-compatible request {request.request_id!r} timed out"
            ) from error
        except Exception as error:
            if "timeout" in type(error).__name__.lower():
                raise BackendTimeoutError(
                    f"OpenAI-compatible request {request.request_id!r} timed out"
                ) from error
            raise BackendError(
                f"OpenAI-compatible request {request.request_id!r} failed"
            ) from error
        latency_ms = (time.perf_counter() - started) * 1000

        try:
            choice = completion.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise BackendResponseError(
                "OpenAI-compatible request "
                f"{request.request_id!r} has an invalid response shape"
            ) from error
        usage = _extract_usage(getattr(completion, "usage", None))
        return ModelResponse(
            request_id=request.request_id,
            content=content,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            latency_ms=latency_ms,
            provenance=ResponseProvenance(
                backend="openai-compatible",
                model=request.model,
                prompt_version=request.prompt_version,
                config_hash=self._config_hash,
            ),
        )


def _extract_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    values = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            values[name] = value
    return values
