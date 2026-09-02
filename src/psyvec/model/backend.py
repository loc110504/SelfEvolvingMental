"""Core model backend contract shared by runtime integrations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


class BackendError(RuntimeError):
    """Base class for visible model backend failures."""


class BackendTimeoutError(BackendError):
    """The backend did not finish within its configured timeout."""


class BackendResponseError(BackendError):
    """The backend returned a missing or malformed response."""


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be text")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    request_id: str
    role: str
    messages: tuple[Message, ...]
    model: str
    prompt_version: str = "agentmental-v1"
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.role.strip():
            raise ValueError("role must not be empty")
        if not self.messages:
            raise ValueError("at least one message is required")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ResponseProvenance:
    backend: str
    model: str
    prompt_version: str
    config_hash: str
    adapter_id: str | None = None
    adapter_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    request_id: str
    content: str
    finish_reason: str | None
    usage: Mapping[str, int]
    latency_ms: float
    provenance: ResponseProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise BackendResponseError("backend returned empty content")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))

    def parse_json(self) -> Any:
        """Parse strict JSON while turning legacy silent fallbacks into failures."""

        try:
            return json.loads(self.content)
        except json.JSONDecodeError as error:
            raise BackendResponseError(
                f"response for request {self.request_id!r} is not valid JSON"
            ) from error


@runtime_checkable
class ModelBackend(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response or raise an explicit BackendError."""
