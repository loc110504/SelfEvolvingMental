"""Layered, secret-safe configuration for model backends."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

_REDACTED = "<redacted>"
_ENVIRONMENT_FIELDS = {
    "PSYVEC_BACKEND": "backend",
    "PSYVEC_USE_BACKEND_ABSTRACTION": "use_backend_abstraction",
    "PSYVEC_API_BASE_URL": "api_base_url",
    "PSYVEC_API_KEY": "api_key",
    "PSYVEC_MODEL": "model",
    "PSYVEC_TIMEOUT_SECONDS": "timeout_seconds",
    "PSYVEC_TEMPERATURE": "temperature",
    "PSYVEC_MAX_TOKENS": "max_tokens",
}
_CONFIG_FIELDS = frozenset(_ENVIRONMENT_FIELDS.values())
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BACKEND_MANIFEST_PATH = _PROJECT_ROOT / "configs" / "base" / "backend.json"
_MANIFEST_SECRET_FIELDS = frozenset({"api_key"})


class ConfigurationError(ValueError):
    """Raised when backend configuration is invalid."""


def load_backend_manifest(
    path: str | Path = _DEFAULT_BACKEND_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load a non-secret backend manifest from a JSON object.

    Credentials are deliberately accepted only from the environment or explicit
    overrides, never from a versioned manifest file.
    """

    manifest_path = Path(path)
    try:
        raw_manifest = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigurationError(
            f"backend manifest file does not exist: {manifest_path}"
        ) from error
    except OSError as error:
        raise ConfigurationError(
            f"backend manifest file could not be read: {manifest_path}"
        ) from error

    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"backend manifest file contains malformed JSON: {manifest_path}"
        ) from error

    if not isinstance(manifest, dict):
        raise ConfigurationError("backend manifest must contain a JSON object")

    unknown = set(manifest) - _CONFIG_FIELDS
    if unknown:
        raise ConfigurationError(
            "unknown manifest backend fields: "
            f"{', '.join(sorted(unknown))}"
        )
    secret_fields = set(manifest) & _MANIFEST_SECRET_FIELDS
    if secret_fields:
        raise ConfigurationError(
            "backend manifest must not contain secret fields: "
            f"{', '.join(sorted(secret_fields))}"
        )
    return manifest


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Resolved Phase 1 backend configuration.

    Secrets are kept in memory but excluded from serialized diagnostics and the
    reproducibility hash.
    """

    backend: str = "legacy-autogen"
    use_backend_abstraction: bool = True
    api_base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    model: str = "qwen2.5-72b"
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ConfigurationError("backend must not be empty")
        if not self.model.strip():
            raise ConfigurationError("model must not be empty")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive")
        if not 0 <= self.temperature <= 2:
            raise ConfigurationError("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ConfigurationError("max_tokens must be positive")

    def redacted_dict(self) -> dict[str, Any]:
        """Return configuration safe for logs and manifests."""

        return {
            "api_base_url": self.api_base_url,
            "api_key": _REDACTED if self.api_key else None,
            "backend": self.backend,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "use_backend_abstraction": self.use_backend_abstraction,
        }

    @property
    def provenance_hash(self) -> str:
        """Hash the canonical non-secret resolved configuration."""

        encoded = json.dumps(
            self.redacted_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_backend_config(
    *,
    environment: Mapping[str, str] | None = None,
    manifest: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    defaults: BackendConfig | None = None,
) -> BackendConfig:
    """Resolve defaults < environment < run manifest < explicit overrides."""

    resolved = defaults or BackendConfig()
    env_values = _read_environment(
        environment if environment is not None else os.environ
    )
    for layer_name, values in (
        ("environment", env_values),
        ("manifest", manifest or {}),
        ("overrides", overrides or {}),
    ):
        unknown = set(values) - _CONFIG_FIELDS
        if unknown:
            raise ConfigurationError(
                f"unknown {layer_name} backend fields: {', '.join(sorted(unknown))}"
            )
        if values:
            resolved = replace(resolved, **_coerce_values(values, layer_name))
    return resolved


def _read_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        field: environment[name]
        for name, field in _ENVIRONMENT_FIELDS.items()
        if name in environment
    }


def _coerce_values(values: Mapping[str, Any], layer_name: str) -> dict[str, Any]:
    coerced = dict(values)
    try:
        if "use_backend_abstraction" in coerced:
            coerced["use_backend_abstraction"] = _parse_bool(
                coerced["use_backend_abstraction"]
            )
        if "timeout_seconds" in coerced:
            coerced["timeout_seconds"] = float(coerced["timeout_seconds"])
        if "temperature" in coerced:
            coerced["temperature"] = float(coerced["temperature"])
        if "max_tokens" in coerced:
            coerced["max_tokens"] = int(coerced["max_tokens"])
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            f"invalid value in {layer_name} configuration"
        ) from error
    return coerced


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigurationError(f"invalid boolean value: {value!r}")
