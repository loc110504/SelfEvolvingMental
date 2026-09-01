"""Framework-neutral model interfaces and Phase 1 compatibility adapters."""

from psyvec.model.adapters import (
    ROLE_ALIASES,
    AdapterManifest,
    AdapterRegistry,
    AdapterRoutingError,
    RoleModelRouter,
    UnknownPolicyError,
    UnknownRoleError,
)
from psyvec.model.backend import (
    BackendError,
    BackendResponseError,
    BackendTimeoutError,
    Message,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    ResponseProvenance,
)
from psyvec.model.factory import OpenAIClientFactory, build_model_backend
from psyvec.model.legacy_autogen_backend import LegacyAutoGenBackend
from psyvec.model.legacy_ports import LegacyModelPort
from psyvec.model.openai_compatible_backend import OpenAICompatibleBackend
from psyvec.model.parity import (
    BackendTraceEntry,
    ParityResult,
    TraceRecorderBackend,
    compare_traces,
)
from psyvec.model.selection import select_backend

__all__ = [
    "ROLE_ALIASES",
    "AdapterManifest",
    "AdapterRegistry",
    "AdapterRoutingError",
    "BackendError",
    "BackendResponseError",
    "BackendTraceEntry",
    "BackendTimeoutError",
    "LegacyAutoGenBackend",
    "LegacyModelPort",
    "Message",
    "ModelBackend",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatibleBackend",
    "OpenAIClientFactory",
    "ParityResult",
    "ResponseProvenance",
    "RoleModelRouter",
    "TraceRecorderBackend",
    "UnknownPolicyError",
    "UnknownRoleError",
    "compare_traces",
    "build_model_backend",
    "select_backend",
]
