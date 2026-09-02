"""Runtime redirection of legacy AgentMental call sites through PsyVEC ports."""

from __future__ import annotations

from psyvec.integration.legacy_bridge import (
    LegacyPortInstaller,
    PatchTarget,
    build_autogen_role_handler,
    build_direct_completion,
)

__all__ = [
    "LegacyPortInstaller",
    "PatchTarget",
    "build_autogen_role_handler",
    "build_direct_completion",
]
