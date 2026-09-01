"""Redirect pinned AgentMental call sites through the Phase 1 backend ports.

The pinned clone under ``AgentMental/`` is never edited: its Phase 0 SHA-256
inventory must keep verifying. Instead the three upstream call shapes are
replaced at runtime, and the feature flag restores the original callables.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from psyvec.model.backend import BackendError
from psyvec.model.legacy_ports import LegacyModelPort

_RoleOf = Callable[[Any], str]


def build_autogen_role_handler(
    port: LegacyModelPort,
    *,
    role_of: _RoleOf,
    request_id_prefix: str = "autogen",
) -> Callable[[Any, Any, Any, str], str | None]:
    """Replace upstream ``makerequest(manager, proxy, recipient, payload)``.

    ``payload`` is forwarded verbatim so prompt content stays byte-identical to
    the characterized baseline.
    """

    counter = itertools.count(1)

    def make_request(
        manager: Any, proxy: Any, recipient: Any, payload: str
    ) -> str | None:
        del manager, proxy
        try:
            response = port.complete(
                request_id=f"{request_id_prefix}-{next(counter)}",
                role=role_of(recipient),
                system_prompt="",
                user_prompt=payload,
            )
        except BackendError:
            return None
        return response.content

    return make_request


def build_direct_completion(
    port: LegacyModelPort,
    *,
    role: str,
    request_id_prefix: str,
) -> Callable[[str, str], str | None]:
    """Replace an upstream direct OpenAI call with a port-backed completion.

    Covers the ``memory.py`` extraction/reassessment and the
    ``generate_response.py`` simulator shapes: system prompt plus user prompt in,
    text out, ``None`` on an explicit backend failure.
    """

    counter = itertools.count(1)

    def complete(system_prompt: str, user_prompt: str) -> str | None:
        try:
            response = port.complete(
                request_id=f"{request_id_prefix}-{next(counter)}",
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except BackendError:
            return None
        return response.content

    return complete


@dataclass(frozen=True, slots=True)
class PatchTarget:
    """One upstream module attribute to redirect."""

    module: Any
    attribute: str
    replacement: Any

    def __post_init__(self) -> None:
        if not hasattr(self.module, self.attribute):
            raise AttributeError(
                f"legacy module has no attribute {self.attribute!r} to redirect"
            )


@dataclass(slots=True)
class LegacyPortInstaller:
    """Install and roll back legacy redirections behind the feature flag."""

    targets: Sequence[PatchTarget]
    enabled: bool = True
    _originals: list[tuple[Any, str, Any]] = field(default_factory=list, init=False)
    _installed: bool = field(default=False, init=False)

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        """Redirect every target, or do nothing when the flag is off."""

        if self._installed or not self.enabled:
            return
        for target in self.targets:
            original = getattr(target.module, target.attribute)
            self._originals.append((target.module, target.attribute, original))
            setattr(target.module, target.attribute, target.replacement)
        self._installed = True

    def restore(self) -> None:
        """Restore the characterized legacy callables exactly as they were."""

        for module, attribute, original in reversed(self._originals):
            setattr(module, attribute, original)
        self._originals.clear()
        self._installed = False

    def __enter__(self) -> LegacyPortInstaller:
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.restore()
