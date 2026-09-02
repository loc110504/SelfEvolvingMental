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
from psyvec.privacy.runtime import RedactingRuntimeLog

_RoleOf = Callable[[Any], str]


def build_autogen_role_handler(
    port: LegacyModelPort,
    *,
    role_of: _RoleOf,
    request_id_prefix: str = "autogen",
    log: RedactingRuntimeLog | None = None,
    participant_id: str | None = None,
) -> Callable[[Any, Any, Any, str], str | None]:
    """Replace upstream ``makerequest(manager, proxy, recipient, payload)``.

    ``payload`` is forwarded verbatim so prompt content stays byte-identical to
    the characterized baseline. When ``log`` is given, the prompt and completion
    reach it hashed, never raw.
    """

    counter = itertools.count(1)

    def make_request(
        manager: Any, proxy: Any, recipient: Any, payload: str
    ) -> str | None:
        del manager, proxy
        request_id = f"{request_id_prefix}-{next(counter)}"
        role = role_of(recipient)
        try:
            response = port.complete(
                request_id=request_id,
                role=role,
                system_prompt="",
                user_prompt=payload,
            )
        except BackendError as error:
            _log_call(
                log,
                request_id=request_id,
                role=role,
                participant_id=participant_id,
                user_prompt=payload,
                completion=None,
                failed=True,
                failure_kind=type(error).__name__,
            )
            return None
        _log_call(
            log,
            request_id=request_id,
            role=role,
            participant_id=participant_id,
            user_prompt=payload,
            completion=response.content,
            failed=False,
            failure_kind=None,
        )
        return response.content

    return make_request


def build_direct_completion(
    port: LegacyModelPort,
    *,
    role: str,
    request_id_prefix: str,
    log: RedactingRuntimeLog | None = None,
    participant_id: str | None = None,
) -> Callable[[str, str], str | None]:
    """Replace an upstream direct OpenAI call with a port-backed completion.

    Covers the ``memory.py`` extraction/reassessment and the
    ``generate_response.py`` simulator shapes: system prompt plus user prompt in,
    text out, ``None`` on an explicit backend failure. ``memory.py`` is the call
    site Doc 00 quirk 7 is about, so ``log`` redacts rather than drops.
    """

    counter = itertools.count(1)

    def complete(system_prompt: str, user_prompt: str) -> str | None:
        request_id = f"{request_id_prefix}-{next(counter)}"
        try:
            response = port.complete(
                request_id=request_id,
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except BackendError as error:
            _log_call(
                log,
                request_id=request_id,
                role=role,
                participant_id=participant_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                completion=None,
                failed=True,
                failure_kind=type(error).__name__,
            )
            return None
        _log_call(
            log,
            request_id=request_id,
            role=role,
            participant_id=participant_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            completion=response.content,
            failed=False,
            failure_kind=None,
        )
        return response.content

    return complete


def _log_call(
    log: RedactingRuntimeLog | None,
    *,
    participant_id: str | None,
    **fields: Any,
) -> None:
    """Emit one redacted runtime record, dropping keys the call site did not set."""

    if log is None:
        return
    if participant_id is not None:
        fields["participant_id"] = participant_id
    log.record(**{key: value for key, value in fields.items() if value is not None})


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
