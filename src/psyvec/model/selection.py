"""Feature-flagged backend selection for Phase 1 rollback."""

from __future__ import annotations

from psyvec.config import BackendConfig
from psyvec.model.backend import ModelBackend


def select_backend(
    config: BackendConfig,
    *,
    abstraction_backend: ModelBackend,
    legacy_backend: ModelBackend,
) -> ModelBackend:
    """Select the new seam or characterized legacy call path explicitly."""

    if config.use_backend_abstraction:
        return abstraction_backend
    return legacy_backend
