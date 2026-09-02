"""Stable, one-way participant identifiers for release artifacts."""

from __future__ import annotations

from psyvec.state.contracts import canonical_hash


def deidentify_participant(participant_id: str) -> str:
    """Return a stable pseudonym that contains no participant identifier."""

    return f"participant-{canonical_hash({'participant_id': participant_id})}"
