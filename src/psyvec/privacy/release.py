"""De-identified release representations of research records."""

from __future__ import annotations

from dataclasses import dataclass

from psyvec.privacy.subjects import deidentify_participant
from psyvec.research.splits import SplitManifest


@dataclass(frozen=True, slots=True)
class ReleasedSplitManifest:
    """The report-safe form of a split manifest."""

    split_id: str
    kind: str
    participant_ids: tuple[str, ...]
    authorized: bool


def release_split_manifest(manifest: SplitManifest) -> ReleasedSplitManifest:
    """Return ``manifest`` with every participant ID replaced by a pseudonym."""

    return ReleasedSplitManifest(
        split_id=manifest.split_id,
        kind=manifest.kind,
        participant_ids=tuple(
            deidentify_participant(participant_id)
            for participant_id in manifest.participant_ids
        ),
        authorized=manifest.authorized,
    )
