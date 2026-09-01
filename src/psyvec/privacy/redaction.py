"""Default-safe conversion of runtime records into loggable records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from psyvec.state.contracts import canonical_hash


def redact_record(
    record: Mapping[str, Any],
    *,
    sensitive_fields: Iterable[str],
    allow_raw_text: bool = False,
) -> dict[str, Any]:
    """Copy ``record`` for logging, hashing declared text fields by default."""

    sensitive = frozenset(sensitive_fields)
    redacted = dict(record)
    for field in sensitive:
        if field in redacted and not allow_raw_text:
            redacted[field] = canonical_hash(redacted[field])
    redacted["raw_text_opt_in"] = allow_raw_text
    return redacted
