"""Redacting sink for the runtime logging path.

Doc 00 quirk 7: upstream logs may carry prompts, answers, memory contents and
demographics. `psyvec.privacy` could already redact such a record; this module
is what actually stands between a runtime call site and a log, so the default
path is redacted rather than redactable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from psyvec.privacy.redaction import redact_record
from psyvec.privacy.subjects import deidentify_participant

#: Fields a runtime call site may hand over that must never be logged raw.
RUNTIME_SENSITIVE_FIELDS: tuple[str, ...] = (
    "system_prompt",
    "user_prompt",
    "completion",
    "memory",
    "demographics",
)

LogSink = Callable[[Mapping[str, Any]], None]


@dataclass(slots=True)
class RedactingRuntimeLog:
    """Hash sensitive fields and pseudonymize participants before the sink sees them.

    ``allow_raw_text`` is the recorded opt-in: it stays on the emitted record as
    ``raw_text_opt_in`` so a reader can tell a redacted log from an approved
    raw-text one.
    """

    sink: LogSink
    sensitive_fields: Iterable[str] = RUNTIME_SENSITIVE_FIELDS
    allow_raw_text: bool = False
    records: list[Mapping[str, Any]] = field(default_factory=list)

    def record(self, **fields: Any) -> Mapping[str, Any]:
        """Redact one runtime record, keep it, and hand it to the sink."""

        payload = dict(fields)
        participant_id = payload.pop("participant_id", None)
        if participant_id is not None:
            payload["participant_ref"] = deidentify_participant(str(participant_id))
        redacted = redact_record(
            payload,
            sensitive_fields=self.sensitive_fields,
            allow_raw_text=self.allow_raw_text,
        )
        self.records.append(redacted)
        self.sink(redacted)
        return redacted


def null_sink(_record: Mapping[str, Any]) -> None:
    """Drop the record. The default when a call site only wants the redaction."""
