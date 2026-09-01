"""Release payload checks for labels and participant-data leaks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psyvec.state.contracts import FORBIDDEN_LABEL_KEYS


@dataclass(frozen=True, slots=True)
class PrivacyFinding:
    """One privacy violation found in a candidate release payload."""

    kind: str
    path: str
    detail: str


def audit_release(
    payload: Mapping[str, Any],
    *,
    participant_ids: Iterable[str],
    sensitive_fields: Iterable[str],
) -> tuple[PrivacyFinding, ...]:
    """Return every label, participant ID, and raw sensitive-text leak."""

    participants = frozenset(participant_ids)
    sensitive = frozenset(sensitive_fields)
    findings: list[PrivacyFinding] = []

    def scan(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key)
                nested_path = f"{path}.{key_text}"
                if key_text in FORBIDDEN_LABEL_KEYS:
                    findings.append(PrivacyFinding("label_key", nested_path, key_text))
                if key_text in sensitive and _is_raw_text(nested):
                    findings.append(
                        PrivacyFinding("sensitive_text", nested_path, key_text)
                    )
                scan(nested, nested_path)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for index, nested in enumerate(value):
                scan(nested, f"{path}[{index}]")
        elif isinstance(value, str) and value in participants:
            findings.append(PrivacyFinding("participant_id", path, value))

    scan(payload, "payload")
    return tuple(findings)


def _is_raw_text(value: Any) -> bool:
    """Recognize text that was not produced by ``redact_record``."""

    return not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
