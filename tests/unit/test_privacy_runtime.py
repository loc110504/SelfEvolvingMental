"""Doc 00 quirk 7: the runtime logging path is redacted by default."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

from psyvec.integration.legacy_bridge import (
    build_autogen_role_handler,
    build_direct_completion,
)
from psyvec.model.backend import (
    BackendError,
    ModelRequest,
    ModelResponse,
    ResponseProvenance,
)
from psyvec.model.legacy_ports import LegacyModelPort
from psyvec.privacy.audit import audit_release
from psyvec.privacy.runtime import RedactingRuntimeLog, null_sink
from psyvec.privacy.subjects import deidentify_participant

PARTICIPANT = "300"
PROMPT = "I have not been sleeping since my brother died"


class RecordingSink:
    def __init__(self) -> None:
        self.records: list[Mapping[str, Any]] = []

    def __call__(self, record: Mapping[str, Any]) -> None:
        self.records.append(record)


class StubBackend:
    """Injected backend. No live model exists anywhere in this suite."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self.fail:
            raise BackendError("stub refused")
        return ModelResponse(
            request_id=request.request_id,
            content="scored 2",
            finish_reason="stop",
            usage={},
            latency_ms=1.0,
            provenance=ResponseProvenance(
                backend="stub",
                model=request.model,
                prompt_version=request.prompt_version,
                config_hash="stub-config",
            ),
        )


def make_port(*, fail: bool = False) -> LegacyModelPort:
    return LegacyModelPort(
        backend=StubBackend(fail=fail),  # type: ignore[arg-type]
        model="stub-model",
        prompt_version="v1",
    )


def is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class RedactingRuntimeLogTests(unittest.TestCase):
    def test_sensitive_fields_are_hashed_and_the_opt_in_is_recorded(self) -> None:
        sink = RecordingSink()
        log = RedactingRuntimeLog(sink=sink)

        record = log.record(request_id="r-1", role="updater", user_prompt=PROMPT)

        self.assertTrue(is_hash(record["user_prompt"]))
        self.assertNotIn(PROMPT, str(record))
        self.assertFalse(record["raw_text_opt_in"])
        self.assertEqual(record["role"], "updater")
        self.assertEqual(sink.records, [record])

    def test_raw_text_requires_an_explicit_recorded_opt_in(self) -> None:
        log = RedactingRuntimeLog(sink=null_sink, allow_raw_text=True)

        record = log.record(request_id="r-1", user_prompt=PROMPT)

        self.assertEqual(record["user_prompt"], PROMPT)
        self.assertTrue(record["raw_text_opt_in"])

    def test_a_participant_id_is_replaced_by_its_pseudonym(self) -> None:
        log = RedactingRuntimeLog(sink=null_sink)

        record = log.record(request_id="r-1", participant_id=PARTICIPANT)

        self.assertNotIn("participant_id", record)
        self.assertEqual(
            record["participant_ref"], deidentify_participant(PARTICIPANT)
        )

    def test_a_redacted_record_passes_the_release_audit(self) -> None:
        log = RedactingRuntimeLog(sink=null_sink)

        record = log.record(
            request_id="r-1",
            participant_id=PARTICIPANT,
            user_prompt=PROMPT,
            completion="scored 2",
        )

        findings = audit_release(
            record,
            participant_ids=(PARTICIPANT,),
            sensitive_fields=("user_prompt", "completion"),
        )
        self.assertEqual(findings, ())


class BridgeLoggingTests(unittest.TestCase):
    def test_the_direct_completion_logs_the_prompt_hashed(self) -> None:
        sink = RecordingSink()
        log = RedactingRuntimeLog(sink=sink)
        complete = build_direct_completion(
            make_port(),
            role="memory",
            request_id_prefix="memory",
            log=log,
            participant_id=PARTICIPANT,
        )

        self.assertEqual(complete("system", PROMPT), "scored 2")

        (record,) = sink.records
        self.assertEqual(record["request_id"], "memory-1")
        self.assertTrue(is_hash(record["user_prompt"]))
        self.assertTrue(is_hash(record["system_prompt"]))
        self.assertTrue(is_hash(record["completion"]))
        self.assertFalse(record["failed"])
        self.assertNotIn(PARTICIPANT, str(record))

    def test_a_backend_failure_is_logged_without_a_completion(self) -> None:
        sink = RecordingSink()
        complete = build_direct_completion(
            make_port(fail=True),
            role="memory",
            request_id_prefix="memory",
            log=RedactingRuntimeLog(sink=sink),
        )

        self.assertIsNone(complete("system", PROMPT))

        (record,) = sink.records
        self.assertTrue(record["failed"])
        self.assertEqual(record["failure_kind"], "BackendError")
        self.assertNotIn("completion", record)

    def test_the_autogen_handler_logs_the_payload_hashed(self) -> None:
        sink = RecordingSink()
        handler = build_autogen_role_handler(
            make_port(),
            role_of=lambda recipient: str(recipient),
            log=RedactingRuntimeLog(sink=sink),
        )

        self.assertEqual(handler(None, None, "updater", PROMPT), "scored 2")

        (record,) = sink.records
        self.assertEqual(record["role"], "updater")
        self.assertTrue(is_hash(record["user_prompt"]))
        self.assertNotIn(PROMPT, str(record))

    def test_logging_stays_optional_and_off_by_default(self) -> None:
        complete = build_direct_completion(
            make_port(), role="memory", request_id_prefix="memory"
        )

        self.assertEqual(complete("system", PROMPT), "scored 2")


if __name__ == "__main__":
    unittest.main()
