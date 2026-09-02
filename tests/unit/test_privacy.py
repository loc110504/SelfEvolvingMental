import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.privacy import (  # noqa: E402
    audit_release,
    deidentify_participant,
    redact_record,
    release_split_manifest,
)
from psyvec.research import SplitManifest  # noqa: E402


class PrivacyTests(unittest.TestCase):
    def test_sensitive_field_is_hashed_by_default(self) -> None:
        raw = "I am Ada and I feel anxious"
        redacted = redact_record({"answer": raw}, sensitive_fields={"answer"})

        self.assertNotEqual(redacted["answer"], raw)
        self.assertNotIn(raw, str(redacted))
        self.assertFalse(redacted["raw_text_opt_in"])

    def test_raw_text_requires_explicit_opt_in_and_is_recorded(self) -> None:
        raw = "private response"
        default = redact_record({"answer": raw}, sensitive_fields={"answer"})
        opted_in = redact_record(
            {"answer": raw}, sensitive_fields={"answer"}, allow_raw_text=True
        )

        self.assertNotEqual(default["answer"], raw)
        self.assertEqual(opted_in["answer"], raw)
        self.assertTrue(opted_in["raw_text_opt_in"])

    def test_participants_are_stably_deidentified_for_release(self) -> None:
        manifest = SplitManifest("split", "evolve", ("real-a", "real-b"), True)
        release = release_split_manifest(manifest)

        self.assertEqual(
            deidentify_participant("real-a"), deidentify_participant("real-a")
        )
        self.assertNotEqual(
            deidentify_participant("real-a"), deidentify_participant("real-b")
        )
        self.assertTrue(
            all(pid not in release.participant_ids for pid in manifest.participant_ids)
        )

    def test_release_audit_reports_every_requested_leak(self) -> None:
        findings = audit_release(
            {"label": "positive", "participant": "real-a", "answer": "private"},
            participant_ids={"real-a"},
            sensitive_fields={"answer"},
        )

        self.assertEqual(
            {finding.kind for finding in findings},
            {"label_key", "participant_id", "sensitive_text"},
        )

    def test_clean_payload_has_no_release_findings(self) -> None:
        findings = audit_release(
            {"participant": deidentify_participant("real-a"), "answer": "a" * 64},
            participant_ids={"real-a"},
            sensitive_fields={"answer"},
        )

        self.assertEqual(findings, ())


if __name__ == "__main__":
    unittest.main()
