"""Grounded Interview Loop policy decisions (Plan_Improve.md Sec 2.4)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.evidence_retrieval import (  # noqa: E402
    EvidenceBundle,
    EvidenceSnippet,
)
from psyvec.evaluation.interview_policy import (  # noqa: E402
    allowed_citation_ids,
    build_client_system_prompt,
    build_followup_instruction,
    build_scorer_prompt,
    default_necessity,
    extract_demographics,
    invalid_citations,
    low_faithfulness_flag,
    mentions_frequency,
)

KNOWN_SNIPPET = EvidenceSnippet(
    turn_id="turn-42", speaker="Participant", text="i sleep fine", source="tag_anchor"
)
CROSS_SNIPPET = EvidenceSnippet(
    turn_id="turn-7",
    speaker="Participant",
    text="diagnosed years ago",
    source="cross_cutting",
)
KNOWN_BUNDLE = EvidenceBundle(
    topic="Sleep Problems",
    status="known",
    snippets=(KNOWN_SNIPPET,),
    cross_cutting=(),
)
MISSING_BUNDLE = EvidenceBundle(
    topic="Sleep Problems",
    status="missing",
    snippets=(),
    cross_cutting=(CROSS_SNIPPET,),
)


class MentionsFrequencyTests(unittest.TestCase):
    def test_detects_explicit_day_count(self) -> None:
        self.assertTrue(mentions_frequency("it happened about 4 days per week"))

    def test_detects_qualitative_phrase(self) -> None:
        self.assertTrue(mentions_frequency("nearly every day I feel this way"))

    def test_false_when_no_frequency_language(self) -> None:
        self.assertFalse(mentions_frequency("I sleep fine, no issues at all lately"))


class DefaultNecessityTests(unittest.TestCase):
    def test_missing_evidence_always_stops(self) -> None:
        self.assertEqual(default_necessity("missing", has_frequency_info=False), 0)
        self.assertEqual(default_necessity("missing", has_frequency_info=True), 0)

    def test_known_evidence_without_frequency_continues(self) -> None:
        self.assertEqual(default_necessity("known", has_frequency_info=False), 1)

    def test_known_evidence_with_frequency_stops(self) -> None:
        self.assertEqual(default_necessity("known", has_frequency_info=True), 0)


class BuildClientSystemPromptTests(unittest.TestCase):
    def test_known_bundle_instructs_consistency_not_invention(self) -> None:
        prompt = build_client_system_prompt("Sleep Problems", KNOWN_BUNDLE)
        self.assertIn("i sleep fine", prompt)
        self.assertIn("Do not contradict them", prompt)

    def test_missing_bundle_instructs_conservative_answer(self) -> None:
        prompt = build_client_system_prompt("Sleep Problems", MISSING_BUNDLE)
        self.assertIn("have not explicitly discussed", prompt)
        self.assertIn("Do NOT invent specific new symptoms", prompt)
        self.assertIn("diagnosed years ago", prompt)


class BuildFollowupInstructionTests(unittest.TestCase):
    def test_targets_frequency_when_missing(self) -> None:
        instruction = build_followup_instruction(
            "Sleep Problems", "I sleep badly sometimes", has_frequency_info=False
        )
        self.assertIn("how many days", instruction)

    def test_targets_severity_when_frequency_already_known(self) -> None:
        instruction = build_followup_instruction(
            "Sleep Problems", "I sleep badly 4 nights a week", has_frequency_info=True
        )
        self.assertIn("severity", instruction)


class BuildScorerPromptTests(unittest.TestCase):
    def test_includes_dialogue_evidence_and_standard(self) -> None:
        prompt = build_scorer_prompt(
            "Sleep Problems", "Q: ...\nA: ...", KNOWN_BUNDLE, "0: fine\n3: severe"
        )
        self.assertIn("Q: ...", prompt)
        self.assertIn("i sleep fine", prompt)
        self.assertIn("0: fine", prompt)
        self.assertIn("evidence_turn_ids", prompt)


class CitationValidationTests(unittest.TestCase):
    def test_allowed_ids_include_specific_and_cross_cutting(self) -> None:
        bundle = EvidenceBundle(
            topic="X",
            status="known",
            snippets=(KNOWN_SNIPPET,),
            cross_cutting=(CROSS_SNIPPET,),
        )
        self.assertEqual(allowed_citation_ids(bundle), frozenset({"turn-42", "turn-7"}))

    def test_invalid_citations_flags_fabricated_turn_id(self) -> None:
        allowed = frozenset({"turn-42"})
        result = invalid_citations(["turn-42", "turn-999"], allowed)
        self.assertEqual(result, ("turn-999",))

    def test_invalid_citations_empty_when_all_valid(self) -> None:
        allowed = frozenset({"turn-42"})
        self.assertEqual(invalid_citations(["turn-42"], allowed), ())


class ExtractDemographicsTests(unittest.TestCase):
    def test_extracts_age_from_im_phrasing(self) -> None:
        interview = [{"roleName": "Participant", "content": "i'm 34 and doing fine"}]
        self.assertIn("Age: 34", extract_demographics(interview))

    def test_extracts_age_from_years_old_phrasing(self) -> None:
        interview = [{"roleName": "Participant", "content": "I am 29 years old"}]
        self.assertIn("Age: 29", extract_demographics(interview))

    def test_undisclosed_when_no_age_found(self) -> None:
        interview = [
            {"roleName": "Participant", "content": "just chatting about weather"}
        ]
        self.assertIn("Age: undisclosed", extract_demographics(interview))

    def test_ignores_ellie_turns(self) -> None:
        interview = [{"roleName": "Ellie", "content": "i'm 5 years into this job"}]
        self.assertIn("Age: undisclosed", extract_demographics(interview))


class LowFaithfulnessFlagTests(unittest.TestCase):
    def test_missing_bundle_never_flagged(self) -> None:
        self.assertFalse(low_faithfulness_flag("anything at all", MISSING_BUNDLE))

    def test_consistent_dialogue_not_flagged(self) -> None:
        # Shares real content words ("sleep", "fine") with the evidence.
        reply = "i sleep fine most nights"
        self.assertFalse(low_faithfulness_flag(reply, KNOWN_BUNDLE))

    def test_unrelated_dialogue_flagged(self) -> None:
        reply = (
            "waking constantly panicking terrified shaking uncontrollably "
            "screaming nightmares vomiting"
        )
        self.assertTrue(low_faithfulness_flag(reply, KNOWN_BUNDLE))

    def test_empty_dialogue_not_flagged(self) -> None:
        self.assertFalse(low_faithfulness_flag("", KNOWN_BUNDLE))


if __name__ == "__main__":
    unittest.main()
