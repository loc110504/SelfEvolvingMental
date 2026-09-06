"""Evidence retrieval over the real DAIC-WOZ transcript (Plan_Improve.md Sec 2.3).

Covers the synthetic-fixture unit behavior (tag-anchor match, keyword
fallback, cross-cutting fallback, missing status, snippet capping, strict
config validation) plus a golden-case regression against the real dev-split
data for participant 404 (Plan_Improve.md Sec 6 Phase 1 step 3): GT PHQ-8 = 0
for every item, yet the previous pipeline's ``real_interview[:50]`` truncation
made the client-persona invent "trouble staying asleep" out of thin air. Turn
145 of the real transcript ("i'm handling my sleep pretty well") sits well
past that truncation point and must be retrieved for Sleep Problems now that
retrieval scans the full transcript.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psyvec.evaluation.evidence_retrieval import (  # noqa: E402
    CROSS_CUTTING,
    extract_ellie_tag,
    format_evidence_for_prompt,
    load_keyword_lexicon,
    load_tag_map,
    retrieve_evidence,
)

TOPICS = ["Sleep Problems", "Depressed Mood", "Loss of Interest", "Fatigue or Low Energy"]

TAG_MAP = {
    "easy_sleep": "Sleep Problems",
    "sleep_affects": "Sleep Problems",
    "feel_down": "Depressed Mood",
    "depression_diagnosed": CROSS_CUTTING,
}

KEYWORD_LEXICON = {
    "Sleep Problems": ("sleep", "insomnia"),
    "Depressed Mood": ("sad", "hopeless"),
    "Loss of Interest": ("interest", "hobby"),
    "Fatigue or Low Energy": ("tired", "exhaust"),
}

SLEEP_Q = "easy_sleep (how easy is it to get a good night's sleep)"
DEPRESSION_DX_Q = "depression_diagnosed (have you been diagnosed with depression)"


def turn(role: str, content: str) -> dict[str, str]:
    return {"roleName": role, "content": content}


class ExtractEllieTagTests(unittest.TestCase):
    def test_parses_tag_and_utterance(self) -> None:
        found = extract_ellie_tag(SLEEP_Q)
        self.assertEqual(
            found, ("easy_sleep", "how easy is it to get a good night's sleep")
        )

    def test_returns_none_for_plain_utterance(self) -> None:
        self.assertIsNone(
            extract_ellie_tag("hi i'm ellie thanks for coming in today")
        )

    def test_returns_none_for_unbalanced_parens(self) -> None:
        self.assertIsNone(extract_ellie_tag("that's good (i think"))


class RetrieveEvidenceTagAnchorTests(unittest.TestCase):
    def test_captures_contiguous_participant_replies_after_anchor(self) -> None:
        interview = [
            turn("Ellie", SLEEP_Q),
            turn("Participant", "it's been pretty good"),
            turn("Participant", "i'm handling my sleep pretty well"),
            turn("Ellie", "okay_confirm (okay)"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        sleep = bundles["Sleep Problems"]
        self.assertEqual(sleep.status, "known")
        self.assertEqual(
            [snippet.text for snippet in sleep.snippets],
            ["it's been pretty good", "i'm handling my sleep pretty well"],
        )
        self.assertTrue(all(s.source == "tag_anchor" for s in sleep.snippets))

    def test_stops_reply_span_at_next_ellie_turn(self) -> None:
        interview = [
            turn("Ellie", SLEEP_Q),
            turn("Participant", "reply one"),
            turn("Ellie", "unrelated (something else)"),
            turn("Participant", "reply two about something unrelated"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        self.assertEqual(
            [s.text for s in bundles["Sleep Problems"].snippets], ["reply one"]
        )

    def test_untagged_ellie_turn_is_not_an_anchor(self) -> None:
        interview = [
            turn("Ellie", "hi i'm ellie thanks for coming in today"),
            turn("Participant", "i sleep fine"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        # Picked up by the keyword fallback ("sleep"), not as a tag_anchor.
        sleep = bundles["Sleep Problems"]
        self.assertEqual(sleep.status, "known")
        self.assertEqual(sleep.snippets[0].source, "keyword")


class RetrieveEvidenceKeywordFallbackTests(unittest.TestCase):
    def test_scans_full_transcript_not_just_a_prefix(self) -> None:
        # 60 filler turns before the one real disclosure — the point of this
        # fix is that retrieval must not be limited to the first N turns.
        filler = [
            turn("Ellie", "small talk")
            if i % 2 == 0
            else turn("Participant", "just chatting")
            for i in range(60)
        ]
        disclosure = turn("Participant", "i've lost interest in my old hobbies")
        interview = [*filler, disclosure]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        loss_of_interest = bundles["Loss of Interest"]
        self.assertEqual(loss_of_interest.status, "known")
        self.assertEqual(
            loss_of_interest.snippets[0].turn_id, f"turn-{len(interview) - 1}"
        )

    def test_does_not_duplicate_a_turn_already_captured_by_tag_anchor(self) -> None:
        interview = [
            turn("Ellie", "feel_down (do you feel down)"),
            turn("Participant", "yeah i've been pretty sad"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        mood = bundles["Depressed Mood"]
        self.assertEqual(len(mood.snippets), 1)
        self.assertEqual(mood.snippets[0].source, "tag_anchor")

    def test_skips_locally_negated_keyword_mention(self) -> None:
        # Regression: participant 303's real transcript ("i try to stay
        # happy i'd rather be happy than sad") used to be retrieved as
        # "known" Depressed Mood evidence on a bare "sad" substring match,
        # even though the participant is contrasting themself away from
        # sadness, not reporting it.
        interview = [
            turn(
                "Participant",
                "i try to stay happy i'd rather be happy than sad my kids keep me going",
            )
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        self.assertEqual(bundles["Depressed Mood"].status, "missing")

    def test_keeps_unnegated_keyword_mention_elsewhere_in_a_long_turn(self) -> None:
        # A negation cue far from the keyword (outside the local window)
        # must not suppress a real, unrelated disclosure later in the turn.
        interview = [
            turn(
                "Participant",
                "i don't know what to say but honestly i've just felt so sad lately",
            )
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        self.assertEqual(bundles["Depressed Mood"].status, "known")

    def test_skips_reply_to_a_hypothetical_ellie_question(self) -> None:
        # Regression: participant 303's real transcript had Ellie ask the
        # hypothetical "what are you like when you don't sleep well", and
        # the participant's reply ("irritated tired lazy") used to be
        # retrieved as "known" Fatigue evidence even though it describes a
        # hypothetical disposition, not a two-week self-report.
        interview = [
            turn("Ellie", "what are you like when you don't sleep well"),
            turn("Participant", "irritated tired lazy"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        self.assertEqual(bundles["Fatigue or Low Energy"].status, "missing")

    def test_keeps_reply_to_a_direct_non_hypothetical_question(self) -> None:
        interview = [
            turn("Ellie", "have you felt tired over the past two weeks"),
            turn("Participant", "yeah i've felt tired most days"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        self.assertEqual(bundles["Fatigue or Low Energy"].status, "known")

    def test_orders_keyword_hits_by_density_then_length(self) -> None:
        interview = [
            turn("Participant", "sleep sleep sleep"),
            turn("Participant", "sleep"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        snippets = bundles["Sleep Problems"].snippets
        self.assertEqual(snippets[0].text, "sleep sleep sleep")


class RetrieveEvidenceMissingAndCrossCuttingTests(unittest.TestCase):
    def test_topic_with_no_hits_is_missing(self) -> None:
        interview = [turn("Participant", "totally unrelated content about weather")]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        self.assertEqual(bundles["Sleep Problems"].status, "missing")
        self.assertEqual(bundles["Sleep Problems"].snippets, ())

    def test_cross_cutting_bucket_is_shared_across_topics(self) -> None:
        interview = [
            turn("Ellie", DEPRESSION_DX_Q),
            turn("Participant", "yes a few years ago"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        for topic in TOPICS:
            self.assertEqual(
                [s.text for s in bundles[topic].cross_cutting],
                ["yes a few years ago"],
            )

    def test_missing_status_still_carries_cross_cutting_context(self) -> None:
        interview = [
            turn("Ellie", DEPRESSION_DX_Q),
            turn("Participant", "yes, it's been hard"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        sleep = bundles["Sleep Problems"]
        self.assertEqual(sleep.status, "missing")
        self.assertEqual(len(sleep.cross_cutting), 1)


class RetrieveEvidenceCappingTests(unittest.TestCase):
    def test_specific_snippets_capped_anchors_before_keywords(self) -> None:
        interview = [
            turn("Ellie", SLEEP_Q),
            turn("Participant", "anchor reply one"),
            turn("Participant", "anchor reply two"),
            turn("Participant", "sleep sleep sleep keyword reply"),
        ]
        bundles = retrieve_evidence(
            interview, TOPICS, TAG_MAP, KEYWORD_LEXICON, max_specific_snippets=2
        )
        snippets = bundles["Sleep Problems"].snippets
        self.assertEqual(len(snippets), 2)
        self.assertEqual(
            [s.text for s in snippets], ["anchor reply one", "anchor reply two"]
        )

    def test_cross_cutting_capped(self) -> None:
        interview = [
            turn("Ellie", DEPRESSION_DX_Q),
            turn("Participant", "first"),
            turn("Ellie", DEPRESSION_DX_Q),
            turn("Participant", "second"),
            turn("Ellie", DEPRESSION_DX_Q),
            turn("Participant", "third"),
        ]
        bundles = retrieve_evidence(
            interview, TOPICS, TAG_MAP, KEYWORD_LEXICON, max_cross_cutting_snippets=2
        )
        self.assertEqual(len(bundles["Sleep Problems"].cross_cutting), 2)


class FormatEvidenceForPromptTests(unittest.TestCase):
    def test_known_bundle_lists_turn_ids(self) -> None:
        interview = [
            turn("Ellie", SLEEP_Q),
            turn("Participant", "i sleep fine"),
        ]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        rendered = format_evidence_for_prompt(bundles["Sleep Problems"])
        self.assertIn("turn-1", rendered)
        self.assertIn("i sleep fine", rendered)

    def test_missing_bundle_without_cross_cutting_says_so(self) -> None:
        interview = [turn("Participant", "totally unrelated")]
        bundles = retrieve_evidence(interview, TOPICS, TAG_MAP, KEYWORD_LEXICON)
        rendered = format_evidence_for_prompt(bundles["Sleep Problems"])
        self.assertIn("No direct disclosure found", rendered)
        self.assertNotIn("turn-", rendered)


class LoadConfigValidationTests(unittest.TestCase):
    def test_load_tag_map_rejects_unknown_topic(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"some_tag": "Not A Real Topic"}, handle)
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                load_tag_map(path, valid_topics=TOPICS)
        finally:
            path.unlink()

    def test_load_keyword_lexicon_rejects_unknown_topic(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"Not A Real Topic": ["x"]}, handle)
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                load_keyword_lexicon(path, valid_topics=TOPICS)
        finally:
            path.unlink()

    def test_load_tag_map_skips_meta_key(self) -> None:
        payload = {"_meta": {"description": "x"}, "easy_sleep": "Sleep Problems"}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            loaded = load_tag_map(path, valid_topics=TOPICS)
            self.assertEqual(loaded, {"easy_sleep": "Sleep Problems"})
        finally:
            path.unlink()


class GoldenCaseParticipant404Test(unittest.TestCase):
    """Regression fixture from Plan_Improve.md Sec 5.1/6: participant 404,
    GT PHQ-8 = 0 on every item, whose real transcript directly contradicts
    what the old pipeline's truncated-context role-play invented."""

    SAMPLE_PATH = PROJECT_ROOT / "data" / "processed_daic_woz" / "dev" / "404.json"

    def setUp(self) -> None:
        if not self.SAMPLE_PATH.is_file():
            self.skipTest(f"golden fixture not present: {self.SAMPLE_PATH}")
        sample = json.loads(self.SAMPLE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sample["phq8_scores"]["PHQ8_Score"], 0)
        scale_file = PROJECT_ROOT / "configs" / "scales" / "PHQ-8.json"
        topics = list(json.loads(scale_file.read_text(encoding="utf-8")).keys())
        tag_map = load_tag_map(
            PROJECT_ROOT / "configs" / "scales" / "topic_tag_map.json",
            valid_topics=topics,
        )
        lexicon = load_keyword_lexicon(
            PROJECT_ROOT / "configs" / "scales" / "topic_keywords.json",
            valid_topics=topics,
        )
        self.bundles = retrieve_evidence(
            sample["real_interview"], topics, tag_map, lexicon
        )

    def test_sleep_evidence_includes_real_turn_past_old_50_turn_cutoff(self) -> None:
        sleep = self.bundles["Sleep Problems"]
        self.assertEqual(sleep.status, "known")
        matched = [
            s for s in sleep.snippets if "handling my sleep pretty well" in s.text
        ]
        self.assertTrue(
            matched,
            "expected the real disclosure at turn 145 to be retrieved for "
            f"Sleep Problems; got snippets: {[s.text for s in sleep.snippets]}",
        )
        # Turn 145 is well past real_interview[:50] — the old pipeline's
        # cutoff that made the client-persona invent "trouble sleeping"
        # instead of using what the participant actually said.
        self.assertGreater(int(matched[0].turn_id.split("-")[1]), 50)


if __name__ == "__main__":
    unittest.main()
