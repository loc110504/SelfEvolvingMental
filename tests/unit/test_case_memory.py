import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
RAW_TURN_TEXT = "I have trouble sleeping nearly every night."

from psyvec.memory import (  # noqa: E402
    AccessPolicy,
    CaseIdMismatchError,
    CaseMemoryAdapter,
    CaseMemoryBuilder,
    UnauthorizedRoleError,
)
from psyvec.state import (  # noqa: E402
    AssessmentState,
    EvidenceSlot,
    LabelLeakageError,
    TopicState,
)


def state() -> AssessmentState:
    slot = EvidenceSlot(
        slot_id="sleep-frequency",
        kind="frequency",
        status="known",
        value="often",
        source_turn_ids=("turn-1",),
    )
    return AssessmentState(
        case_id="case-1",
        scale_version="PHQ-8",
        topic_states=(
            TopicState("sleep", "v1", evidence_slots=(slot,)),
            TopicState("mood", "v1"),
        ),
        dialogue_turn_ids=("turn-1", "turn-2"),
    )


class CaseMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = AccessPolicy({"evaluator"})
        self.memory = CaseMemoryBuilder().build(
            state(),
            topic_summaries={"sleep": "Difficulty sleeping."},
            access_policy=self.policy,
        )

    def test_projection_indexes_evidence_and_keeps_turn_ids_only(self) -> None:
        self.assertEqual(self.memory.evidence_index["sleep"], ("sleep-frequency",))
        self.assertEqual(self.memory.evidence_index["mood"], ())
        self.assertEqual(self.memory.turn_refs, ("turn-1", "turn-2"))
        self.assertNotIn(RAW_TURN_TEXT, repr(self.memory))

    def test_permitted_role_reads_memory(self) -> None:
        result = CaseMemoryAdapter().read(
            self.memory, role="evaluator", case_id="case-1"
        )
        self.assertIs(result, self.memory)

    def test_role_outside_access_policy_is_refused(self) -> None:
        with self.assertRaises(UnauthorizedRoleError):
            CaseMemoryAdapter().read(self.memory, role="scorer", case_id="case-1")

    def test_different_case_id_is_refused(self) -> None:
        with self.assertRaises(CaseIdMismatchError):
            CaseMemoryAdapter().read(
                self.memory, role="evaluator", case_id="case-2"
            )

    def test_summary_with_forbidden_label_key_is_refused(self) -> None:
        with self.assertRaises(LabelLeakageError):
            CaseMemoryBuilder().build(
                state(),
                topic_summaries={"label": "depressed"},
                access_policy=self.policy,
            )


if __name__ == "__main__":
    unittest.main()
