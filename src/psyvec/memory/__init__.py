"""Case-scoped, policy-filtered memory."""

from __future__ import annotations

from psyvec.memory.case_memory import (
    AccessPolicy,
    CaseIdMismatchError,
    CaseMemory,
    CaseMemoryAccessError,
    CaseMemoryAdapter,
    CaseMemoryBuilder,
    ExpiredCaseMemoryError,
    UnauthorizedRoleError,
)
from psyvec.memory.patient_evidence import (
    ContradictionEdge,
    EvidenceRecord,
    PatientEvidenceMemory,
    add_evidence,
    detect_contradictions,
    make_record_id,
)

__all__ = [
    "AccessPolicy",
    "CaseIdMismatchError",
    "CaseMemory",
    "CaseMemoryAccessError",
    "CaseMemoryAdapter",
    "CaseMemoryBuilder",
    "ContradictionEdge",
    "EvidenceRecord",
    "ExpiredCaseMemoryError",
    "PatientEvidenceMemory",
    "UnauthorizedRoleError",
    "add_evidence",
    "detect_contradictions",
    "make_record_id",
]
