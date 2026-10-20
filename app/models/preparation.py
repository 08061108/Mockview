"""Preparation-case, target, clarification, and operation contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field
from app.models.evidence import EvidenceAssessment, EvidenceChunk
from app.models.jd import SkillMatrix
from app.models.resume import ResumeProfile


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


PreparationStatus = Literal[
    "draft", "analyzing", "waiting_input", "ready", "failed", "cancelled"
]
TargetStatus = Literal["pending", "ready", "partial", "missing_evidence"]
OperationStatus = Literal[
    "accepted", "running", "waiting_input", "succeeded", "failed_retryable", "failed", "cancelled"
]


class MaterialCitation(BaseModel):
    material_type: Literal["jd", "resume", "project_note", "observation"]
    revision: int
    quote: str
    source_id: str | None = None
    start: int | None = None
    end: int | None = None


class AssessmentTarget(BaseModel):
    target_id: str
    case_id: str
    material_revision: int
    requirement_id: str
    skill_id: str
    scope: str
    importance: float = Field(ge=0.0, le=1.0)
    sources: list[Literal["job_requirement", "project_claim", "historical_observation"]]
    citations: list[MaterialCitation] = Field(default_factory=list)
    verification_goal: str
    constraints: list[str] = Field(default_factory=list)
    gap_kind: Literal["technical_knowledge", "personal_experience"] = "technical_knowledge"
    resume_signal: Literal["claimed", "unmentioned", "not_applicable"] = "not_applicable"
    status: TargetStatus = "pending"
    version: int = 1


class ClarificationRequest(BaseModel):
    clarification_id: str
    case_id: str
    material_revision: int
    question: str
    missing_information: str
    status: Literal["pending", "answered", "cancelled"] = "pending"
    answer: str | None = None


class PreparationCase(BaseModel):
    case_id: str
    user_id: str
    parent_case_id: str | None = None
    jd_text: str
    resume_text: str = ""
    resume_parse_warnings: list[str] = Field(default_factory=list)
    project_notes: str = ""
    preferences: dict[str, Any] = Field(default_factory=dict)
    material_revision: int = 1
    knowledge_release_id: str | None = None
    status: PreparationStatus = "draft"
    targets: list[AssessmentTarget] = Field(default_factory=list)
    clarifications: list[ClarificationRequest] = Field(default_factory=list)
    evidence_chunks: dict[str, list[EvidenceChunk]] = Field(default_factory=dict)
    evidence_assessments: dict[str, EvidenceAssessment] = Field(default_factory=dict)
    skill_matrix: SkillMatrix | None = None
    resume_profile: ResumeProfile | None = None
    analysis_strategy_version: str = "material-analysis-v1"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class PreparationCreate(BaseModel):
    jd_text: str = Field(min_length=1, max_length=50000)
    resume_text: str = Field(default="", max_length=100000)
    project_notes: str = Field(default="", max_length=100000)
    preferences: dict[str, Any] = Field(default_factory=dict)


class PreparationUpdate(BaseModel):
    jd_text: str | None = Field(default=None, max_length=50000)
    resume_text: str | None = Field(default=None, max_length=100000)
    project_notes: str | None = Field(default=None, max_length=100000)
    preferences: dict[str, Any] | None = None


class RetrainingRequest(BaseModel):
    source_session_id: str = Field(min_length=1, max_length=128)
    selected_skills: list[str] = Field(min_length=1)
    preferences: dict[str, Any] = Field(default_factory=dict)


class OperationRecord(BaseModel):
    operation_id: str
    user_id: str
    kind: str
    case_id: str | None = None
    status: OperationStatus = "accepted"
    input_revision: int | None = None
    graph_version: str = "preparation-p4-v1"
    state_version: str = "preparation-state-p4-v1"
    input_hash: str
    result: dict[str, Any] | None = None
    error: str | None = None
    error_details: dict[str, Any] | None = None
    budget: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ToolCallRecord(BaseModel):
    tool_call_id: str
    operation_id: str
    tool_name: str
    input_hash: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    status: Literal["accepted", "running", "succeeded", "failed"] = "accepted"
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    budget_counted: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AgentDecision(BaseModel):
    tool: Literal[
        "read_material",
        "get_observations",
        "retrieve_knowledge",
        "search_public_knowledge",
        "request_clarification",
        "finish_assessment",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason_summary: str
