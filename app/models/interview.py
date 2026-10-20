"""Data models for interview session state and assessment."""

from __future__ import annotations

from typing import Literal
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class AnswerAssessment(BaseModel):
    """Assessment of a single candidate answer (internal, not shown to candidate)."""

    assessment_id: str = ""
    question_id: int | str
    turn_id: str = ""
    score: int = Field(..., ge=1, le=10, description="Score on 1-10 scale")
    covered_points: list[str] = Field(default_factory=list, description="Reference points covered")
    missed_points: list[str] = Field(default_factory=list, description="Reference points missed")
    should_follow_up: bool = Field(default=False)
    follow_up_reason: str | None = Field(default=None)
    answered_aspects: list[str] = Field(default_factory=list)
    needs_clarification: list[str] = Field(default_factory=list)
    follow_up_kind: Literal["clarification", "depth", "hint"] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FinalQuestionAssessment(BaseModel):
    """One effective final assessment for the complete dialogue of a question."""

    observation_id: str
    question_id: str
    display_id: int
    primary_skill: str
    score: int | None = Field(default=None, ge=1, le=10)
    scorable: bool = True
    covered_points: list[str] = Field(default_factory=list)
    missed_points: list[str] = Field(default_factory=list)
    answer_evidence: list[str] = Field(default_factory=list)
    alternative_reasoning: list[str] = Field(default_factory=list)
    hint_dependency: Literal["none", "clarification", "substantial"] = "none"
    uncertainty: list[str] = Field(default_factory=list)
    rubric_version: str = "p1-v1"


class FinalQuestionAssessmentDraft(BaseModel):
    score: int | None = Field(default=None, ge=1, le=10, description="1-10 integer; null only when scorable=false")
    scorable: bool = True
    covered_points: list[str] = Field(default_factory=list)
    missed_points: list[str] = Field(default_factory=list)
    answer_evidence: list[str] = Field(default_factory=list)
    alternative_reasoning: list[str] = Field(default_factory=list)
    hint_dependency: Literal["none", "clarification", "substantial"] = "none"
    uncertainty: list[str] = Field(default_factory=list)

    @field_validator("hint_dependency", mode="before")
    @classmethod
    def normalize_empty_hint_dependency(cls, value):
        return "none" if value is None or value == "" else value


class ChatMessage(BaseModel):
    message_id: str = ""
    role: Literal["interviewer", "candidate", "system"]
    content: str
    question_id: str | None = None
    turn_id: str | None = None
    kind: Literal["main", "clarification", "depth", "hint", "system"] | None = None
    answer_id: str | None = None
    created_at: datetime | None = None


class InterviewSession(BaseModel):
    """Metadata for a running interview session."""

    session_id: str
    user_id: str = "local-user"
    preparation_case_id: str | None = None
    jd_text: str
    status: Literal[
        "pending",
        "analyzing",
        "interviewing",
        "evaluating",
        "pausing",
        "paused",
        "completed",
        "failed",
    ] = "pending"
    current_question_index: int = 0
    follow_up_count: int = 0
    max_follow_ups: int = 2
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    assessments: list[AnswerAssessment] = Field(default_factory=list)
