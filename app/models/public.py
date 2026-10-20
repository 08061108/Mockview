"""Public, deliberately whitelisted interview API models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PublicError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    retry_after_seconds: int | None = None


class PublicSessionOverview(BaseModel):
    session_id: str
    preparation_case_id: str | None = None
    title: str
    mode: Literal["practice", "professional"]
    status: str
    graph_started: bool
    has_report: bool
    question_count: int = 0
    assessment_count: int = 0
    current_question_index: int = 0
    max_follow_ups: int = 2
    jd_preview: str = ""
    overall_score: float | None = None
    grade: str | None = None
    completion_reason: Literal["normal", "early", "no_assessment"] | None = None
    pending_action: Literal["answer_retry_required"] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    paused_at: str | None = None


class PublicCurrentTurn(BaseModel):
    message_id: str = ""
    content: str
    question_id: str | None = None
    turn_id: str | None = None
    kind: str | None = None
    question_index: int = 0
    total_questions: int = 0
    skill_tags: list[str] = Field(default_factory=list)
    difficulty: str = ""
    follow_up_number: int | None = None


class PublicSessionState(BaseModel):
    session: PublicSessionOverview
    graph_started: bool
    current_turn: PublicCurrentTurn | None = None
    pending_action: Literal["answer_retry_required"] | None = None
    has_report: bool = False
    interview_complete: bool = False


class PublicTranscriptItem(BaseModel):
    message_id: str
    role: Literal["interviewer", "candidate", "system"]
    content: str
    question_id: str | None = None
    turn_id: str | None = None
    kind: str | None = None
    answer_id: str | None = None
    created_at: datetime | None = None
    processing_status: Literal["accepted", "succeeded", "failed_retryable"] | None = None


class PublicQuestionReview(BaseModel):
    question_id: str
    display_id: int
    primary_skill: str
    score: int | None = None
    scorable: bool = True


class PublicSkillScore(BaseModel):
    skill_name: str
    score: float
    evidence: str


class PublicReportHighlight(BaseModel):
    point: str
    evidence: str


class PublicRoundScore(BaseModel):
    round_name: str
    score: float
    grade: str
    summary: str


class PublicMissedKnowledge(BaseModel):
    question: str
    question_id: str | None = None
    score: int | None = None
    scorable: bool = True
    missed_points: list[str] = Field(default_factory=list)
    reference_answer: str


class PublicInterviewReport(BaseModel):
    overall_score: float | None = None
    grade: str | None = None
    overall_assessment: str
    skill_scores: list[PublicSkillScore] = Field(default_factory=list)
    strengths: list[PublicReportHighlight] = Field(default_factory=list)
    improvements: list[PublicReportHighlight] = Field(default_factory=list)
    round_scores: list[PublicRoundScore] = Field(default_factory=list)
    technical_depth_score: float | None = None
    technical_breadth_score: float | None = None
    hiring_recommendation: str | None = None
    total_questions: int | None = None
    coverage: dict = Field(default_factory=dict)
    missed_knowledge: list[PublicMissedKnowledge] = Field(default_factory=list)
    study_suggestions: list[str] = Field(default_factory=list)


class PublicTranscript(BaseModel):
    session: PublicSessionOverview
    items: list[PublicTranscriptItem] = Field(default_factory=list)
    question_reviews: list[PublicQuestionReview] = Field(default_factory=list)
    has_report: bool = False
    report: PublicInterviewReport | None = None
