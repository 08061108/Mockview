"""Data models for interview questions."""

from __future__ import annotations

from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field, field_validator


class QuestionItem(BaseModel):
    id: int = Field(..., description="Question sequence number (1-based)")
    question_id: str = Field(
        default="",
        description="Stable business identifier; display id may repeat across rounds",
    )
    target_id: str | None = Field(
        default=None,
        description="Exact target_id copied from the provided frozen assessment-target list",
    )
    content: str = Field(..., description="面试题正文，必须使用简体中文")
    skill_tags: list[str] = Field(..., description="Skills this question tests")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty level")
    reference_points: list[str] = Field(
        ..., min_length=1, description="优秀回答应覆盖的关键点，必须使用简体中文"
    )
    follow_up_directions: list[str] = Field(
        default_factory=list, description="可能的追问方向，必须使用简体中文"
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="Reference question IDs adapted from the RAG question bank",
    )
    source_categories: list[str] = Field(
        default_factory=list,
        description="Question bank categories for the adapted references",
    )
    primary_skill: str | None = Field(
        default=None,
        description="The single skill that receives this question's score",
    )
    evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Evidence chunks supporting factual rubric points",
    )
    rubric_citations: dict[str, list[str]] = Field(
        default_factory=dict,
        description="JSON object mapping each exact reference-point string to an array of evidence chunk ID strings; never a list or scalar",
    )
    scenario_assumptions: list[str] = Field(default_factory=list)
    acceptable_alternatives: list[str] = Field(default_factory=list)
    validation_status: Literal["pending", "supported", "conflicting", "uncertain"] = "pending"

    @field_validator("skill_tags", mode="before")
    @classmethod
    def flatten_skill_tags(cls, value):
        if not isinstance(value, list):
            return value
        return [str(item) for group in value for item in (group if isinstance(group, list) else [group])]

    @field_validator("rubric_citations", mode="before")
    @classmethod
    def reject_ambiguous_generated_citations(cls, value):
        # Planner citations are server-owned and recomputed from selected evidence.
        return value if isinstance(value, dict) else {}

    def ensure_stable_id(self, namespace: str) -> "QuestionItem":
        """Assign a repeatable ID before execution; never ask the LLM to invent it."""
        if self.question_id:
            return self
        stable = uuid5(NAMESPACE_URL, f"mockview:{namespace}:{self.id}:{self.content}").hex
        return self.model_copy(
            update={
                "question_id": stable,
                "primary_skill": self.primary_skill or (self.skill_tags[0] if self.skill_tags else None),
            }
        )


class QuestionPlan(BaseModel):
    """The full interview question plan produced by the Question Planner agent."""

    questions: list[QuestionItem] = Field(..., min_length=1, max_length=10)
    total_estimated_minutes: int = Field(
        default=20, description="Estimated total interview duration"
    )


class QuestionPlanCoverage(BaseModel):
    """Deterministic coverage summary for one planning stage."""

    status: Literal["complete", "partial", "not_applicable"]
    target_ids: list[str] = Field(default_factory=list)
    covered_target_ids: list[str] = Field(default_factory=list)
    uncovered_target_ids: list[str] = Field(default_factory=list)
    evidence_eligible_target_ids: list[str] = Field(default_factory=list)
    evidence_unresolved_target_ids: list[str] = Field(default_factory=list)
    retained_question_count: int = 0
    rejected_question_count: int = 0


class QuestionSlot(BaseModel):
    """A stable place in an interview blueprint, without generated wording."""

    slot_id: str
    question_id: str
    position: int = Field(ge=1)
    round_number: int = Field(ge=1, le=2)
    target_id: str | None = None
    primary_skill: str
    difficulty: Literal["easy", "medium", "hard"]
    weight: float = Field(default=1.0, gt=0)
    objective: str


class QuestionBlueprint(BaseModel):
    """Frozen global coverage plan; question wording is generated slot-by-slot."""

    blueprint_id: str
    input_version: str
    round_number: int
    planned_count: int = Field(ge=1, le=10)
    slots: list[QuestionSlot] = Field(min_length=1, max_length=10)


class GeneratedQuestionDraft(BaseModel):
    """Model-owned wording and rubric for one server-owned blueprint slot."""

    content: str = Field(..., description="面试题正文，必须使用简体中文")
    skill_tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"]
    reference_points: list[str] = Field(min_length=1)
    follow_up_directions: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    scenario_assumptions: list[str] = Field(default_factory=list)
    acceptable_alternatives: list[str] = Field(default_factory=list)
