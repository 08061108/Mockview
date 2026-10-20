"""LangGraph shared state definition for the interview workflow."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, TypedDict

from app.models.interview import AnswerAssessment, ChatMessage, FinalQuestionAssessment
from app.models.jd import SkillMatrix
from app.models.question import QuestionBlueprint, QuestionItem, QuestionPlanCoverage
from app.models.report import InterviewReport, PracticeReport, ProfessionalReport
from app.models.resume import ResumeJDMatch, ResumeParseResult, ResumeProfile
from app.models.preparation import AssessmentTarget
from app.models.evidence import EvidenceAssessment, QuestionValidation


def _append_messages(
    existing: list[ChatMessage], new: list[ChatMessage]
) -> list[ChatMessage]:
    """Merge messages by stable ID while preserving first-seen order."""
    merged: list[ChatMessage] = []
    positions: dict[str, int] = {}
    for raw in [*(existing or []), *(new or [])]:
        message = raw if isinstance(raw, ChatMessage) else ChatMessage.model_validate(raw)
        key = message.message_id or hashlib.sha256(
            json.dumps(message.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        if key in positions:
            merged[positions[key]] = message
        else:
            positions[key] = len(merged)
            merged.append(message)
    return merged


def _merge_assessments(
    existing: list[AnswerAssessment], new: list[AnswerAssessment]
) -> list[AnswerAssessment]:
    merged: list[AnswerAssessment] = []
    positions: dict[str, int] = {}
    for raw in [*(existing or []), *(new or [])]:
        item = raw if isinstance(raw, AnswerAssessment) else AnswerAssessment.model_validate(raw)
        key = item.assessment_id or hashlib.sha256(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        if key in positions:
            merged[positions[key]] = item
        else:
            positions[key] = len(merged)
            merged.append(item)
    return merged


def _merge_final_assessments(
    existing: dict[str, FinalQuestionAssessment],
    new: dict[str, FinalQuestionAssessment],
) -> dict[str, FinalQuestionAssessment]:
    result = dict(existing or {})
    for key, raw in (new or {}).items():
        result[str(key)] = (
            raw if isinstance(raw, FinalQuestionAssessment) else FinalQuestionAssessment.model_validate(raw)
        )
    return result


class InterviewState(TypedDict, total=False):
    """Shared state flowing through the LangGraph interview graph.

    Each agent reads the fields it needs and writes the fields it owns.
    Using total=False so we can build the state incrementally.
    """

    # --- Mode ---
    interview_mode: Literal["practice", "professional"]

    # --- Input ---
    session_id: str
    user_id: str
    jd_text: str
    resume_text: str
    resume_parse_result: ResumeParseResult
    material_revision: int | None
    analysis_strategy_version: str
    knowledge_release_id: str | None

    # --- Long-term memory ---
    memory_context: str
    retrieved_memories: dict

    # --- Resume Analyst output (professional mode) ---
    resume_profile: ResumeProfile
    resume_jd_match: ResumeJDMatch

    # --- JD Analyst output ---
    skill_matrix: SkillMatrix
    assessment_targets: list[AssessmentTarget]
    evidence_chunks: dict[str, list]
    evidence_assessments: dict[str, EvidenceAssessment]
    preparation_case_id: str
    question_validations: dict[str, QuestionValidation]

    # --- Question Planner output ---
    question_plan: list[QuestionItem]
    question_plan_coverage: dict[str, QuestionPlanCoverage]
    question_blueprint: QuestionBlueprint
    question_reference_snapshot: list[dict]
    question_generation_status: Literal["idle", "generating", "ready", "failed"]
    question_generation_error: str
    question_generation_attempts: dict[str, int]
    question_retry_commands: dict[str, int]

    # --- Interview loop state ---
    current_question_index: int
    follow_up_count: int
    max_follow_ups: int
    conversation_history: Annotated[list[ChatMessage], _append_messages]
    current_candidate_answer: str

    # --- Assessment state ---
    assessments: Annotated[list[AnswerAssessment], _merge_assessments]
    final_question_assessments: Annotated[
        dict[str, FinalQuestionAssessment], _merge_final_assessments
    ]
    current_turn_id: str
    current_answer_id: str
    current_question_id: str

    # --- Two-round state (professional mode) ---
    current_round: int
    round1_question_plan: list[QuestionItem]
    round1_assessments: list[AnswerAssessment]
    round1_final_question_assessments: dict[str, FinalQuestionAssessment]
    round1_summary_text: str
    round2_question_plan: list[QuestionItem]

    # --- Evaluator output (professional mode) ---
    final_report: InterviewReport
    professional_report: ProfessionalReport

    # --- Evaluator output (practice mode) ---
    practice_report: PracticeReport

    # --- Control flow ---
    interview_complete: bool
    graph_version: str
    state_version: str
    rubric_version: str
