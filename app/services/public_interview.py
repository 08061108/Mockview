"""Build every externally visible interview payload from explicit allow-lists."""

from __future__ import annotations

from typing import Any

from app.models.interview import ChatMessage, FinalQuestionAssessment, InterviewSession
from app.models.public import (
    PublicCurrentTurn,
    PublicInterviewReport,
    PublicMissedKnowledge,
    PublicQuestionReview,
    PublicReportHighlight,
    PublicRoundScore,
    PublicSessionOverview,
    PublicSkillScore,
    PublicTranscript,
    PublicTranscriptItem,
)


FORBIDDEN_PUBLIC_FIELDS = frozenset({
    "question_plan", "reference_points", "acceptable_alternatives",
    "rubric_citations", "evidence_chunk_ids", "evidence_chunks", "rubric",
    "prompt", "analysis_snapshot", "model_thinking", "follow_up_reason",
})


def _value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def build_public_report(report: Any, *, completed: bool) -> PublicInterviewReport | None:
    if not report or not completed:
        return None
    return PublicInterviewReport(
        overall_score=_value(report, "overall_score"),
        grade=_value(report, "grade"),
        overall_assessment=str(_value(report, "overall_assessment", "")),
        skill_scores=[PublicSkillScore.model_validate({
            "skill_name": _value(item, "skill_name", ""),
            "score": _value(item, "score", 0),
            "evidence": _value(item, "evidence", ""),
        }) for item in (_value(report, "skill_scores", []) or [])],
        strengths=[PublicReportHighlight.model_validate({
            "point": _value(item, "point", ""), "evidence": _value(item, "evidence", "")
        }) for item in (_value(report, "strengths", []) or [])],
        improvements=[PublicReportHighlight.model_validate({
            "point": _value(item, "point", ""), "evidence": _value(item, "evidence", "")
        }) for item in (_value(report, "improvements", []) or [])],
        round_scores=[PublicRoundScore.model_validate({
            "round_name": _value(item, "round_name", ""),
            "score": _value(item, "score", 0),
            "grade": _value(item, "grade", ""),
            "summary": _value(item, "summary", ""),
        }) for item in (_value(report, "round_scores", []) or [])],
        technical_depth_score=_value(report, "technical_depth_score"),
        technical_breadth_score=_value(report, "technical_breadth_score"),
        hiring_recommendation=_value(report, "hiring_recommendation"),
        total_questions=_value(report, "total_questions"),
        coverage={key: _value(report, "coverage", {}).get(key) for key in
                  ("covered_skills", "uncovered_skills", "scored_questions", "unscored_questions")
                  if key in (_value(report, "coverage", {}) or {})},
        missed_knowledge=[PublicMissedKnowledge.model_validate({
            "question": _value(item, "question", ""),
            "question_id": _value(item, "question_id"),
            "score": _value(item, "score"),
            "scorable": _value(item, "scorable", _value(item, "score") is not None),
            "missed_points": list(_value(item, "missed_points", []) or []),
            # This is a report-authored learning answer, never a rubric projection.
            "reference_answer": _value(item, "reference_answer", ""),
        }) for item in (_value(report, "missed_knowledge", []) or [])],
        study_suggestions=list(_value(report, "study_suggestions", []) or []),
    )


def current_public_turn(state: dict[str, Any]) -> PublicCurrentTurn | None:
    for raw in reversed(state.get("conversation_history", []) or []):
        try:
            msg = raw if isinstance(raw, ChatMessage) else ChatMessage.model_validate(raw)
        except Exception:
            continue
        if msg.role == "interviewer":
            index = int(state.get("current_question_index", 0) or 0)
            plan = state.get("question_plan", []) or []
            question = plan[index] if index < len(plan) else None
            tags = _value(question, "skill_tags", []) if question is not None else []
            difficulty = _value(question, "difficulty", "") if question is not None else ""
            return PublicCurrentTurn(
                message_id=msg.message_id, content=msg.content,
                question_id=msg.question_id, turn_id=msg.turn_id,
                kind=msg.kind or ("follow_up" if int(state.get("follow_up_count", 0) or 0) > 0 else "question"),
                question_index=index + 1,
                total_questions=len(plan), skill_tags=list(tags or []),
                difficulty=str(difficulty or ""),
                follow_up_number=(int(state.get("follow_up_count", 0) or 0) or None),
            )
    return None


def build_public_transcript(
    *, session: InterviewSession, overview: PublicSessionOverview,
    state: dict[str, Any], pending_answers: dict[str, dict[str, Any]], report: Any,
) -> PublicTranscript:
    graph_items: list[PublicTranscriptItem] = []
    answer_ids: set[str] = set()
    message_ids: set[str] = set()
    for index, raw in enumerate(state.get("conversation_history", []) or session.conversation_history):
        try:
            msg = raw if isinstance(raw, ChatMessage) else ChatMessage.model_validate(raw)
        except Exception:
            continue
        message_id = msg.message_id or f"legacy-{index}"
        if message_id in message_ids:
            continue
        message_ids.add(message_id)
        if msg.answer_id:
            answer_ids.add(msg.answer_id)
        graph_items.append(PublicTranscriptItem(
            message_id=message_id, role=msg.role, content=msg.content,
            question_id=msg.question_id, turn_id=msg.turn_id, kind=msg.kind,
            answer_id=msg.answer_id, created_at=msg.created_at,
            processing_status="succeeded" if msg.answer_id else None,
        ))
    # Pending answers are inserted immediately after their matching interviewer turn.
    for answer_id, raw in pending_answers.items():
        if answer_id in answer_ids:
            continue
        item = PublicTranscriptItem(
            message_id=f"answer-{answer_id}", role="candidate",
            content=str(raw.get("content", "")), turn_id=raw.get("turn_id"),
            answer_id=answer_id,
            processing_status=raw.get("status", "accepted"),
        )
        insert_at = len(graph_items)
        if item.turn_id:
            for pos, existing in enumerate(graph_items):
                if existing.role == "interviewer" and existing.turn_id == item.turn_id:
                    insert_at = pos + 1
        graph_items.insert(insert_at, item)
    reviews: list[PublicQuestionReview] = []
    if session.status == "completed":
        values = state.get("final_question_assessments", {}) or {}
        iterable = values.values() if isinstance(values, dict) else values
        for raw in iterable:
            try:
                assessment = raw if isinstance(raw, FinalQuestionAssessment) else FinalQuestionAssessment.model_validate(raw)
            except Exception:
                continue
            reviews.append(PublicQuestionReview(
                question_id=assessment.question_id, display_id=assessment.display_id,
                primary_skill=assessment.primary_skill, score=assessment.score,
                scorable=assessment.scorable,
            ))
    public_report = build_public_report(report, completed=session.status == "completed")
    return PublicTranscript(
        session=overview, items=graph_items, question_reviews=reviews,
        has_report=public_report is not None, report=public_report,
    )


def assert_no_forbidden_public_fields(value: Any) -> None:
    """Test/helper guard that detects accidental nested internal-field exposure."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        forbidden = FORBIDDEN_PUBLIC_FIELDS.intersection(value)
        if forbidden:
            raise AssertionError(f"Forbidden public fields: {sorted(forbidden)}")
        for child in value.values():
            assert_no_forbidden_public_fields(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_forbidden_public_fields(child)
