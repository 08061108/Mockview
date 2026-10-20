"""Persistence helpers for interview session metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.models.interview import AnswerAssessment, ChatMessage, InterviewSession
from app.models.resume import ResumeParseResult
from app.models.preparation import AssessmentTarget
from app.models.evidence import EvidenceAssessment, EvidenceChunk
from app.services.database import get_pool, session_store_enabled

logger = logging.getLogger(__name__)


@dataclass
class PersistedSession:
    session: InterviewSession
    mode: str
    preparation_case_id: str | None = None
    assessment_targets: list[AssessmentTarget] | None = None
    evidence_chunks: dict[str, list[EvidenceChunk]] | None = None
    evidence_assessments: dict[str, EvidenceAssessment] | None = None
    pending_answers: dict[str, dict[str, Any]] | None = None
    analysis_snapshot: dict[str, Any] | None = None
    material_revision: int | None = None
    knowledge_release_id: str | None = None
    operation_state: dict[str, Any] | None = None
    resume_text: str = ""
    resume_parse_result: ResumeParseResult | None = None
    graph_started: bool = False
    last_state: dict[str, Any] | None = None
    persisted_assessment_count: int = 0
    final_memory_saved: bool = False
    error_message: str | None = None
    created_at: Any | None = None
    updated_at: Any | None = None
    completed_at: Any | None = None
    completion_reason: str | None = None
    paused_at: Any | None = None


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _session_from_row(row: dict[str, Any]) -> PersistedSession:
    conversation = [
        msg if isinstance(msg, ChatMessage) else ChatMessage.model_validate(msg)
        for msg in (row.get("conversation_history") or [])
    ]
    assessments = [
        item if isinstance(item, AnswerAssessment) else AnswerAssessment.model_validate(item)
        for item in (row.get("assessments") or [])
    ]
    session = InterviewSession(
        session_id=row["session_id"],
        user_id=row["user_id"],
        preparation_case_id=row.get("preparation_case_id"),
        jd_text=row["jd_text"],
        status=row["status"],
        current_question_index=row.get("current_question_index") or 0,
        follow_up_count=row.get("follow_up_count") or 0,
        max_follow_ups=row.get("max_follow_ups") or 2,
        conversation_history=conversation,
        assessments=assessments,
    )
    raw_resume = row.get("resume_parse_result")
    resume_parse_result = (
        ResumeParseResult.model_validate(raw_resume)
        if raw_resume
        else None
    )
    return PersistedSession(
        session=session,
        mode=row["mode"],
        preparation_case_id=row.get("preparation_case_id"),
        assessment_targets=[
            AssessmentTarget.model_validate(item)
            for item in (row.get("assessment_targets") or [])
        ],
        evidence_chunks={
            key: [EvidenceChunk.model_validate(item) for item in values]
            for key, values in (row.get("evidence_chunks") or {}).items()
        },
        evidence_assessments={
            key: EvidenceAssessment.model_validate(value)
            for key, value in (row.get("evidence_assessments") or {}).items()
        },
        pending_answers=row.get("pending_answers") or {},
        analysis_snapshot=row.get("analysis_snapshot") or {},
        material_revision=row.get("material_revision"),
        knowledge_release_id=row.get("knowledge_release_id"),
        operation_state=row.get("operation_state") or {},
        resume_text=row.get("resume_text") or "",
        resume_parse_result=resume_parse_result,
        graph_started=bool(row.get("graph_started")),
        last_state=row.get("last_state") or {},
        persisted_assessment_count=row.get("persisted_assessment_count") or 0,
        final_memory_saved=bool(row.get("final_memory_saved")),
        error_message=row.get("error_message"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        completed_at=row.get("completed_at"),
        completion_reason=row.get("completion_reason"),
        paused_at=row.get("paused_at"),
    )


async def save_session_record(
    *,
    session: InterviewSession,
    mode: str,
    resume_text: str = "",
    resume_parse_result: ResumeParseResult | None = None,
    graph_started: bool = False,
    last_state: dict[str, Any] | None = None,
    persisted_assessment_count: int = 0,
    final_memory_saved: bool = False,
    error_message: str | None = None,
    preparation_case_id: str | None = None,
    assessment_targets: list[AssessmentTarget] | None = None,
    evidence_chunks: dict[str, list[EvidenceChunk]] | None = None,
    evidence_assessments: dict[str, EvidenceAssessment] | None = None,
    pending_answers: dict[str, dict[str, Any]] | None = None,
    analysis_snapshot: dict[str, Any] | None = None,
    material_revision: int | None = None,
    knowledge_release_id: str | None = None,
    operation_state: dict[str, Any] | None = None,
    completion_reason: str | None = None,
    paused_at: Any | None = None,
) -> None:
    """Upsert one interview session metadata row."""
    if not session_store_enabled():
        return
    pool = get_pool()
    if pool is None:
        logger.debug("Postgres session store enabled but pool is not initialized")
        return

    from psycopg.types.json import Jsonb

    completed_expr = "NOW()" if session.status == "completed" else "NULL"
    async with pool.connection() as conn:
        await conn.execute(
            f"""
            INSERT INTO interview_sessions (
                session_id,
                user_id,
                preparation_case_id,
                assessment_targets,
                evidence_chunks,
                evidence_assessments,
                pending_answers,
                analysis_snapshot,
                material_revision,
                knowledge_release_id,
                operation_state,
                mode,
                jd_text,
                max_follow_ups,
                status,
                current_question_index,
                follow_up_count,
                graph_started,
                resume_text,
                resume_parse_result,
                conversation_history,
                assessments,
                last_state,
                persisted_assessment_count,
                final_memory_saved,
                error_message,
                completed_at,
                completion_reason,
                paused_at
            )
            VALUES (
                %(session_id)s,
                %(user_id)s,
                %(preparation_case_id)s,
                %(assessment_targets)s,
                %(evidence_chunks)s,
                %(evidence_assessments)s,
                %(pending_answers)s,
                %(analysis_snapshot)s,
                %(material_revision)s,
                %(knowledge_release_id)s,
                %(operation_state)s,
                %(mode)s,
                %(jd_text)s,
                %(max_follow_ups)s,
                %(status)s,
                %(current_question_index)s,
                %(follow_up_count)s,
                %(graph_started)s,
                %(resume_text)s,
                %(resume_parse_result)s,
                %(conversation_history)s,
                %(assessments)s,
                %(last_state)s,
                %(persisted_assessment_count)s,
                %(final_memory_saved)s,
                %(error_message)s,
                {completed_expr},
                %(completion_reason)s,
                %(paused_at)s
            )
            ON CONFLICT (session_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                preparation_case_id = EXCLUDED.preparation_case_id,
                assessment_targets = EXCLUDED.assessment_targets,
                evidence_chunks = EXCLUDED.evidence_chunks,
                evidence_assessments = EXCLUDED.evidence_assessments,
                pending_answers = EXCLUDED.pending_answers,
                analysis_snapshot = EXCLUDED.analysis_snapshot,
                material_revision = EXCLUDED.material_revision,
                knowledge_release_id = EXCLUDED.knowledge_release_id,
                operation_state = EXCLUDED.operation_state,
                mode = EXCLUDED.mode,
                jd_text = EXCLUDED.jd_text,
                max_follow_ups = EXCLUDED.max_follow_ups,
                status = EXCLUDED.status,
                current_question_index = EXCLUDED.current_question_index,
                follow_up_count = EXCLUDED.follow_up_count,
                graph_started = EXCLUDED.graph_started,
                resume_text = EXCLUDED.resume_text,
                resume_parse_result = EXCLUDED.resume_parse_result,
                conversation_history = EXCLUDED.conversation_history,
                assessments = EXCLUDED.assessments,
                last_state = EXCLUDED.last_state,
                persisted_assessment_count = EXCLUDED.persisted_assessment_count,
                final_memory_saved = EXCLUDED.final_memory_saved,
                error_message = EXCLUDED.error_message,
                updated_at = NOW(),
                completed_at = CASE
                    WHEN EXCLUDED.status = 'completed'
                    THEN COALESCE(interview_sessions.completed_at, NOW())
                    ELSE interview_sessions.completed_at
                END,
                completion_reason = EXCLUDED.completion_reason,
                paused_at = EXCLUDED.paused_at
            """,
            {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "preparation_case_id": preparation_case_id,
                "assessment_targets": Jsonb(_dump(assessment_targets or [])),
                "evidence_chunks": Jsonb(_dump(evidence_chunks or {})),
                "evidence_assessments": Jsonb(_dump(evidence_assessments or {})),
                "pending_answers": Jsonb(_dump(pending_answers or {})),
                "analysis_snapshot": Jsonb(_dump(analysis_snapshot or {})),
                "material_revision": material_revision,
                "knowledge_release_id": knowledge_release_id,
                "operation_state": Jsonb(_dump(operation_state or {})),
                "mode": mode,
                "jd_text": session.jd_text,
                "max_follow_ups": session.max_follow_ups,
                "status": session.status,
                "current_question_index": session.current_question_index,
                "follow_up_count": session.follow_up_count,
                "graph_started": graph_started,
                "resume_text": resume_text,
                "resume_parse_result": (
                    Jsonb(_dump(resume_parse_result)) if resume_parse_result else None
                ),
                "conversation_history": Jsonb(_dump(session.conversation_history)),
                "assessments": Jsonb(_dump(session.assessments)),
                "last_state": Jsonb(_dump(last_state or {})),
                "persisted_assessment_count": persisted_assessment_count,
                "final_memory_saved": final_memory_saved,
                "error_message": error_message,
                "completion_reason": completion_reason,
                "paused_at": paused_at,
            },
        )


async def load_session_record(session_id: str) -> PersistedSession | None:
    if not session_store_enabled():
        return None
    pool = get_pool()
    if pool is None:
        logger.debug("Postgres session store enabled but pool is not initialized")
        return None

    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM interview_sessions WHERE session_id = %s",
            (session_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return _session_from_row(row)


async def list_session_records_for_user(
    user_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[PersistedSession]:
    """List persisted session metadata for one user, newest first."""
    if not session_store_enabled():
        return []
    pool = get_pool()
    if pool is None:
        logger.debug("Postgres session store enabled but pool is not initialized")
        return []

    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT *
            FROM interview_sessions
            WHERE user_id = %s
            ORDER BY updated_at DESC, created_at DESC
            LIMIT %s OFFSET %s
            """,
            (user_id, limit, offset),
        )
        rows = await cursor.fetchall()
    return [_session_from_row(row) for row in rows]
