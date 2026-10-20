"""PostgreSQL persistence for preparation cases; no-op when unavailable."""

from __future__ import annotations

from app.models.preparation import (
    AssessmentTarget,
    ClarificationRequest,
    PreparationCase,
)
from app.services.database import get_pool, session_store_enabled


async def save_preparation_case(case: PreparationCase) -> None:
    if not session_store_enabled() or get_pool() is None:
        return
    from psycopg.types.json import Jsonb

    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO preparation_cases (
                    case_id, user_id, parent_case_id, material_revision, status, jd_text,
                    resume_text, resume_parse_warnings, project_notes, preferences, knowledge_release_id,
                    evidence_chunks, evidence_assessments, skill_matrix, resume_profile,
                    analysis_strategy_version, created_at, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (case_id) DO UPDATE SET
                    user_id=EXCLUDED.user_id,
                    parent_case_id=EXCLUDED.parent_case_id,
                    material_revision=EXCLUDED.material_revision,
                    status=EXCLUDED.status,
                    jd_text=EXCLUDED.jd_text,
                    resume_text=EXCLUDED.resume_text,
                    resume_parse_warnings=EXCLUDED.resume_parse_warnings,
                    project_notes=EXCLUDED.project_notes,
                    preferences=EXCLUDED.preferences,
                    knowledge_release_id=EXCLUDED.knowledge_release_id,
                    evidence_chunks=EXCLUDED.evidence_chunks,
                    evidence_assessments=EXCLUDED.evidence_assessments,
                    skill_matrix=EXCLUDED.skill_matrix,
                    resume_profile=EXCLUDED.resume_profile,
                    analysis_strategy_version=EXCLUDED.analysis_strategy_version,
                    updated_at=EXCLUDED.updated_at
                """,
                (
                    case.case_id,
                    case.user_id,
                    case.parent_case_id,
                    case.material_revision,
                    case.status,
                    case.jd_text,
                    case.resume_text,
                    Jsonb(case.resume_parse_warnings),
                    case.project_notes,
                    Jsonb(case.preferences),
                    case.knowledge_release_id,
                    Jsonb({
                        key: [chunk.model_dump(mode="json") for chunk in chunks]
                        for key, chunks in case.evidence_chunks.items()
                    }),
                    Jsonb({
                        key: value.model_dump(mode="json")
                        for key, value in case.evidence_assessments.items()
                    }),
                    Jsonb(case.skill_matrix.model_dump(mode="json")) if case.skill_matrix else None,
                    Jsonb(case.resume_profile.model_dump(mode="json")) if case.resume_profile else None,
                    case.analysis_strategy_version,
                    case.created_at,
                    case.updated_at,
                ),
            )
            for target in case.targets:
                await conn.execute(
                    """
                    INSERT INTO assessment_targets (
                        target_id, case_id, material_revision, requirement_id, skill_id,
                        scope, importance, sources, citations, verification_goal,
                        constraints, status, version
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (target_id) DO UPDATE SET
                        sources=EXCLUDED.sources,
                        citations=EXCLUDED.citations,
                        verification_goal=EXCLUDED.verification_goal,
                        constraints=EXCLUDED.constraints,
                        status=EXCLUDED.status,
                        version=EXCLUDED.version
                    """,
                    (
                        target.target_id,
                        target.case_id,
                        target.material_revision,
                        target.requirement_id,
                        target.skill_id,
                        target.scope,
                        target.importance,
                        Jsonb(target.sources),
                        Jsonb([item.model_dump(mode="json") for item in target.citations]),
                        target.verification_goal,
                        Jsonb(target.constraints),
                        target.status,
                        target.version,
                    ),
                )
            for item in case.clarifications:
                await conn.execute(
                    """
                    INSERT INTO preparation_clarifications (
                        clarification_id, case_id, material_revision, question,
                        missing_information, status, answer
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (clarification_id) DO UPDATE SET
                        status=EXCLUDED.status,
                        answer=EXCLUDED.answer,
                        updated_at=NOW()
                    """,
                    (
                        item.clarification_id,
                        item.case_id,
                        item.material_revision,
                        item.question,
                        item.missing_information,
                        item.status,
                        item.answer,
                    ),
                )


async def load_preparation_case(case_id: str, user_id: str) -> PreparationCase | None:
    if not session_store_enabled() or get_pool() is None:
        return None
    pool = get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM preparation_cases WHERE case_id=%s AND user_id=%s",
            (case_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        target_cursor = await conn.execute(
            "SELECT * FROM assessment_targets WHERE case_id=%s AND material_revision=%s ORDER BY importance DESC, target_id",
            (case_id, row["material_revision"]),
        )
        targets = await target_cursor.fetchall()
        clarification_cursor = await conn.execute(
            "SELECT * FROM preparation_clarifications WHERE case_id=%s AND material_revision=%s ORDER BY created_at, clarification_id",
            (case_id, row["material_revision"]),
        )
        clarifications = await clarification_cursor.fetchall()
    return PreparationCase(
        case_id=row["case_id"],
        user_id=row["user_id"],
        parent_case_id=row.get("parent_case_id"),
        material_revision=row["material_revision"],
        knowledge_release_id=row.get("knowledge_release_id"),
        status=row["status"],
        jd_text=row["jd_text"],
        resume_text=row["resume_text"],
        resume_parse_warnings=row.get("resume_parse_warnings") or [],
        project_notes=row["project_notes"],
        preferences=row["preferences"] or {},
        evidence_chunks=row.get("evidence_chunks") or {},
        evidence_assessments=row.get("evidence_assessments") or {},
        skill_matrix=row.get("skill_matrix"),
        resume_profile=row.get("resume_profile"),
        analysis_strategy_version=row.get("analysis_strategy_version") or "material-analysis-v1",
        targets=[AssessmentTarget.model_validate(dict(item)) for item in targets],
        clarifications=[ClarificationRequest.model_validate(dict(item)) for item in clarifications],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
