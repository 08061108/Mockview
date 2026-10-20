"""Preparation cases, clarifications, and resumable operation endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Annotated
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.agents.job_assessment import JobAssessmentAgent, ToolContext
from app.config import settings
from app.llm.client import model_trace
from app.services import preparation_limits as limits
from app.models.preparation import PreparationCreate, PreparationUpdate, RetrainingRequest
from app.models.user import UserInDB
from app.models.interview import FinalQuestionAssessment
from app.domain.skill_identity import canonical_skill_id
from app.memory.service import get_memory_service
from app.services.scoring import aggregate_scores
from app.security import get_current_user
from app.services.operation_store import get_operation_store
from app.services.preparation_repository import save_preparation_case
from app.services.preparation_service import get_preparation_service
from app.services.session_manager import get_session_manager
from app.utils.file_parser import parse_resume_with_metadata

router = APIRouter()
_analysis_tasks: dict[str, asyncio.Task] = {}


class StartOperationRequest(BaseModel):
    operation_id: str | None = Field(default=None, min_length=1, max_length=128)


class ClarificationAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=20000)


def _agent_loop_enabled(operation) -> bool:
    result = operation.result if operation and isinstance(operation.result, dict) else {}
    agent_loop = result.get("agent_loop")
    if isinstance(agent_loop, dict) and isinstance(agent_loop.get("enabled"), bool):
        return agent_loop["enabled"]
    # Compatibility for historical waiting operations that had already entered
    # the Agent before the default-off switch was introduced.
    if result.get("agent_result") is not None:
        return True
    return settings.assessment_autonomous_loop_enabled


def _compact_assessment_result(value: dict | None) -> dict | None:
    """Persist/API-expose decisions and outcomes without raw evidence bodies."""
    if not isinstance(value, dict):
        return None
    compact: dict = {
        key: value[key]
        for key in ("status", "complete")
        if key in value
    }
    result = value.get("result")
    if isinstance(result, dict):
        compact["result"] = {
            key: list(result.get(key) or [])
            for key in ("completed_target_ids", "unresolved_target_ids")
        }
    compact["targets"] = [
        {
            key: item[key]
            for key in (
                "target_id",
                "gap_kind",
                "resume_signal",
                "verdict",
                "complete",
                "stop_reason",
            )
            if key in item
        }
        for item in value.get("targets", [])
        if isinstance(item, dict)
    ]
    compact["trajectory"] = [
        {
            key: item[key]
            for key in (
                "target_id",
                "tool",
                "reason",
                "decision_source",
                "reused",
                "decision_reused",
            )
            if key in item
        }
        for item in value.get("trajectory", [])
        if isinstance(item, dict)
    ]
    return compact


def _analysis_result(
    case,
    *,
    pending_clarifications: int,
    agent_enabled: bool,
    agent_status: str,
    evidence_validation: dict | None = None,
    evidence_sufficiency: str = "unknown",
    assessment_result: dict | None = None,
    agent_result: dict | None = None,
) -> dict:
    assessments = dict(case.evidence_assessments or {})
    target_ids = [item.target_id for item in case.targets]
    sufficient_ids = [
        target_id
        for target_id in target_ids
        if target_id in assessments
        and assessments[target_id].verdict == "sufficient"
    ]
    unresolved_ids = [target_id for target_id in target_ids if target_id not in sufficient_ids]
    result = {
        "case_id": case.case_id,
        "material_revision": case.material_revision,
        "target_count": len(case.targets),
        "clarification_count": len(case.clarifications),
        "pending_clarification_count": pending_clarifications,
        "material_analysis": {
            "status": "completed",
            "target_count": len(case.targets),
        },
        "evidence_validation": evidence_validation
        or {"status": "not_performed", "mode": "none"},
        "evidence_sufficiency": evidence_sufficiency,
        "evidence_coverage": {
            "target_count": len(target_ids),
            "eligible_target_count": len(sufficient_ids),
            "unresolved_target_count": len(unresolved_ids),
            "eligible_target_ids": sufficient_ids,
            "unresolved_target_ids": unresolved_ids,
        },
        "agent_loop": {"enabled": agent_enabled, "status": agent_status},
    }
    compact_agent = _compact_assessment_result(agent_result)
    compact_assessment = _compact_assessment_result(assessment_result)
    if compact_agent is not None:
        result["agent_result"] = compact_agent
    if compact_assessment is not None:
        result["assessment_result"] = compact_assessment
    return result


async def _evidence_result(operations, operation_id: str, case) -> tuple[dict, str]:
    assessments = dict(case.evidence_assessments or {})
    if not assessments:
        return {"status": "not_performed", "mode": "none"}, "unknown"

    target_ids = {item.target_id for item in case.targets}
    assessed_ids = target_ids.intersection(assessments)
    validation_status = (
        "completed" if target_ids and assessed_ids == target_ids else "partial"
    )
    records = await operations.tool_calls(operation_id, "assess_evidence")
    modes = {
        str(item.result.get("validation_mode"))
        for item in records
        if item.status == "succeeded"
        and isinstance(item.result, dict)
        and item.result.get("validation_mode") in {"deterministic", "semantic"}
    }
    validation_mode = next(iter(modes)) if len(modes) == 1 else "mixed" if modes else "unknown"
    validation = {
        "status": validation_status,
        "mode": validation_mode,
        "validated_target_count": len(assessed_ids),
        "target_count": len(target_ids),
    }
    if validation_status != "completed":
        return validation, "unknown"

    verdicts = {assessments[target_id].verdict for target_id in assessed_ids}
    if verdicts == {"sufficient"}:
        sufficiency = "sufficient"
    elif "conflicting" in verdicts:
        sufficiency = "conflicting"
    elif "partial" in verdicts or "sufficient" in verdicts:
        sufficiency = "partial"
    else:
        sufficiency = "insufficient"
    return validation, sufficiency


async def _set_case_status(
    case_id: str,
    user_id: str,
    next_status: str,
    expected_revision: int | None = None,
) -> None:
    case = await get_preparation_service().get(case_id, user_id)
    if not case or (
        expected_revision is not None and case.material_revision != expected_revision
    ):
        return
    case.status = next_status  # type: ignore[assignment]
    case.updated_at = datetime.now(timezone.utc)
    await save_preparation_case(case)


async def _run_analysis(operation_id: str, case_id: str, user_id: str) -> None:
    operations = get_operation_store()
    service = get_preparation_service()
    operation = await operations.get(operation_id, user_id)
    expected_revision = operation.input_revision if operation else None
    agent_enabled = _agent_loop_enabled(operation)
    try:
        await operations.transition(operation_id, "running")
        case = await service.get(case_id, user_id)
        if not case:
            raise KeyError(case_id)

        pending_clarifications = [
            item for item in case.clarifications if item.status == "pending"
        ]
        if case.targets and pending_clarifications:
            await operations.transition(
                operation_id,
                "waiting_input",
                result=_analysis_result(
                    case,
                    pending_clarifications=len(pending_clarifications),
                    agent_enabled=agent_enabled,
                    agent_status="not_started" if agent_enabled else "disabled",
                ),
            )
            return

        # A resumed operation must continue from the frozen targets after the
        # user answered clarifications. Re-running material analysis here would
        # regenerate the same questions and create an endless HITL loop.
        if (
            not case.targets
            or (
                case.skill_matrix is not None
                and case.analysis_strategy_version != "material-analysis-v1"
            )
        ):
            case = await service.analyze(case_id, user_id)
        pending_clarifications = [
            item for item in case.clarifications if item.status == "pending"
        ]
        if case.status == "waiting_input" or pending_clarifications:
            await operations.transition(
                operation_id,
                "waiting_input",
                result=_analysis_result(
                    case,
                    pending_clarifications=len(pending_clarifications),
                    agent_enabled=agent_enabled,
                    agent_status="not_started" if agent_enabled else "disabled",
                ),
            )
            return

        # Evidence preparation always runs. The feature flag controls only the
        # autonomous LLM Tool Calling gap loop; deterministic Hybrid retrieval,
        # evidence assessment and conservative HITL remain the default path.
        agent_result = await JobAssessmentAgent(operations).run(
            ToolContext(
                operation_id=operation_id,
                user_id=user_id,
                case_id=case_id,
                material_revision=case.material_revision,
                targets=case.targets,
            )
        )
        latest_case = await service.get(case_id, user_id) or case
        evidence_validation, evidence_sufficiency = await _evidence_result(
            operations, operation_id, latest_case
        )
        next_status = (
            "waiting_input"
            if agent_result.get("status") == "waiting_input"
            else "succeeded"
        )
        current = await operations.get(operation_id, user_id)
        if current and current.status != "cancelled":
            await operations.transition(
                operation_id,
                next_status,
                result=_analysis_result(
                    latest_case,
                    pending_clarifications=len(
                        [
                            item
                            for item in latest_case.clarifications
                            if item.status == "pending"
                        ]
                    ),
                    agent_enabled=agent_enabled,
                    agent_status=(
                        "waiting_input"
                        if agent_enabled
                        and agent_result.get("status") == "waiting_input"
                        else "executed" if agent_enabled else "disabled"
                    ),
                    evidence_validation=evidence_validation,
                    evidence_sufficiency=evidence_sufficiency,
                    assessment_result=agent_result,
                    agent_result=agent_result if agent_enabled else None,
                ),
            )
    except asyncio.CancelledError:
        await _set_case_status(case_id, user_id, "cancelled", expected_revision)
        current = await operations.get(operation_id, user_id)
        if current and current.status != "cancelled":
            await operations.transition(operation_id, "cancelled")
        raise
    except RuntimeError as exc:
        current = await operations.get(operation_id, user_id)
        if current and current.status != "cancelled":
            await _set_case_status(case_id, user_id, "failed", expected_revision)
            exhausted = str(exc).startswith("operation budget exhausted:")
            await operations.transition(operation_id, "failed" if exhausted else "failed_retryable", error=str(exc))
    except Exception as exc:
        current = await operations.get(operation_id, user_id)
        if current and current.status != "cancelled":
            await _set_case_status(case_id, user_id, "failed", expected_revision)
            await operations.transition(operation_id, "failed", error=f"{type(exc).__name__}: {exc}")


def _schedule_analysis(operation_id: str, case_id: str, user_id: str) -> asyncio.Task:
    existing = _analysis_tasks.get(operation_id)
    if existing and not existing.done():
        return existing
    async def bounded_run():
        try:
            with model_trace(operation_id=operation_id, enforce_budget=True):
                await asyncio.wait_for(
                    _run_analysis(operation_id, case_id, user_id),
                    timeout=settings.preparation_timeout_seconds,
                )
        except TimeoutError:
            current = await get_operation_store().get(operation_id, user_id)
            await _set_case_status(case_id, user_id, "failed", current.input_revision if current else None)
            await get_operation_store().transition(operation_id, "failed_retryable", error="Preparation analysis timed out")
        finally:
            limits.release(user_id, operation_id)

    task = asyncio.create_task(
        bounded_run(),
        name=f"preparation-analysis:{operation_id}",
    )
    _analysis_tasks[operation_id] = task
    def finished(done):
        if _analysis_tasks.get(operation_id) is done:
            _analysis_tasks.pop(operation_id, None)
            limits.release(user_id, operation_id)
    task.add_done_callback(finished)
    return task


@router.post("/cases", status_code=status.HTTP_201_CREATED)
async def create_case(
    request: PreparationCreate,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    await limits.enforce("preparation:create", current_user.id, settings.rate_limit_start_per_minute, 60)
    return await get_preparation_service().create(current_user.id, request)


@router.get("/cases/{case_id}")
async def get_case(
    case_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    case = await get_preparation_service().get(case_id, current_user.id)
    if not case:
        raise HTTPException(status_code=404, detail="Preparation case not found")
    return case


@router.patch("/cases/{case_id}")
async def update_case(
    case_id: str,
    request: PreparationUpdate,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    try:
        return await get_preparation_service().update(case_id, current_user.id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Preparation case not found") from exc


@router.get("/cases/{case_id}/progress")
async def get_case_progress(case_id: str, current_user: Annotated[UserInDB, Depends(get_current_user)]):
    case = await get_case(case_id, current_user)
    operation = await get_operation_store().latest_preparation(case_id, current_user.id, case.material_revision)
    return {"case": case, "operation": operation}


@router.post("/cases/{case_id}/resume")
async def upload_case_resume(
    case_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    resume_file: UploadFile = File(...),
):
    service = get_preparation_service()
    case = await service.get(case_id, current_user.id)
    if not case:
        raise HTTPException(status_code=404, detail="Preparation case not found")
    operation_id = uuid4().hex
    limits.reserve(current_user.id, operation_id)
    try:
        await limits.enforce("interview:resume", current_user.id, settings.rate_limit_resume_per_hour, 3600)
        content, suffix = await limits.read_resume(resume_file)
        if suffix != ".pdf":
            await limits.enforce("resume:vision", current_user.id, settings.rate_limit_vision_per_day, 86400)
        operations = get_operation_store()
        await operations.accept(user_id=current_user.id, kind="parse_resume", case_id=case_id,
            input_revision=case.material_revision, operation_id=operation_id,
            payload={"case_id": case_id, "revision": case.material_revision},
            budget={"model_requests": settings.resume_model_request_budget})
        await operations.transition(operation_id, "running")
        upload_dir = settings.upload_root / "preparation" / case_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{operation_id}{suffix}"
        file_path.write_bytes(content)
        try:
            with model_trace(operation_id=operation_id, enforce_budget=True):
                parsed = await asyncio.wait_for(parse_resume_with_metadata(file_path), settings.resume_parse_timeout_seconds)
        except TimeoutError as exc:
            await operations.transition(operation_id, "failed_retryable", error="Resume parsing timed out")
            raise HTTPException(504, "Resume parsing timed out") from exc
        except RuntimeError as exc:
            await operations.transition(operation_id, "failed", error="Resume parsing request budget exhausted" if str(exc).startswith("operation budget exhausted:") else "Resume parsing failed")
            if str(exc).startswith("operation budget exhausted:"):
                raise HTTPException(429, "Resume parsing request budget exhausted") from exc
            raise
        except Exception:
            await operations.transition(operation_id, "failed", error="Resume parsing failed")
            raise
        if not parsed.normalized_text.strip():
            await operations.transition(operation_id, "failed", error="No readable resume text")
            raise HTTPException(422, "No readable resume text was extracted; please provide readable text")
        updated = await service.update(case_id, current_user.id, PreparationUpdate(resume_text=parsed.normalized_text))
        updated.resume_parse_warnings = list(parsed.warnings)
        await save_preparation_case(updated)
        await operations.transition(operation_id, "succeeded")
        return updated
    finally:
        limits.release(current_user.id, operation_id)


@router.post("/cases/{case_id}/analysis", status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(
    case_id: str,
    request: StartOperationRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    request = request.model_copy(update={"operation_id": request.operation_id or uuid4().hex})
    async with limits.admission(current_user.id, f"case-analysis:{case_id}"):
        return await _start_analysis(case_id, request, current_user)


async def _start_analysis(case_id: str, request: StartOperationRequest, current_user: UserInDB):
    service = get_preparation_service()
    case = await service.get(case_id, current_user.id)
    if not case:
        raise HTTPException(status_code=404, detail="Preparation case not found")
    operations = get_operation_store()
    current = await operations.latest_preparation(case_id, current_user.id, case.material_revision)
    if current:
        return {"operation_id": current.operation_id, "status": current.status,
                "status_url": f"/api/preparation/operations/{current.operation_id}"}
    operation_id = request.operation_id or uuid4().hex
    existing = await operations.get(operation_id, current_user.id)
    reserved = False
    try:
        if not existing or existing.status == "accepted" and operation_id not in _analysis_tasks:
            limits.reserve(current_user.id, operation_id)
            reserved = True
            await limits.enforce("preparation:analysis", current_user.id, settings.rate_limit_start_per_minute, 60)
        operation = await operations.accept(
            user_id=current_user.id,
            kind="prepare_case",
            case_id=case_id,
            input_revision=case.material_revision,
            operation_id=operation_id,
            payload={
                "case_id": case_id,
                "material_revision": case.material_revision,
                "jd_text": case.jd_text,
                "resume_text": case.resume_text,
                "project_notes": case.project_notes,
            },
            budget={
                "decisions": settings.assessment_max_decisions,
                "model_requests": settings.assessment_model_request_budget,
                "searches": settings.assessment_search_budget,
                "extracts": settings.assessment_extract_budget,
                "semantic_checks": settings.assessment_semantic_budget,
            },
        )
    except ValueError as exc:
        if reserved:
            limits.release(current_user.id, operation_id)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except BaseException:
        if reserved:
            limits.release(current_user.id, operation_id)
        raise
    if operation.status == "accepted":
        _schedule_analysis(operation.operation_id, case_id, current_user.id)
    return {
        "operation_id": operation.operation_id,
        "status": operation.status,
        "status_url": f"/api/preparation/operations/{operation.operation_id}",
    }


@router.post("/cases/{case_id}/clarifications/{clarification_id}")
async def answer_clarification(
    case_id: str,
    clarification_id: str,
    request: ClarificationAnswer,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    try:
        async with limits.admission(current_user.id, f"clarification:{case_id}"):
            return await get_preparation_service().answer_clarification(
                case_id, clarification_id, current_user.id, request.answer
            )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Case or clarification not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


async def _repractice_source(session_id: str, user_id: str):
    manager = get_session_manager()
    session = await manager.ensure_session_loaded(session_id)
    if not session or session.user_id != user_id:
        raise HTTPException(404, "Interview session not found")
    if session.status != "completed" or not session.preparation_case_id:
        raise HTTPException(409, "Repractice requires a completed interview linked to a preparation case")
    source = await get_preparation_service().get(session.preparation_case_id, user_id)
    if not source:
        raise HTTPException(404, "Source preparation case not found")
    revision = manager.preparation_revision(session_id)
    if revision is None or revision != source.material_revision:
        raise HTTPException(409, "Source materials changed since interview; original revision unavailable")
    state = await manager.get_last_state(session_id)
    items = [FinalQuestionAssessment.model_validate(item) for item in
             (state.get("final_question_assessments") or {}).values()]
    valid = []
    memory = get_memory_service()
    for item in items:
        observation = await memory._aget_memory_item(item.observation_id)
        if observation and (observation.user_id != user_id or
                            observation.structured.get("validity", "valid") != "valid"):
            continue
        if item.scorable and item.score is not None:
            valid.append(item)
    summary = aggregate_scores(valid, source.skill_matrix)
    target_skills = {canonical_skill_id(target.scope) for target in source.targets}
    weak = [{"skill_name": name, "score": score} for name, score in summary.skill_scores.items()
            if score < 8 and canonical_skill_id(name) in target_skills]
    return source, revision, valid, weak


@router.get("/sessions/{session_id}/repractice-options")
async def repractice_options(session_id: str, current_user: Annotated[UserInDB, Depends(get_current_user)]):
    source, revision, _, weak = await _repractice_source(session_id, current_user.id)
    return {"source_session_id": session_id, "case_id": source.case_id,
            "material_revision": revision, "weak_skills": weak}


@router.post("/cases/{case_id}/repractice", status_code=status.HTTP_201_CREATED)
async def create_repractice(
    case_id: str,
    request: RetrainingRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    try:
        async with limits.admission(current_user.id, f"repractice:{request.source_session_id}"):
            source, revision, valid, weak = await _repractice_source(request.source_session_id, current_user.id)
            if source.case_id != case_id:
                raise HTTPException(409, "Session does not belong to this preparation case")
            selected = {canonical_skill_id(name) for name in request.selected_skills}
            if not selected <= {canonical_skill_id(row["skill_name"]) for row in weak}:
                raise HTTPException(422, "Select only skills with valid final scores below 8")
            return await get_preparation_service().create_repractice(
                case_id, current_user.id, request, source_revision=revision,
                source_question_ids=[item.question_id for item in valid if canonical_skill_id(item.primary_skill) in selected],
            )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Preparation case not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


class CreateInterviewRequest(BaseModel):
    mode: str = "practice"
    max_follow_ups: int = Field(default=2, ge=0, le=5)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


@router.post("/cases/{case_id}/interviews", status_code=status.HTTP_201_CREATED)
async def create_interview_from_case(
    case_id: str,
    request: CreateInterviewRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    async with limits.admission(current_user.id, f"case-interview:{case_id}"):
        return await _create_interview_from_case(case_id, request, current_user)


async def _create_interview_from_case(case_id: str, request: CreateInterviewRequest, current_user: UserInDB):
    await limits.enforce("interview:start", current_user.id, settings.rate_limit_start_per_minute, 60)
    case = await get_preparation_service().get(case_id, current_user.id)
    if not case:
        raise HTTPException(status_code=404, detail="Preparation case not found")
    pending_clarifications = [
        item for item in case.clarifications if item.status == "pending"
    ]
    if case.status != "ready" or pending_clarifications or not case.targets:
        raise HTTPException(
            status_code=409,
            detail="Preparation is not ready; complete pending clarification and evidence assessment first",
        )
    sufficient_target_ids = {
        target_id
        for target_id, assessment in (case.evidence_assessments or {}).items()
        if assessment.verdict == "sufficient"
    }
    if not sufficient_target_ids:
        raise HTTPException(
            status_code=409,
            detail="Preparation evidence assessment has no question-eligible targets",
        )
    if request.mode not in {"practice", "professional"}:
        raise HTTPException(status_code=422, detail="mode must be practice or professional")
    session_id = (uuid5(NAMESPACE_URL, f"mockview:interview:{current_user.id}:{case_id}:{case.material_revision}:{request.request_id}").hex
                  if request.request_id else uuid4().hex)
    manager = get_session_manager()
    if request.request_id:
        existing = await manager.ensure_session_loaded(session_id)
        if existing:
            if existing.max_follow_ups != request.max_follow_ups or manager.get_session_mode(session_id) != request.mode:
                raise HTTPException(409, "Interview request_id already used with different settings")
            await manager.persist_session(session_id)
            return {"session_id": session_id, "case_id": case_id, "mode": request.mode,
                    "websocket_url": f"/api/ws/interview/{session_id}"}
    session = manager.create_session(
        session_id=session_id,
        user_id=current_user.id,
        jd_text=case.jd_text,
        resume_text=case.resume_text,
        mode=request.mode,  # type: ignore[arg-type]
        max_follow_ups=request.max_follow_ups,
        preparation_case_id=case.case_id,
        assessment_targets=case.targets,
        evidence_chunks=case.evidence_chunks,
        evidence_assessments=case.evidence_assessments,
        skill_matrix=case.skill_matrix,
        resume_profile=case.resume_profile,
        material_revision=case.material_revision,
        analysis_strategy_version=case.analysis_strategy_version,
        knowledge_release_id=case.knowledge_release_id,
    )
    await manager.persist_session(session_id)
    return {
        "session_id": session.session_id,
        "case_id": case.case_id,
        "mode": request.mode,
        "websocket_url": f"/api/ws/interview/{session.session_id}",
    }


@router.get("/operations/{operation_id}")
async def get_operation(
    operation_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    operation = await get_operation_store().get(operation_id, current_user.id)
    if not operation:
        raise HTTPException(status_code=404, detail="Operation not found")
    return operation


@router.get("/operations/{operation_id}/events")
async def get_operation_events(
    operation_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    events = await get_operation_store().events(operation_id, current_user.id)
    if events is None:
        raise HTTPException(status_code=404, detail="Operation not found")
    return {"operation_id": operation_id, "events": events}


@router.post("/operations/{operation_id}/continue", status_code=status.HTTP_202_ACCEPTED)
async def continue_operation(
    operation_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    async with limits.admission(current_user.id, operation_id):
        return await _continue_operation(operation_id, current_user)


async def _continue_operation(operation_id: str, current_user: UserInDB):
    operations = get_operation_store()
    operation = await operations.get(operation_id, current_user.id)
    if not operation:
        raise HTTPException(status_code=404, detail="Operation not found")
    if operation.kind != "prepare_case":
        raise HTTPException(status_code=409, detail="Only preparation analysis can be continued here")
    if operation.status not in {"failed_retryable", "waiting_input"}:
        return operation
    case = await get_preparation_service().get(operation.case_id or "", current_user.id)
    if not case:
        raise HTTPException(status_code=404, detail="Preparation case not found")
    if case.material_revision != operation.input_revision:
        raise HTTPException(status_code=409, detail="Materials changed; start a new operation")
    limits.reserve(current_user.id, operation_id)
    try:
        await limits.enforce("preparation:analysis", current_user.id, settings.rate_limit_start_per_minute, 60)
        await operations.transition(operation_id, "accepted", error=None)
        _schedule_analysis(operation_id, case.case_id, current_user.id)
    except BaseException:
        limits.release(current_user.id, operation_id)
        raise
    return await operations.get(operation_id, current_user.id)


@router.post("/operations/{operation_id}/cancel")
async def cancel_operation(
    operation_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    operation = await get_operation_store().get(operation_id, current_user.id)
    if not operation:
        raise HTTPException(status_code=404, detail="Operation not found")
    if operation.status in {"succeeded", "failed", "cancelled"}:
        return operation
    cancelled = await get_operation_store().transition(operation_id, "cancelled")
    if operation.case_id:
        await _set_case_status(
            operation.case_id,
            current_user.id,
            "cancelled",
            operation.input_revision,
        )
    task = _analysis_tasks.get(operation_id)
    if task and not task.done():
        task.cancel()
    return cancelled
