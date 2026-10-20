"""REST endpoints for interview operations."""

from __future__ import annotations

import uuid
import inspect
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.cache.rate_limiter import check_rate_limit
from app.cache.session_cache import save_session_meta
from app.config import settings
from app.errors import PublicAppError, classify_public_error, document_error, public_http_exception
from app.models.interview import ChatMessage
from app.models.public import PublicSessionOverview, PublicSessionState
from app.models.user import UserInDB
from app.security import get_current_user
from app.services.report_exporter import markdown_to_html, report_to_markdown
from app.services.session_manager import get_session_manager
from app.services.model_capabilities import get_model_capabilities
from app.services.public_interview import build_public_report, build_public_transcript, current_public_turn
from app.utils.file_parser import parse_resume_with_metadata

router = APIRouter()


class StartInterviewRequest(BaseModel):
    jd_text: str = Field(..., min_length=10, description="Job description text")
    max_follow_ups: int = Field(default=2, ge=0, le=5)
    mode: str = Field(default="practice", description="Interview mode: practice or professional")


class StartInterviewResponse(BaseModel):
    session_id: str
    message: str
    websocket_url: str
    mode: str


class SubmitAnswerRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Candidate answer text")
    answer_id: str | None = None
    turn_id: str | None = None


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _last_interviewer_message(state: dict) -> str | None:
    for msg in reversed(state.get("conversation_history", []) or []):
        if isinstance(msg, ChatMessage) and msg.role == "interviewer":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "interviewer":
            return msg.get("content", "")
    return None


def _question_meta(state: dict) -> tuple[int, int, list[str], str]:
    plan = state.get("question_plan", []) or []
    idx = int(state.get("current_question_index", 0) or 0)
    blueprint = state.get("question_blueprint")
    total = len(plan)
    if blueprint:
        total = int(
            getattr(blueprint, "planned_count", 0)
            or (blueprint.get("planned_count", 0) if isinstance(blueprint, dict) else 0)
            or len(plan)
        )
    if idx < len(plan):
        question = plan[idx]
        skill_tags = getattr(question, "skill_tags", None)
        difficulty = getattr(question, "difficulty", None)
        if isinstance(question, dict):
            skill_tags = question.get("skill_tags", skill_tags)
            difficulty = question.get("difficulty", difficulty)
        return idx, total, skill_tags or [], difficulty or ""
    return idx, total, [], ""


def _current_turn_payload(state: dict) -> dict | None:
    message = _last_interviewer_message(state)
    if not message:
        return None
    idx, total, tags, difficulty = _question_meta(state)
    follow_up_count = int(state.get("follow_up_count", 0) or 0)
    payload = {
        "question_index": idx + 1,
        "content": message,
        "kind": "follow_up" if follow_up_count > 0 else "question",
    }
    if follow_up_count > 0:
        payload["follow_up_number"] = follow_up_count
    else:
        payload.update(
            {
                "total_questions": total,
                "skill_tags": tags,
                "difficulty": difficulty,
            }
        )
    return payload


def _current_plan_coverage(state: dict) -> dict:
    coverage_by_stage = state.get("question_plan_coverage", {}) or {}
    if state.get("interview_mode") == "professional":
        stage = (
            "professional:round2"
            if "professional:round2" in coverage_by_stage
            else "professional:round1"
        )
    else:
        stage = "practice"
    raw = coverage_by_stage.get(stage)
    coverage = _dump(raw) if raw else {}
    return {
        "question_plan_status": coverage.get("status", "not_applicable"),
        "covered_target_ids": coverage.get("covered_target_ids", []),
        "uncovered_target_ids": coverage.get("uncovered_target_ids", []),
        "evidence_eligible_target_ids": coverage.get(
            "evidence_eligible_target_ids", []
        ),
        "evidence_unresolved_target_ids": coverage.get(
            "evidence_unresolved_target_ids", []
        ),
    }


def _rate_limit_identity(
    request: Request,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str:
    host = request.client.host if request.client else "unknown"
    if session_id:
        return f"session:{session_id}"
    if user_id:
        return f"user:{user_id}"
    return f"ip:{host}"


def _assert_session_owner(session, current_user: UserInDB) -> None:
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session access denied")


async def _enforce_rate_limit(
    name: str,
    identity: str,
    limit: int,
    window_seconds: int,
) -> None:
    result = await check_rate_limit(name, identity, limit, window_seconds)
    if result.allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Too many requests. Please retry after {result.retry_after_seconds} seconds.",
        headers={"Retry-After": str(result.retry_after_seconds)},
    )


def _report_from_state_or_manager(session_id: str, state: dict) -> Any:
    mgr = get_session_manager()
    return (
        state.get("practice_report")
        or state.get("professional_report")
        or state.get("final_report")
        or mgr.get_report_for_session(session_id)
    )


def _public_overview(mgr: Any, session_id: str, state: dict | None = None) -> PublicSessionOverview:
    if hasattr(mgr, "get_public_overview"):
        return mgr.get_public_overview(session_id)
    session = mgr.get_session(session_id)
    state = state or {}
    report = _report_from_state_or_manager(session_id, state)
    return PublicSessionOverview(
        session_id=session.session_id,
        title="面试会话",
        mode=mgr.get_session_mode(session_id),
        status=session.status,
        graph_started=mgr.has_graph_started(session_id),
        has_report=bool(report),
        question_count=_question_meta(state)[1],
        assessment_count=len(state.get("assessments", []) or []),
        current_question_index=session.current_question_index,
        max_follow_ups=session.max_follow_ups,
        jd_preview=" ".join(session.jd_text.split())[:180],
    )


def _public_report(mgr: Any, session_id: str):
    if hasattr(mgr, "get_public_report"):
        return mgr.get_public_report(session_id)
    session = mgr.get_session(session_id)
    return build_public_report(
        mgr.get_report_for_session(session_id), completed=session.status == "completed"
    )


async def _interview_payload(session_id: str, state: dict) -> dict:
    mgr = get_session_manager()
    if hasattr(mgr, "get_public_state"):
        public_state = await mgr.get_public_state(session_id)
    else:
        overview = _public_overview(mgr, session_id, state)
        public_state = PublicSessionState(
            session=overview, graph_started=overview.graph_started,
            current_turn=current_public_turn(state), has_report=overview.has_report,
            interview_complete=overview.status == "completed",
        )
    report = _public_report(mgr, session_id)
    return {
        "session": public_state.session,
        "state": public_state,
        "next": public_state.current_turn,
        "report": report,
    }


async def _store_and_parse_resume(session_id: str, upload: UploadFile):
    if not upload.filename:
        raise document_error("INVALID_DOCUMENT", "请选择要上传的简历文件。", 422)
    safe_name = Path(upload.filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise document_error("UNSUPPORTED_FILE_TYPE", "不支持该文件类型，请上传 PDF、PNG、JPG 或 JPEG。", 415)
    if suffix in {".png", ".jpg", ".jpeg"} and get_model_capabilities()["vision"] == "unavailable":
        raise PublicAppError("MODEL_NOT_FOUND", "配置的图片识别模型不可用，请联系管理员检查模型配置。", 503)
    upload_dir = settings.upload_root / "resumes" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name
    content = await upload.read()
    if not content:
        raise document_error("EMPTY_DOCUMENT", "上传的简历文件为空。", 422)
    file_path.write_bytes(content)
    try:
        return await parse_resume_with_metadata(file_path)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise classify_public_error(exc) from exc


@router.post("/start", response_model=StartInterviewResponse)
async def start_interview(
    req: StartInterviewRequest,
    request: Request,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Create a new interview session. Returns a session_id for WebSocket connection."""
    await _enforce_rate_limit(
        "interview:start",
        _rate_limit_identity(request, user_id=current_user.id),
        settings.rate_limit_start_per_minute,
        60,
    )
    mgr = get_session_manager()
    session_id = str(uuid.uuid4())
    mode = req.mode if req.mode in ("practice", "professional") else "practice"
    session = mgr.create_session(
        session_id=session_id,
        jd_text=req.jd_text,
        max_follow_ups=req.max_follow_ups,
        mode=mode,
        user_id=current_user.id,
    )
    await mgr.persist_session(session_id)
    await save_session_meta(session, mode)
    return StartInterviewResponse(
        session_id=session_id,
        message="Session created. Connect via WebSocket to begin the interview.",
        websocket_url=f"/api/ws/interview/{session_id}",
        mode=mode,
    )


@router.post("/start-with-resume")
async def start_interview_with_resume(
    request: Request,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    jd_text: str = Form(..., min_length=10),
    resume_file: UploadFile = File(...),
    max_follow_ups: int = Form(default=2, ge=0, le=5),
    mode: str = Form(default="professional"),
):
    """Create a session and attach a parsed resume in one multipart request."""
    identity = _rate_limit_identity(request, user_id=current_user.id)
    await _enforce_rate_limit(
        "interview:start",
        identity,
        settings.rate_limit_start_per_minute,
        60,
    )
    await _enforce_rate_limit(
        "interview:resume",
        identity,
        settings.rate_limit_resume_per_hour,
        3600,
    )
    mgr = get_session_manager()
    session_id = str(uuid.uuid4())
    normalized_mode = mode if mode in ("practice", "professional") else "professional"
    parse_result = await _store_and_parse_resume(session_id, resume_file)
    session = mgr.create_session(
        session_id=session_id,
        jd_text=jd_text,
        max_follow_ups=max_follow_ups,
        mode=normalized_mode,
        resume_text=parse_result.normalized_text,
        resume_parse_result=parse_result,
        user_id=current_user.id,
    )
    await mgr.persist_session(session_id)
    await save_session_meta(session, normalized_mode)
    return {
        "session_id": session_id,
        "message": "Session created with parsed resume.",
        "websocket_url": f"/api/ws/interview/{session_id}",
        "mode": normalized_mode,
        "resume": _dump(parse_result),
    }


@router.get("/sessions")
async def list_sessions(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    limit: int = 50,
    offset: int = 0,
):
    """List interview sessions owned by the current user."""
    normalized_limit = max(1, min(limit, 100))
    normalized_offset = max(0, offset)
    mgr = get_session_manager()
    sessions = await mgr.list_user_sessions(
        current_user.id,
        limit=normalized_limit,
        offset=normalized_offset,
    )
    return {
        "items": sessions,
        "limit": normalized_limit,
        "offset": normalized_offset,
    }


@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Get the current state of an interview session."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    return _public_overview(mgr, session_id)


@router.post("/session/{session_id}/resume")
async def upload_resume(
    session_id: str,
    request: Request,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    resume_file: UploadFile | None = File(default=None),
):
    """Resume a paused interview, or attach a resume before graph start."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    if resume_file is None:
        try:
            return await mgr.resume_interview(session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail={
                "code": "INTERVIEW_CONFLICT", "message": str(exc),
                "retryable": False, "retry_after_seconds": None,
            }) from exc
    await _enforce_rate_limit(
        "interview:resume",
        _rate_limit_identity(request, user_id=session.user_id, session_id=session_id),
        settings.rate_limit_resume_per_hour,
        3600,
    )
    try:
        parse_result = await _store_and_parse_resume(session_id, resume_file)
        mgr.set_resume(
            session_id=session_id,
            resume_text=parse_result.normalized_text,
            resume_parse_result=parse_result,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "INTERVIEW_CONFLICT", "message": str(exc), "retryable": False, "retry_after_seconds": None},
        ) from exc
    except Exception as exc:
        raise public_http_exception(exc) from exc
    return {
        "session_id": session_id,
        "resume": _dump(parse_result),
        "message": "Resume parsed and attached to session.",
    }


@router.post("/session/{session_id}/pause")
async def pause_session(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    try:
        return await mgr.request_pause(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail={
            "code": "INTERVIEW_CONFLICT", "message": str(exc),
            "retryable": False, "retry_after_seconds": None,
        }) from exc


@router.post("/session/{session_id}/start")
async def start_session_graph(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Start the LangGraph interview and return the first interviewer turn."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    try:
        if hasattr(mgr, "ensure_start_task"):
            state = await mgr.ensure_start_task(session_id)
        else:
            state = await mgr.start_interview_graph(session_id)
    except Exception as exc:
        raise public_http_exception(exc) from exc
    return await _interview_payload(session_id, state)


@router.post("/session/{session_id}/answer")
async def submit_session_answer(
    session_id: str,
    req: SubmitAnswerRequest,
    request: Request,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Submit one answer through REST and return the next turn or final report."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    await _enforce_rate_limit(
        "interview:answer",
        _rate_limit_identity(request, user_id=session.user_id, session_id=session_id),
        settings.rate_limit_answer_per_minute,
        60,
    )
    answer = req.content.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    try:
        answer_id = req.answer_id or uuid.uuid4().hex
        if hasattr(mgr, "accept_answer_submission"):
            await mgr.accept_answer_submission(session_id, answer, answer_id, req.turn_id)
        if hasattr(mgr, "ensure_answer_task"):
            state = await mgr.ensure_answer_task(session_id, answer, answer_id)
        else:
            parameters = inspect.signature(mgr.submit_answer).parameters
            if "answer_id" in parameters:
                state = await mgr.submit_answer(session_id, answer, answer_id=answer_id)
            else:
                state = await mgr.submit_answer(session_id, answer)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "INTERVIEW_CONFLICT", "message": str(exc), "retryable": False, "retry_after_seconds": None}) from exc
    except Exception as exc:
        raise public_http_exception(exc) from exc
    return await _interview_payload(session_id, state)


@router.post("/session/{session_id}/stop")
async def stop_session(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Stop a running interview and generate an early report when possible."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    try:
        state = await mgr.ensure_end_task(session_id)
    except Exception as exc:
        raise public_http_exception(exc) from exc
    return await _interview_payload(session_id, state)


@router.post("/session/{session_id}/end")
async def end_session(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Preferred early-completion endpoint; /stop is a compatibility alias."""
    return await stop_session(session_id, current_user)


@router.get("/session/{session_id}/state")
async def get_session_state(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Return lightweight session metadata plus the latest LangGraph state snapshot."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    return await mgr.get_public_state(session_id)


@router.get("/session/{session_id}/transcript")
async def get_session_transcript(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    return await mgr.get_public_transcript(session_id)


@router.get("/report/{session_id}")
async def get_report(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Get the final interview report for a completed session."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Interview not yet completed")
    report = _public_report(mgr, session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/report/{session_id}/export")
async def export_report(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    format: str = "markdown",
):
    """Export a completed interview report as Markdown or printable HTML."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Interview not yet completed")
    report = _public_report(mgr, session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    mode = mgr.get_session_mode(session_id)
    if hasattr(mgr, "get_public_transcript"):
        transcript = await mgr.get_public_transcript(session_id)
    else:
        state = await mgr.get_last_state(session_id)
        transcript = build_public_transcript(
            session=session, overview=_public_overview(mgr, session_id, state),
            state=state, pending_answers={}, report=mgr.get_report_for_session(session_id),
        )
    markdown = report_to_markdown(report, mode=mode, session_id=session_id, transcript=transcript)
    normalized_format = format.lower().strip()
    filename_base = f"interview-report-{session_id[:8]}"
    if normalized_format in {"html", "pdf"}:
        html = markdown_to_html(markdown)
        return Response(
            content=html,
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Disposition": (
                    "inline" if normalized_format == "pdf" else "attachment"
                )
                + f'; filename="{filename_base}.html"'
            },
        )
    if normalized_format in {"markdown", "md"}:
        return Response(
            content=markdown,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.md"'},
        )
    raise HTTPException(status_code=400, detail="Unsupported export format")
