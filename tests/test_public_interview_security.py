from __future__ import annotations

from datetime import datetime, timezone

import pytest
from unittest.mock import AsyncMock

from app.errors import PublicAppError, classify_public_error
from app.models.interview import ChatMessage, InterviewSession
from app.models.public import PublicSessionOverview
from app.services.public_interview import (
    assert_no_forbidden_public_fields,
    build_public_report,
    build_public_transcript,
)
from app.services.session_manager import SessionManager, _SessionData


def _overview(*, completed: bool = True, has_report: bool = True):
    return PublicSessionOverview(
        session_id="s1", title="后端工程师", mode="practice",
        status="completed" if completed else "interviewing",
        graph_started=True, has_report=has_report,
    )


def test_public_report_is_allow_listed_and_keeps_only_report_authored_learning_answer():
    raw = {
        "overall_score": 7.5,
        "grade": "B",
        "overall_assessment": "整体表现稳定。",
        "skill_scores": [],
        "strengths": [],
        "improvements": [],
        "missed_knowledge": [{
            "question": "如何保证幂等？", "score": 6,
            "missed_points": ["请求标识"],
            "reference_answer": "使用稳定请求标识和持久化操作状态。",
            "reference_points": ["内部评分点"],
            "evidence_chunk_ids": ["secret"],
        }],
        "question_plan": [{"rubric": "secret"}],
    }
    public = build_public_report(raw, completed=True)
    assert public is not None
    assert public.missed_knowledge[0].reference_answer.startswith("使用稳定请求标识")
    dumped = public.model_dump(mode="json")
    assert_no_forbidden_public_fields(dumped)
    assert "question_plan" not in dumped


def test_transcript_merges_pending_answer_after_turn_and_hides_reviews_until_complete():
    session = InterviewSession(
        session_id="s1", jd_text="x" * 20, status="interviewing",
        conversation_history=[],
    )
    state = {"conversation_history": [ChatMessage(
        message_id="q1", role="interviewer", content="问题一",
        question_id="question-1", turn_id="turn-1", kind="main",
        created_at=datetime.now(timezone.utc),
    )], "final_question_assessments": {"question-1": {
        "observation_id": "o1", "question_id": "question-1", "display_id": 1,
        "primary_skill": "Python", "score": 8,
        "covered_points": ["internal"], "rubric_version": "private",
    }}}
    transcript = build_public_transcript(
        session=session, overview=_overview(completed=False, has_report=False),
        state=state,
        pending_answers={"a1": {"answer_id": "a1", "turn_id": "turn-1", "content": "我的回答", "status": "failed_retryable"}},
        report=None,
    )
    assert [item.role for item in transcript.items] == ["interviewer", "candidate"]
    assert transcript.items[1].processing_status == "failed_retryable"
    assert transcript.question_reviews == []
    assert_no_forbidden_public_fields(transcript)


@pytest.mark.parametrize(
    ("exc", "code", "status"),
    [
        (type("Provider404", (Exception,), {"status_code": 404})("model not found"), "MODEL_NOT_FOUND", 503),
        (type("Provider401", (Exception,), {"status_code": 401})("bad key"), "MODEL_AUTH_FAILED", 503),
        (type("Provider429", (Exception,), {"status_code": 429, "headers": {"Retry-After": "9"}})("rate limit"), "MODEL_RATE_LIMITED", 429),
        (TimeoutError("late"), "MODEL_TIMEOUT", 504),
        (type("Provider500", (Exception,), {"status_code": 500})("upstream"), "MODEL_UNAVAILABLE", 502),
        (type("APIConnectionError", (Exception,), {})("network"), "MODEL_UNAVAILABLE", 502),
    ],
)
def test_model_error_classifier_never_exposes_provider_body(exc, code, status):
    public = classify_public_error(exc)
    assert public.code == code
    assert public.status_code == status
    assert "bad key" not in public.message
    assert "upstream" not in public.message


def test_public_app_error_serializes_stable_shape():
    error = PublicAppError("EMPTY_DOCUMENT", "文件为空。", 422)
    assert error.public().model_dump() == {
        "code": "EMPTY_DOCUMENT", "message": "文件为空。", "retryable": False,
        "retry_after_seconds": None,
    }


@pytest.mark.asyncio
async def test_pause_is_immediate_while_waiting_and_resume_is_explicit():
    manager = SessionManager.__new__(SessionManager)
    data = _SessionData(session=InterviewSession(
        session_id="pause-1", jd_text="x" * 20, status="interviewing"
    ), graph_started=True)
    manager._sessions = {"pause-1": data}
    manager._persist_session = AsyncMock()
    manager.get_last_state = AsyncMock(return_value={})

    paused = await manager.request_pause("pause-1")
    assert paused.session.status == "paused"
    assert data.paused_at is not None

    resumed = await manager.resume_interview("pause-1")
    assert resumed.session.status == "interviewing"
    assert data.paused_at is None


@pytest.mark.asyncio
async def test_pause_during_operation_waits_and_end_wins():
    import asyncio

    manager = SessionManager.__new__(SessionManager)
    data = _SessionData(session=InterviewSession(
        session_id="pause-2", jd_text="x" * 20, status="evaluating"
    ), graph_started=True)
    manager._sessions = {"pause-2": data}
    manager._persist_session = AsyncMock()
    manager.get_last_state = AsyncMock(return_value={})
    gate = asyncio.Event()

    async def active():
        await gate.wait()
        return {}

    data.answer_task = asyncio.create_task(active())
    pausing = await manager.request_pause("pause-2")
    assert pausing.session.status == "pausing"
    data.end_requested = True
    data.pause_requested = False
    gate.set()
    await data.answer_task
    await asyncio.sleep(0)
    assert data.session.status != "paused"


@pytest.mark.asyncio
async def test_scanned_pdf_is_not_disguised_as_png(monkeypatch, tmp_path):
    from app.resume import parser

    async def no_text(_path):
        raise ValueError("no extractable text")

    vision = AsyncMock()
    monkeypatch.setattr(parser, "parse_pdf", no_text)
    monkeypatch.setattr(parser, "parse_image_with_vision", vision)
    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"%PDF-test")

    with pytest.raises(PublicAppError) as caught:
        await parser.parse_resume_document(path)
    assert caught.value.code == "EMPTY_DOCUMENT"
    assert "PNG" in caught.value.message
    vision.assert_not_awaited()
