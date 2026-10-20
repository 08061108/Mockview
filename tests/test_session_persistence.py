from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.interview import ChatMessage, InterviewSession
from app.models.jd import SkillItem, SkillMatrix
from app.services.session_manager import SessionManager
from app.services.session_repository import PersistedSession


def _pausing_persisted_session(*, committed_answer: bool = False) -> PersistedSession:
    answer_id = "answer-restart-1"
    history = [
        ChatMessage(
            message_id="question-1",
            role="interviewer",
            content="请说明如何保证接口幂等。",
            turn_id="turn-1",
            question_id="question-1",
            kind="main",
        )
    ]
    if committed_answer:
        history.append(
            ChatMessage(
                message_id="candidate-1",
                role="candidate",
                content="使用稳定请求标识。",
                turn_id="turn-1",
                question_id="question-1",
                answer_id=answer_id,
            )
        )
    session = InterviewSession(
        session_id="restart-pausing",
        user_id="student-restart",
        jd_text="Python FastAPI backend engineer position",
        status="pausing",
    )
    return PersistedSession(
        session=session,
        mode="practice",
        graph_started=True,
        pending_answers={
            answer_id: {
                "answer_id": answer_id,
                "turn_id": "turn-1",
                "content": "使用稳定请求标识。",
                "status": "accepted",
            }
        },
        operation_state={"pause_requested": True},
        last_state={"conversation_history": history},
    )


class FakeAsyncGraph:
    def __init__(self):
        self.updated = False

    async def aupdate_state(self, config, values):
        self.updated = True
        assert config["configurable"]["thread_id"] == "async-session"
        assert values == {"current_candidate_answer": "answer text"}

    def update_state(self, *_args, **_kwargs):
        raise AssertionError("sync update_state must not be used with async graph")

    async def ainvoke(self, *_args, **_kwargs):
        return {
            "interview_complete": False,
            "current_question_index": 0,
            "follow_up_count": 0,
            "conversation_history": [],
            "assessments": [],
        }


@pytest.mark.asyncio
async def test_session_manager_restores_persisted_metadata(monkeypatch):
    session = InterviewSession(
        session_id="persisted-session",
        user_id="student-1",
        jd_text="Python FastAPI LangGraph developer position",
        status="interviewing",
        current_question_index=1,
        follow_up_count=0,
        max_follow_ups=2,
        conversation_history=[
            ChatMessage(role="interviewer", content="Please introduce your project.")
        ],
    )

    async def fake_load_session_record(session_id: str):
        assert session_id == session.session_id
        return PersistedSession(
            session=session,
            mode="professional",
            resume_text="Candidate resume text",
            resume_parse_result=None,
            graph_started=True,
            last_state={"current_question_index": 1},
            persisted_assessment_count=1,
            final_memory_saved=True,
        )

    monkeypatch.setattr(
        "app.services.session_manager.load_session_record",
        fake_load_session_record,
    )

    manager = SessionManager()

    restored = await manager.ensure_session_loaded(session.session_id)

    assert restored == session
    assert manager.get_session(session.session_id) == session
    assert manager.get_session_mode(session.session_id) == "professional"
    assert manager.has_graph_started(session.session_id) is True
    assert (await manager.get_last_state(session.session_id))["current_question_index"] == 1

    data = manager._sessions[session.session_id]
    assert data.persisted_assessment_count == 1
    assert data.final_memory_saved is True


@pytest.mark.asyncio
async def test_session_manager_public_persist_delegates_to_repository(monkeypatch):
    captured = {}

    async def fake_save_session_record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "app.services.session_manager.save_session_record",
        fake_save_session_record,
    )

    manager = SessionManager()
    matrix = SkillMatrix(
        position_title="Backend Engineer",
        experience_level="mid",
        skills=[
            SkillItem(
                name="FastAPI",
                category="framework",
                weight=1.0,
                is_required=True,
            )
        ],
    )
    session = manager.create_session(
        session_id="new-session",
        user_id="student-2",
        jd_text="Python FastAPI LangGraph developer position",
        mode="practice",
        skill_matrix=matrix,
        material_revision=3,
        analysis_strategy_version="material-analysis-v1",
        knowledge_release_id="release-3",
    )

    await manager.persist_session(session.session_id)

    assert captured["session"] == session
    assert captured["mode"] == "practice"
    assert captured["resume_text"] == ""
    assert captured["graph_started"] is False
    assert captured["persisted_assessment_count"] == 0
    assert captured["final_memory_saved"] is False
    assert captured["analysis_snapshot"]["skill_matrix"] == matrix
    assert captured["analysis_snapshot"]["analysis_strategy_version"] == "material-analysis-v1"
    assert captured["material_revision"] == 3
    assert captured["knowledge_release_id"] == "release-3"


@pytest.mark.asyncio
async def test_session_manager_submit_answer_uses_async_update_state(monkeypatch):
    async def fake_save_session_record(**_kwargs):
        return None

    monkeypatch.setattr(
        "app.services.session_manager.save_session_record",
        fake_save_session_record,
    )

    manager = SessionManager()
    session = manager.create_session(
        session_id="async-session",
        user_id="student-3",
        jd_text="Python FastAPI LangGraph developer position",
        mode="practice",
    )
    data = manager._sessions[session.session_id]
    data.graph_started = True
    graph = FakeAsyncGraph()
    manager._practice_graph = graph

    result = await manager.submit_answer(session.session_id, "answer text")

    assert graph.updated is True
    assert result["interview_complete"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("completion_source", ["operation", "checkpoint"])
async def test_restart_finishes_pausing_when_answer_is_durably_committed(
    monkeypatch, completion_source
):
    persisted = _pausing_persisted_session(
        committed_answer=completion_source == "checkpoint"
    )

    async def fake_load_session_record(_session_id):
        return persisted

    operation = (
        SimpleNamespace(status="succeeded")
        if completion_source == "operation"
        else None
    )
    operation_store = SimpleNamespace(get=AsyncMock(return_value=operation))
    monkeypatch.setattr(
        "app.services.session_manager.load_session_record", fake_load_session_record
    )
    monkeypatch.setattr(
        "app.services.session_manager.get_operation_store", lambda: operation_store
    )
    manager = SessionManager.__new__(SessionManager)
    manager._sessions = {}
    manager._persist_session = AsyncMock()

    restored = await manager.ensure_session_loaded("restart-pausing")

    assert restored is not None
    assert restored.status == "paused"
    data = manager._sessions["restart-pausing"]
    assert data.pending_answers["answer-restart-1"]["status"] == "succeeded"
    assert data.pause_requested is False
    assert data.paused_at is not None
    manager._persist_session.assert_awaited_once_with("restart-pausing")


@pytest.mark.asyncio
async def test_restart_keeps_uncommitted_answer_for_explicit_retry(monkeypatch):
    persisted = _pausing_persisted_session()

    async def fake_load_session_record(_session_id):
        return persisted

    operation_store = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status="running"))
    )
    monkeypatch.setattr(
        "app.services.session_manager.load_session_record", fake_load_session_record
    )
    monkeypatch.setattr(
        "app.services.session_manager.get_operation_store", lambda: operation_store
    )
    manager = SessionManager.__new__(SessionManager)
    manager._sessions = {}
    manager._persist_session = AsyncMock()

    restored = await manager.ensure_session_loaded("restart-pausing")

    assert restored is not None
    assert restored.status == "paused"
    data = manager._sessions["restart-pausing"]
    pending = data.pending_answers["answer-restart-1"]
    assert pending["status"] == "failed_retryable"
    assert pending["content"] == "使用稳定请求标识。"
    assert manager.get_public_overview("restart-pausing").pending_action == "answer_retry_required"
