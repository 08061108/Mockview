import asyncio
from contextlib import contextmanager

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import interview_rest, interview_ws
from app.main import app
from app.models.interview import ChatMessage, InterviewSession
from app.models.public import PublicSessionOverview, PublicSessionState
from app.models.resume import ResumeParseMetadata, ResumeParseResult
from app.models.user import UserInDB
from app.security import create_access_token, get_current_user
from app.services.public_interview import (
    assert_no_forbidden_public_fields,
    build_public_report,
    build_public_transcript,
    current_public_turn,
)


def _fake_user(user_id="user-1") -> UserInDB:
    return UserInDB(
        id=user_id,
        username=f"{user_id}@example.test",
        password_hash="hash",
        display_name=user_id,
        role="user",
        is_active=True,
    )


def _client_with_user(user: UserInDB | None = None) -> TestClient:
    current_user = user or _fake_user()

    async def override_current_user():
        return current_user

    app.dependency_overrides[get_current_user] = override_current_user
    return TestClient(app)


def _clear_overrides():
    app.dependency_overrides.clear()


@contextmanager
def _ws_connect(client, session_id, user=None):
    with client.websocket_connect(f"/api/ws/interview/{session_id}") as ws:
        ws.send_json({"type": "auth", "token": create_access_token(user or _fake_user())})
        yield ws


async def _fake_get_user_by_id(user_id: str):
    return _fake_user(user_id)


class FakeInterviewManager:
    def __init__(self):
        self.sessions = {}
        self.graph_started = False
        self.resume_text = ""
        self.report = {
            "overall_score": 8.0,
            "grade": "B",
            "overall_assessment": "The candidate showed solid backend engineering thinking.",
        }
        self.last_state = {}

    def create_session(
        self,
        session_id,
        jd_text,
        max_follow_ups=2,
        mode="practice",
        resume_text="",
        resume_parse_result=None,
        user_id="local-user",
    ):
        session = InterviewSession(
            session_id=session_id,
            user_id=user_id,
            jd_text=jd_text,
            max_follow_ups=max_follow_ups,
        )
        self.sessions[session_id] = session
        self.mode = mode
        self.resume_text = resume_text
        self.resume_parse_result = resume_parse_result
        return session

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    async def ensure_session_loaded(self, session_id):
        return self.get_session(session_id)

    async def persist_session(self, _session_id):
        return None

    def get_session_mode(self, _session_id):
        return getattr(self, "mode", "practice")

    def has_graph_started(self, _session_id):
        return self.graph_started

    async def get_last_state(self, _session_id):
        return self.last_state

    def set_resume(self, session_id, resume_text, resume_parse_result=None):
        if self.graph_started:
            raise RuntimeError("Resume cannot be changed after the interview has started")
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        self.resume_text = resume_text
        self.resume_parse_result = resume_parse_result

    def get_report_for_session(self, _session_id):
        return self.report

    async def list_user_sessions(self, user_id, limit=50, offset=0):
        items = []
        for session in self.sessions.values():
            if session.user_id != user_id:
                continue
            items.append(
                {
                    "session_id": session.session_id,
                    "title": session.jd_text[:36],
                    "mode": getattr(self, "mode", "practice"),
                    "status": session.status,
                    "graph_started": self.graph_started,
                    "has_report": session.status == "completed",
                    "question_count": len(self.last_state.get("question_plan", []) or []),
                    "assessment_count": len(self.last_state.get("assessments", []) or []),
                    "current_question_index": session.current_question_index,
                    "max_follow_ups": session.max_follow_ups,
                    "jd_preview": session.jd_text[:180],
                    "overall_score": self.report["overall_score"] if session.status == "completed" else None,
                    "grade": self.report["grade"] if session.status == "completed" else None,
                    "error_message": None,
                    "created_at": None,
                    "updated_at": None,
                    "completed_at": None,
                }
            )
        return items[offset : offset + limit]

    async def start_interview_graph(self, session_id):
        self.graph_started = True
        self.sessions[session_id].status = "interviewing"
        self.last_state = {
            "interview_complete": False,
            "current_question_index": 0,
            "follow_up_count": 0,
            "current_round": 1,
            "question_plan": [
                {
                    "skill_tags": ["FastAPI", "LangGraph"],
                    "difficulty": "medium",
                }
            ],
            "conversation_history": [
                ChatMessage(
                    role="interviewer",
                    content="Explain how your FastAPI layer drives the Agent workflow.",
                )
            ],
            "assessments": [],
        }
        return self.last_state

    async def submit_answer(self, session_id, answer):
        self.sessions[session_id].status = "completed"
        self.last_state = {
            "interview_complete": True,
            "practice_report": self.report,
            "current_round": 1,
            "question_plan": [],
            "conversation_history": [
                ChatMessage(role="interviewer", content="Explain FastAPI."),
                ChatMessage(role="candidate", content=answer),
            ],
            "assessments": [{"question_id": 1, "score": 8}],
        }
        return self.last_state

    async def stop_interview(self, session_id):
        self.sessions[session_id].status = "completed"
        self.last_state = {
            "interview_complete": True,
            "practice_report": self.report,
            "current_round": 1,
            "question_plan": [],
            "conversation_history": [],
            "assessments": [],
        }
        return self.last_state


class PublicContractManager(FakeInterviewManager):
    """Exercise actual public DTO builders while keeping model calls out of API tests."""

    def get_public_overview(self, session_id):
        session = self.sessions[session_id]
        report = self.get_public_report(session_id)
        return PublicSessionOverview(
            session_id=session_id,
            title="后端工程师",
            mode=getattr(self, "mode", "practice"),
            status=session.status,
            graph_started=self.graph_started,
            has_report=report is not None,
            assessment_count=len(self.last_state.get("final_question_assessments", {})),
        )

    async def get_public_state(self, session_id):
        overview = self.get_public_overview(session_id)
        return PublicSessionState(
            session=overview,
            graph_started=self.graph_started,
            current_turn=current_public_turn(self.last_state),
            has_report=overview.has_report,
            interview_complete=overview.status == "completed",
        )

    def get_public_report(self, session_id):
        return build_public_report(
            self.report,
            completed=self.sessions[session_id].status == "completed",
        )

    async def get_public_transcript(self, session_id):
        session = self.sessions[session_id]
        return build_public_transcript(
            session=session,
            overview=self.get_public_overview(session_id),
            state=self.last_state,
            pending_answers={},
            report=self.report,
        )


class BackgroundResumeManager(FakeInterviewManager):
    """Simulate reconnecting while the initial question stream finishes elsewhere."""

    def __init__(self):
        super().__init__()
        self._start_task = None

    async def get_last_state(self, _session_id):
        return {} if self._start_task is None else self.last_state

    def has_active_operation(self, _session_id):
        return True

    def subscribe_events(self, _session_id):
        queue = asyncio.Queue()
        queue.put_nowait({"event": "start", "stream_id": "old-stream", "kind": "question"})
        queue.put_nowait({"event": "end", "stream_id": "old-stream", "kind": "question"})
        return queue

    def unsubscribe_events(self, _session_id, _queue):
        return None

    def ensure_start_task(self, session_id):
        if self._start_task is None:

            async def finish():
                await asyncio.sleep(0)
                await self.start_interview_graph(session_id)
                return self.last_state

            self._start_task = asyncio.create_task(finish())
        return self._start_task


class EndOnlyStreamingManager(BackgroundResumeManager):
    """Simulate a reconnect that receives only a queued stream end event."""

    def has_active_operation(self, _session_id):
        return False

    def subscribe_events(self, _session_id):
        queue = asyncio.Queue()
        queue.put_nowait(
            {
                "event": {
                    "event": "end",
                    "stream_id": "stale-stream",
                    "kind": "question",
                    "content": "stale streamed question",
                    "done": True,
                },
                "sequence": 2,
            }
        )
        return queue


def test_rest_interview_contract(monkeypatch):
    manager = FakeInterviewManager()
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)
    client = _client_with_user()

    try:
        created = client.post(
            "/api/interview/start",
            json={
                "jd_text": "Python FastAPI LangGraph AI application developer position",
                "mode": "practice",
                "max_follow_ups": 1,
            },
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        assert manager.sessions[session_id].user_id == "user-1"

        started = client.post(f"/api/interview/session/{session_id}/start")
        assert started.status_code == 200
        assert started.json()["next"]["kind"] == "question"
        assert started.json()["next"]["skill_tags"] == ["FastAPI", "LangGraph"]

        answered = client.post(
            f"/api/interview/session/{session_id}/answer",
            json={"content": "I use FastAPI endpoints to invoke the session manager."},
        )
        assert answered.status_code == 200
        assert answered.json()["state"]["interview_complete"] is True
        assert answered.json()["report"]["overall_score"] == 8.0

        report = client.get(f"/api/interview/report/{session_id}")
        assert report.status_code == 200
        assert report.json()["grade"] == "B"
    finally:
        _clear_overrides()


def test_resume_upload_contract(monkeypatch):
    manager = FakeInterviewManager()
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)

    async def fake_parse(_session_id, upload):
        return ResumeParseResult(
            raw_text="Python FastAPI LangGraph",
            normalized_text="Python FastAPI LangGraph",
            metadata=ResumeParseMetadata(
                file_name=upload.filename or "resume.pdf",
                file_type=".pdf",
                parser="fake",
                raw_char_count=24,
                normalized_char_count=24,
            ),
        )

    monkeypatch.setattr(interview_rest, "_store_and_parse_resume", fake_parse)
    client = _client_with_user()

    try:
        created = client.post(
            "/api/interview/start",
            json={"jd_text": "Python FastAPI LangGraph AI application developer position"},
        )
        session_id = created.json()["session_id"]

        uploaded = client.post(
            f"/api/interview/session/{session_id}/resume",
            files={"resume_file": ("resume.pdf", b"%PDF fake", "application/pdf")},
        )

        assert uploaded.status_code == 200
        assert manager.resume_text == "Python FastAPI LangGraph"
        assert uploaded.json()["resume"]["metadata"]["parser"] == "fake"
    finally:
        _clear_overrides()


def test_websocket_interview_contract(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="ws-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    with _ws_connect(client, session.session_id) as ws:
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json()["stage"] == "questions_ready"
        question = ws.receive_json()
        assert question["type"] == "question"
        assert question["skill_tags"] == ["FastAPI", "LangGraph"]

        ws.send_json({"type": "answer", "content": "I keep the graph state in SessionManager."})
        assert ws.receive_json()["stage"] == "processing"
        assert ws.receive_json()["type"] == "interview_end"
        report = ws.receive_json()
        assert report["type"] == "report"
        assert report["data"]["grade"] == "B"


def test_websocket_reconnect_delivers_question_after_background_stream(monkeypatch):
    class OrderedRecoveryManager(BackgroundResumeManager):
        def __init__(self):
            super().__init__()
            self.recovery_order = []

        def subscribe_events(self, session_id):
            self.recovery_order.append("subscribe")
            return super().subscribe_events(session_id)

        async def get_last_state(self, session_id):
            self.recovery_order.append("read_state")
            return await super().get_last_state(session_id)

    manager = OrderedRecoveryManager()
    session = manager.create_session(
        session_id="ws-background-resume-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    with _ws_connect(client, session.session_id) as ws:
        assert ws.receive_json()["stage"] == "resumed"
        question = ws.receive_json()
        assert question["type"] == "question"
        assert question["content"] == "Explain how your FastAPI layer drives the Agent workflow."
    assert manager.recovery_order.index("subscribe") < manager.recovery_order.index(
        "read_state"
    )


def test_websocket_drops_orphan_stream_end_and_sends_complete_question(monkeypatch):
    manager = EndOnlyStreamingManager()
    session = manager.create_session(
        session_id="ws-orphan-stream-end-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    with _ws_connect(client, session.session_id) as ws:
        assert ws.receive_json()["stage"] == "analyzing_jd"
        question = ws.receive_json()
        assert question["type"] == "question"
        assert question["content"] == "Explain how your FastAPI layer drives the Agent workflow."


def test_websocket_start_handshake_does_not_duplicate_question(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="ws-start-handshake-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    with _ws_connect(client, session.session_id) as ws:
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json()["stage"] == "questions_ready"
        assert ws.receive_json()["type"] == "question"

        ws.send_json({"type": "start_interview"})
        ws.send_json({"type": "answer", "content": "I keep the graph state in SessionManager."})
        assert ws.receive_json()["stage"] == "processing"
        assert ws.receive_json()["type"] == "interview_end"


def test_rest_rejects_cross_user_session(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="other-user-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-2",
    )
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)
    client = _client_with_user(_fake_user("user-1"))

    try:
        response = client.get(f"/api/interview/session/{session.session_id}")
        assert response.status_code == 403
    finally:
        _clear_overrides()


def test_rest_lists_only_current_user_sessions(monkeypatch):
    manager = FakeInterviewManager()
    own = manager.create_session(
        session_id="own-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    own.status = "completed"
    manager.create_session(
        session_id="other-session",
        jd_text="Java backend developer position",
        user_id="user-2",
    )
    manager.last_state = {
        "question_plan": [{"skill_tags": ["FastAPI"]}],
        "assessments": [{"question_id": 1, "score": 8}],
    }
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)
    client = _client_with_user(_fake_user("user-1"))

    try:
        response = client.get("/api/interview/sessions")
        assert response.status_code == 200
        items = response.json()["items"]
        assert [item["session_id"] for item in items] == ["own-session"]
        assert items[0]["has_report"] is True
        assert items[0]["overall_score"] == 8.0
        assert items[0]["grade"] == "B"
    finally:
        _clear_overrides()


def test_rest_exports_completed_report(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="export-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    session.status = "completed"
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)
    client = _client_with_user(_fake_user("user-1"))

    try:
        markdown = client.get(f"/api/interview/report/{session.session_id}/export")
        assert markdown.status_code == 200
        assert "text/markdown" in markdown.headers["content-type"]
        assert "AI 模拟面试报告" in markdown.text
        assert "8.0/10" in markdown.text

        html = client.get(f"/api/interview/report/{session.session_id}/export?format=pdf")
        assert html.status_code == 200
        assert "text/html" in html.headers["content-type"]
        assert "<html" in html.text
    finally:
        _clear_overrides()


def _completed_public_contract_manager():
    manager = PublicContractManager()
    session = manager.create_session(
        session_id="public-contract-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    session.status = "completed"
    manager.graph_started = True
    manager.report = {
        "overall_score": 8.0,
        "grade": "B",
        "overall_assessment": "接口设计清晰。",
        "missed_knowledge": [{
            "question": "如何保证幂等？",
            "score": 7,
            "missed_points": ["稳定请求标识"],
            "reference_answer": "保存请求标识和操作结果。",
            "reference_points": ["PRIVATE_REFERENCE_POINT"],
            "evidence_chunk_ids": ["PRIVATE_EVIDENCE_ID"],
        }],
        "question_plan": [{"rubric": "PRIVATE_RUBRIC"}],
    }
    manager.last_state = {
        "interview_complete": True,
        "question_plan": [{
            "skill_tags": ["FastAPI"],
            "difficulty": "medium",
            "reference_points": ["PRIVATE_STATE_REFERENCE"],
        }],
        "conversation_history": [
            ChatMessage(
                message_id="q-public-1",
                role="interviewer",
                content="如何保证接口幂等？",
                question_id="question-1",
                turn_id="turn-1",
                kind="main",
            ),
            ChatMessage(
                message_id="a-public-1",
                role="candidate",
                content="我会保存请求标识和操作结果。",
                question_id="question-1",
                turn_id="turn-1",
                answer_id="answer-1",
            ),
        ],
        "final_question_assessments": {
            "question-1": {
                "observation_id": "observation-1",
                "question_id": "question-1",
                "display_id": 1,
                "primary_skill": "FastAPI",
                "score": 8,
                "covered_points": ["PRIVATE_COVERED_POINT"],
                "rubric_version": "PRIVATE_RUBRIC_VERSION",
            }
        },
    }
    return manager, session


def test_public_rest_and_export_outputs_recursively_exclude_internal_fields(monkeypatch):
    manager, session = _completed_public_contract_manager()
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)
    client = _client_with_user(_fake_user("user-1"))

    try:
        for path in (
            f"/api/interview/session/{session.session_id}",
            f"/api/interview/session/{session.session_id}/state",
            f"/api/interview/session/{session.session_id}/transcript",
            f"/api/interview/report/{session.session_id}",
        ):
            response = client.get(path)
            assert response.status_code == 200
            assert_no_forbidden_public_fields(response.json())
        state = client.get(f"/api/interview/session/{session.session_id}/state").json()
        assert "state" not in state

        transcript = client.get(
            f"/api/interview/session/{session.session_id}/transcript"
        ).json()
        assert [item["role"] for item in transcript["items"]] == [
            "interviewer",
            "candidate",
        ]
        assert transcript["question_reviews"][0]["score"] == 8
        assert transcript["report"]["missed_knowledge"][0]["reference_answer"]

        for format_name in ("markdown", "html"):
            exported = client.get(
                f"/api/interview/report/{session.session_id}/export?format={format_name}"
            )
            assert exported.status_code == 200
            assert "我会保存请求标识和操作结果" in exported.text
            for private_value in (
                "PRIVATE_REFERENCE_POINT",
                "PRIVATE_EVIDENCE_ID",
                "PRIVATE_RUBRIC",
                "PRIVATE_STATE_REFERENCE",
                "PRIVATE_COVERED_POINT",
                "PRIVATE_RUBRIC_VERSION",
            ):
                assert private_value not in exported.text
    finally:
        _clear_overrides()


def test_websocket_transcript_snapshot_uses_public_projection(monkeypatch):
    manager, session = _completed_public_contract_manager()
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    with _ws_connect(client, session.session_id) as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "transcript_snapshot"
        assert_no_forbidden_public_fields(snapshot)
        assert [item["role"] for item in snapshot["data"]["items"]] == [
            "interviewer",
            "candidate",
        ]
        assert ws.receive_json()["stage"] == "resumed"
        assert ws.receive_json()["type"] == "interview_end"
        report = ws.receive_json()
        assert report["type"] == "report"
        assert_no_forbidden_public_fields(report)


def test_websocket_requires_authentication(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="ws-auth-required-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    client = TestClient(app)

    try:
        with client.websocket_connect(f"/api/ws/interview/{session.session_id}") as ws:
            ws.send_json({"type": "auth"})
            ws.receive_json()
            raise AssertionError("WebSocket should reject a missing auth token")
    except WebSocketDisconnect as exc:
        assert exc.code == 4401


def test_websocket_rejects_cross_user_session(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="ws-cross-user-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-2",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    try:
        with _ws_connect(client, session.session_id, _fake_user("user-1")) as ws:
            ws.receive_json()
            raise AssertionError("WebSocket should reject cross-user access")
    except WebSocketDisconnect as exc:
        assert exc.code == 4403
