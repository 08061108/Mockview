from __future__ import annotations

import asyncio

import pytest

from app.agents import interviewer
from app.models.question import QuestionItem
from app.services.session_manager import SessionManager
from app.services.operation_store import reset_operation_store
from app.streaming import (
    SessionEventBus,
    emit_status_event,
    reset_stream_sink,
    set_stream_sink,
)
from app.api import interview_ws


class FakeStreamingLLM:
    def __init__(self, chunks: list[str] | None = None, stream_error: Exception | None = None):
        self.chunks = chunks or []
        self.stream_error = stream_error
        self.chat_calls = 0

    async def chat(self, **_kwargs) -> str:
        self.chat_calls += 1
        return "fallback question"

    async def chat_stream(self, **_kwargs):
        if self.stream_error:
            raise self.stream_error
        for chunk in self.chunks:
            yield chunk


def _state() -> dict:
    return {
        "question_plan": [
            QuestionItem(
                id=1,
                content="请介绍一下你的项目。",
                skill_tags=["Python", "LangGraph"],
                difficulty="medium",
                reference_points=["项目架构"],
            )
        ],
        "current_question_index": 0,
        "conversation_history": [],
        "max_follow_ups": 2,
    }


@pytest.mark.asyncio
async def test_ask_question_streams_chunks_and_returns_complete_state(monkeypatch):
    llm = FakeStreamingLLM(["第一段", "第二段"])
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(interviewer, "get_llm_client", lambda: llm)
    token = set_stream_sink(sink)
    try:
        result = await interviewer.ask_question(_state())
    finally:
        reset_stream_sink(token)

    assert [event["event"] for event in events] == ["start", "delta", "delta", "end"]
    assert "".join(event["chunk"] for event in events if event["event"] == "delta") == "第一段第二段"
    assert events[-1]["content"] == "第一段第二段"
    assert all(event["question_index"] == 1 for event in events)
    assert all(event["total_questions"] == 1 for event in events)
    assert all(event["skill_tags"] == ["Python", "LangGraph"] for event in events)
    assert all(event["difficulty"] == "medium" for event in events)
    assert all(event["question_id"] for event in events)
    assert all(event["turn_id"].endswith(":main") for event in events)
    assert result["conversation_history"][0].content == "第一段第二段"
    assert llm.chat_calls == 0


@pytest.mark.asyncio
async def test_streaming_does_not_replay_unknown_failure_before_first_chunk(monkeypatch):
    llm = FakeStreamingLLM(stream_error=RuntimeError("connection interrupted"))
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(interviewer, "get_llm_client", lambda: llm)
    token = set_stream_sink(sink)
    try:
        with pytest.raises(RuntimeError, match="connection interrupted"):
            await interviewer.ask_question(_state())
    finally:
        reset_stream_sink(token)

    assert [event["event"] for event in events] == ["start"]
    assert llm.chat_calls == 0


@pytest.mark.asyncio
async def test_streaming_falls_back_only_when_provider_explicitly_rejects_stream(monkeypatch):
    class UnsupportedStreamError(RuntimeError):
        status_code = 400

    llm = FakeStreamingLLM(
        stream_error=UnsupportedStreamError("stream is not supported by this model")
    )
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(interviewer, "get_llm_client", lambda: llm)
    token = set_stream_sink(sink)
    try:
        result = await interviewer.ask_question(_state())
    finally:
        reset_stream_sink(token)

    assert result["conversation_history"][0].content == "fallback question"
    assert [event["event"] for event in events] == ["start", "delta", "end"]
    assert llm.chat_calls == 1


@pytest.mark.asyncio
async def test_streaming_falls_back_to_chat_when_provider_returns_no_text(monkeypatch):
    llm = FakeStreamingLLM([])
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(interviewer, "get_llm_client", lambda: llm)
    token = set_stream_sink(sink)
    try:
        result = await interviewer.ask_question(_state())
    finally:
        reset_stream_sink(token)

    assert result["conversation_history"][0].content == "fallback question"
    assert [event["event"] for event in events] == ["start", "delta", "end"]
    assert events[1]["chunk"] == "fallback question"
    assert events[-1]["content"] == "fallback question"
    assert llm.chat_calls == 1


@pytest.mark.asyncio
async def test_status_event_uses_the_same_transport_sink():
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    token = set_stream_sink(sink)
    try:
        await emit_status_event("planning_questions", "正在规划问题。")
    finally:
        reset_stream_sink(token)

    assert events == [
        {
            "type": "status",
            "stage": "planning_questions",
            "message": "正在规划问题。",
        }
    ]


@pytest.mark.asyncio
async def test_session_event_bus_fans_out_bounded_events():
    bus = SessionEventBus(max_queue_size=2)
    first = bus.subscribe()
    second = bus.subscribe()

    await bus.publish({"type": "status", "stage": "one", "message": "1"})
    await bus.publish({"type": "status", "stage": "two", "message": "2"})
    await bus.publish({"type": "status", "stage": "three", "message": "3"})

    assert (await first.get())["sequence"] == 2
    assert (await second.get())["event"]["stage"] == "two"

    bus.unsubscribe(second)
    await bus.publish({"type": "status", "stage": "four", "message": "4"})
    assert (await first.get())["sequence"] == 3


@pytest.mark.asyncio
async def test_session_event_sequence_persists_replayable_events_without_token_content(monkeypatch):
    saved = []

    async def latest(_session_id):
        return 5

    async def save(session_id, sequence, event_id, event):
        saved.append((session_id, sequence, event_id, event))

    monkeypatch.setattr("app.streaming.latest_session_event_sequence", latest)
    monkeypatch.setattr("app.streaming.save_session_event", save)
    bus = SessionEventBus("session-events", max_queue_size=8)
    queue = bus.subscribe(after_sequence=5)
    await bus.publish({"type": "status", "stage": "planning", "message": "ready"})
    await bus.publish({"event": "delta", "stream_id": "stream-1", "chunk": "secret token"})
    await bus.publish({"event": "end", "stream_id": "stream-1", "content": "complete"})

    assert [item[1] for item in saved] == [6, 8]
    assert all("secret token" not in str(item) for item in saved)
    assert [(await queue.get())["sequence"] for _ in range(3)] == [6, 7, 8]


@pytest.mark.asyncio
async def test_background_answer_task_deduplicates_same_answer_id(monkeypatch):
    manager = SessionManager()
    session = manager.create_session(
        session_id="background-answer-session",
        jd_text="Python FastAPI LangGraph developer position",
    )
    data = manager._sessions[session.session_id]
    data.graph_started = True

    gate = __import__("asyncio").Event()
    calls = 0

    async def fake_submit(_session_id, _answer, *, stream_sink=None):
        nonlocal calls
        calls += 1
        await gate.wait()
        if stream_sink:
            await stream_sink({"type": "status", "stage": "done", "message": "done"})
        return {"interview_complete": False, "conversation_history": []}

    monkeypatch.setattr(manager, "submit_answer", fake_submit)
    first = manager.ensure_answer_task(session.session_id, "answer", "answer-1")
    second = manager.ensure_answer_task(session.session_id, "answer", "answer-1")
    assert first is second

    with pytest.raises(RuntimeError, match="Another answer"):
        manager.ensure_answer_task(session.session_id, "different", "answer-2")

    gate.set()
    result = await first
    assert result["interview_complete"] is False
    assert calls == 1

    cached = await manager.ensure_answer_task(session.session_id, "answer", "answer-1")
    assert cached["interview_complete"] is False
    assert calls == 1


@pytest.mark.asyncio
async def test_background_start_task_is_shared_and_publishes_events(monkeypatch):
    manager = SessionManager()
    session = manager.create_session(
        session_id="background-start-session",
        jd_text="Python FastAPI LangGraph developer position",
    )
    queue = manager.subscribe_events(session.session_id)
    calls = 0

    async def fake_start(_session_id, *, stream_sink=None):
        nonlocal calls
        calls += 1
        await stream_sink(
            {
                "event": "start",
                "stream_id": "stream-1",
                "kind": "question",
            }
        )
        await asyncio.sleep(0)
        await stream_sink(
            {
                "event": "end",
                "stream_id": "stream-1",
                "kind": "question",
                "content": "question",
                "done": True,
            }
        )
        return {"interview_complete": False}

    monkeypatch.setattr(manager, "start_interview_graph", fake_start)
    first = manager.ensure_start_task(session.session_id)
    second = manager.ensure_start_task(session.session_id)

    assert first is second
    result = await first
    first_event = await queue.get()
    second_event = await queue.get()

    assert result["interview_complete"] is False
    assert calls == 1
    assert first_event["event"]["event"] == "start"
    assert second_event["event"]["event"] == "end"


@pytest.mark.asyncio
async def test_websocket_heartbeat_keeps_exactly_one_queue_consumer(monkeypatch):
    class CountingQueue(asyncio.Queue):
        def __init__(self):
            super().__init__()
            self.get_calls = 0

        async def get(self):
            self.get_calls += 1
            return await super().get()

    class WebSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    class Manager:
        def unsubscribe_events(self, _session_id, _queue):
            return None

    async def finish_later():
        await asyncio.sleep(0.035)
        return {"interview_complete": False}

    monkeypatch.setattr(interview_ws, "WS_WAIT_HEARTBEAT_SECONDS", 0.01)
    queue = CountingQueue()
    result = await interview_ws._wait_for_session_task(
        WebSocket(),
        "s1",
        Manager(),
        asyncio.create_task(finish_later()),
        queue,
        {},
    )
    assert result["interview_complete"] is False
    assert queue.get_calls == 1


@pytest.mark.asyncio
async def test_task_completion_forwards_event_already_taken_by_consumer():
    class WebSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    class Manager:
        def unsubscribe_events(self, _session_id, _queue):
            return None

    queue = asyncio.Queue()
    await queue.put(
        {
            "sequence": 1,
            "event": {
                "type": "status",
                "stage": "question_ready",
                "message": "ready",
            },
        }
    )

    async def complete_now():
        await asyncio.sleep(0)
        return {"interview_complete": False}

    websocket = WebSocket()
    result = await interview_ws._wait_for_session_task(
        websocket,
        "simultaneous",
        Manager(),
        asyncio.create_task(complete_now()),
        queue,
        {},
    )
    assert result["interview_complete"] is False
    assert any(message.get("stage") == "question_ready" for message in websocket.messages)


@pytest.mark.asyncio
async def test_failed_start_is_not_recreated_until_explicit_retry(monkeypatch):
    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    monkeypatch.setattr("app.services.operation_store.append_operation_event", noop)
    reset_operation_store()
    manager = SessionManager()
    session = manager.create_session("failed-start", "JD for failure")
    calls = 0

    async def fail(_session_id, *, stream_sink=None):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(manager, "start_interview_graph", fail)
    first = manager.ensure_start_task(session.session_id)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await first
    second = manager.ensure_start_task(session.session_id)
    assert second is first
    assert calls == 1

    retried = await manager.retry_start_task(session.session_id, "retry-command-1")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await retried
    assert calls == 2
