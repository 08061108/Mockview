from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents.job_assessment import AssessmentTools, ToolContext
from app.models.evidence import SearchSnapshot
from app.models.preparation import PreparationCreate
from app.services.operation_store import OperationStore, get_operation_store
from app.services.preparation_service import PreparationService, build_assessment_targets
from app.models.jd import SkillItem, SkillMatrix
from app.models.interview import ChatMessage
from app.models.question import QuestionBlueprint, QuestionItem, QuestionSlot
from app.services.session_manager import SessionManager


@pytest.mark.asyncio
async def test_h1_pending_answer_waits_for_explicit_retry_after_reload(monkeypatch):
    manager = SessionManager()
    session = manager.create_session("h1", "JD text for recovery test")
    data = manager._sessions[session.session_id]
    data.graph_started = True
    data.last_state = {
        "question_plan": [
            QuestionItem(
                id=1,
                content="请解释恢复语义。",
                skill_tags=["recovery"],
                primary_skill="recovery",
                difficulty="medium",
                reference_points=["幂等"],
            )
        ],
        "current_question_index": 0,
    }

    persisted = []

    async def fake_persist(_session_id):
        persisted.append(dict(data.pending_answers))

    calls = 0

    async def fake_submit(_session_id, answer, *, answer_id=None, stream_sink=None):
        nonlocal calls
        calls += 1
        return {
            "conversation_history": [
                ChatMessage(
                    message_id=f"{answer_id}:candidate",
                    role="candidate",
                    content=answer,
                    answer_id=answer_id,
                )
            ],
            "interview_complete": False,
        }

    monkeypatch.setattr(manager, "_persist_session", fake_persist)
    monkeypatch.setattr(manager, "submit_answer", fake_submit)
    await manager.accept_answer_submission("h1", "original answer", "answer-1")
    assert persisted and persisted[-1]["answer-1"]["status"] == "accepted"
    assert manager.ensure_pending_answer_task("h1") is None
    assert manager.pending_retry_kind("h1") == "answer"
    task = await manager.retry_pending_answer_task("h1", "retry-answer-1")
    result = await task
    assert calls == 1
    assert result["conversation_history"][0].content == "original answer"
    assert data.pending_answers["answer-1"]["status"] == "succeeded"
    duplicate = await manager.retry_pending_answer_task("h1", "retry-answer-1")
    assert duplicate is task


@pytest.mark.asyncio
async def test_h2_h4_same_answer_id_reuses_result_and_conflicting_content_is_rejected(monkeypatch):
    manager = SessionManager()
    session = manager.create_session("h24", "JD text for idempotency test")
    data = manager._sessions[session.session_id]
    data.graph_started = True
    data.last_state = {
        "conversation_history": [
            ChatMessage(
                message_id="answer-1:candidate",
                role="candidate",
                content="saved answer",
                answer_id="answer-1",
            )
        ],
        "interview_complete": False,
    }
    called = False

    async def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(manager, "submit_answer", should_not_run)
    result = await manager.ensure_answer_task("h24", "saved answer", "answer-1")
    assert result == data.last_state
    assert called is False
    with pytest.raises(ValueError, match="different content"):
        manager.ensure_answer_task("h24", "changed answer", "answer-1")


@pytest.mark.asyncio
async def test_h3_checkpoint_result_backfills_business_state(monkeypatch):
    manager = SessionManager()
    session = manager.create_session("h3", "JD text for checkpoint test")
    data = manager._sessions[session.session_id]
    data.graph_started = True
    checkpoint = {
        "conversation_history": [ChatMessage(role="interviewer", content="restored")],
        "interview_complete": True,
        "final_question_assessments": {},
    }

    async def fake_state(_session_id):
        return checkpoint

    cached = []

    async def fake_cache(_session_id, state):
        cached.append(state)

    monkeypatch.setattr(manager, "get_graph_state", fake_state)
    monkeypatch.setattr(manager, "_cache_runtime_state", fake_cache)
    result = await manager.get_last_state("h3")
    assert result is checkpoint
    assert data.session.status == "completed"
    assert cached == [checkpoint]


@pytest.mark.asyncio
async def test_same_pending_answer_id_with_different_payload_conflicts(monkeypatch):
    manager = SessionManager()
    manager.create_session("conflict", "JD text")

    async def noop(_session_id):
        return None

    monkeypatch.setattr(manager, "_persist_session", noop)
    await manager.accept_answer_submission("conflict", "first", "a1", "t1")
    with pytest.raises(ValueError):
        await manager.accept_answer_submission("conflict", "second", "a1", "t1")


@pytest.mark.asyncio
async def test_h5_partial_search_and_extract_results_are_reused(monkeypatch):
    async def noop(*_args, **_kwargs):
        return None

    async def no_load(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.save_tool_call", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", no_load)
    monkeypatch.setattr("app.services.operation_store.load_tool_call", no_load)
    monkeypatch.setattr("app.services.preparation_service.save_preparation_case", noop)
    monkeypatch.setattr("app.agents.job_assessment.save_preparation_case", noop)
    monkeypatch.setattr("app.agents.job_assessment.save_document", noop)

    operations = OperationStore()
    operation = await operations.accept(
        user_id="u1",
        kind="prepare",
        payload={"case": "c1"},
        operation_id="op-h5",
        budget={"decisions": 8, "searches": 2, "extracts": 4},
    )
    preparation = PreparationService()
    case = await preparation.create(
        "u1", PreparationCreate(jd_text="需要 RAG 检索纠错经验")
    )
    matrix = SkillMatrix(
        position_title="AI Engineer",
        experience_level="mid",
        skills=[SkillItem(name="RAG", category="framework", weight=1, is_required=True)],
    )
    case.targets = build_assessment_targets(case, matrix)
    preparation._cases[case.case_id] = case

    class FakePublic:
        search_calls = 0
        extract_calls: dict[str, int] = {}

        async def search(self, query, max_results=5):
            self.search_calls += 1
            return [{"url": "https://a.example"}, {"url": "https://b.example"}]

        async def extract(self, urls):
            url = urls[0]
            self.extract_calls[url] = self.extract_calls.get(url, 0) + 1
            if url.endswith("b.example") and self.extract_calls[url] == 1:
                raise RuntimeError("synthetic extract interruption")
            return [
                SearchSnapshot(
                    query="RAG",
                    url=url,
                    title=url,
                    content="RAG 检索不足时查询改写并在证据不足时停止。",
                    fetched_at="2026-10-13T00:00:00Z",
                    content_hash=url[-9:],
                )
            ]

    public = FakePublic()
    context = ToolContext(
        operation_id=operation.operation_id,
        user_id="u1",
        case_id=case.case_id,
        material_revision=1,
        targets=case.targets,
    )
    tools = AssessmentTools(
        context, preparation=preparation, public=public, operations=operations
    )
    args = {"query": "RAG 检索纠错", "target_id": case.targets[0].target_id}
    with pytest.raises(RuntimeError, match="interruption"):
        await tools.search_public_knowledge(args)
    result = await tools.search_public_knowledge(args)
    assert public.search_calls == 1
    assert public.extract_calls["https://a.example"] == 1
    assert public.extract_calls["https://b.example"] == 2
    assert len(result["chunks"]) == 2


@pytest.mark.asyncio
async def test_startup_resumes_pending_checkpoint_without_reloading_memory(monkeypatch):
    manager = SessionManager()
    manager.create_session("resume-checkpoint", "JD text for frozen checkpoint")
    checkpoint = {
        "session_id": "resume-checkpoint",
        "user_id": "u1",
        "interview_mode": "practice",
        "memory_context": "frozen-memory",
        "retrieved_memories": {"frozen": True},
        "question_plan": [],
        "conversation_history": [],
        "interview_complete": False,
    }

    class FakeGraph:
        invoked_with = "not-called"

        async def aget_state(self, _config):
            return SimpleNamespace(values=checkpoint, next=("generate_question",))

        async def ainvoke(self, value, _config):
            self.invoked_with = value
            return checkpoint

    class FrozenMemory:
        async def abuild_context(self, *_args, **_kwargs):
            raise AssertionError("mutable memory must not be read while resuming")

    async def noop(*_args, **_kwargs):
        return None

    graph = FakeGraph()
    manager._practice_graph = graph
    monkeypatch.setattr(
        "app.services.session_manager.get_memory_service", lambda: FrozenMemory()
    )
    monkeypatch.setattr(manager, "_cache_runtime_state", noop)
    monkeypatch.setattr(manager, "_persist_session", noop)
    result = await manager.start_interview_graph("resume-checkpoint")
    assert result is checkpoint
    assert graph.invoked_with is None


@pytest.mark.asyncio
async def test_duplicate_question_retry_command_reuses_completed_task(monkeypatch):
    manager = SessionManager()
    manager.create_session("retry-idempotent", "JD text for retry")
    data = manager._sessions["retry-idempotent"]
    data.graph_started = True
    slot = QuestionSlot(
        slot_id="slot-1",
        question_id="question-1",
        position=1,
        round_number=1,
        primary_skill="Python",
        difficulty="medium",
        objective="验证 Python 基础",
    )
    state = {
        "question_blueprint": QuestionBlueprint(
            blueprint_id="blueprint-1",
            input_version="input-v1",
            round_number=1,
            planned_count=1,
            slots=[slot],
        ),
        "question_plan": [],
        "current_question_index": 0,
        "question_generation_attempts": {slot.slot_id: 1},
        "question_retry_commands": {},
        "question_generation_status": "failed",
    }

    class FakeGraph:
        calls = 0

        async def aupdate_state(self, _config, update):
            state.update(update)

        async def ainvoke(self, _value, _config):
            self.calls += 1
            state.update(
                {
                    "question_plan": [
                        QuestionItem(
                            id=1,
                            question_id="question-1",
                            content="请解释 Python GIL。",
                            skill_tags=["Python"],
                            primary_skill="Python",
                            difficulty="medium",
                            reference_points=["线程", "解释器锁"],
                        )
                    ],
                    "question_generation_status": "ready",
                    "interview_complete": False,
                }
            )
            return state

    async def get_state(_session_id):
        return state

    async def noop(*_args, **_kwargs):
        return None

    graph = FakeGraph()
    manager._practice_graph = graph
    monkeypatch.setattr(manager, "get_graph_state", get_state)
    monkeypatch.setattr(manager, "_cache_runtime_state", noop)
    monkeypatch.setattr(manager, "_persist_session", noop)
    first = await manager.retry_question_task("retry-idempotent", "retry-command-1")
    await first
    duplicate = await manager.retry_question_task(
        "retry-idempotent", "retry-command-1"
    )
    assert duplicate is first
    assert graph.calls == 1


@pytest.mark.asyncio
async def test_material_analysis_is_reused_for_same_revision_and_strategy(monkeypatch):
    service = PreparationService()
    case = await service.create(
        "u1", PreparationCreate(jd_text="需要熟悉 RAG 检索与证据校验")
    )
    matrix = SkillMatrix(
        position_title="AI Engineer",
        experience_level="mid",
        skills=[
            SkillItem(
                name="RAG",
                category="framework",
                weight=1.0,
                is_required=True,
            )
        ],
    )
    frozen = case.model_copy(
        update={
            "skill_matrix": matrix,
            "targets": build_assessment_targets(case, matrix),
            "status": "failed",
        }
    )
    service._cases[case.case_id] = frozen

    async def must_not_analyze(_state):
        raise AssertionError("same material analysis should be reused")

    monkeypatch.setattr("app.services.preparation_service.analyze_jd", must_not_analyze)
    reused = await service.analyze(case.case_id, "u1")
    assert reused.skill_matrix == matrix
    assert reused.targets == frozen.targets
    assert reused.material_revision == case.material_revision


@pytest.mark.asyncio
async def test_persisted_start_retry_command_resumes_same_unstarted_attempt(monkeypatch):
    from app.services.operation_store import reset_operation_store

    reset_operation_store()
    manager = SessionManager()
    manager.create_session("retry-start-crash-window", "JD text for startup retry")
    data = manager._sessions["retry-start-crash-window"]
    data.start_attempt = 2
    data.start_retry_commands["retry-start-command"] = 2
    calls = 0

    async def fake_start(_session_id, *, stream_sink=None):
        nonlocal calls
        calls += 1
        return {"interview_complete": False, "conversation_history": []}

    monkeypatch.setattr(manager, "start_interview_graph", fake_start)
    task = await manager.retry_start_task(
        "retry-start-crash-window", "retry-start-command"
    )
    result = await task
    assert result["interview_complete"] is False
    assert calls == 1
    operation = await get_operation_store().get(
        "interview-start:retry-start-crash-window:attempt:2", "local-user"
    )
    assert operation is not None
    assert operation.status == "succeeded"


@pytest.mark.asyncio
async def test_process_recovery_marks_accepted_operation_retryable(monkeypatch):
    async def noop(*_args, **_kwargs):
        return None

    async def no_persisted_recovery():
        return 0

    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    monkeypatch.setattr("app.services.operation_store.append_operation_event", noop)
    monkeypatch.setattr(
        "app.services.operation_store.mark_unfinished_operations_retryable",
        no_persisted_recovery,
    )
    store = OperationStore()
    await store.accept(
        user_id="u1",
        kind="interview_start",
        case_id="recover-accepted",
        operation_id="accepted-before-crash",
        payload={"session_id": "recover-accepted"},
    )
    assert await store.recover_running() == 1
    recovered = await store.get("accepted-before-crash", "u1")
    assert recovered is not None
    assert recovered.status == "failed_retryable"
    assert recovered.error_details["type"] == "ProcessInterrupted"
