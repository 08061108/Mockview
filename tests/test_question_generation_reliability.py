from __future__ import annotations

import asyncio

import pytest

from app.agents.interviewer import route_after_final_assessment
from app.agents.graph import build_practice_graph
from app.agents.question_planner import (
    generate_current_question,
    plan_questions,
    plan_questions_round2,
)
from app.llm.scheduler import (
    get_model_request_gate,
    model_request_priority,
    reset_model_request_gate,
)
from app.models.evidence import EvidenceChunk
from app.models.jd import SkillItem, SkillMatrix
from app.models.preparation import AssessmentTarget
from app.models.question import GeneratedQuestionDraft
from app.models.question import QuestionItem
from app.services.operation_store import OperationStore, reset_operation_store
from app.services.question_artifact_store import (
    SlotArtifact,
    get_question_artifact_store,
    reset_question_artifact_store,
)
from langgraph.checkpoint.memory import MemorySaver


@pytest.mark.asyncio
async def test_blueprint_delivers_only_first_question_and_routing_uses_planned_count(monkeypatch):
    target = AssessmentTarget(
        target_id="target-rag",
        case_id="case-1",
        material_revision=1,
        requirement_id="req-rag",
        skill_id="RAG",
        scope="RAG",
        importance=1.0,
        sources=["job_requirement"],
        verification_goal="验证 RAG 检索不足时的查询改写与停止边界",
    )
    chunk = EvidenceChunk(
        chunk_id="chunk-rag",
        document_id="doc-rag",
        text="RAG 检索不足时执行查询改写，证据仍不足时停止并说明边界。",
        source_type="private_material",
    )

    async def no_retrieval(*_args, **_kwargs):
        return []

    class FakeLLM:
        calls = 0

        async def chat_structured(self, **_kwargs):
            self.calls += 1
            return GeneratedQuestionDraft(
                content="RAG 检索结果不足时，你会如何改写查询并决定停止？",
                skill_tags=["RAG"],
                difficulty="medium",
                reference_points=["查询改写", "证据仍不足时停止"],
            )

    fake = FakeLLM()

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.agents.question_planner._retrieve_reference_questions", no_retrieval)
    monkeypatch.setattr("app.agents.question_planner.get_llm_client", lambda: fake)
    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    monkeypatch.setattr("app.services.operation_store.append_operation_event", noop)
    reset_operation_store()
    reset_question_artifact_store()
    state = {
        "session_id": "progressive-session",
        "user_id": "u1",
        "interview_mode": "practice",
        "skill_matrix": SkillMatrix(
            position_title="AI Engineer",
            experience_level="mid",
            skills=[SkillItem(name="RAG", category="framework", weight=1.0, is_required=True)],
        ),
        "assessment_targets": [target],
        "evidence_chunks": {target.target_id: [chunk]},
        "current_round": 1,
    }
    planned = await plan_questions(state)
    assert planned["question_plan"] == []
    assert planned["question_blueprint"].planned_count == 6
    revised = await plan_questions(
        {
            **state,
            "material_revision": 2,
            "analysis_strategy_version": "material-analysis-v1",
        }
    )
    assert (
        revised["question_blueprint"].input_version
        != planned["question_blueprint"].input_version
    )

    ready = await generate_current_question({**state, **planned})
    assert len(ready["question_plan"]) == 1
    assert (
        ready["question_plan"][0].question_id
        == planned["question_blueprint"].slots[0].question_id
    )
    assert fake.calls == 1
    routed = {**state, **planned, **ready, "current_question_index": 0}
    assert route_after_final_assessment(routed) == "ask_question"


@pytest.mark.asyncio
async def test_practice_graph_stops_for_answer_after_only_first_slot(monkeypatch):
    target = AssessmentTarget(
        target_id="target-graph-rag",
        case_id="case-graph",
        material_revision=1,
        requirement_id="req-graph-rag",
        skill_id="RAG",
        scope="RAG",
        importance=1.0,
        sources=["job_requirement"],
        verification_goal="验证检索证据不足时的处理策略",
    )
    chunk = EvidenceChunk(
        chunk_id="chunk-graph-rag",
        document_id="doc-graph-rag",
        text="检索证据不足时先改写查询；仍不足时停止并说明结论边界。",
        source_type="private_material",
    )

    async def no_retrieval(*_args, **_kwargs):
        return []

    async def noop(*_args, **_kwargs):
        return None

    class FakeLLM:
        calls = 0

        async def chat_structured(self, **_kwargs):
            self.calls += 1
            return GeneratedQuestionDraft(
                content="当检索证据不足时，你如何改写查询并控制结论边界？",
                skill_tags=["RAG"],
                difficulty="medium",
                reference_points=["查询改写", "证据边界"],
            )

    fake = FakeLLM()
    monkeypatch.setattr("app.agents.question_planner._retrieve_reference_questions", no_retrieval)
    monkeypatch.setattr("app.agents.question_planner.get_llm_client", lambda: fake)
    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    monkeypatch.setattr("app.services.operation_store.append_operation_event", noop)
    reset_operation_store()
    reset_question_artifact_store()
    graph = build_practice_graph().compile(
        interrupt_before=["assess_answer"],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "progressive-graph-session"}}
    result = await graph.ainvoke(
        {
            "session_id": "progressive-graph-session",
            "user_id": "u1",
            "interview_mode": "practice",
            "jd_text": "招聘熟悉 RAG 的工程师",
            "skill_matrix": SkillMatrix(
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
            ),
            "assessment_targets": [target],
            "evidence_chunks": {target.target_id: [chunk]},
            "evidence_assessments": {},
            "conversation_history": [],
            "assessments": [],
            "final_question_assessments": {},
            "current_question_index": 0,
            "follow_up_count": 0,
            "max_follow_ups": 2,
            "current_round": 1,
            "interview_complete": False,
        },
        config,
    )
    snapshot = await graph.aget_state(config)
    assert fake.calls == 1
    assert result["question_blueprint"].planned_count == 6
    assert len(result["question_plan"]) == 1
    assert len(result["conversation_history"]) == 1
    assert snapshot.next == ("assess_answer",)


def test_legacy_full_plan_without_blueprint_keeps_original_routing():
    plan = [
        QuestionItem(
            id=index,
            content=f"legacy question {index}",
            skill_tags=["Python"],
            primary_skill="Python",
            difficulty="medium",
            reference_points=["point"],
        )
        for index in (1, 2)
    ]
    assert (
        route_after_final_assessment(
            {
                "interview_mode": "practice",
                "question_plan": plan,
                "current_question_index": 0,
            }
        )
        == "ask_question"
    )
    assert (
        route_after_final_assessment(
            {
                "interview_mode": "practice",
                "question_plan": plan,
                "current_question_index": 1,
            }
        )
        == "evaluate"
    )


@pytest.mark.asyncio
async def test_round2_blueprint_is_derived_from_round1_probe_areas(monkeypatch):
    async def no_retrieval(*_args, **_kwargs):
        return []

    monkeypatch.setattr("app.agents.question_planner._retrieve_reference_questions", no_retrieval)
    base = {
        "session_id": "professional-round2",
        "user_id": "u1",
        "interview_mode": "professional",
        "skill_matrix": SkillMatrix(
            position_title="AI Engineer",
            experience_level="mid",
            skills=[
                SkillItem(
                    name="RAG",
                    category="framework",
                    weight=0.8,
                    is_required=True,
                ),
                SkillItem(
                    name="FastAPI",
                    category="framework",
                    weight=1.0,
                    is_required=True,
                ),
            ],
        ),
        "assessment_targets": [],
        "current_round": 2,
        "round1_summary_text": (
            '{"round1_score":7,"round1_grade":"B",'
            '"technical_depth_assessment":"技术深度尚可",'
            '"project_understanding":"项目理解基本完整",'
            '"strengths_observed":["工程实现"],'
            '"areas_to_probe":["RAG 证据冲突处理"],'
            '"proceed_to_round2":true,'
            '"round1_feedback":"继续验证系统边界"}'
        ),
    }
    planned = await plan_questions_round2(base)
    blueprint = planned["question_blueprint"]
    assert blueprint.round_number == 2
    assert blueprint.slots[0].primary_skill == "rag"
    assert "RAG 证据冲突处理" in blueprint.slots[0].objective


@pytest.mark.asyncio
async def test_model_gate_serves_waiting_foreground_before_background(monkeypatch):
    from app.llm import scheduler

    monkeypatch.setattr(scheduler.settings, "model_max_concurrency", 1)
    monkeypatch.setattr(scheduler.settings, "question_prefetch_global_limit", 1)
    reset_model_request_gate()
    gate = get_model_request_gate()
    release = asyncio.Event()
    order: list[str] = []

    async def holder():
        async with gate.slot():
            order.append("holder")
            await release.wait()

    async def contender(name: str, priority: str):
        async with model_request_priority(priority):
            async with gate.slot():
                order.append(name)

    holding = asyncio.create_task(holder())
    await asyncio.sleep(0)
    background = asyncio.create_task(contender("background", "background"))
    await asyncio.sleep(0)
    foreground = asyncio.create_task(contender("foreground", "foreground"))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(holding, background, foreground)
    assert order == ["holder", "foreground", "background"]
    reset_model_request_gate()


@pytest.mark.asyncio
async def test_stale_question_attempt_cannot_overwrite_new_attempt():
    reset_question_artifact_store()
    store = get_question_artifact_store()
    first = await store.begin(
        session_id="s1",
        blueprint_id="b1",
        slot_id="slot-1",
        input_version="v1",
        attempt=1,
    )
    second = await store.begin(
        session_id="s1",
        blueprint_id="b1",
        slot_id="slot-1",
        input_version="v1",
        attempt=2,
    )
    with pytest.raises(RuntimeError, match="stale question result"):
        await store.commit(first, {"question_id": "old"})
    committed = await store.commit(second, {"question_id": "new"})
    assert committed.question == {"question_id": "new"}
    duplicate = await store.commit(second, {"question_id": "new"})
    assert duplicate == committed
    with pytest.raises(RuntimeError, match="different artifact"):
        await store.commit(second, {"question_id": "changed"})


@pytest.mark.asyncio
async def test_terminated_session_rejects_question_result():
    reset_question_artifact_store()
    store = get_question_artifact_store()
    artifact = await store.begin(
        session_id="s2",
        blueprint_id="b2",
        slot_id="slot-1",
        input_version="v1",
        attempt=1,
    )
    await store.terminate_session("s2")
    with pytest.raises(RuntimeError, match="no longer valid"):
        await store.commit(artifact, {"question_id": "late"})


@pytest.mark.asyncio
async def test_question_artifact_failure_distinguishes_retryable_from_terminal():
    reset_question_artifact_store()
    store = get_question_artifact_store()
    retryable = await store.begin(
        session_id="s-retryable",
        blueprint_id="b-retryable",
        slot_id="slot-1",
        input_version="v1",
        attempt=1,
    )
    await store.fail(retryable, "temporary retrieval failure")
    stored_retryable = await store.get("s-retryable", "b-retryable", "slot-1")
    assert stored_retryable is not None
    assert stored_retryable.status == "failed_retryable"

    terminal = await store.begin(
        session_id="s-terminal",
        blueprint_id="b-terminal",
        slot_id="slot-1",
        input_version="v1",
        attempt=1,
    )
    await store.fail(terminal, "evidence missing", retryable=False)
    stored_terminal = await store.get("s-terminal", "b-terminal", "slot-1")
    assert stored_terminal is not None
    assert stored_terminal.status == "failed"


@pytest.mark.asyncio
async def test_termination_before_prefetch_registration_rejects_late_begin():
    reset_question_artifact_store()
    store = get_question_artifact_store()
    await store.terminate_session("s-before-begin")
    with pytest.raises(RuntimeError, match="terminated"):
        await store.begin(
            session_id="s-before-begin",
            blueprint_id="b-before-begin",
            slot_id="slot-1",
            input_version="v1",
            attempt=1,
        )


@pytest.mark.asyncio
async def test_rejected_database_begin_does_not_cache_speculative_attempt(monkeypatch):
    reset_question_artifact_store()
    store = get_question_artifact_store()
    first = await store.begin(
        session_id="s-db-authority",
        blueprint_id="b-db-authority",
        slot_id="slot-1",
        input_version="v1",
        attempt=1,
    )

    async def rejected(_candidate):
        return SlotArtifact(**{**first.__dict__, "terminated": True})

    monkeypatch.setattr(store, "_begin_postgres", rejected)
    with pytest.raises(RuntimeError, match="terminated"):
        await store.begin(
            session_id="s-db-authority",
            blueprint_id="b-db-authority",
            slot_id="slot-1",
            input_version="v1",
            attempt=2,
        )
    cached = await store.get("s-db-authority", "b-db-authority", "slot-1")
    assert cached is not None
    assert cached.current_attempt == 1
    assert cached.terminated is True


@pytest.mark.asyncio
async def test_postgres_conditional_path_rejects_cross_store_stale_commit(monkeypatch):
    class Cursor:
        def __init__(self, row=None):
            self.row = row

        async def fetchone(self):
            return self.row

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Connection:
        row = None

        def transaction(self):
            return Transaction()

        async def execute(self, sql, params=()):
            compact = " ".join(sql.split())
            if compact.startswith("INSERT INTO interview_question_slots"):
                session_id, blueprint_id, slot_id, input_version, attempt, checked_session = params
                assert checked_session == session_id
                if self.row is None:
                    self.row = {
                        "session_id": session_id,
                        "blueprint_id": blueprint_id,
                        "slot_id": slot_id,
                        "input_version": input_version,
                        "current_attempt": attempt,
                        "status": "generating",
                        "question": None,
                        "error": None,
                        "terminated": False,
                    }
                elif (
                    self.row["current_attempt"] < attempt
                    and self.row["input_version"] == input_version
                    and not self.row["terminated"]
                ):
                    self.row.update(
                        current_attempt=attempt,
                        status="generating",
                        question=None,
                        error=None,
                    )
                return Cursor()
            if compact.startswith("SELECT session_id"):
                return Cursor(dict(self.row) if self.row else None)
            if compact.startswith("UPDATE interview_question_slots SET status='ready'"):
                (
                    question,
                    session_id,
                    blueprint_id,
                    slot_id,
                    input_version,
                    attempt,
                    expected_question,
                ) = params
                question_value = getattr(question, "obj", question)
                expected_value = getattr(expected_question, "obj", expected_question)
                valid = bool(
                    self.row
                    and self.row["session_id"] == session_id
                    and self.row["blueprint_id"] == blueprint_id
                    and self.row["slot_id"] == slot_id
                    and self.row["input_version"] == input_version
                    and self.row["current_attempt"] == attempt
                    and not self.row["terminated"]
                    and (
                        self.row["status"] == "generating"
                        or (
                            self.row["status"] == "ready"
                            and self.row["question"] == expected_value
                        )
                    )
                )
                if not valid:
                    return Cursor()
                self.row.update(
                    status="ready",
                    question=question_value,
                    error=None,
                )
                return Cursor(dict(self.row))
            raise AssertionError(f"unexpected SQL: {compact}")

    class ConnectionContext:
        def __init__(self, connection):
            self.connection = connection

        async def __aenter__(self):
            return self.connection

        async def __aexit__(self, *_args):
            return False

    class Pool:
        def __init__(self):
            self.connection_state = Connection()

        def connection(self):
            return ConnectionContext(self.connection_state)

    from app.services.question_artifact_store import QuestionArtifactStore

    pool = Pool()
    monkeypatch.setattr(
        "app.services.question_artifact_store.session_store_enabled", lambda: True
    )
    monkeypatch.setattr(
        "app.services.question_artifact_store.get_pool", lambda: pool
    )
    older_store = QuestionArtifactStore()
    newer_store = QuestionArtifactStore()
    older = await older_store.begin(
        session_id="db-session",
        blueprint_id="db-blueprint",
        slot_id="slot-1",
        input_version="v1",
        attempt=1,
    )
    newer = await newer_store.begin(
        session_id="db-session",
        blueprint_id="db-blueprint",
        slot_id="slot-1",
        input_version="v1",
        attempt=2,
    )
    with pytest.raises(RuntimeError, match="stale or terminated"):
        await older_store.commit(older, {"question_id": "old"})
    committed = await newer_store.commit(newer, {"question_id": "new"})
    assert committed.current_attempt == 2
    assert committed.question == {"question_id": "new"}


@pytest.mark.asyncio
async def test_postgres_ready_artifact_cannot_be_overwritten_by_same_attempt(monkeypatch):
    # Reuse the SQL-emulating pool from the cross-store test by exercising the
    # real store against a tiny connection implementation focused on this race.
    row = {
        "session_id": "same-attempt-session",
        "blueprint_id": "same-attempt-blueprint",
        "slot_id": "slot-1",
        "input_version": "v1",
        "current_attempt": 1,
        "status": "generating",
        "question": None,
        "error": None,
        "terminated": False,
    }

    class Cursor:
        def __init__(self, value=None):
            self.value = value

        async def fetchone(self):
            return self.value

    class Context:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self.value

        async def __aexit__(self, *_args):
            return False

    class Connection:
        def transaction(self):
            return Context(self)

        async def execute(self, sql, params=()):
            compact = " ".join(sql.split())
            if compact.startswith("INSERT INTO interview_question_slots"):
                return Cursor()
            if compact.startswith("SELECT session_id"):
                return Cursor(dict(row))
            if compact.startswith("UPDATE interview_question_slots SET status='ready'"):
                question = getattr(params[0], "obj", params[0])
                expected = getattr(params[-1], "obj", params[-1])
                if row["status"] == "ready" and row["question"] != expected:
                    return Cursor()
                row.update(status="ready", question=question, error=None)
                return Cursor(dict(row))
            raise AssertionError(compact)

    class Pool:
        def __init__(self):
            self.conn = Connection()

        def connection(self):
            return Context(self.conn)

    from app.services.question_artifact_store import QuestionArtifactStore

    pool = Pool()
    monkeypatch.setattr(
        "app.services.question_artifact_store.session_store_enabled", lambda: True
    )
    monkeypatch.setattr(
        "app.services.question_artifact_store.get_pool", lambda: pool
    )
    first_store = QuestionArtifactStore()
    second_store = QuestionArtifactStore()
    first = await first_store.begin(
        session_id=row["session_id"],
        blueprint_id=row["blueprint_id"],
        slot_id=row["slot_id"],
        input_version=row["input_version"],
        attempt=1,
    )
    second = await second_store.begin(
        session_id=row["session_id"],
        blueprint_id=row["blueprint_id"],
        slot_id=row["slot_id"],
        input_version=row["input_version"],
        attempt=1,
    )
    await first_store.commit(first, {"question_id": "winner"})
    with pytest.raises(RuntimeError, match="stale or terminated"):
        await second_store.commit(second, {"question_id": "loser"})
    assert row["question"] == {"question_id": "winner"}


@pytest.mark.asyncio
async def test_foreground_attaches_to_same_inflight_prefetch(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def no_retrieval(*_args, **_kwargs):
        return []

    class SlowLLM:
        calls = 0

        async def chat_structured(self, **_kwargs):
            self.calls += 1
            started.set()
            await release.wait()
            return GeneratedQuestionDraft(
                content="请说明 FastAPI 请求链路中的异常处理。",
                skill_tags=["FastAPI"],
                difficulty="medium",
                reference_points=["异常分类", "可观测性"],
            )

    async def noop(*_args, **_kwargs):
        return None

    fake = SlowLLM()
    monkeypatch.setattr("app.agents.question_planner._retrieve_reference_questions", no_retrieval)
    monkeypatch.setattr("app.agents.question_planner.get_llm_client", lambda: fake)
    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    monkeypatch.setattr("app.services.operation_store.append_operation_event", noop)
    reset_operation_store()
    reset_question_artifact_store()
    base = {
        "session_id": "prefetch-session",
        "user_id": "u1",
        "interview_mode": "practice",
        "skill_matrix": SkillMatrix(
            position_title="Backend Engineer",
            experience_level="mid",
            skills=[SkillItem(name="FastAPI", category="framework", weight=1.0, is_required=True)],
        ),
        "assessment_targets": [],
        "evidence_chunks": {},
        "current_round": 1,
    }
    planned = await plan_questions(base)
    state = {**base, **planned}
    background = asyncio.create_task(
        generate_current_question({**state, "question_generation_background": True})
    )
    await started.wait()
    foreground = asyncio.create_task(generate_current_question(state))
    await asyncio.sleep(0)
    release.set()
    background_result, foreground_result = await asyncio.gather(background, foreground)
    assert fake.calls == 1
    assert background_result["question_plan"][0] == foreground_result["question_plan"][0]


@pytest.mark.asyncio
async def test_ready_artifact_is_reused_after_operation_store_restart(monkeypatch):
    async def no_retrieval(*_args, **_kwargs):
        return []

    async def noop(*_args, **_kwargs):
        return None

    class InitialLLM:
        async def chat_structured(self, **_kwargs):
            return GeneratedQuestionDraft(
                content="请解释 FastAPI 中间件的请求顺序。",
                skill_tags=["FastAPI"],
                difficulty="medium",
                reference_points=["入站顺序", "出站顺序"],
            )

    class MustNotRunLLM:
        async def chat_structured(self, **_kwargs):
            raise AssertionError("ready artifact must be reused without a model call")

    monkeypatch.setattr("app.agents.question_planner._retrieve_reference_questions", no_retrieval)
    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    monkeypatch.setattr("app.services.operation_store.append_operation_event", noop)
    reset_operation_store()
    reset_question_artifact_store()
    base = {
        "session_id": "artifact-restart-session",
        "user_id": "u1",
        "interview_mode": "practice",
        "skill_matrix": SkillMatrix(
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
        ),
        "assessment_targets": [],
        "evidence_chunks": {},
        "current_round": 1,
    }
    planned = await plan_questions(base)
    state = {**base, **planned}
    monkeypatch.setattr(
        "app.agents.question_planner.get_llm_client", lambda: InitialLLM()
    )
    first = await generate_current_question(state)
    reset_operation_store()
    monkeypatch.setattr(
        "app.agents.question_planner.get_llm_client", lambda: MustNotRunLLM()
    )
    recovered = await generate_current_question(state)
    assert recovered["question_plan"] == first["question_plan"]


@pytest.mark.asyncio
async def test_session_usage_summary_includes_requests_tokens_and_tools(monkeypatch):
    async def noop(*_args, **_kwargs):
        return None

    async def no_operations(_case_id):
        return []

    async def no_tool_calls(_operation_id):
        return []

    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.save_tool_call", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_tool_call", noop)
    monkeypatch.setattr("app.services.operation_store.list_operations_for_case", no_operations)
    monkeypatch.setattr("app.services.operation_store.list_tool_calls", no_tool_calls)
    store = OperationStore()
    await store.accept(
        user_id="u1",
        kind="interview_question",
        case_id="usage-session",
        operation_id="usage-operation",
        payload={"slot": "1"},
        budget={"model_requests": 2},
    )
    await store.record_model_request_started(
        "usage-operation",
        request_kind="structured:Question",
        model="qwen-test",
        queue_elapsed_seconds=0.25,
    )
    await store.record_model_completion(
        "usage-operation",
        usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        elapsed_seconds=1.25,
    )
    await store.accept_tool_call(
        operation_id="usage-operation",
        tool_name="retrieve_questions",
        payload={"query": "RAG"},
        tool_call_id="usage-tool",
    )
    await store.finish_tool_call("usage-tool", result={"count": 2})
    summary = await store.session_usage_summary("usage-session")
    assert summary["physical_requests"] == 1
    assert summary["completed_requests"] == 1
    assert summary["total_tokens"] == 28
    assert summary["tool_calls"] == 1
    assert summary["models"] == {"qwen-test": 1}
    assert summary["model_queue_seconds"] == 0.25
    assert summary["max_model_queue_seconds"] == 0.25
    assert summary["billing_cost_usd"] is None
