"""P1 closure: synthetic final scores, durable JSON doubles, real business handlers."""
import asyncio
import json
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver

from app.api import preparation as api
from app.agents import evaluator, job_assessment, question_planner
from app.memory.models import MemoryContext
from app.models.interview import FinalQuestionAssessment, InterviewSession
from app.models.jd import SkillItem, SkillMatrix
from app.models.preparation import PreparationCase, RetrainingRequest, OperationRecord, ClarificationRequest
from app.models.question import GeneratedQuestionDraft, QuestionItem
from app.models.report import PracticeReport, MissedKnowledge
from app.models.user import UserInDB
from app.rag.retriever import RetrievedQuestion, RetrievalOutcome
from app.services.operation_store import OperationStore
from app.services.preparation_service import PreparationService, build_assessment_targets
from app.services.session_manager import SessionManager
from app.services.session_repository import PersistedSession, _dump
from app.services.public_interview import build_public_report
from app.services.report_exporter import report_to_markdown


def final(qid, skill, score, **kwargs):
    return FinalQuestionAssessment(observation_id=f"obs:{qid}", question_id=qid,
        display_id=1, primary_skill=skill, score=score, **kwargs)


@pytest.fixture
def world(monkeypatch, tmp_path):
    async def save_case(case):
        (tmp_path / f"case-{case.case_id}.json").write_text(case.model_dump_json(), encoding="utf-8")

    async def load_case(cid, uid):
        path = tmp_path / f"case-{cid}.json"
        case = PreparationCase.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
        return case if case and case.user_id == uid else None

    async def save_operation(op):
        (tmp_path / f"op-{hashlib.sha256(op.operation_id.encode()).hexdigest()}.json").write_text(op.model_dump_json(), encoding="utf-8")

    async def load_operation(oid):
        path = tmp_path / f"op-{hashlib.sha256(oid.encode()).hexdigest()}.json"
        return OperationRecord.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    async def list_ops(cid):
        rows = [OperationRecord.model_validate_json(p.read_text(encoding="utf-8")) for p in tmp_path.glob("op-*.json")]
        return [row for row in rows if row.case_id == cid]

    async def save_session(**kwargs):
        (tmp_path / f"session-{kwargs['session'].session_id}.json").write_text(json.dumps(_dump(kwargs)), encoding="utf-8")

    async def load_session(sid):
        path = tmp_path / f"session-{sid}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        data["session"] = InterviewSession.model_validate(data["session"])
        return PersistedSession(**data)

    for module in ("app.services.preparation_service", "app.api.preparation", "app.agents.job_assessment"):
        monkeypatch.setattr(f"{module}.save_preparation_case", save_case)
    monkeypatch.setattr("app.services.preparation_service.load_preparation_case", load_case)
    for name, value in {"save_operation": save_operation, "load_operation": load_operation,
                        "list_operations_for_case": list_ops}.items():
        monkeypatch.setattr(f"app.services.operation_store.{name}", value)
    monkeypatch.setattr("app.services.session_manager.save_session_record", save_session)
    monkeypatch.setattr("app.services.session_manager.load_session_record", load_session)
    monkeypatch.setattr("app.services.session_manager.get_checkpointer", lambda: InMemorySaver())
    monkeypatch.setattr(SessionManager, "_schedule_persist", lambda *_: None)
    monkeypatch.setattr(SessionManager, "_schedule_question_prefetch", lambda *_: None)
    monkeypatch.setattr("app.services.preparation_limits.check_rate_limit", AsyncMock(return_value=SimpleNamespace(allowed=True)))
    api.limits._active.clear()
    api._analysis_tasks.clear()
    service, operations, manager = PreparationService(), OperationStore(), SessionManager()
    for module in (api, job_assessment):
        monkeypatch.setattr(module, "get_preparation_service", lambda: service)
    for module in (api, question_planner):
        monkeypatch.setattr(module, "get_operation_store", lambda: operations)
    monkeypatch.setattr(api, "get_session_manager", lambda: manager)
    memory = SimpleNamespace(_aget_memory_item=AsyncMock(return_value=None),
        abuild_context=AsyncMock(return_value=MemoryContext(user_id="u1")), format_context=lambda _: "")
    monkeypatch.setattr(api, "get_memory_service", lambda: memory)
    monkeypatch.setattr("app.services.session_manager.get_memory_service", lambda: memory)
    monkeypatch.setattr(job_assessment, "get_knowledge_service", lambda: SimpleNamespace(hydrate=AsyncMock(), active_release=lambda: None))
    from app.services.question_artifact_store import QuestionArtifactStore
    monkeypatch.setattr(question_planner, "get_question_artifact_store", lambda: QuestionArtifactStore())
    matrix = SkillMatrix(position_title="合成 Agent 岗位", experience_level="junior", skills=[
        SkillItem(name="RAG", category="domain", weight=.8, is_required=True),
        SkillItem(name="Python", category="language", weight=.2, is_required=True)])
    source = PreparationCase(case_id="source", user_id="u1", jd_text="合成 RAG Python 岗位",
                             material_revision=3, status="ready", skill_matrix=matrix)
    source.targets = build_assessment_targets(source, matrix)
    service._cases[source.case_id] = source
    session = manager.create_session("original", source.jd_text, user_id="u1", preparation_case_id="source",
        material_revision=3, skill_matrix=matrix, assessment_targets=source.targets)
    session.status = "completed"
    finals = [final("q-rag", "rag", 5), final("q-python", "python", 9), final("q-unscored", "rag", None, scorable=False)]
    manager._sessions["original"].last_state = {"final_question_assessments": {row.question_id: row for row in finals}}
    user = UserInDB(id="u1", username="synthetic", password_hash="unused", display_name="合成用户", role="user", is_active=True)
    return SimpleNamespace(service=service, operations=operations, manager=manager, source=source,
                           user=user, memory=memory, save_case=save_case, save_operation=save_operation, tmp=tmp_path)


async def finish_analysis(oid):
    task = api._analysis_tasks.get(oid)
    if task:
        await task
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_repractice_report_to_evidence_to_real_graph_first_question(world, monkeypatch):
    w = world
    await w.save_case(w.source)
    await w.manager.persist_session("original")
    # Reload both identities, as happens when viewing an old report after a restart.
    w.service._cases.clear()
    w.manager._sessions.clear()
    options = await api.repractice_options("original", w.user)
    assert options["case_id"] == "source" and options["material_revision"] == 3
    assert options["weak_skills"] == [{"skill_name": "RAG", "score": 5.0}]
    req = RetrainingRequest(source_session_id="original", selected_skills=["RAG"])
    a, b = await asyncio.gather(api.create_repractice("source", req, w.user), api.create_repractice("source", req, w.user))
    assert a.case_id == b.case_id and a.material_revision == 3
    assert a.preferences["source_question_ids"] == ["q-rag"]
    assert "avoid_question_ids" not in a.preferences
    assert all(t.material_revision == 3 and t.case_id == a.case_id for t in a.targets)
    assert a.targets[0].target_id != w.source.targets[0].target_id
    with pytest.raises(HTTPException) as denied:
        await api.create_interview_from_case(a.case_id, api.CreateInterviewRequest(), w.user)
    assert denied.value.status_code == 409
    body = "RAG 通过向量检索与 BM25 关键词检索召回候选，使用 RRF 倒数排名融合排序并去重。"
    retrieved = RetrievedQuestion(id="synthetic-rag", content=body,
        metadata={"question": "如何实现 RAG 混合检索？", "reference_points": body,
                  "category": "RAG", "difficulty": "medium"}, score=1, source="hybrid")
    outcome = RetrievalOutcome(items=[retrieved], candidate_status="found", mode="hybrid",
        vector_status="ok", bm25_status="ok", input_fingerprint="synthetic", query_count=1,
        duration_ms=0, index_version="synthetic")
    retriever = SimpleNamespace(retrieve_multi_outcome=AsyncMock(return_value=outcome))
    monkeypatch.setattr(job_assessment, "get_retriever", lambda: retriever)
    monkeypatch.setattr(question_planner, "get_retriever", lambda: retriever)
    monkeypatch.setattr(question_planner, "hydrate_parent_results", lambda rows, **_: rows)
    started = await api.start_analysis(a.case_id, api.StartOperationRequest(), w.user)
    await finish_analysis(started["operation_id"])
    progress = await api.get_case_progress(a.case_id, w.user)
    assert progress["operation"].status == "succeeded", progress["operation"].error
    assert progress["operation"].result["evidence_coverage"]["eligible_target_count"] == 1
    budget = dict(progress["operation"].budget)
    repeated = await api.start_analysis(a.case_id, api.StartOperationRequest(), w.user)
    assert repeated["operation_id"] == started["operation_id"]
    assert (await api.get_case_progress(a.case_id, w.user))["operation"].budget == budget
    request = api.CreateInterviewRequest(request_id="same-click")
    one, two = await asyncio.gather(api.create_interview_from_case(a.case_id, request, w.user),
                                    api.create_interview_from_case(a.case_id, request, w.user))
    assert one["session_id"] == two["session_id"]
    sid = one["session_id"]
    w.manager._sessions.pop(sid)
    assert (await api.create_interview_from_case(a.case_id, request, w.user))["session_id"] == sid
    async def model(**kwargs):
        assert kwargs["response_model"] is GeneratedQuestionDraft
        return GeneratedQuestionDraft(content="如何通过 RRF 融合 RAG 检索候选？", difficulty="medium",
            skill_tags=["RAG"], reference_points=[body], source_ids=["synthetic-rag"])
    stub = SimpleNamespace(chat_structured=AsyncMock(side_effect=model))
    monkeypatch.setattr(question_planner, "get_llm_client", lambda: stub)
    state = await w.manager.start_interview_graph(sid)
    assert state["conversation_history"][-1].role == "interviewer"
    assert state["question_plan"][0].validation_status == "supported"
    assert state["question_plan"][0].target_id == a.targets[0].target_id
    assert state["preparation_case_id"] == a.case_id and state["material_revision"] == 3
    assert stub.chat_structured.call_count == 1


@pytest.mark.asyncio
async def test_repractice_rejects_cross_user_wrong_case_changed_revision_and_disputed(world):
    w = world
    req = RetrainingRequest(source_session_id="original", selected_skills=["RAG"])
    for cid, user, request, status in [
        ("source", w.user.model_copy(update={"id": "other"}), req, 404),
        ("wrong-case", w.user, req, 409),
        ("source", w.user, req.model_copy(update={"selected_skills": ["Python"]}), 422),
    ]:
        with pytest.raises(HTTPException) as error:
            await api.create_repractice(cid, request, user)
        assert error.value.status_code == status
    w.memory._aget_memory_item.return_value = SimpleNamespace(user_id="u1", structured={"validity": "disputed"})
    assert (await api.repractice_options("original", w.user))["weak_skills"] == []
    w.source.material_revision = 4
    with pytest.raises(HTTPException) as error:
        await api.create_repractice("source", req, w.user)
    assert error.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["running", "waiting_input", "succeeded", "failed", "failed_retryable"])
async def test_progress_restores_exact_operation_without_reaccept_or_budget_reset(world, status):
    w = world
    await w.save_case(w.source)
    op = await w.operations.accept(user_id="u1", kind="prepare_case", case_id="source", input_revision=3,
        operation_id="frozen", payload={"revision": 3}, budget={"model_requests": 4})
    await w.operations.consume_budget(op.operation_id, "model_requests")
    await w.operations.transition(op.operation_id, status)
    w.operations._operations.clear()
    w.service._cases.clear()
    for _ in range(2):
        progress = await api.get_case_progress("source", w.user)
        assert progress["operation"].operation_id == "frozen"
        assert progress["operation"].status == status
        assert progress["operation"].budget["model_requests"] == 3
    assert (await api.start_analysis("source", api.StartOperationRequest(), w.user))["operation_id"] == "frozen"
    assert len(list(w.tmp.glob("op-*.json"))) == 1
    with pytest.raises(HTTPException) as error:
        await api.get_case_progress("source", w.user.model_copy(update={"id": "other"}))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_hitl_reload_answer_retry_and_continue_preserve_budget(world, monkeypatch):
    w = world
    w.source.status = "waiting_input"
    w.source.clarifications = [ClarificationRequest(clarification_id="clarify", case_id="source",
        material_revision=3, question="补充合成职责", missing_information="职责")]
    await w.save_case(w.source)
    op = await w.operations.accept(user_id="u1", kind="prepare_case", case_id="source", input_revision=3,
        operation_id="hitl", payload={}, budget={"model_requests": 3})
    await w.operations.consume_budget(op.operation_id, "model_requests")
    await w.operations.transition(op.operation_id, "waiting_input")
    w.service._cases.clear()
    w.operations._operations.clear()
    progress = await api.get_case_progress("source", w.user)
    assert progress["case"].clarifications[0].status == "pending"
    request = api.ClarificationAnswer(answer="合成职责说明")
    a = await api.answer_clarification("source", "clarify", request, w.user)
    b = await api.answer_clarification("source", "clarify", request, w.user)
    assert a.project_notes == b.project_notes == "合成职责说明"
    gate = asyncio.Event()
    async def resumed(oid, *_):
        await w.operations.transition(oid, "running")
        await gate.wait()
        await w.operations.consume_budget(oid, "model_requests")
        await w.operations.transition(oid, "succeeded")
    monkeypatch.setattr(api, "_run_analysis", AsyncMock(side_effect=resumed))
    await asyncio.gather(api.continue_operation("hitl", w.user), api.continue_operation("hitl", w.user))
    gate.set()
    await finish_analysis("hitl")
    assert api._run_analysis.call_count == 1
    assert (await api.get_case_progress("source", w.user))["operation"].budget["model_requests"] == 1


@pytest.mark.asyncio
async def test_final_scores_control_report_history_export_and_unscored(world, monkeypatch):
    w = world
    q = QuestionItem(id=1, question_id="q1", content="合成 RAG 题", skill_tags=["RAG"], difficulty="medium", reference_points=["证据"])
    q2 = q.model_copy(update={"id": 2, "question_id": "q2", "content": "无法评分题"})
    finals = [final("q1", "rag", 5, missed_points=["权威遗漏点"]), final("q2", "python", None, scorable=False)]
    generated = PracticeReport(overall_score=10, grade="A", total_questions=99, skill_scores=[],
        overall_assessment="按有效证据继续学习", study_suggestions=["复习"], missed_knowledge=[
            MissedKnowledge(question_id="q1", question="模型改写题面", score=9, missed_points=["假遗漏点"], reference_answer="合成参考答案"),
            MissedKnowledge(question_id="invented", question="虚构题", score=7, missed_points=[], reference_answer="虚构参考")])
    stub = SimpleNamespace(chat_structured=AsyncMock(return_value=generated))
    monkeypatch.setattr(evaluator, "get_llm_client", lambda: stub)
    state = {"skill_matrix": w.source.skill_matrix, "question_plan": [q, q2],
             "final_question_assessments": {row.question_id: row for row in finals}}
    report = (await evaluator.evaluate_practice(state))["practice_report"]
    assert (report.overall_score, report.grade, report.total_questions) == (5, "D", 2)
    assert report.missed_knowledge[0].score == 5
    assert report.missed_knowledge[0].missed_points == ["权威遗漏点"]
    assert report.missed_knowledge[1].score is None and not report.missed_knowledge[1].scorable
    assert len(report.missed_knowledge) == 2
    assert report.skill_scores[0].score == 5
    state.update(practice_report=report, interview_complete=True)
    w.manager._sessions["original"].last_state = state
    await w.manager.persist_session("original")
    w.manager._sessions.clear()
    await w.manager.ensure_session_loaded("original")
    public = w.manager.get_public_report("original")
    history = w.manager.get_public_overview("original")
    transcript = await w.manager.get_public_transcript("original")
    assert history.overall_score == public.overall_score == transcript.report.overall_score == 5
    assert history.question_count == history.assessment_count == public.total_questions == 2
    assert public.coverage["unscored_questions"] == 1
    markdown = report_to_markdown(public, mode="practice", session_id="original")
    assert "得分：5/10" in markdown and "得分：未评分" in markdown and "假遗漏点" not in markdown
    prompt = stub.chat_structured.call_args.kwargs["messages"][-1]["content"]
    assert "final_score=None" in prompt and "唯一权威数值" in prompt
    # Existing reports remain snapshots: no implicit re-score or rewritten missed score.
    old = generated.model_dump()
    old.pop("coverage")
    before = json.dumps(old, ensure_ascii=False)
    legacy = build_public_report(old, completed=True)
    assert legacy.overall_score == 10 and legacy.missed_knowledge[0].score == 9
    assert json.dumps(old, ensure_ascii=False) == before
