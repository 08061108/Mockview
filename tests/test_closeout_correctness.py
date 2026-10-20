"""Focused offline regressions for closeout P0-01/02/03 (all model calls stubbed)."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agents import evaluator, interviewer
from app.agents.evidence_pipeline import (
    assess_evidence_deterministic, retrieved_to_evidence, validate_question_deterministic,
)
from app.agents.question_planner import _build_blueprint
from app.models.evidence import EvidenceAssessment, EvidenceChunk
from app.models.interview import AnswerAssessment, ChatMessage, FinalQuestionAssessment, FinalQuestionAssessmentDraft
from app.models.jd import SkillItem, SkillMatrix
from app.models.preparation import PreparationCase
from app.models.question import QuestionItem
from app.models.report import PracticeReport, InterviewReport, ProfessionalReport
from app.rag.chunking import build_question_chunks
from app.rag.retriever import RetrievedQuestion
from app.services.preparation_service import build_assessment_targets
from app.services.scoring import aggregate_scores, aggregate_professional_rounds
from app.services.session_manager import SessionManager


def matrix():
    return SkillMatrix(position_title="Agent engineer", experience_level="junior", skills=[
        SkillItem(name="RAG", weight=.8, category="domain", is_required=True),
        SkillItem(name="Python", weight=.2, category="language", is_required=True),
    ])


def final(skill, score, qid="q1", **kwargs):
    return FinalQuestionAssessment(observation_id=f"observation:{qid}", question_id=qid,
        display_id=1, primary_skill=skill, score=score, **kwargs)


def state_for_end():
    return {
        "session_id": "end-test", "interview_mode": "practice", "current_round": 1,
        "skill_matrix": matrix(), "current_question_index": 0, "follow_up_count": 1,
        "current_turn_id": "followup", "question_plan": [QuestionItem(
            id=1, question_id="q1", content="解释 RAG 检索和融合", skill_tags=["RAG"],
            primary_skill="RAG", difficulty="medium", reference_points=["检索融合"],
        )],
        "conversation_history": [
            ChatMessage(message_id="main", role="interviewer", question_id="q1", turn_id="main", content="主问题"),
            ChatMessage(message_id="a1:candidate", role="candidate", question_id="q1", turn_id="main", answer_id="a1", content="先检索候选，再融合排序"),
            ChatMessage(message_id="followup", role="interviewer", question_id="q1", turn_id="followup", content="如何去重？"),
        ],
        "assessments": [AnswerAssessment(question_id="q1", turn_id="main", score=6, should_follow_up=True)],
        "final_question_assessments": {}, "interview_complete": False,
    }


@pytest.fixture
def llm(monkeypatch):
    async def structured(*, response_model, **kwargs):
        if response_model is AnswerAssessment:
            return AnswerAssessment(question_id="q1", score=8, should_follow_up=False)
        if response_model is FinalQuestionAssessmentDraft:
            return FinalQuestionAssessmentDraft(score=8, answer_evidence=["先检索候选，再融合排序"])
        common = dict(skill_scores=[], overall_assessment="离线合成报告")
        if response_model is PracticeReport:
            return PracticeReport(total_questions=999, missed_knowledge=[], study_suggestions=["复习检索融合"], **common)
        if response_model is InterviewReport:
            return InterviewReport(strengths=[dict(point="合成优势", evidence="合成证据")], improvements=[dict(point="合成建议", evidence="合成证据")], **common)
        if response_model is ProfessionalReport:
            return ProfessionalReport(strengths=[dict(point="合成优势", evidence="合成证据")], improvements=[dict(point="合成建议", evidence="合成证据")], hiring_recommendation="离线合成建议", **common)
        raise AssertionError(f"Unexpected model: {response_model}")
    stub = SimpleNamespace(chat_structured=AsyncMock(side_effect=structured))
    monkeypatch.setattr(interviewer, "get_llm_client", lambda: stub)
    monkeypatch.setattr(evaluator, "get_llm_client", lambda: stub)
    return stub


async def setup_manager(monkeypatch, state, mode="practice", *, checkpoint=True):
    from langgraph.checkpoint.memory import InMemorySaver
    monkeypatch.setattr("app.services.session_manager.get_checkpointer", lambda: InMemorySaver())
    manager = SessionManager()
    monkeypatch.setattr(manager, "_schedule_persist", lambda *_: None)
    manager.create_session("end-test", "Synthetic JD", mode=mode)
    data = manager._sessions["end-test"]
    data.last_state = copy.deepcopy(state)
    data.graph_started = True
    data.session.status = "interviewing"
    persisted = []

    async def persist(_sid):
        persisted.append(copy.deepcopy(data.last_state))

    monkeypatch.setattr(manager, "_persist_session", persist)
    monkeypatch.setattr(manager, "_persist_new_assessment_memories", AsyncMock())
    monkeypatch.setattr(manager, "_persist_final_memory", AsyncMock())
    monkeypatch.setattr(manager, "_cache_runtime_state", AsyncMock())
    monkeypatch.setattr("app.services.session_manager.get_question_artifact_store", lambda: SimpleNamespace(terminate_session=AsyncMock()))
    graph = manager._get_graph(mode)
    if checkpoint:
        await graph.aupdate_state(manager._config("end-test"), state, as_node="ask_follow_up")
    return manager, data, graph, persisted


@pytest.mark.asyncio
async def test_skill_identity_flows_targets_blueprint_final_report(llm):
    sm = matrix()
    targets = build_assessment_targets(PreparationCase(case_id="c", user_id="u", jd_text="RAG Python"), sm)
    state = state_for_end()
    state.update(preparation_case_id="c", assessment_targets=targets, evidence_assessments={
        target.target_id: EvidenceAssessment(target_id=target.target_id, verdict="sufficient") for target in targets
    })
    blueprint = _build_blueprint(state, namespace="practice", round_number=1, planned_count=2)
    assert [target.skill_id for target in targets] == ["rag", "python"]
    assert [slot.primary_skill for slot in blueprint.slots] == ["rag", "python"]
    finals = [final("rag", 10, answer_evidence=["RAG 的回答"]), final("python", 2, "q2")]
    state["final_question_assessments"] = {item.question_id: item for item in finals}
    report = (await evaluator.evaluate_practice(state))["practice_report"]
    assert report.overall_score == 8.4  # Old name/ID mismatch produced 6.0.
    assert report.grade == "B"  # Existing threshold A >= 8.5 is unchanged.
    assert report.coverage == dict(covered_skills=["RAG", "Python"], uncovered_skills=[], scored_questions=2, unscored_questions=0)
    assert report.skill_scores[0].evidence == "RAG 的回答"


def test_old_display_names_and_canonical_ids_share_one_weight():
    rows = [final("RAG", 10), final("rag", 6, "q2"), final("Python", 2, "q3"), final("python", None, "q4", scorable=False)]
    original = [item.model_dump() for item in rows]
    result = aggregate_scores(iter(rows), matrix())
    assert result.skill_scores == {"RAG": 8, "Python": 2}
    assert result.overall_score == 6.8
    assert result.uncovered_skills == []
    assert (result.scored_questions, result.unscored_questions) == (3, 1)
    assert [item.model_dump() for item in rows] == original
    assert aggregate_professional_rounds(8, 6) == 7.1
    assert aggregate_scores([final("Other", 7)], matrix()).uncovered_skills == ["RAG", "Python"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["practice", "professional"])
async def test_early_end_settles_once_before_report_and_updates_checkpoint(monkeypatch, llm, mode):
    state = state_for_end()
    manager, data, graph, persisted = await setup_manager(monkeypatch, state, mode)
    original = llm.chat_structured.side_effect

    async def checked(**kwargs):
        if kwargs["response_model"] is not FinalQuestionAssessmentDraft:
            checkpoint = await graph.aget_state(manager._config("end-test"))
            assert checkpoint.values["final_question_assessments"]["q1"].score == 8
            assert persisted[-1]["final_question_assessments"]["q1"].score == 8
        return await original(**kwargs)

    llm.chat_structured.side_effect = checked
    task = manager.ensure_end_task("end-test")
    assert manager.ensure_end_task("end-test") is task
    result = await task
    assert len(result["final_question_assessments"]) == 1
    report = result["practice_report" if mode == "practice" else "final_report"]
    assert report.overall_score == 8
    assert report.coverage["scored_questions"] == 1
    assert report.coverage["unscored_questions"] == 0
    assert await manager.stop_interview("end-test") == result
    assert llm.chat_structured.call_count == 2
    checkpoint = (await graph.aget_state(manager._config("end-test"))).values
    assert checkpoint["interview_complete"] is True
    assert len(checkpoint["conversation_history"]) == 3
    assert len(checkpoint["assessments"]) == 1
    # Business status stale after crash, checkpoint already completed: no model calls.
    data.session.status = "interviewing"
    assert (await manager.stop_interview("end-test"))["interview_complete"]
    assert llm.chat_structured.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("already_final", [False, True])
async def test_json_business_snapshot_recovers_without_checkpoint(monkeypatch, llm, already_final):
    from app.services.session_repository import _dump
    state = state_for_end()
    if already_final:
        state["final_question_assessments"] = {"q1": final("RAG", 8)}
    restored = json.loads(json.dumps(_dump(state)))
    manager, _, graph, _ = await setup_manager(monkeypatch, restored, checkpoint=False)
    result = await manager.stop_interview("end-test")
    assert result["practice_report"].total_questions == 1
    assert result["practice_report"].overall_score == 8
    assert llm.chat_structured.call_count == (1 if already_final else 2)
    saved = (await graph.aget_state(manager._config("end-test"))).values
    assert saved["interview_complete"] and saved["question_plan"][0].question_id == "q1"
    assert len(saved["conversation_history"]) == 3
    assert len(saved["final_question_assessments"]) == 1


@pytest.mark.asyncio
async def test_checkpoint_final_survives_business_save_failure(monkeypatch, llm):
    manager, data, graph, _ = await setup_manager(monkeypatch, state_for_end())
    monkeypatch.setattr(manager, "_persist_session", AsyncMock(side_effect=RuntimeError("synthetic save failure")))
    with pytest.raises(RuntimeError, match="synthetic save failure"):
        await manager.stop_interview("end-test")
    saved = (await graph.aget_state(manager._config("end-test"))).values
    assert saved["final_question_assessments"]["q1"].score == 8
    assert not saved["interview_complete"]
    assert llm.chat_structured.call_count == 1
    monkeypatch.setattr(manager, "_persist_session", AsyncMock())
    data.last_state = {}  # Simulate loss of the business cache after the failed save.
    result = await manager.stop_interview("end-test")
    assert result["practice_report"].total_questions == 1
    assert llm.chat_structured.call_count == 2


@pytest.mark.asyncio
async def test_legacy_numeric_process_assessment_retained_as_unscored(monkeypatch, llm):
    state = state_for_end()
    state["question_plan"][0].question_id = ""
    state["assessments"][0].question_id = 1
    state["conversation_history"] = []
    manager, _, _, _ = await setup_manager(monkeypatch, state)
    result = await manager.stop_interview("end-test")
    report = result["practice_report"]
    assert report.total_questions == 1 and report.coverage["unscored_questions"] == 1
    assert report.overall_score is None
    assert llm.chat_structured.call_count == 1  # No invented transcript for scoring.


@pytest.mark.asyncio
async def test_report_retry_reuses_durable_final(monkeypatch, llm):
    manager, data, graph, persisted = await setup_manager(monkeypatch, state_for_end())
    original = llm.chat_structured.side_effect
    fail_report = True

    async def fail_once(**kwargs):
        nonlocal fail_report
        if kwargs["response_model"] is PracticeReport and fail_report:
            fail_report = False
            raise RuntimeError("synthetic report failure")
        return await original(**kwargs)

    llm.chat_structured.side_effect = fail_once
    with pytest.raises(RuntimeError, match="synthetic report failure"):
        await manager.stop_interview("end-test")
    assert persisted[-1]["final_question_assessments"]["q1"].score == 8
    # Same saved checkpoint reloaded by a new manager; no repeat single-question scoring.
    restored = copy.deepcopy(persisted[-1])
    manager2, _, _, _ = await setup_manager(monkeypatch, restored)
    result = await manager2.stop_interview("end-test")
    assert result["practice_report"].total_questions == 1
    assert sum(c.kwargs["response_model"] is FinalQuestionAssessmentDraft for c in llm.chat_structured.call_args_list) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "null_score", "missing_transcript"])
async def test_unavailable_final_is_explicitly_unscored(monkeypatch, llm, failure):
    state = state_for_end()
    if failure == "missing_transcript":
        state["conversation_history"] = []
    original = llm.chat_structured.side_effect

    async def unavailable(**kwargs):
        if kwargs["response_model"] is FinalQuestionAssessmentDraft:
            if failure == "exception":
                raise RuntimeError("synthetic evaluator unavailable")
            return FinalQuestionAssessmentDraft(score=None, scorable=True)
        return await original(**kwargs)

    llm.chat_structured.side_effect = unavailable
    manager, _, _, _ = await setup_manager(monkeypatch, state)
    result = await manager.stop_interview("end-test")
    assessment = result["final_question_assessments"]["q1"]
    assert not assessment.scorable and assessment.score is None and assessment.uncertainty
    report = result["practice_report"]
    assert report.total_questions == 1 and report.overall_score is None
    assert report.coverage["unscored_questions"] == 1 and report.coverage["scored_questions"] == 0


@pytest.mark.asyncio
async def test_unanswered_current_question_is_not_counted(monkeypatch, llm):
    state = state_for_end()
    state["assessments"] = []
    state["conversation_history"] = state["conversation_history"][:1]
    manager, _, _, _ = await setup_manager(monkeypatch, state)
    result = await manager.stop_interview("end-test")
    assert result["practice_report"] is None
    llm.chat_structured.assert_not_called()


@pytest.mark.asyncio
async def test_normal_finalization_is_reused_by_early_end(monkeypatch, llm):
    state = state_for_end()
    state.update(await interviewer.finalize_question_assessment(state))
    manager, _, _, _ = await setup_manager(monkeypatch, state)
    result = await manager.stop_interview("end-test")
    assert result["practice_report"].total_questions == 1
    assert llm.chat_structured.call_count == 2


@pytest.mark.asyncio
async def test_normal_graph_finishes_answer_final_and_report(monkeypatch, llm):
    state = state_for_end()
    state.update(current_candidate_answer="按父文档 ID 去重", current_answer_id="a2")
    manager, data, graph, _ = await setup_manager(monkeypatch, state)
    result = await graph.ainvoke(None, manager._config("end-test"))
    assert result["interview_complete"]
    assert result["practice_report"].total_questions == 1
    assert result["practice_report"].overall_score == 8
    assert len(result["final_question_assessments"]) == 1
    assert llm.chat_structured.call_count == 3
    # End after normal graph completion only restores the completed snapshot.
    ended = await manager.stop_interview("end-test")
    assert ended["interview_complete"] and llm.chat_structured.call_count == 3


@pytest.mark.asyncio
async def test_early_end_round_two_preserves_both_rounds(monkeypatch, llm):
    state = state_for_end()
    r1 = final("RAG", 10, "round1-q1")
    state.update(interview_mode="professional", current_round=2,
        round1_final_question_assessments={r1.question_id: r1},
        final_question_assessments={r1.question_id: r1},
        round1_assessments=[AnswerAssessment(question_id=r1.question_id, score=10)],
    )
    state["assessments"].insert(0, state["round1_assessments"][0])
    manager, _, _, _ = await setup_manager(monkeypatch, state, "professional")
    result = await manager.stop_interview("end-test")
    report = result["professional_report"]
    assert report.technical_depth_score == 10 and report.technical_breadth_score == 8
    assert report.overall_score == 9.1
    assert report.coverage["scored_questions"] == 2


@pytest.mark.asyncio
async def test_advance_to_unanswered_question_does_not_count_an_extra_final(monkeypatch, llm):
    state = state_for_end()
    state["final_question_assessments"] = {"q1": final("rag", 8)}
    state["question_plan"].append(state["question_plan"][0].model_copy(update={"id": 2, "question_id": "q2"}))
    state["current_question_index"] = 1
    state["current_turn_id"] = "q2:main"
    manager, _, _, _ = await setup_manager(monkeypatch, state)
    result = await manager.stop_interview("end-test")
    assert set(result["final_question_assessments"]) == {"q1"}
    assert result["practice_report"].total_questions == 1
    assert llm.chat_structured.call_count == 1


@pytest.mark.asyncio
async def test_end_waits_for_inflight_answer_and_recovers_accepted_turn(monkeypatch, llm):
    state = state_for_end()
    manager, data, _, _ = await setup_manager(monkeypatch, state)
    started, released = asyncio.Event(), asyncio.Event()

    async def inflight():
        started.set()
        await released.wait()
        data.pending_answers["a2"] = dict(turn_id="followup", content="按父文档 ID 去重", status="failed")
        data.pending_answers["other"] = dict(turn_id="other-question", content="不属于本题", status="accepted")

    data.answer_task = asyncio.create_task(inflight())
    await started.wait()
    end = manager.ensure_end_task("end-test")
    await asyncio.sleep(0)
    assert not end.done()
    llm.chat_structured.assert_not_called()
    released.set()
    result = await end
    assert len(result["final_question_assessments"]) == 1
    transcript = llm.chat_structured.call_args_list[0].kwargs["messages"][1]["content"]
    assert "按父文档 ID 去重" in transcript and "不属于本题" not in transcript
    assert sum(m.answer_id == "a2" for m in result["conversation_history"]) == 1


def rewrite_fixture():
    root = Path(__file__).resolve().parents[1]
    parent = json.loads((root / "tests/fixtures/query_rewrite.json").read_text(encoding="utf-8"))
    sm = SkillMatrix(position_title="Engineer", experience_level="junior", skills=[
        SkillItem(name="查询改写", weight=1, category="domain", is_required=True)
    ])
    target = build_assessment_targets(PreparationCase(case_id="rewrite", user_id="u", jd_text="查询改写"), sm)[0]
    raw_chunks = build_question_chunks(parent, "RAG", "query_rewrite.json")
    chunks = [retrieved_to_evidence(RetrievedQuestion(id=c.id, content=c.text, metadata=c.metadata, score=1))
        for c in raw_chunks if c.id.endswith(("__answer_point_1", "__question_summary", "__generated_query_1"))]
    question = QuestionItem(id=1, target_id=target.target_id, content="如何设计查询改写？", skill_tags=["查询改写"], difficulty="medium",
        reference_points=["将缩写扩展并补充领域专业术语", "把单一查询拆分为多个子查询并行检索", "对多路改写结果进行融合与去重"],
        evidence_chunk_ids=[c.chunk_id for c in chunks])
    return parent, target, chunks, question, raw_chunks


def test_nine_parent_contaminated_citation_edges_become_three_exact_edges():
    _, target, chunks, question, raw = rewrite_fixture()
    result = validate_question_deterministic(question, target, chunks)
    assert result.unsupported_points == []
    assert len(result.citations) == 3
    assert sum(len(ids) for ids in result.citations.values()) == 3
    assert all(ids == ["synthetic-query-rewrite__question_summary"] for ids in result.citations.values())
    for chunk in chunks:
        assert chunk.text == next(c.metadata["chunk_text"] for c in raw if c.id == chunk.chunk_id)
    assert assess_evidence_deterministic(target, chunks).verdict == "sufficient"
    without_summary = [c for c in chunks if not c.chunk_id.endswith("__question_summary")]
    rejected = validate_question_deterministic(question, target, without_summary)
    assert len(rejected.unsupported_points) == 3 and rejected.citations == {}


def test_parent_body_is_only_citable_under_real_parent_id():
    parent, target, _, question, raw = rewrite_fixture()
    evidence = retrieved_to_evidence(RetrievedQuestion(id=parent["id"], content=parent["content"], metadata=raw[0].metadata, score=1))
    assert evidence.chunk_id == evidence.document_id == parent["id"]
    assert evidence.location == "parent_question"
    question.evidence_chunk_ids = [evidence.chunk_id]
    result = validate_question_deterministic(question, target, [evidence])
    assert result.unsupported_points == []
    assert all(ids == [parent["id"]] for ids in result.citations.values())


def test_missing_or_legacy_enriched_child_body_cannot_supply_support():
    parent, target, _, question, raw = rewrite_fixture()
    metadata = dict(raw[0].metadata)
    metadata.pop("chunk_text")
    missing = retrieved_to_evidence(RetrievedQuestion(id=raw[0].id, content=raw[0].text, metadata=metadata, score=1))
    assert missing.citation_text == ""
    wrapper = "Source ID: x\nCategory: RAG\nReference points: " + "; ".join(parent["reference_points"])
    legacy = EvidenceChunk(chunk_id="p__child", document_id="p", text=wrapper, source_type="question_bank")
    assert legacy.citation_text == ""
    result = validate_question_deterministic(question, target, [legacy, missing])
    assert len(result.unsupported_points) == 3
    from app.agents.question_planner import _format_prepared_evidence
    prompt = _format_prepared_evidence({"evidence_chunks": {target.target_id: [missing, legacy]}})
    assert "Reference points:" not in prompt and wrapper not in prompt


def test_source_context_identifies_target_but_does_not_prove_rubric():
    _, target, _, question, _ = rewrite_fixture()
    exact = EvidenceChunk(chunk_id="child", document_id="p", text="Expand abbreviations and domain terms",
        source_context="Query rewriting", source_type="question_bank", publication_status="published")
    question.reference_points = ["将缩写扩展并补充领域专业术语"]
    question.evidence_chunk_ids = [exact.chunk_id]
    assert validate_question_deterministic(question, target, [exact]).unsupported_points == []
    misleading = exact.model_copy(update={"text": "An unrelated sentence", "source_context": "Query rewriting. Expand abbreviations and domain terms"})
    assert validate_question_deterministic(question, target, [misleading]).unsupported_points == question.reference_points
