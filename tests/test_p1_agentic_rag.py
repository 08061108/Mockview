from __future__ import annotations

import pytest

from app.agents.evidence_pipeline import (
    assess_evidence_deterministic,
    resolve_question_evidence,
)
from app.agents.job_assessment import AssessmentTools, JobAssessmentAgent, ToolContext
from app.agents.question_planner import _build_blueprint, generate_current_question, plan_questions
from app.models.evidence import EvidenceChunk
from app.models.jd import SkillItem, SkillMatrix
from app.models.preparation import (
    AssessmentTarget,
    ClarificationRequest,
    MaterialCitation,
    PreparationCase,
)
from app.models.question import GeneratedQuestionDraft, QuestionItem, QuestionPlan
from app.models.resume import ProjectInfo, ResumeProfile
from app.rag.retriever import RetrievedQuestion
from app.resume.matcher import match_resume_to_jd
from app.services.operation_store import OperationStore
from app.services.preparation_service import build_assessment_targets
from app.services.question_artifact_store import QuestionArtifactStore


def _matrix() -> SkillMatrix:
    return SkillMatrix(
        position_title="AI Engineer",
        experience_level="mid",
        skills=[
            SkillItem(name="检索增强生成", category="framework", weight=1, is_required=True),
            SkillItem(name="Redis", category="domain", weight=0.8, is_required=True),
        ],
    )


def test_resume_alias_match_and_unmentioned_is_not_inability():
    profile = ResumeProfile(
        skills=["RAG"],
        projects=[
            ProjectInfo(
                name="知识库",
                tech_stack=["vector search"],
                description="实现企业知识问答",
                key_contributions=["负责检索链路"],
                potential_deep_dive_points=[],
            )
        ],
    )

    matched = match_resume_to_jd(profile, _matrix())

    assert "检索增强生成" in matched.matched_skills
    assert matched.match_evidence["检索增强生成"]
    assert matched.unmentioned_skills == ["Redis"]
    assert matched.missing_skills == matched.unmentioned_skills
    assert "不具备" not in " ".join(matched.interview_focus)


def test_target_generation_distinguishes_claimed_and_unmentioned_skills():
    profile = ResumeProfile(skills=["RAG"])
    case = PreparationCase(
        case_id="case-targets",
        user_id="u1",
        jd_text="需要检索增强生成与 Redis",
        resume_text="负责 RAG 项目的召回链路",
    )

    targets = build_assessment_targets(case, _matrix(), profile)
    by_scope = {item.scope: item for item in targets}

    assert by_scope["检索增强生成"].resume_signal == "claimed"
    assert by_scope["检索增强生成"].gap_kind == "personal_experience"
    assert by_scope["Redis"].resume_signal == "unmentioned"
    assert by_scope["Redis"].gap_kind == "technical_knowledge"
    assert "不得将未写明的经历推断为不会" in by_scope["Redis"].constraints


def test_unreviewed_external_evidence_never_satisfies_or_supports_rubric():
    target = AssessmentTarget(
        target_id="target-rag",
        case_id="case-1",
        material_revision=1,
        requirement_id="req-rag",
        skill_id="rag",
        scope="RAG",
        importance=1,
        sources=["job_requirement"],
        verification_goal="验证 RAG 查询改写",
    )
    candidate = EvidenceChunk(
        chunk_id="candidate-1",
        document_id="candidate-doc",
        text="RAG 查询改写和召回融合",
        source_type="public_web",
        publication_status="candidate",
    )
    question = QuestionItem(
        id=1,
        target_id=target.target_id,
        content="如何做 RAG 查询改写？",
        skill_tags=["RAG"],
        difficulty="medium",
        reference_points=["查询改写"],
        evidence_chunk_ids=[candidate.chunk_id],
    )

    assessment = assess_evidence_deterministic(target, [candidate])
    selected = resolve_question_evidence(
        question, [candidate], target_id=target.target_id
    )

    assert assessment.verdict == "insufficient"
    assert assessment.relevant_chunk_ids == []
    assert selected == []


class _FakePreparation:
    def __init__(self, case: PreparationCase):
        self.case = case

    async def get(self, case_id: str, user_id: str):
        if (case_id, user_id) == (self.case.case_id, self.case.user_id):
            return self.case
        return None


class _FakeKnowledge:
    async def hydrate(self):
        return None

    def active_release(self):
        return None

    async def search_release(self, _release_id, _query, top_k=8):
        return []


@pytest.mark.asyncio
async def test_answered_clarification_is_bound_only_to_its_target(monkeypatch):
    first = AssessmentTarget(
        target_id="target-first",
        case_id="case-bound-clarification",
        material_revision=1,
        requirement_id="req-first",
        skill_id="rag",
        scope="RAG",
        importance=1,
        sources=["job_requirement", "project_claim"],
        verification_goal="验证 RAG 项目职责",
        gap_kind="personal_experience",
        resume_signal="claimed",
    )
    second = first.model_copy(
        update={
            "target_id": "target-second",
            "requirement_id": "req-second",
            "skill_id": "agent",
            "scope": "Agent",
            "verification_goal": "验证 Agent 项目职责",
        }
    )
    case = PreparationCase(
        case_id=first.case_id,
        user_id="u1",
        jd_text="需要 RAG 和 Agent",
        targets=[first, second],
        clarifications=[
            ClarificationRequest(
                clarification_id="operation:clarification:target-first",
                case_id=first.case_id,
                material_revision=1,
                question="请说明 RAG 职责",
                missing_information="RAG 职责",
                status="answered",
                answer="我负责 RAG 查询改写与召回评测",
            )
        ],
    )
    store = OperationStore()

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.agents.job_assessment.save_preparation_case", noop)
    await store.accept(
        user_id="u1",
        kind="prepare_case",
        case_id=case.case_id,
        input_revision=1,
        operation_id="operation",
        payload={"case_id": case.case_id},
    )
    context = ToolContext(
        operation_id="operation",
        user_id="u1",
        case_id=case.case_id,
        material_revision=1,
        targets=[first, second],
    )
    tools = AssessmentTools(
        context,
        operations=store,
        preparation=_FakePreparation(case),
    )
    await tools.seed_material_evidence(first)
    await tools.seed_material_evidence(second)

    assert any(
        item.retrieval_sources == ["user_clarification"]
        for item in case.evidence_chunks[first.target_id]
    )
    assert not any(
        item.retrieval_sources == ["user_clarification"]
        for item in case.evidence_chunks.get(second.target_id, [])
    )


class _FakeRetriever:
    def __init__(self, *, empty: bool = False):
        self.empty = empty
        self.calls: list[str] = []

    async def retrieve(self, query: str, top_k: int = 8):
        self.calls.append(query)
        if self.empty:
            return []
        scope = query.split()[0]
        return [
            RetrievedQuestion(
                id=f"chunk-{scope}",
                content=f"{scope} 原理、实现、边界与常见故障处理",
                metadata={"parent_id": f"doc-{scope}", "chunk_type": "answer"},
                score=1.0,
                source="hybrid",
            )
        ]


async def _operation_store(monkeypatch, *, decisions: int = 8) -> OperationStore:
    store = OperationStore()

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.services.operation_store.save_operation", noop)
    monkeypatch.setattr("app.services.operation_store.load_operation", noop)
    monkeypatch.setattr("app.services.operation_store.save_tool_call", noop)
    monkeypatch.setattr("app.services.operation_store.load_tool_call", noop)
    await store.accept(
        user_id="u1",
        kind="prepare_case",
        case_id="case-agent",
        input_revision=1,
        payload={"case_id": "case-agent"},
        operation_id="agent-op",
        budget={"decisions": decisions, "model_requests": 4, "semantic_checks": 0},
    )
    await store.transition("agent-op", "running")
    return store


@pytest.mark.asyncio
async def test_autonomous_gap_uses_llm_tool_result_to_complete_target(monkeypatch):
    target = _technical_target("target-rag", "RAG")
    case = PreparationCase(
        case_id="case-agent",
        user_id="u1",
        jd_text="需要 RAG",
        status="ready",
        targets=[target],
    )
    store = await _operation_store(monkeypatch)

    class SequenceRetriever:
        def __init__(self):
            self.calls = []

        async def retrieve(self, query: str, top_k: int = 8):
            self.calls.append(query)
            if len(self.calls) == 1:
                return []
            return [
                RetrievedQuestion(
                    id="rag-evidence",
                    content="RAG Query Rewriting 查询改写与检索边界",
                    metadata={"parent_id": "rag-doc", "chunk_type": "answer"},
                    score=1.0,
                    source="hybrid",
                )
            ]

    class FakeLLM:
        calls = 0

        async def chat_tool_call(self, **_kwargs):
            self.calls += 1
            return {
                "name": "retrieve_knowledge",
                "arguments": {
                    "query": "RAG Query Rewriting 查询改写",
                    "target_id": target.target_id,
                    "top_k": 8,
                    "reason_summary": "首轮证据不足，改写查询后再次检索。",
                },
            }

    retriever = SequenceRetriever()

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.agents.job_assessment.settings.assessment_autonomous_loop_enabled", True)
    monkeypatch.setattr("app.agents.job_assessment.get_llm_client", lambda: FakeLLM())
    monkeypatch.setattr("app.agents.job_assessment.save_preparation_case", noop)
    monkeypatch.setattr("app.agents.job_assessment.get_preparation_service", lambda: _FakePreparation(case))
    monkeypatch.setattr("app.agents.job_assessment.get_retriever", lambda: retriever)
    monkeypatch.setattr("app.agents.job_assessment.get_knowledge_service", lambda: _FakeKnowledge())

    result = await JobAssessmentAgent(store).run(
        ToolContext("agent-op", "u1", case.case_id, 1, [target])
    )

    assert result["complete"] is True
    assert len(retriever.calls) == 2
    assert result["trajectory"][-1]["decision_source"] == "llm_tool_call"
    assert result["trajectory"][-1]["tool"] == "retrieve_knowledge"
    assert case.evidence_assessments[target.target_id].verdict == "sufficient"

    class FakePlannerLLM:
        async def chat_structured(self, **_kwargs):
            return GeneratedQuestionDraft(
                content="RAG 首轮召回不足时如何执行查询改写？",
                skill_tags=["RAG"],
                difficulty="medium",
                reference_points=["查询改写"],
            )

    async def no_extra_retrieval(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        "app.agents.question_planner._retrieve_reference_questions",
        no_extra_retrieval,
    )
    monkeypatch.setattr(
        "app.agents.question_planner.get_llm_client", lambda: FakePlannerLLM()
    )
    planning_state = {
        "session_id": "closure-session",
        "user_id": "u1",
        "preparation_case_id": case.case_id,
        "skill_matrix": SkillMatrix(
                position_title="AI Engineer",
                experience_level="mid",
                skills=[
                    SkillItem(
                        name="RAG", category="framework", weight=1, is_required=True
                    )
                ],
        ),
        "assessment_targets": case.targets,
        "evidence_chunks": case.evidence_chunks,
        "evidence_assessments": case.evidence_assessments,
        "interview_mode": "practice",
        "max_follow_ups": 2,
    }
    planned = await plan_questions(planning_state)
    assert {slot.target_id for slot in planned["question_blueprint"].slots} == {
        target.target_id
    }
    artifact_store = QuestionArtifactStore()
    monkeypatch.setattr(
        "app.agents.question_planner.get_operation_store", lambda: store
    )
    monkeypatch.setattr(
        "app.agents.question_planner.get_question_artifact_store",
        lambda: artifact_store,
    )
    monkeypatch.setattr(
        "app.services.question_artifact_store.session_store_enabled", lambda: False
    )
    generated = await generate_current_question({**planning_state, **planned})
    assert generated["question_plan"][0].target_id == target.target_id
    assert generated["question_plan"][0].validation_status == "supported"
    coverage = generated["question_plan_coverage"]["practice"]
    assert coverage.evidence_eligible_target_ids == [target.target_id]
    assert coverage.covered_target_ids == [target.target_id]


@pytest.mark.asyncio
async def test_autonomous_gap_rejects_public_search_for_personal_claim_then_hitl(monkeypatch):
    target = _technical_target("target-personal", "RAG").model_copy(
        update={
            "gap_kind": "personal_experience",
            "resume_signal": "claimed",
            "sources": ["job_requirement", "project_claim"],
            "citations": [],
        }
    )
    case = PreparationCase(
        case_id="case-agent",
        user_id="u1",
        jd_text="需要 RAG 项目经验",
        resume_text="参与 RAG 项目",
        status="ready",
        targets=[target],
    )
    store = await _operation_store(monkeypatch)
    retriever = _FakeRetriever()

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat_tool_call(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "name": "search_public_knowledge",
                    "arguments": {
                        "query": "搜索某人的 RAG 项目",
                        "target_id": target.target_id,
                        "reason_summary": "尝试错误的公网补查。",
                    },
                }
            return {
                "name": "request_clarification",
                "arguments": {
                    "target_id": target.target_id,
                    "question": "请补充你的具体职责和工程取舍。",
                    "missing_information": "个人职责和工程取舍",
                    "reason_summary": "个人经历只能由用户补充。",
                },
            }

    llm = FakeLLM()

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.agents.job_assessment.settings.assessment_autonomous_loop_enabled", True)
    monkeypatch.setattr("app.agents.job_assessment.settings.public_search_enabled", True)
    monkeypatch.setattr("app.agents.job_assessment.get_llm_client", lambda: llm)
    monkeypatch.setattr("app.agents.job_assessment.save_preparation_case", noop)
    monkeypatch.setattr("app.agents.job_assessment.get_preparation_service", lambda: _FakePreparation(case))
    monkeypatch.setattr("app.agents.job_assessment.get_retriever", lambda: retriever)
    monkeypatch.setattr("app.agents.job_assessment.get_knowledge_service", lambda: _FakeKnowledge())

    result = await JobAssessmentAgent(store).run(
        ToolContext("agent-op", "u1", case.case_id, 1, [target])
    )

    assert result["status"] == "waiting_input"
    assert llm.calls == 2
    assert result["trajectory"][-2]["tool"] == "invalid_decision"
    assert result["trajectory"][-1]["tool"] == "request_clarification"

    case.clarifications = [
        item.model_copy(
            update={
                "status": "answered",
                "answer": (
                    "我负责设计 RAG 查询改写和混合召回，使用 Recall@5 "
                    "评测并根据延迟指标选择了并行检索方案。"
                ),
            }
        )
        for item in case.clarifications
    ]
    case.status = "ready"
    resumed = await JobAssessmentAgent(store).run(
        ToolContext("agent-op", "u1", case.case_id, 1, [target])
    )
    assert resumed["complete"] is True
    assert llm.calls == 2
    assert case.evidence_assessments[target.target_id].verdict == "sufficient"
    blueprint = _build_blueprint(
        {
            "preparation_case_id": case.case_id,
            "assessment_targets": case.targets,
            "evidence_assessments": case.evidence_assessments,
            "skill_matrix": _matrix(),
        },
        namespace="practice",
        round_number=1,
        planned_count=1,
    )
    assert blueprint.slots[0].target_id == target.target_id


@pytest.mark.asyncio
async def test_autonomous_recovery_reuses_persisted_model_decision(monkeypatch):
    target = _technical_target("target-unknown", "NebulaFluxXYZ")
    case = PreparationCase(
        case_id="case-agent",
        user_id="u1",
        jd_text="需要 NebulaFluxXYZ",
        status="ready",
        targets=[target],
    )
    store = await _operation_store(monkeypatch)
    retriever = _FakeRetriever(empty=True)

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat_tool_call(self, **_kwargs):
            self.calls += 1
            return {
                "name": "finish_assessment",
                "arguments": {
                    "target_ids": [target.target_id],
                    "summary": "本地证据不足，停止并保留缺口。",
                    "reason_summary": "没有新增证据来源。",
                },
            }

    llm = FakeLLM()

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.agents.job_assessment.settings.assessment_autonomous_loop_enabled", True)
    monkeypatch.setattr("app.agents.job_assessment.get_llm_client", lambda: llm)
    monkeypatch.setattr("app.agents.job_assessment.save_preparation_case", noop)
    monkeypatch.setattr("app.agents.job_assessment.get_preparation_service", lambda: _FakePreparation(case))
    monkeypatch.setattr("app.agents.job_assessment.get_retriever", lambda: retriever)
    monkeypatch.setattr("app.agents.job_assessment.get_knowledge_service", lambda: _FakeKnowledge())
    context = ToolContext("agent-op", "u1", case.case_id, 1, [target])

    first = await JobAssessmentAgent(store).run(context)
    case.evidence_chunks = {}
    case.evidence_assessments = {}
    case.targets = [target.model_copy(update={"status": "pending"})]
    second = await JobAssessmentAgent(store).run(context)

    assert first["complete"] is False
    assert second["complete"] is False
    assert llm.calls == 1
    assert second["trajectory"][-1]["decision_reused"] is True


def _technical_target(target_id: str, scope: str) -> AssessmentTarget:
    return AssessmentTarget(
        target_id=target_id,
        case_id="case-agent",
        material_revision=1,
        requirement_id=f"req-{target_id}",
        skill_id=scope.casefold(),
        scope=scope,
        importance=1,
        sources=["job_requirement"],
        verification_goal=f"验证 {scope} 原理、实现与边界",
        resume_signal="unmentioned",
    )


@pytest.mark.asyncio
async def test_agent_completes_multiple_targets_and_reuses_persisted_tools(
    monkeypatch,
):
    targets = [_technical_target("target-rag", "RAG"), _technical_target("target-redis", "Redis")]
    case = PreparationCase(
        case_id="case-agent",
        user_id="u1",
        jd_text="需要 RAG 与 Redis",
        status="ready",
        targets=targets,
    )
    store = await _operation_store(monkeypatch, decisions=8)
    retriever = _FakeRetriever()

    async def save(case_arg):
        case.__dict__.update(case_arg.__dict__)

    monkeypatch.setattr("app.agents.job_assessment.save_preparation_case", save)
    monkeypatch.setattr(
        "app.agents.job_assessment.get_preparation_service",
        lambda: _FakePreparation(case),
    )
    monkeypatch.setattr("app.agents.job_assessment.get_retriever", lambda: retriever)
    monkeypatch.setattr("app.agents.job_assessment.get_knowledge_service", lambda: _FakeKnowledge())
    context = ToolContext("agent-op", "u1", case.case_id, 1, targets)

    first = await JobAssessmentAgent(store).run(context)
    calls_after_first = len(retriever.calls)
    case.evidence_chunks = {}
    case.evidence_assessments = {}
    case.targets = [item.model_copy(update={"status": "pending"}) for item in targets]
    second = await JobAssessmentAgent(store).run(context)

    assert first["complete"] is True
    assert {item["target_id"] for item in first["targets"]} == {
        "target-rag",
        "target-redis",
    }
    assert all(item["complete"] for item in first["targets"])
    assert second["complete"] is True
    assert len(retriever.calls) == calls_after_first
    assert second["trajectory"]
    assert all(item["reused"] for item in second["trajectory"])


@pytest.mark.asyncio
async def test_agent_does_not_mark_partial_as_complete_and_stops_on_budget(monkeypatch):
    targets = [_technical_target("target-rag", "RAG"), _technical_target("target-redis", "Redis")]
    case = PreparationCase(
        case_id="case-agent", user_id="u1", jd_text="需要 RAG", status="ready", targets=targets
    )
    store = await _operation_store(monkeypatch, decisions=1)
    retriever = _FakeRetriever(empty=True)

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.agents.job_assessment.save_preparation_case", noop)
    monkeypatch.setattr(
        "app.agents.job_assessment.get_preparation_service", lambda: _FakePreparation(case)
    )
    monkeypatch.setattr("app.agents.job_assessment.get_retriever", lambda: retriever)
    monkeypatch.setattr("app.agents.job_assessment.get_knowledge_service", lambda: _FakeKnowledge())

    result = await JobAssessmentAgent(store).run(
        ToolContext("agent-op", "u1", case.case_id, 1, targets)
    )

    assert result["complete"] is False
    assert result["result"]["completed_target_ids"] == []
    assert all(item["complete"] is False for item in result["targets"])
    assert any(item.get("stop_reason") == "budget_exhausted" for item in result["targets"])


@pytest.mark.asyncio
async def test_personal_experience_gap_requests_user_clarification(monkeypatch):
    target = _technical_target("target-project", "RAG").model_copy(
        update={
            "gap_kind": "personal_experience",
            "resume_signal": "claimed",
            "sources": ["job_requirement", "project_claim"],
            "citations": [
                MaterialCitation(
                    material_type="resume",
                    revision=1,
                    quote="参与 RAG 项目",
                )
            ],
        }
    )
    case = PreparationCase(
        case_id="case-agent", user_id="u1", jd_text="需要 RAG", status="ready", targets=[target]
    )
    store = await _operation_store(monkeypatch, decisions=4)

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.agents.job_assessment.save_preparation_case", noop)
    monkeypatch.setattr(
        "app.agents.job_assessment.get_preparation_service", lambda: _FakePreparation(case)
    )
    monkeypatch.setattr("app.agents.job_assessment.get_retriever", lambda: _FakeRetriever(empty=True))
    monkeypatch.setattr("app.agents.job_assessment.get_knowledge_service", lambda: _FakeKnowledge())

    result = await JobAssessmentAgent(store).run(
        ToolContext("agent-op", "u1", case.case_id, 1, [target])
    )

    assert result["status"] == "waiting_input"
    assert result["complete"] is False
    assert result["targets"][0]["stop_reason"] == "user_clarification_required"
    assert len(case.clarifications) == 1
    assert case.clarifications[0].clarification_id.endswith(target.target_id)
