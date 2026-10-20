from __future__ import annotations

import pytest

from app.agents.evidence_pipeline import assess_evidence_deterministic
from app.agents.evidence_pipeline import validate_questions_semantic
from app.agents.question_planner import _finalize_plan
from app.models.evidence import EvidenceChunk, QuestionValidation, SearchSnapshot
from app.models.preparation import AssessmentTarget
from app.models.question import QuestionItem, QuestionPlan
from app.services.public_knowledge import (
    build_public_technical_query,
    sanitize_public_query,
    snapshots_to_evidence,
)
from app.llm.transport import build_model_http_client


def _target(*, gap_kind: str = "technical_knowledge") -> AssessmentTarget:
    return AssessmentTarget(
        target_id="target-rag",
        case_id="case-synthetic",
        material_revision=1,
        requirement_id="req-rag",
        skill_id="rag",
        scope="RAG 检索",
        importance=1.0,
        sources=["job_requirement", *(["project_claim"] if gap_kind == "personal_experience" else [])],
        verification_goal="验证查询改写、召回融合与故障边界",
        gap_kind=gap_kind,
        resume_signal="claimed" if gap_kind == "personal_experience" else "unmentioned",
    )


def test_public_query_is_target_bound_and_rejects_personal_experience():
    query = build_public_technical_query(_target(), ["technical_knowledge"])
    assert "RAG 检索" in query
    assert "technical_knowledge" in query

    with pytest.raises(ValueError, match="personal experience"):
        build_public_technical_query(_target(gap_kind="personal_experience"), [])


def test_public_query_sanitizes_contact_and_url():
    value = sanitize_public_query(
        "RAG alice@example.com 13800138000 https://private.example/path"
    )
    assert "alice@example.com" not in value
    assert "13800138000" not in value
    assert "private.example" not in value


def test_public_snapshots_are_bounded_and_remain_unreviewed(monkeypatch):
    monkeypatch.setattr("app.services.public_knowledge.settings.public_snapshot_max_chars", 20)
    monkeypatch.setattr("app.services.public_knowledge.settings.public_snapshot_total_max_chars", 25)
    snapshots = [
        SearchSnapshot(
            query="q",
            url=f"https://example.com/{index}",
            title="title",
            content="x" * 100,
            fetched_at="2026-10-17T00:00:00+00:00",
            content_hash=str(index) * 64,
        )
        for index in (1, 2)
    ]
    chunks = snapshots_to_evidence(snapshots)
    assert sum(len(item.text) for item in chunks) == 25
    assert all(item.publication_status == "candidate" for item in chunks)


def test_bare_resume_mention_does_not_prove_personal_experience():
    target = _target(gap_kind="personal_experience")
    technical = EvidenceChunk(
        chunk_id="published-1",
        document_id="doc-1",
        text="RAG 检索需要查询改写与召回融合",
        source_type="question_bank",
        publication_status="published",
    )
    bare_claim = EvidenceChunk(
        chunk_id="private-1",
        document_id="case-synthetic",
        text="参与 RAG 项目",
        source_type="private_material",
        publication_status="private",
        scope="private",
    )
    detailed_claim = bare_claim.model_copy(
        update={
            "chunk_id": "private-2",
            "text": "我负责设计 RAG 查询改写和混合召回，优化后 p95 延迟降低 20%，并处理召回为空的降级边界。",
        }
    )

    before = assess_evidence_deterministic(target, [technical, bare_claim])
    after = assess_evidence_deterministic(target, [technical, detailed_claim])

    assert before.verdict == "partial"
    assert "personal_experience" in before.missing_aspects
    assert after.verdict == "sufficient"


@pytest.mark.asyncio
async def test_semantic_shadow_cannot_change_formal_question(monkeypatch):
    target = _target()
    chunk = EvidenceChunk(
        chunk_id="published-rag",
        document_id="doc-rag",
        text="RAG 查询改写能够处理低质量查询。",
        source_type="question_bank",
        publication_status="published",
    )
    question = QuestionItem(
        id=1,
        target_id=target.target_id,
        content="RAG 查询质量不足时如何处理？",
        skill_tags=["RAG"],
        difficulty="medium",
        reference_points=["查询改写"],
        evidence_chunk_ids=[chunk.chunk_id],
    )

    async def contradictory_shadow(*_args, **_kwargs):
        return {
            question.ensure_stable_id("practice").question_id: QuestionValidation(
                question_id=question.ensure_stable_id("practice").question_id,
                target_match="uncertain",
                rubric_support="uncertain",
                answer_relevance="uncertain",
                unsupported_points=["查询改写"],
                verdict="contradicted",
            )
        }

    monkeypatch.setattr(
        "app.agents.question_planner.validate_questions_semantic",
        contradictory_shadow,
    )
    monkeypatch.setattr(
        "app.agents.question_planner.settings.question_semantic_validation_enabled",
        False,
    )
    monkeypatch.setattr(
        "app.agents.question_planner.settings.question_semantic_shadow_enabled",
        True,
    )

    plan, coverage = await _finalize_plan(
        QuestionPlan(questions=[question]),
        "practice",
        state={
            "assessment_targets": [target],
            "evidence_chunks": {target.target_id: [chunk]},
        },
        retrieved=[],
    )

    assert coverage.status == "complete"
    assert plan.questions[0].validation_status == "supported"
    assert plan.questions[0].reference_points == ["查询改写"]
    assert plan.questions[0].rubric_citations == {"查询改写": [chunk.chunk_id]}


def test_explicit_model_proxy_is_scoped_and_ignores_ambient_proxy(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.llm.transport.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        "app.llm.transport.settings.model_proxy_url",
        "http://host.docker.internal:7897",
    )
    monkeypatch.setattr("app.llm.transport.settings.model_force_ipv4", True)
    monkeypatch.setattr("app.llm.transport.settings.model_trust_env", True)

    build_model_http_client()

    assert captured["proxy"] == "http://host.docker.internal:7897"
    assert captured["transport"] is None
    assert captured["trust_env"] is False


@pytest.mark.asyncio
async def test_semantic_validator_cannot_claim_supported_with_empty_citation(monkeypatch):
    target = _target()
    chunk = EvidenceChunk(
        chunk_id="published-rag",
        document_id="doc-rag",
        text="RAG Query Rewriting 改写查询。",
        source_type="question_bank",
        publication_status="published",
    )
    question = QuestionItem(
        id=1,
        question_id="q-shadow",
        target_id=target.target_id,
        content="RAG 如何改写查询？",
        skill_tags=["RAG"],
        difficulty="medium",
        reference_points=["Query Rewriting", "不存在的评分点"],
        evidence_chunk_ids=[chunk.chunk_id],
    )

    class FakeLLM:
        async def chat_structured(self, **_kwargs):
            from app.models.evidence import QuestionValidationBatch

            return QuestionValidationBatch(
                validations=[
                    QuestionValidation(
                        question_id="q-shadow",
                        target_match="supported",
                        rubric_support="supported",
                        answer_relevance="supported",
                        citations={
                            "Query Rewriting": [chunk.chunk_id],
                            "不存在的评分点": [],
                        },
                        verdict="supported",
                    )
                ]
            )

    monkeypatch.setattr("app.agents.evidence_pipeline.get_llm_client", lambda: FakeLLM())
    result = await validate_questions_semantic([question], [target], [chunk])

    assert result["q-shadow"].verdict == "uncertain"
    assert result["q-shadow"].unsupported_points == ["不存在的评分点"]
