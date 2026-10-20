from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models.jd import SkillItem, SkillMatrix
from app.models.preparation import AssessmentTarget
from app.rag.postprocess import hydrate_parent_results
from app.rag.query_builder import build_retrieval_queries
from app.rag.reranker import rerank_results
from app.rag.retriever import (
    HybridRetriever,
    RetrievalUnavailableError,
    RetrievedQuestion,
)
from scripts.evaluate_rag_stabilization import quality_compare


class FakeStore:
    def __init__(self, *, vector_results=None, vector_error: Exception | None = None):
        self.documents = [
            {
                "id": "redis-1",
                "document": "Redis cache aside distributed lock",
                "metadata": {"parent_id": "redis", "category": "Redis"},
            },
            {
                "id": "python-1",
                "document": "Python asyncio event loop coroutine",
                "metadata": {"parent_id": "python", "category": "Python"},
            },
            {
                "id": "mysql-1",
                "document": "MySQL MVCC transaction index",
                "metadata": {"parent_id": "mysql", "category": "Database"},
            },
        ]
        self.vector_results = list(vector_results or [])
        self.vector_error = vector_error
        self.index_version = "index-v1"
        self.index_manifest = {"status": "ready", "index_version": "index-v1", "embedding_model": "test"}

    @property
    def count(self):
        return len(self.documents)

    def get_all_documents(self):
        return {
            "ids": [item["id"] for item in self.documents],
            "documents": [item["document"] for item in self.documents],
            "metadatas": [item["metadata"] for item in self.documents],
        }

    def query_by_embedding(self, *_args):
        if self.vector_error:
            raise self.vector_error
        return self.vector_results


class FakeEmbedding:
    model = "test"

    def __init__(self):
        self.batches: list[list[str]] = []

    async def embed_texts(self, texts):
        self.batches.append(list(texts))
        return [[0.1, 0.2] for _ in texts]


@pytest.mark.asyncio
async def test_vector_failure_falls_back_to_bm25_when_bm25_has_candidates():
    retriever = HybridRetriever(
        vector_store=FakeStore(vector_error=ConnectionError("down")),
        embedding_service=FakeEmbedding(),
    )

    outcome = await retriever.retrieve_outcome("Redis distributed lock")

    assert outcome.mode == "bm25_only"
    assert outcome.vector_status == "failed"
    assert outcome.bm25_status == "ok"
    assert [item.id for item in outcome.items] == ["redis-1"]


@pytest.mark.asyncio
async def test_failed_vector_and_empty_bm25_is_unavailable_not_evidence_missing():
    retriever = HybridRetriever(
        vector_store=FakeStore(vector_error=ConnectionError("down")),
        embedding_service=FakeEmbedding(),
    )

    with pytest.raises(RetrievalUnavailableError) as raised:
        await retriever.retrieve_outcome("Kafka exactly once")

    assert raised.value.code == "RETRIEVAL_UNAVAILABLE"
    assert raised.value.retryable is True


@pytest.mark.asyncio
async def test_two_successful_empty_branches_are_conclusive_and_cached():
    embedding = FakeEmbedding()
    retriever = HybridRetriever(
        vector_store=FakeStore(vector_results=[]),
        embedding_service=embedding,
    )

    first = await retriever.retrieve_outcome("Kafka exactly once")
    second = await retriever.retrieve_outcome("Kafka exactly once")

    assert first.candidate_status == "conclusive_empty"
    assert first.vector_status == "empty"
    assert first.bm25_status == "empty"
    assert second.input_fingerprint == first.input_fingerprint
    assert second.cache_hit is True
    assert second.timings_ms["total"] > 0
    assert all(
        second.timings_ms[key] == 0
        for key in (
            "bm25_build",
            "bm25_search",
            "embedding",
            "vector_search",
            "fusion",
        )
    )
    assert len(embedding.batches) == 1


@pytest.mark.asyncio
async def test_bm25_and_vector_pipeline_overlap_and_total_is_wall_clock():
    bm25_started = threading.Event()
    release_bm25 = threading.Event()
    vector_started = threading.Event()

    class CoordinatedStore(FakeStore):
        def query_by_embedding(self, *_args):
            vector_started.set()
            return []

    class CoordinatedEmbedding(FakeEmbedding):
        async def embed_texts(self, texts):
            self.batches.append(list(texts))
            assert await asyncio.to_thread(bm25_started.wait, 1)
            await asyncio.sleep(0.02)
            return [[0.1, 0.2] for _ in texts]

    retriever = HybridRetriever(CoordinatedStore(), CoordinatedEmbedding())

    def blocked_bm25(*_args):
        bm25_started.set()
        assert release_bm25.wait(1)
        return []

    retriever._bm25_search = blocked_bm25  # type: ignore[method-assign]
    task = asyncio.create_task(retriever.retrieve_outcome("no matching evidence"))
    assert await asyncio.to_thread(vector_started.wait, 1)
    assert task.done() is False
    release_bm25.set()
    outcome = await asyncio.wait_for(task, 1)

    timings = outcome.timings_ms
    assert timings["bm25_search"] > 0
    assert timings["embedding"] > 0
    assert timings["vector_search"] > 0
    assert timings["total"] > 0
    assert timings["total"] < sum(
        timings[key]
        for key in (
            "bm25_build",
            "bm25_search",
            "embedding",
            "vector_search",
            "fusion",
        )
    )


def test_query_expansion_is_bounded_and_independent_from_ranking():
    matrix = SkillMatrix(
        position_title="AI Engineer",
        experience_level="mid",
        skills=[
            SkillItem(name="RAG", category="framework", weight=1, is_required=True),
            SkillItem(name="Redis", category="domain", weight=0.7, is_required=False),
        ],
    )
    targets = [
        AssessmentTarget(
            target_id=f"target-{index}",
            case_id="case-1",
            material_revision=1,
            requirement_id=f"req-{index}",
            skill_id=f"skill-{index}",
            scope=f"scope-{index}",
            importance=importance,
            sources=["job_requirement"],
            verification_goal=f"goal-{index}",
        )
        for index, importance in enumerate((0.3, 1.0, 0.8, 0.5))
    ]
    state = {"skill_matrix": matrix, "assessment_targets": targets}

    legacy = build_retrieval_queries(state, "practice", strategy="legacy")
    expanded = build_retrieval_queries(
        state,
        "practice",
        strategy="target_expansion_v1",
    )
    candidate = Settings(
        _env_file=None,
        rag_query_strategy="target_expansion_v1",
        rag_ranking_strategy="legacy",
    )

    assert [item.text for item in expanded[: len(legacy)]] == [
        item.text for item in legacy
    ]
    assert [item.text for item in expanded[len(legacy) :]] == [
        "scope-1 goal-1",
        "scope-2 goal-2",
        "scope-3 goal-3",
    ]
    assert candidate.rag_query_strategy == "target_expansion_v1"
    assert candidate.rag_ranking_strategy == "legacy"


def test_deprecated_expanded_ranking_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rag_ranking_strategy="expanded")


@pytest.mark.asyncio
async def test_bm25_rebuilds_when_index_version_changes():
    store = FakeStore(vector_error=ConnectionError("down"))
    retriever = HybridRetriever(store, FakeEmbedding())
    await retriever.retrieve_outcome("Redis distributed lock")
    store.documents.append({
        "id": "kafka-1",
        "document": "Kafka idempotent consumer ordering",
        "metadata": {"parent_id": "kafka", "category": "Message Queue"},
    })
    store.index_version = "index-v2"
    store.index_manifest["index_version"] = "index-v2"

    outcome = await retriever.retrieve_outcome("Kafka idempotent consumer")

    assert outcome.index_version == "index-v2"
    assert outcome.items[0].id == "kafka-1"


@pytest.mark.asyncio
async def test_embedding_model_mismatch_uses_bm25_without_vector_query():
    store = FakeStore(vector_error=AssertionError("vector query must be skipped"))
    store.index_manifest["embedding_model"] = "different-model"
    retriever = HybridRetriever(store, FakeEmbedding())

    outcome = await retriever.retrieve_outcome("Redis distributed lock")

    assert outcome.mode == "bm25_only"
    assert outcome.vector_status == "failed"
    assert outcome.items


@pytest.mark.asyncio
async def test_embedding_batches_are_bounded_and_isolated(monkeypatch):
    class PartiallyFailingEmbedding(FakeEmbedding):
        async def embed_texts(self, texts):
            self.batches.append(list(texts))
            if "q0" in texts:
                raise ConnectionError("one batch failed")
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("app.rag.retriever.settings.embedding_query_batch_size", 10)
    embedding = PartiallyFailingEmbedding()
    retriever = HybridRetriever(FakeStore(), embedding)

    values = await retriever._embed_queries_isolated([f"q{index}" for index in range(12)])

    assert sorted(len(batch) for batch in embedding.batches) == [2, 10]
    assert all(isinstance(item, BaseException) for item in values[:10])
    assert all(not isinstance(item, BaseException) for item in values[10:])


@pytest.mark.asyncio
async def test_oversized_embedding_batch_is_split_without_deadlock(monkeypatch):
    class TooLargeError(Exception):
        status_code = 413

    class LimitedEmbedding(FakeEmbedding):
        async def embed_texts(self, texts):
            self.batches.append(list(texts))
            if len(texts) > 2:
                raise TooLargeError("payload too large")
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("app.rag.retriever.settings.embedding_query_batch_size", 10)
    monkeypatch.setattr("app.rag.retriever.settings.embedding_query_max_concurrency", 1)
    embedding = LimitedEmbedding()
    retriever = HybridRetriever(FakeStore(), embedding)

    values = await retriever._embed_queries_isolated([f"split-{index}" for index in range(5)])

    assert all(not isinstance(item, BaseException) for item in values)
    assert any(len(batch) > 2 for batch in embedding.batches)
    assert sum(len(batch) for batch in embedding.batches if len(batch) <= 2) >= 5


def _result(item_id: str, parent_id: str, score: float, chunk_type: str) -> RetrievedQuestion:
    return RetrievedQuestion(
        id=item_id,
        content=item_id,
        metadata={
            "parent_id": parent_id,
            "question_id": parent_id,
            "chunk_id": item_id,
            "chunk_type": chunk_type,
            "skill_tags": "[]",
        },
        score=score,
        source="hybrid",
    )


def test_bounded_parent_hydration_does_not_reward_chunk_count_without_limit():
    crowded = [
        _result(f"crowded-{index}", "crowded", 0.01, f"answer-{index}")
        for index in range(6)
    ]
    precise = [_result("precise-1", "precise", 0.03, "question_stem")]

    legacy = hydrate_parent_results([*crowded, *precise], top_k=2)
    bounded = hydrate_parent_results([*crowded, *precise], top_k=2, strategy="bounded")

    assert legacy[0].id == "crowded"
    assert bounded[0].id == "precise"


def test_normalized_business_reranking_is_bounded():
    state = {
        "skill_matrix": SkillMatrix(
            position_title="Backend",
            experience_level="mid",
            skills=[SkillItem(name="Redis", category="domain", weight=1.0, is_required=True)],
        )
    }
    strong = _result("strong", "strong", 1.0, "answer_point")
    weak = replace(
        _result("weak", "weak", 0.01, "question_stem"),
        metadata={
            **_result("weak", "weak", 0.01, "question_stem").metadata,
            "content": "Redis",
            "skill_tags": '["Redis"]',
        },
    )

    ranked = rerank_results(state, [strong, weak], "practice", strategy="normalized")

    assert ranked[0].id == "strong"
    assert ranked[0].score >= 0.85
    assert ranked[1].score <= 0.16


def test_quality_selection_is_lexicographic_and_ignores_equal_quality_latency():
    baseline = {"recall@5": 0.5, "mrr@10": 0.8, "ndcg@10": 0.7, "p95_ms": 100}

    assert quality_compare(
        {**baseline, "recall@5": 0.6, "mrr@10": 0.1}, baseline
    ) == 1
    assert quality_compare(
        {**baseline, "mrr@10": 0.81, "ndcg@10": 0.1}, baseline
    ) == 1
    assert quality_compare(
        {**baseline, "ndcg@10": 0.71}, baseline
    ) == 1
    assert quality_compare(
        {**baseline, "p95_ms": 1}, baseline
    ) == 0
