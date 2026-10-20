"""Hybrid retriever combining vector search and BM25 keyword search with RRF fusion."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from rank_bm25 import BM25Okapi

from app.config import settings
from app.errors import PublicAppError
from app.rag.embeddings import EmbeddingService, get_embedding_service
from app.rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedQuestion:
    """A single retrieved question with fusion score."""

    id: str
    content: str
    metadata: dict
    score: float
    source: str = ""  # "vector" | "bm25" | "hybrid"


CandidateStatus = Literal["found", "conclusive_empty"]
RetrievalMode = Literal["hybrid", "vector_only", "bm25_only"]
BranchStatus = Literal["ok", "empty", "failed", "skipped"]


@dataclass
class RetrievalOutcome:
    items: list[RetrievedQuestion]
    candidate_status: CandidateStatus
    mode: RetrievalMode
    vector_status: BranchStatus
    bm25_status: BranchStatus
    input_fingerprint: str
    query_count: int
    duration_ms: int
    index_version: str
    failure_codes: tuple[str, ...] = ()
    cache_hit: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)


class RetrievalUnavailableError(PublicAppError):
    def __init__(self, *, reason: str = "retrieval branch unavailable") -> None:
        super().__init__(
            code="RETRIEVAL_UNAVAILABLE",
            message="检索服务暂时不可用，请稍后重试。",
            status_code=503,
            retryable=True,
        )
        self.reason = reason


def _tokenize(text: str) -> list[str]:
    """Preserve legacy English ranking while adding Chinese and special tech tokens."""
    # The original tokenizer is retained as the base so the frozen English-heavy
    # retrieval set does not move merely because punctuation handling changed.
    tokens = [item.casefold() for item in re.split(r"\W+", text) if item]
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(sequence[index : index + 2] for index in range(max(0, len(sequence) - 1)))
    # Preserve identifiers that the legacy split loses entirely (C++, C#, .NET).
    for item in re.findall(r"(?:[A-Za-z][A-Za-z0-9_]*\+\+|[A-Za-z][A-Za-z0-9_]*#|\.NET)", text):
        normalized = item.casefold()
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens


class HybridRetriever:
    """Two-stage retriever: vector + BM25, fused with Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1 / (k + rank_i(d))
    where k is a constant (default 60) and rank_i is the rank from source i.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._store = vector_store or get_vector_store()
        self._embedding = embedding_service or get_embedding_service()
        self._rrf_k = rrf_k

        self._bm25: BM25Okapi | None = None
        self._bm25_legacy: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []
        self._bm25_built = False
        self._bm25_index_version: str | None = None
        self._empty_outcomes: dict[str, RetrievalOutcome] = {}
        self._query_embedding_cache: dict[str, list[float]] = {}
        self.embedding_batch_calls = 0

    def build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the vector store."""
        all_docs = self._store.get_all_documents()
        if not all_docs["ids"]:
            logger.warning("No documents in vector store, BM25 index empty")
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_built = False
            return

        self._bm25_docs = []
        corpus: list[list[str]] = []
        legacy_corpus: list[list[str]] = []

        for i, doc_id in enumerate(all_docs["ids"]):
            doc_text = all_docs["documents"][i]
            meta = all_docs["metadatas"][i]
            self._bm25_docs.append({
                "id": doc_id,
                "document": doc_text,
                "metadata": meta,
            })
            corpus.append(_tokenize(doc_text))
            legacy_corpus.append(_legacy_tokenize(doc_text))

        self._bm25 = BM25Okapi(corpus)
        self._bm25_legacy = BM25Okapi(legacy_corpus)
        self._bm25_built = True
        self._bm25_index_version = self._index_version()
        logger.info("BM25 index built with %d documents", len(corpus))

    def _ensure_bm25_index(self) -> None:
        """Build BM25 once on first production retrieval."""
        if self._bm25_built and self._bm25_index_version != self._index_version():
            self.invalidate_bm25_index()
        if self._bm25_built or self._store.count == 0:
            return
        self.build_bm25_index()

    def invalidate_bm25_index(self) -> None:
        self._bm25 = None
        self._bm25_legacy = None
        self._bm25_docs = []
        self._bm25_built = False
        self._bm25_index_version = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        where: dict | None = None,
    ) -> list[RetrievedQuestion]:
        """Compatibility wrapper returning only retrieved items.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            vector_weight: Weight for vector search ranks in RRF.
            bm25_weight: Weight for BM25 ranks in RRF.
            where: Optional ChromaDB metadata filter.
        """
        outcome = await self.retrieve_outcome(
            query,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            where=where,
        )
        return outcome.items

    async def retrieve_outcome(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        where: dict | None = None,
        *,
        fingerprint_context: dict[str, Any] | None = None,
    ) -> RetrievalOutcome:
        return await self.retrieve_multi_outcome(
            [query],
            top_k_per_query=top_k,
            final_top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            where=where,
            fingerprint_context=fingerprint_context,
        )

    async def retrieve_multi(
        self,
        queries: list[Any],
        top_k_per_query: int = 8,
        final_top_k: int = 15,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        where: dict | None = None,
    ) -> list[RetrievedQuestion]:
        outcome = await self.retrieve_multi_outcome(
            queries,
            top_k_per_query=top_k_per_query,
            final_top_k=final_top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            where=where,
        )
        return outcome.items

    async def retrieve_multi_outcome(
        self,
        queries: list[Any],
        top_k_per_query: int = 8,
        final_top_k: int = 15,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        where: dict | None = None,
        *,
        fingerprint_context: dict[str, Any] | None = None,
    ) -> RetrievalOutcome:
        started = time.perf_counter()
        timings_ms = self._blank_timings()
        normalized: list[tuple[str, float]] = []
        seen: set[str] = set()
        for query in queries:
            text, priority = self._query_text_and_priority(query)
            text = " ".join(text.split())
            if text and text.casefold() not in seen:
                normalized.append((text, priority))
                seen.add(text.casefold())
        fingerprint = self._fingerprint(normalized, where, fingerprint_context)
        cached = self._empty_outcomes.get(fingerprint)
        if cached is not None:
            timings_ms["total"] = self._elapsed_ms(started)
            return replace(
                cached,
                duration_ms=round(timings_ms["total"]),
                cache_hit=True,
                timings_ms=timings_ms,
            )
        if not normalized:
            outcome = self._empty_outcome(fingerprint, 0, started)
            self._empty_outcomes[fingerprint] = outcome
            return outcome

        bm25_build_started = time.perf_counter()
        try:
            await asyncio.to_thread(self._ensure_bm25_index)
            bm25_build_failed = False
        except Exception as exc:
            bm25_build_failed = True
            logger.warning("BM25 index build failed type=%s", type(exc).__name__)
        timings_ms["bm25_build"] = self._elapsed_ms(bm25_build_started)

        fetch_n = top_k_per_query * 3

        async def run_bm25_branch() -> list[Any]:
            if bm25_build_failed:
                return [RuntimeError("bm25 build unavailable") for _ in normalized]
            branch_started = time.perf_counter()
            tasks = [
                asyncio.to_thread(self._bm25_search, text, fetch_n, where)
                for text, _priority in normalized
            ]
            values = await asyncio.gather(*tasks, return_exceptions=True)
            timings_ms["bm25_search"] = self._elapsed_ms(branch_started)
            return list(values)

        async def run_vector_branch() -> list[Any]:
            embedding_started = time.perf_counter()
            try:
                compatible, incompatibility = self._vector_index_compatible()
                if compatible:
                    embeddings = await self._embed_queries_isolated(
                        [text for text, _ in normalized]
                    )
                    expected_dimension = self._index_manifest().get(
                        "embedding_dimension"
                    )
                    if expected_dimension:
                        embeddings = [
                            value
                            if isinstance(value, BaseException)
                            or len(value) == int(expected_dimension)
                            else RuntimeError(
                                "vector index embedding dimension mismatch"
                            )
                            for value in embeddings
                        ]
                else:
                    embeddings = [RuntimeError(incompatibility) for _ in normalized]
            except Exception as exc:
                embeddings = [exc for _ in normalized]
            timings_ms["embedding"] = self._elapsed_ms(embedding_started)

            vector_tasks: list[asyncio.Future] = []
            vector_indexes: list[int] = []
            for index, embedding in enumerate(embeddings):
                if isinstance(embedding, BaseException):
                    continue
                vector_indexes.append(index)
                vector_tasks.append(
                    asyncio.to_thread(
                        self._store.query_by_embedding,
                        embedding,
                        fetch_n,
                        where,
                    )
                )
            vector_raw: list[Any] = [
                item
                if isinstance(item, BaseException)
                else RuntimeError("embedding unavailable")
                for item in embeddings
            ]
            if not vector_tasks:
                return vector_raw
            vector_started = time.perf_counter()
            values = await asyncio.gather(*vector_tasks, return_exceptions=True)
            timings_ms["vector_search"] = self._elapsed_ms(vector_started)
            for index, value in zip(vector_indexes, values):
                vector_raw[index] = value
            return vector_raw

        bm25_raw, vector_raw = await asyncio.gather(
            run_bm25_branch(),
            run_vector_branch(),
        )

        fusion_started = time.perf_counter()
        fused_multi: dict[str, RetrievedQuestion] = {}
        vector_found = bm25_found = False
        vector_failed = False
        bm25_failed = bm25_build_failed
        failures: set[str] = set()
        for index, (_text, priority) in enumerate(normalized):
            vector_value = vector_raw[index]
            bm25_value = bm25_raw[index]
            if isinstance(vector_value, BaseException):
                vector_failed = True
                failures.add(self._failure_code(vector_value, "vector"))
                vector_results: list[dict] = []
            else:
                vector_results = vector_value
                vector_found = vector_found or bool(vector_results)
            if isinstance(bm25_value, BaseException):
                bm25_failed = True
                failures.add(self._failure_code(bm25_value, "bm25"))
                bm25_results: list[dict] = []
            else:
                bm25_results = bm25_value
                bm25_found = bm25_found or bool(bm25_results)
            fused = self._rrf_fuse(
                vector_results, bm25_results, vector_weight, bm25_weight
            )
            ranked = sorted(
                fused.values(), key=lambda value: value.score, reverse=True
            )[:top_k_per_query]
            for item in ranked:
                weighted = item.score * priority
                if item.id not in fused_multi:
                    fused_multi[item.id] = replace(item, score=weighted)
                else:
                    fused_multi[item.id].score += weighted
                    sources = set(fused_multi[item.id].source.split("+"))
                    sources.update(item.source.split("+"))
                    fused_multi[item.id].source = "+".join(
                        sorted(value for value in sources if value)
                    )

        items = sorted(
            fused_multi.values(), key=lambda value: value.score, reverse=True
        )[:final_top_k]
        timings_ms["fusion"] = self._elapsed_ms(fusion_started)
        if not items and (vector_failed or bm25_failed):
            timings_ms["total"] = self._elapsed_ms(started)
            logger.warning(
                "RAG retrieval unavailable queries=%d timings_ms=%s failures=%s",
                len(normalized),
                timings_ms,
                sorted(failures),
            )
            raise RetrievalUnavailableError(
                reason=",".join(sorted(failures))
                or "partial branch failure with no candidates"
            )
        vector_status: BranchStatus = (
            "failed" if vector_failed else "ok" if vector_found else "empty"
        )
        bm25_status: BranchStatus = (
            "failed" if bm25_failed else "ok" if bm25_found else "empty"
        )
        mode: RetrievalMode = (
            "hybrid" if vector_found and bm25_found
            else "vector_only" if vector_found
            else "bm25_only"
        )
        timings_ms["total"] = self._elapsed_ms(started)
        outcome = RetrievalOutcome(
            items=items,
            candidate_status="found" if items else "conclusive_empty",
            mode=mode,
            vector_status=vector_status,
            bm25_status=bm25_status,
            input_fingerprint=fingerprint,
            query_count=len(normalized),
            duration_ms=round(timings_ms["total"]),
            index_version=self._index_version(),
            failure_codes=tuple(sorted(failures)),
            cache_hit=False,
            timings_ms=timings_ms,
        )
        if outcome.candidate_status == "conclusive_empty":
            self._empty_outcomes[fingerprint] = outcome
        logger.info(
            "RAG retrieval completed queries=%d results=%d mode=%s vector=%s bm25=%s timings_ms=%s",
            len(normalized), len(items), mode, vector_status, bm25_status, timings_ms,
        )
        return outcome

    async def _embed_queries_isolated(self, texts: list[str]) -> list[list[float] | BaseException]:
        size = max(1, settings.embedding_query_batch_size)
        semaphore = asyncio.Semaphore(max(1, settings.embedding_query_max_concurrency))
        model = str(getattr(self._embedding, "model", settings.embedding_model))
        results: list[list[float] | BaseException | None] = [None] * len(texts)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        for index, value in enumerate(texts):
            key = f"{model}\0{' '.join(value.split()).casefold()}"
            cached = self._query_embedding_cache.get(key)
            if cached is not None:
                results[index] = cached
            else:
                missing_indexes.append(index)
                missing_texts.append(value)

        async def run_batch(batch: list[str]) -> list[list[float] | BaseException]:
            error: Exception | None = None
            async with semaphore:
                try:
                    self.embedding_batch_calls += 1
                    if hasattr(self._embedding, "embed_texts"):
                        return list(await self._embedding.embed_texts(batch))
                    return [await self._embedding.embed_query(item) for item in batch]
                except Exception as exc:
                    error = exc
            assert error is not None
            if len(batch) > 1 and self._is_batch_too_large(error):
                midpoint = len(batch) // 2
                left, right = await asyncio.gather(
                    run_batch(batch[:midpoint]), run_batch(batch[midpoint:])
                )
                return [*left, *right]
            logger.warning("Embedding batch failed size=%d type=%s", len(batch), type(error).__name__)
            return [error for _ in batch]

        batches = [missing_texts[index:index + size] for index in range(0, len(missing_texts), size)]
        values = await asyncio.gather(*(run_batch(batch) for batch in batches))
        flattened = [item for batch in values for item in batch]
        for index, text, value in zip(missing_indexes, missing_texts, flattened):
            results[index] = value
            if not isinstance(value, BaseException):
                key = f"{model}\0{' '.join(text.split()).casefold()}"
                self._query_embedding_cache[key] = value
        return [
            item if item is not None else RuntimeError("embedding result missing")
            for item in results
        ]

    @staticmethod
    def _is_batch_too_large(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        text = str(exc).casefold()
        return status == 413 or (
            status == 400 and any(marker in text for marker in ("batch", "too many", "input limit", "maximum input"))
        )

    def _fingerprint(
        self,
        queries: list[tuple[str, float]],
        where: dict | None,
        context: dict[str, Any] | None,
    ) -> str:
        payload = {
            "queries": queries,
            "where": where or {},
            "context": context or {},
            "index_version": self._index_version(),
            "embedding_model": getattr(self._embedding, "model", settings.embedding_model),
            "strategy": settings.rag_query_strategy_version,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _empty_outcome(self, fingerprint: str, query_count: int, started: float) -> RetrievalOutcome:
        timings_ms = self._blank_timings()
        timings_ms["total"] = self._elapsed_ms(started)
        return RetrievalOutcome(
            items=[], candidate_status="conclusive_empty", mode="bm25_only",
            vector_status="empty", bm25_status="empty", input_fingerprint=fingerprint,
            query_count=query_count, duration_ms=round(timings_ms["total"]),
            index_version=self._index_version(),
            timings_ms=timings_ms,
        )

    @staticmethod
    def _blank_timings() -> dict[str, float]:
        return {
            "bm25_build": 0.0,
            "bm25_search": 0.0,
            "embedding": 0.0,
            "vector_search": 0.0,
            "fusion": 0.0,
            "total": 0.0,
        }

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    def _index_version(self) -> str:
        return str(getattr(self._store, "index_version", f"legacy_unknown:{self._store.count}"))

    def _index_manifest(self) -> dict[str, Any]:
        value = getattr(self._store, "index_manifest", {})
        return value if isinstance(value, dict) else {}

    def _vector_index_compatible(self) -> tuple[bool, str]:
        manifest = self._index_manifest()
        if not manifest or manifest.get("status") == "legacy_unknown":
            return True, ""
        expected_model = str(manifest.get("embedding_model") or "")
        actual_model = str(getattr(self._embedding, "model", settings.embedding_model))
        if expected_model and expected_model != actual_model:
            return False, "vector index embedding model mismatch"
        return True, ""

    @staticmethod
    def _failure_code(exc: BaseException, branch: str) -> str:
        return f"{branch}_{type(exc).__name__.lower()}"

    async def retrieve_vector_only(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[RetrievedQuestion]:
        """Pure vector search (fallback when BM25 index is empty)."""
        query_embedding = await self._embedding.embed_query(query)
        results = await asyncio.to_thread(
            self._store.query_by_embedding,
            query_embedding,
            top_k,
            where,
        )
        return [
            RetrievedQuestion(
                id=r["id"],
                content=r["document"],
                metadata=r["metadata"],
                score=1.0 - r["distance"],
                source="vector",
            )
            for r in results
        ]

    def _bm25_search(
        self,
        query: str,
        n_results: int,
        where: dict | None = None,
    ) -> list[dict]:
        """BM25 keyword search."""
        if self._bm25 is None or not self._bm25_docs:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        use_legacy = not re.search(r"[\u4e00-\u9fff]", query)
        index = self._bm25_legacy if use_legacy and self._bm25_legacy is not None else self._bm25
        query_tokens = _legacy_tokenize(query) if use_legacy else tokens
        scores = index.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                break
            doc = self._bm25_docs[idx]
            if where and not _metadata_matches_where(doc["metadata"], where):
                continue
            results.append({
                "id": doc["id"],
                "document": doc["document"],
                "metadata": doc["metadata"],
                "bm25_score": float(scores[idx]),
            })
            if len(results) >= n_results:
                break
        return results

    def _rrf_fuse(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        vector_weight: float,
        bm25_weight: float,
    ) -> dict[str, RetrievedQuestion]:
        """Reciprocal Rank Fusion across two result lists."""
        k = self._rrf_k
        fused: dict[str, RetrievedQuestion] = {}

        for rank, item in enumerate(vector_results):
            doc_id = item["id"]
            rrf_score = vector_weight / (k + rank + 1)
            if doc_id not in fused:
                fused[doc_id] = RetrievedQuestion(
                    id=doc_id,
                    content=item["document"],
                    metadata=item["metadata"],
                    score=0.0,
                    source="vector",
                )
            fused[doc_id].score += rrf_score

        for rank, item in enumerate(bm25_results):
            doc_id = item["id"]
            rrf_score = bm25_weight / (k + rank + 1)
            if doc_id not in fused:
                fused[doc_id] = RetrievedQuestion(
                    id=doc_id,
                    content=item["document"],
                    metadata=item["metadata"],
                    score=0.0,
                    source="bm25",
                )
                fused[doc_id].score += rrf_score
            else:
                fused[doc_id].score += rrf_score
                fused[doc_id].source = "hybrid"

        return fused

    @staticmethod
    def _query_text_and_priority(query: Any) -> tuple[str, float]:
        if hasattr(query, "text"):
            return str(query.text), float(getattr(query, "priority", 1.0))
        return str(query), 1.0


def _metadata_matches_where(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """Small subset of Chroma-style metadata filters for BM25 parity."""
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$nin" in expected and actual in expected["$nin"]:
                return False
            continue
        if actual != expected:
            return False
    return True


def _legacy_tokenize(text: str) -> list[str]:
    """Frozen P0 tokenizer used for non-Chinese queries to prevent regression."""
    return [item.casefold() for item in re.split(r"\W+", text) if item]


_default_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = HybridRetriever()
    return _default_retriever
