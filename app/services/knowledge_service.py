"""Candidate review and atomic knowledge release publishing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import settings
from app.models.knowledge import KnowledgeCandidateCreate, KnowledgeDocument, KnowledgeRelease
from app.models.evidence import EvidenceChunk
from app.rag.embeddings import get_embedding_service
from app.rag.retriever import _tokenize
from app.services.knowledge_repository import (
    list_referenced_release_ids,
    list_releases,
    load_active_release,
    load_documents,
    load_release,
    save_release,
)
from app.services.database import session_store_enabled

logger = logging.getLogger(__name__)


class ReleaseBuilder(Protocol):
    async def build(
        self,
        release: KnowledgeRelease,
        documents: list[KnowledgeDocument],
        previous_release: KnowledgeRelease | None = None,
    ) -> tuple[int, int]: ...


class ChromaReleaseBuilder:
    """Builds a new collection and BM25 snapshot without touching the active one."""

    async def build(
        self,
        release: KnowledgeRelease,
        documents: list[KnowledgeDocument],
        previous_release: KnowledgeRelease | None = None,
    ) -> tuple[int, int]:
        texts = [item.content for item in documents]
        ids = [f"{item.document_id}:v{item.version}" for item in documents]
        reusable: dict[str, list[float]] = {}
        if (
            previous_release
            and previous_release.embedding_model == release.embedding_model
            and previous_release.tokenization_version == release.tokenization_version
            and previous_release.chunking_version == release.chunking_version
        ):
            try:
                reusable = await asyncio.to_thread(
                    self._load_reusable_embeddings, previous_release, ids
                )
            except Exception:
                logger.exception(
                    "Unable to reuse vectors from release %s; embedding all documents",
                    previous_release.release_id,
                )
        missing_indexes = [index for index, item_id in enumerate(ids) if item_id not in reusable]
        missing_vectors = await get_embedding_service().embed_texts(
            [texts[index] for index in missing_indexes]
        ) if missing_indexes else []
        generated = dict(zip((ids[index] for index in missing_indexes), missing_vectors))
        embeddings = [reusable.get(item_id) or generated[item_id] for item_id in ids]
        return await asyncio.to_thread(
            self._build_indices, release, documents, texts, embeddings
        )

    @staticmethod
    def _load_reusable_embeddings(
        previous_release: KnowledgeRelease,
        ids: list[str],
    ) -> dict[str, list[float]]:
        if not ids:
            return {}
        client = chromadb.PersistentClient(
            path=str(settings.resolve_project_path(settings.chroma_persist_dir)),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_collection(previous_release.chroma_collection)
        result = collection.get(ids=ids, include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None:
            return {}
        return {
            str(item_id): [float(value) for value in vector]
            for item_id, vector in zip(result.get("ids") or [], embeddings)
        }

    @staticmethod
    def _build_indices(
        release: KnowledgeRelease,
        documents: list[KnowledgeDocument],
        texts: list[str],
        embeddings: list[list[float]],
    ) -> tuple[int, int]:
        client = chromadb.PersistentClient(
            path=str(settings.resolve_project_path(settings.chroma_persist_dir)),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            release.chroma_collection, metadata={"hnsw:space": "cosine"}
        )
        if documents:
            collection.upsert(
                ids=[f"{item.document_id}:v{item.version}" for item in documents],
                documents=texts,
                metadatas=[
                    {
                        "document_id": item.document_id,
                        "version": item.version,
                        "source_type": item.source_type,
                        "source_url": item.source_url or "",
                    }
                    for item in documents
                ],
                embeddings=embeddings,
            )
        snapshot_path = settings.resolve_project_path(release.bm25_snapshot)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(
                [
                    {
                        "id": f"{item.document_id}:v{item.version}",
                        "document_id": item.document_id,
                        "version": item.version,
                        "text": item.content,
                    }
                    for item in documents
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        dimension = len(embeddings[0]) if embeddings else 0
        return collection.count(), dimension


class KnowledgeService:
    def __init__(self, builder: ReleaseBuilder | None = None) -> None:
        self.documents: dict[str, KnowledgeDocument] = {}
        self.releases: dict[str, KnowledgeRelease] = {}
        self.active_release_id: str | None = None
        self.builder = builder or ChromaReleaseBuilder()
        self._hydrated = False
        self._publish_lock = asyncio.Lock()

    async def hydrate(self) -> None:
        if self._hydrated:
            return
        for item in await load_documents():
            self.documents[item.document_id] = item
        active = await load_active_release()
        if active:
            self.releases[active.release_id] = active
            self.active_release_id = active.release_id
        self._hydrated = True

    def add_candidate(
        self, request: KnowledgeCandidateCreate, owner_user_id: str | None = None
    ) -> KnowledgeDocument:
        digest = hashlib.sha256(request.content.encode("utf-8")).hexdigest()
        existing = next(
            (item for item in self.documents.values() if item.content_hash == digest), None
        )
        if existing:
            return existing
        item = KnowledgeDocument(
            document_id=uuid4().hex,
            owner_user_id=owner_user_id,
            scope="private" if owner_user_id else "public",
            source_type=request.source_type,
            source_url=request.source_url,
            title=request.title,
            content=request.content,
            content_hash=digest,
            metadata=request.metadata,
        )
        self.documents[item.document_id] = item
        return item

    def list_candidates(self) -> list[KnowledgeDocument]:
        return sorted(
            [item for item in self.documents.values() if item.status == "candidate"],
            key=lambda item: item.created_at,
        )

    def review(self, document_id: str, status: str) -> KnowledgeDocument:
        if status not in {"approved", "rejected", "withdrawn"}:
            raise ValueError("invalid review status")
        item = self.documents[document_id]
        if item.scope != "public" and status == "approved":
            raise ValueError("private material cannot be published")
        updated = item.model_copy(
            update={"status": status, "updated_at": datetime.now(timezone.utc)}
        )
        self.documents[document_id] = updated
        return updated

    async def publish(self) -> KnowledgeRelease:
        """Serialize builds so only one release can swap the active pointer."""
        async with self._publish_lock:
            await self.hydrate()
            return await self._publish_locked()

    async def _publish_locked(self) -> KnowledgeRelease:
        approved = [
            item for item in self.documents.values()
            if item.status == "approved" and item.scope == "public"
        ]
        release_id = uuid4().hex
        release = KnowledgeRelease(
            release_id=release_id,
            document_versions={item.document_id: item.version for item in approved},
            chroma_collection=f"knowledge_release_{release_id}",
            bm25_snapshot=f"data/knowledge/releases/{release_id}.bm25.json",
            embedding_model=settings.embedding_model,
        )
        previous = self.active_release_id
        previous_release = self.releases.get(previous or "")
        self.releases[release_id] = release
        await save_release(release, activate=False)
        try:
            vector_count, dimension = await self.builder.build(
                release, approved, previous_release
            )
            snapshot = settings.resolve_project_path(release.bm25_snapshot)
            bm25_rows = json.loads(snapshot.read_text(encoding="utf-8"))
            expected_ids = {
                f"{item.document_id}:v{item.version}" for item in approved
            }
            bm25_ids = {str(item["id"]) for item in bm25_rows}
            if vector_count != len(expected_ids) or bm25_ids != expected_ids:
                raise RuntimeError("Chroma and BM25 manifests do not match")
            release = release.model_copy(
                update={
                    "status": "active",
                    "embedding_dimension": dimension,
                    "activated_at": datetime.now(timezone.utc),
                }
            )
            self.releases[release_id] = release
            self.active_release_id = release_id
            if previous and previous in self.releases:
                self.releases[previous] = self.releases[previous].model_copy(
                    update={"status": "ready"}
                )
            await save_release(release, activate=True)
            try:
                await self._cleanup_unreferenced_releases(
                    current_release_id=release_id,
                    previous_release_id=previous,
                )
            except Exception:
                logger.exception("Knowledge release cleanup failed after activation")
            return release
        except Exception as exc:
            failed = release.model_copy(
                update={"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
            self.releases[release_id] = failed
            self.active_release_id = previous
            await save_release(failed, activate=False)
            return failed

    async def _cleanup_unreferenced_releases(
        self,
        *,
        current_release_id: str,
        previous_release_id: str | None,
    ) -> None:
        """Retain current, previous, and every preparation-pinned release."""
        if not session_store_enabled():
            return
        keep = {current_release_id}
        if previous_release_id:
            keep.add(previous_release_id)
        keep.update(await list_referenced_release_ids())
        for release in await list_releases():
            if release.release_id in keep or release.status != "ready":
                continue
            await asyncio.to_thread(self._delete_release_assets, release)
            retired = release.model_copy(update={"status": "retired"})
            self.releases[release.release_id] = retired
            await save_release(retired, activate=False)

    @staticmethod
    def _delete_release_assets(release: KnowledgeRelease) -> None:
        if release.chroma_collection.startswith("knowledge_release_"):
            client = chromadb.PersistentClient(
                path=str(settings.resolve_project_path(settings.chroma_persist_dir)),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            try:
                client.delete_collection(release.chroma_collection)
            except Exception:
                logger.info(
                    "Knowledge collection already absent: %s",
                    release.chroma_collection,
                )
        snapshot = settings.resolve_project_path(release.bm25_snapshot).resolve()
        release_root = settings.resolve_project_path("data/knowledge/releases").resolve()
        if snapshot.is_relative_to(release_root) and snapshot.is_file():
            snapshot.unlink()

    def active_release(self) -> KnowledgeRelease | None:
        return self.releases.get(self.active_release_id or "")

    def allowed_document_ids(
        self, release: KnowledgeRelease | None = None
    ) -> set[str]:
        release = release or self.active_release()
        if not release:
            return set()
        return {
            document_id for document_id in release.document_versions
            if self.documents.get(document_id)
            and self.documents[document_id].status != "withdrawn"
        }

    async def search_candidates(self, query: str, top_k: int = 8) -> list[EvidenceChunk]:
        """Reuse review-pending public sources without publishing them globally."""
        await self.hydrate()
        documents = [
            item
            for item in self.documents.values()
            if item.scope == "public" and item.status == "candidate"
        ]
        if not query.strip() or not documents:
            return []

        def rank() -> list[tuple[KnowledgeDocument, float]]:
            tokenized = [_tokenize(item.content) for item in documents]
            query_tokens = _tokenize(query)
            bm25 = BM25Okapi(tokenized)
            scores = [float(value) for value in bm25.get_scores(_tokenize(query))]
            scores = [
                max(
                    score,
                    len(set(query_tokens).intersection(tokens))
                    / max(1, len(set(query_tokens))),
                )
                for score, tokens in zip(scores, tokenized)
            ]
            ordered = sorted(
                zip(documents, scores), key=lambda pair: pair[1], reverse=True
            )
            return [pair for pair in ordered if pair[1] > 0][:top_k]

        ranked = await asyncio.to_thread(rank)
        return [
            EvidenceChunk(
                chunk_id=f"candidate:{item.document_id}:v{item.version}",
                document_id=item.document_id,
                document_version=str(item.version),
                text=item.content,
                location=item.source_url or item.document_id,
                source_type="candidate",
                source_url=item.source_url,
                record_id=item.document_id,
                publication_status="candidate",
                scope="public",
                retrieval_sources=["candidate_bm25"],
                ranking={"bm25": score},
            )
            for item, score in ranked
        ]

    async def search_active(self, query: str, top_k: int = 8) -> list[EvidenceChunk]:
        """Search one immutable active release without mixing index versions."""
        await self.hydrate()
        release = self.active_release()
        if not release:
            return []
        return await self.search_release(release.release_id, query, top_k=top_k)

    async def search_release(
        self, release_id: str, query: str, top_k: int = 8
    ) -> list[EvidenceChunk]:
        """Search the exact release pinned by a preparation operation."""
        await self.hydrate()
        release = self.releases.get(release_id)
        if not release:
            release = await load_release(release_id)
            if release:
                self.releases[release_id] = release
        if not release or release.status not in {"active", "ready"} or not query.strip():
            return []
        allowed = self.allowed_document_ids(release)
        if not allowed:
            return []
        snapshot_path = settings.resolve_project_path(release.bm25_snapshot)
        if not snapshot_path.exists():
            raise RuntimeError(f"active BM25 snapshot is missing: {release.release_id}")
        rows = await asyncio.to_thread(self._load_snapshot, snapshot_path)
        rows = [item for item in rows if str(item.get("document_id")) in allowed]
        if not rows:
            return []
        expected_ids = {str(item["id"]) for item in rows}
        embedding = await get_embedding_service().embed_query(query)
        vector_raw, collection_count = await asyncio.to_thread(
            self._query_release_collection,
            release,
            embedding,
            top_k,
        )
        if collection_count != len(release.document_versions):
            raise RuntimeError(f"active Chroma release is incomplete: {release.release_id}")
        vector_items: list[tuple[str, str, dict, float]] = []
        for index, item_id in enumerate(vector_raw["ids"][0]):
            if item_id not in expected_ids:
                continue
            vector_items.append(
                (
                    item_id,
                    vector_raw["documents"][0][index],
                    vector_raw["metadatas"][0][index],
                    1.0 - float(vector_raw["distances"][0][index]),
                )
            )

        keyword_scores, keyword_order = await asyncio.to_thread(
            self._rank_keywords, rows, query
        )
        fused: dict[str, dict] = {}
        for rank, (item_id, text, metadata, raw_score) in enumerate(vector_items, 1):
            fused[item_id] = {
                "text": text,
                "metadata": metadata,
                "score": 0.5 / (60 + rank),
                "raw_vector_score": raw_score,
                "sources": ["vector"],
            }
        for rank, index in enumerate(keyword_order, 1):
            if keyword_scores[index] <= 0:
                break
            row = rows[index]
            item_id = str(row["id"])
            item = fused.setdefault(
                item_id,
                {
                    "text": str(row["text"]),
                    "metadata": {
                        "document_id": row["document_id"],
                        "version": row["version"],
                    },
                    "score": 0.0,
                    "sources": [],
                },
            )
            item["score"] += 0.5 / (60 + rank)
            item["sources"].append("bm25")

        chunks: list[EvidenceChunk] = []
        for item_id, item in sorted(
            fused.items(), key=lambda pair: pair[1]["score"], reverse=True
        )[:top_k]:
            metadata = item["metadata"] or {}
            document_id = str(metadata.get("document_id") or item_id.split(":v", 1)[0])
            document = self.documents.get(document_id)
            if document_id not in allowed or not document:
                continue
            chunks.append(
                EvidenceChunk(
                    chunk_id=item_id,
                    document_id=document_id,
                    document_version=str(metadata.get("version") or document.version),
                    release_id=release.release_id,
                    text=item["text"],
                    location=document.source_url or release.release_id,
                    source_type=(
                        "public_web" if document.source_type == "public_web" else "candidate"
                    ),
                    source_url=document.source_url,
                    record_id=document_id,
                    publication_status="published",
                    scope="public",
                    retrieval_sources=item["sources"],
                    ranking={"rrf": float(item["score"])},
                )
            )
        return chunks

    @staticmethod
    def _load_snapshot(path: Path) -> list[dict]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _query_release_collection(
        release: KnowledgeRelease,
        embedding: list[float],
        top_k: int,
    ) -> tuple[dict, int]:
        client = chromadb.PersistentClient(
            path=str(settings.resolve_project_path(settings.chroma_persist_dir)),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_collection(release.chroma_collection)
        count = collection.count()
        if count != len(release.document_versions):
            return {}, count
        raw = collection.query(
            query_embeddings=[embedding],
            n_results=min(max(top_k * 3, top_k), count),
            include=["documents", "metadatas", "distances"],
        )
        return raw, count

    @staticmethod
    def _rank_keywords(rows: list[dict], query: str) -> tuple[list[float], list[int]]:
        bm25 = BM25Okapi([_tokenize(str(item["text"])) for item in rows])
        scores = [float(value) for value in bm25.get_scores(_tokenize(query))]
        order = sorted(range(len(rows)), key=lambda index: scores[index], reverse=True)
        return scores, order


_service: KnowledgeService | None = None


def get_knowledge_service() -> KnowledgeService:
    global _service
    if _service is None:
        _service = KnowledgeService()
    return _service


def reset_knowledge_service(builder: ReleaseBuilder | None = None) -> None:
    global _service
    _service = KnowledgeService(builder)
