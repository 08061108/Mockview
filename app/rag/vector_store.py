"""ChromaDB vector store wrapper for the interview question bank."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "interview_questions"


class VectorStore:
    """Thin wrapper around ChromaDB with persistent storage."""

    def __init__(self, persist_dir: str | None = None) -> None:
        persist_path = persist_dir or settings.chroma_persist_dir
        self._persist_path = Path(persist_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._persist_path / "rag_index_manifest.json"

        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection '%s' has %d documents",
            COLLECTION_NAME,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    @property
    def index_manifest(self) -> dict:
        """Return a persisted index contract without making legacy indexes unusable."""
        try:
            value = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {
                "index_version": f"legacy_unknown:{self.count}",
                "status": "legacy_unknown",
                "chunk_count": self.count,
            }

    @property
    def index_version(self) -> str:
        return str(self.index_manifest.get("index_version") or f"legacy_unknown:{self.count}")

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with pre-computed embeddings."""
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._write_manifest(ids, documents, metadatas, embeddings)
        logger.info("Upserted %d documents into '%s'", len(ids), COLLECTION_NAME)

    def _write_manifest(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        digest = hashlib.sha256()
        stored = self.get_all_documents()
        stored_rows = sorted(
            zip(stored.get("ids", []), stored.get("documents", []), stored.get("metadatas", [])),
            key=lambda item: item[0],
        )
        for item_id, document, _metadata in stored_rows:
            digest.update(item_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document.encode("utf-8"))
            digest.update(b"\0")
        corpus_hash = digest.hexdigest()
        stored_metadatas = [row[2] for row in stored_rows]
        chunk_strategies = sorted({
            str(item.get("chunk_strategy"))
            for item in stored_metadatas
            if item.get("chunk_strategy")
        })
        payload = {
            "schema_version": 1,
            "index_version": corpus_hash[:16],
            "corpus_sha256": corpus_hash,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": len(embeddings[0]) if embeddings else None,
            "document_count": len({str(item.get("parent_id") or item.get("question_id") or "") for item in stored_metadatas}),
            "chunk_count": self.count,
            "chunk_strategies": chunk_strategies,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = self._manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self._manifest_path)

    def query_by_embedding(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Query by embedding vector, return ranked results."""
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        items: list[dict] = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return items

    def query_by_text(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Query using ChromaDB's built-in embedding (if configured)."""
        kwargs: dict = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        items: list[dict] = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return items

    def get_all_documents(self) -> dict:
        """Retrieve all documents (for BM25 index building)."""
        return self._collection.get(include=["documents", "metadatas"])

    def delete_collection(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._manifest_path.unlink(missing_ok=True)
        logger.info("Collection '%s' reset", COLLECTION_NAME)


_default_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
