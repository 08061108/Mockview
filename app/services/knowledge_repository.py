"""Persistence for knowledge documents, releases, and the active pointer."""

from __future__ import annotations

from app.models.knowledge import KnowledgeDocument, KnowledgeRelease
from app.services.database import get_pool, session_store_enabled


async def save_document(item: KnowledgeDocument) -> None:
    if not session_store_enabled() or get_pool() is None:
        return
    from psycopg.types.json import Jsonb

    async with get_pool().connection() as conn:
        await conn.execute(
            """
            INSERT INTO knowledge_documents (
                document_id, version, owner_user_id, scope, source_type, source_url,
                title, content, content_hash, status, metadata, created_at, updated_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (document_id, version) DO UPDATE SET
                status=EXCLUDED.status,
                title=EXCLUDED.title,
                content=EXCLUDED.content,
                metadata=EXCLUDED.metadata,
                updated_at=EXCLUDED.updated_at
            """,
            (
                item.document_id, item.version, item.owner_user_id, item.scope,
                item.source_type, item.source_url, item.title, item.content,
                item.content_hash, item.status, Jsonb(item.metadata),
                item.created_at, item.updated_at,
            ),
        )


async def save_release(item: KnowledgeRelease, *, activate: bool = False) -> None:
    if not session_store_enabled() or get_pool() is None:
        return
    from psycopg.types.json import Jsonb

    async with get_pool().connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO knowledge_releases (
                    release_id, status, document_versions, chroma_collection,
                    bm25_snapshot, embedding_model, embedding_dimension,
                    tokenization_version, chunking_version, error, created_at, activated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (release_id) DO UPDATE SET
                    status=EXCLUDED.status,
                    embedding_dimension=EXCLUDED.embedding_dimension,
                    error=EXCLUDED.error,
                    activated_at=EXCLUDED.activated_at
                """,
                (
                    item.release_id, item.status, Jsonb(item.document_versions),
                    item.chroma_collection, item.bm25_snapshot, item.embedding_model,
                    item.embedding_dimension, item.tokenization_version,
                    item.chunking_version, item.error, item.created_at, item.activated_at,
                ),
            )
            if activate:
                await conn.execute(
                    """
                    UPDATE knowledge_releases
                    SET status='ready'
                    WHERE status='active' AND release_id<>%s
                    """,
                    (item.release_id,),
                )
                await conn.execute(
                    """
                    INSERT INTO knowledge_release_pointer (singleton, release_id)
                    VALUES (TRUE, %s)
                    ON CONFLICT (singleton) DO UPDATE SET
                        release_id=EXCLUDED.release_id,
                        updated_at=NOW()
                    """,
                    (item.release_id,),
                )


async def load_active_release() -> KnowledgeRelease | None:
    if not session_store_enabled() or get_pool() is None:
        return None
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            """
            SELECT r.* FROM knowledge_release_pointer p
            JOIN knowledge_releases r ON r.release_id = p.release_id
            WHERE p.singleton=TRUE
            """
        )
        row = await cursor.fetchone()
    return KnowledgeRelease.model_validate(dict(row)) if row else None


async def load_release(release_id: str) -> KnowledgeRelease | None:
    if not session_store_enabled() or get_pool() is None:
        return None
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM knowledge_releases WHERE release_id=%s", (release_id,)
        )
        row = await cursor.fetchone()
    return KnowledgeRelease.model_validate(dict(row)) if row else None


async def list_releases() -> list[KnowledgeRelease]:
    if not session_store_enabled() or get_pool() is None:
        return []
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM knowledge_releases ORDER BY created_at, release_id"
        )
        rows = await cursor.fetchall()
    return [KnowledgeRelease.model_validate(dict(row)) for row in rows]


async def list_referenced_release_ids() -> set[str]:
    if not session_store_enabled() or get_pool() is None:
        return set()
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            """
            SELECT DISTINCT knowledge_release_id
            FROM preparation_cases
            WHERE knowledge_release_id IS NOT NULL
            """
        )
        rows = await cursor.fetchall()
    return {str(row["knowledge_release_id"]) for row in rows}


async def load_documents() -> list[KnowledgeDocument]:
    if not session_store_enabled() or get_pool() is None:
        return []
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            """
            SELECT DISTINCT ON (document_id) *
            FROM knowledge_documents
            ORDER BY document_id, version DESC
            """
        )
        rows = await cursor.fetchall()
    return [KnowledgeDocument.model_validate(dict(row)) for row in rows]
