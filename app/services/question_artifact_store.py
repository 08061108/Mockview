"""Conditional persistence for versioned progressive-question slot artifacts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.services.database import get_pool, session_store_enabled


@dataclass
class SlotArtifact:
    session_id: str
    blueprint_id: str
    slot_id: str
    input_version: str
    current_attempt: int
    status: str
    question: dict[str, Any] | None = None
    error: str | None = None
    terminated: bool = False


class QuestionArtifactStore:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str, str], SlotArtifact] = {}
        self._terminated_sessions: set[str] = set()
        self._lock = asyncio.Lock()

    async def begin(
        self,
        *,
        session_id: str,
        blueprint_id: str,
        slot_id: str,
        input_version: str,
        attempt: int,
    ) -> SlotArtifact:
        key = (session_id, blueprint_id, slot_id)
        async with self._lock:
            if session_id in self._terminated_sessions:
                raise RuntimeError("interview session is terminated")
            current = self._items.get(key)
            if current:
                if current.terminated:
                    raise RuntimeError("interview session is terminated")
                if current.input_version != input_version:
                    raise RuntimeError("question slot input version is no longer valid")
                if attempt < current.current_attempt:
                    raise RuntimeError("stale question-generation attempt")
                if attempt > current.current_attempt:
                    candidate = SlotArtifact(
                        session_id=session_id,
                        blueprint_id=blueprint_id,
                        slot_id=slot_id,
                        input_version=input_version,
                        current_attempt=attempt,
                        status="generating",
                    )
                else:
                    candidate = current
            else:
                candidate = SlotArtifact(
                    session_id=session_id,
                    blueprint_id=blueprint_id,
                    slot_id=slot_id,
                    input_version=input_version,
                    current_attempt=attempt,
                    status="generating",
                )
            persisted = await self._begin_postgres(candidate)
            database_active = session_store_enabled() and get_pool() is not None
            if database_active and persisted is None:
                raise RuntimeError("interview session is no longer valid")
            if persisted:
                # The database is authoritative after restart or a concurrent
                # termination. Cache it before raising so a failed begin cannot
                # leave a speculative attempt in memory.
                self._items[key] = persisted
                if persisted.terminated:
                    raise RuntimeError("interview session is terminated")
                if persisted.input_version != input_version:
                    raise RuntimeError("question slot input version is no longer valid")
                if persisted.current_attempt != attempt:
                    raise RuntimeError("stale question-generation attempt")
                candidate = persisted
            self._items[key] = candidate
            return candidate

    async def get(self, session_id: str, blueprint_id: str, slot_id: str) -> SlotArtifact | None:
        key = (session_id, blueprint_id, slot_id)
        async with self._lock:
            current = self._items.get(key)
            if current:
                return current
            if not session_store_enabled() or get_pool() is None:
                return None
            async with get_pool().connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT session_id, blueprint_id, slot_id, input_version,
                           current_attempt, status, question, error, terminated
                    FROM interview_question_slots
                    WHERE session_id=%s AND blueprint_id=%s AND slot_id=%s
                    """,
                    (session_id, blueprint_id, slot_id),
                )
                row = await cursor.fetchone()
            if not row:
                return None
            current = SlotArtifact(**dict(row))
            self._items[key] = current
            return current

    async def commit(self, artifact: SlotArtifact, question: dict[str, Any]) -> SlotArtifact:
        key = (artifact.session_id, artifact.blueprint_id, artifact.slot_id)
        async with self._lock:
            current = self._items.get(key)
            if not current:
                raise RuntimeError("question slot was not registered")
            if current.terminated or current.input_version != artifact.input_version:
                raise RuntimeError("question slot input is no longer valid")
            if current.current_attempt != artifact.current_attempt:
                raise RuntimeError("stale question result rejected")
            if current.status == "ready":
                if current.question == question:
                    return current
                raise RuntimeError("question slot already contains a different artifact")
            persisted = await self._commit_postgres(current, question)
            updated = persisted or SlotArtifact(
                **{**current.__dict__, "status": "ready", "question": question, "error": None}
            )
            self._items[key] = updated
            return updated

    async def fail(
        self,
        artifact: SlotArtifact,
        error: str,
        *,
        retryable: bool = True,
    ) -> None:
        key = (artifact.session_id, artifact.blueprint_id, artifact.slot_id)
        async with self._lock:
            current = self._items.get(key)
            if not current or current.current_attempt != artifact.current_attempt:
                return
            status = "failed_retryable" if retryable else "failed"
            persisted = await self._fail_postgres(current, error, status=status)
            if persisted is False:
                return
            self._items[key] = SlotArtifact(
                **{**current.__dict__, "status": status, "error": error}
            )

    async def terminate_session(self, session_id: str) -> None:
        async with self._lock:
            self._terminated_sessions.add(session_id)
            for key, item in list(self._items.items()):
                if item.session_id == session_id:
                    self._items[key] = SlotArtifact(**{**item.__dict__, "terminated": True})
            if session_store_enabled() and get_pool() is not None:
                async with get_pool().connection() as conn:
                    await conn.execute(
                        """
                        UPDATE interview_question_slots
                        SET terminated=TRUE, updated_at=NOW()
                        WHERE session_id=%s
                        """,
                        (session_id,),
                    )

    async def _begin_postgres(self, item: SlotArtifact) -> SlotArtifact | None:
        if not session_store_enabled() or get_pool() is None:
            return None
        async with get_pool().connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO interview_question_slots (
                        session_id, blueprint_id, slot_id, input_version,
                        current_attempt, status
                    )
                    SELECT %s,%s,%s,%s,%s,'generating'
                    FROM interview_sessions
                    WHERE session_id=%s
                      AND status NOT IN ('completed','cancelled')
                    ON CONFLICT (session_id, blueprint_id, slot_id) DO UPDATE SET
                        current_attempt=EXCLUDED.current_attempt,
                        status='generating', question=NULL, error=NULL, updated_at=NOW()
                    WHERE interview_question_slots.current_attempt < EXCLUDED.current_attempt
                      AND interview_question_slots.input_version = EXCLUDED.input_version
                      AND interview_question_slots.terminated = FALSE
                    """,
                    (
                        item.session_id,
                        item.blueprint_id,
                        item.slot_id,
                        item.input_version,
                        item.current_attempt,
                        item.session_id,
                    ),
                )
                cursor = await conn.execute(
                    """
                    SELECT session_id, blueprint_id, slot_id, input_version,
                           current_attempt, status, question, error, terminated
                    FROM interview_question_slots
                    WHERE session_id=%s AND blueprint_id=%s AND slot_id=%s
                    FOR UPDATE
                    """,
                    (item.session_id, item.blueprint_id, item.slot_id),
                )
                row = await cursor.fetchone()
        return SlotArtifact(**dict(row)) if row else None

    async def _commit_postgres(
        self, item: SlotArtifact, question: dict[str, Any]
    ) -> SlotArtifact | None:
        if not session_store_enabled() or get_pool() is None:
            return None
        from psycopg.types.json import Jsonb

        async with get_pool().connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    UPDATE interview_question_slots
                    SET status='ready', question=%s, error=NULL, updated_at=NOW()
                    WHERE session_id=%s AND blueprint_id=%s AND slot_id=%s
                      AND input_version=%s AND current_attempt=%s
                      AND terminated=FALSE
                      AND (
                          status='generating'
                          OR (status='ready' AND question=%s)
                      )
                    RETURNING session_id, blueprint_id, slot_id, input_version,
                              current_attempt, status, question, error, terminated
                    """,
                    (
                        Jsonb(question),
                        item.session_id,
                        item.blueprint_id,
                        item.slot_id,
                        item.input_version,
                        item.current_attempt,
                        Jsonb(question),
                    ),
                )
                row = await cursor.fetchone()
        if not row:
            raise RuntimeError("stale or terminated question result rejected")
        return SlotArtifact(**dict(row))

    async def _fail_postgres(
        self,
        item: SlotArtifact,
        error: str,
        *,
        status: str = "failed_retryable",
    ) -> bool | None:
        if not session_store_enabled() or get_pool() is None:
            return None
        async with get_pool().connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE interview_question_slots
                SET status=%s, error=%s, updated_at=NOW()
                WHERE session_id=%s AND blueprint_id=%s AND slot_id=%s
                  AND input_version=%s AND current_attempt=%s
                  AND terminated=FALSE AND status='generating'
                RETURNING slot_id
                """,
                (
                    status,
                    error,
                    item.session_id,
                    item.blueprint_id,
                    item.slot_id,
                    item.input_version,
                    item.current_attempt,
                ),
            )
            return await cursor.fetchone() is not None


_store: QuestionArtifactStore | None = None


def get_question_artifact_store() -> QuestionArtifactStore:
    global _store
    if _store is None:
        _store = QuestionArtifactStore()
    return _store


def reset_question_artifact_store() -> None:
    global _store
    _store = None
