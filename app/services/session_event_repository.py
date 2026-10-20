"""Durable, replayable non-token WebSocket events for interview sessions."""

from __future__ import annotations

from typing import Any

from app.services.database import get_pool, session_store_enabled


async def latest_session_event_sequence(session_id: str) -> int:
    if not session_store_enabled() or get_pool() is None:
        return 0
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT sequence FROM interview_session_event_sequences WHERE session_id=%s",
            (session_id,),
        )
        row = await cursor.fetchone()
    return int(row["sequence"] if row else 0)


async def save_session_event(
    session_id: str,
    sequence: int,
    event_id: str,
    event: dict[str, Any],
) -> None:
    if not session_store_enabled() or get_pool() is None:
        return
    from psycopg.types.json import Jsonb

    async with get_pool().connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO interview_session_event_sequences (session_id, sequence)
                VALUES (%s,%s)
                ON CONFLICT (session_id) DO UPDATE SET
                    sequence=GREATEST(interview_session_event_sequences.sequence, EXCLUDED.sequence),
                    updated_at=NOW()
                """,
                (session_id, sequence),
            )
            await conn.execute(
                """
                INSERT INTO interview_session_events (
                    session_id, sequence, event_id, event_type, payload
                ) VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (
                    session_id,
                    sequence,
                    event_id,
                    str(event.get("type") or event.get("event") or "event"),
                    Jsonb(event),
                ),
            )


async def list_session_events(
    session_id: str,
    *,
    after_sequence: int = 0,
    limit: int = 256,
) -> list[dict[str, Any]]:
    if not session_store_enabled() or get_pool() is None:
        return []
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            """
            SELECT sequence, event_id, payload
            FROM interview_session_events
            WHERE session_id=%s AND sequence>%s
            ORDER BY sequence
            LIMIT %s
            """,
            (session_id, max(after_sequence, 0), min(max(limit, 1), 1000)),
        )
        rows = await cursor.fetchall()
    return [
        {
            "sequence": int(row["sequence"]),
            "event_id": row["event_id"],
            "event": row["payload"],
        }
        for row in rows
    ]
