"""PostgreSQL persistence for resumable operations and tool-call results."""

from __future__ import annotations

from uuid import uuid4

from app.models.preparation import OperationRecord, ToolCallRecord
from app.services.database import get_pool, session_store_enabled


async def append_operation_event(
    event_id: str,
    operation_id: str,
    status: str,
    error: str | None,
    created_at,
) -> None:
    if not session_store_enabled() or get_pool() is None:
        return
    async with get_pool().connection() as conn:
        await conn.execute(
            """
            INSERT INTO agent_operation_events (
                event_id, operation_id, status, error, created_at
            ) VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (event_id) DO NOTHING
            """,
            (event_id, operation_id, status, error, created_at),
        )


async def list_operation_events(operation_id: str) -> list[dict]:
    if not session_store_enabled() or get_pool() is None:
        return []
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            """
            SELECT event_id, event_order, operation_id, status, error, created_at
            FROM agent_operation_events
            WHERE operation_id=%s
            ORDER BY event_order
            """,
            (operation_id,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def save_operation(item: OperationRecord) -> None:
    if not session_store_enabled() or get_pool() is None:
        return
    from psycopg.types.json import Jsonb

    async with get_pool().connection() as conn:
        await conn.execute(
            """
            INSERT INTO agent_operations (
                operation_id,user_id,case_id,kind,status,input_revision,
                graph_version,state_version,input_hash,result,error,error_details,budget,metrics,created_at,updated_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (operation_id) DO UPDATE SET
                status=EXCLUDED.status,result=EXCLUDED.result,error=EXCLUDED.error,
                error_details=EXCLUDED.error_details,
                budget=EXCLUDED.budget,metrics=EXCLUDED.metrics,
                updated_at=EXCLUDED.updated_at
            """,
            (
                item.operation_id,item.user_id,item.case_id,item.kind,item.status,
                item.input_revision,item.graph_version,item.state_version,item.input_hash,
                Jsonb(item.result) if item.result is not None else None,item.error,
                Jsonb(item.error_details) if item.error_details is not None else None,
                Jsonb(item.budget),Jsonb(item.metrics),item.created_at,item.updated_at,
            ),
        )


async def load_operation(operation_id: str) -> OperationRecord | None:
    if not session_store_enabled() or get_pool() is None:
        return None
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM agent_operations WHERE operation_id=%s", (operation_id,)
        )
        row = await cursor.fetchone()
    return OperationRecord.model_validate(dict(row)) if row else None


async def list_operations_for_case(case_id: str) -> list[OperationRecord]:
    if not session_store_enabled() or get_pool() is None:
        return []
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            """
            SELECT * FROM agent_operations
            WHERE case_id=%s
            ORDER BY created_at, operation_id
            """,
            (case_id,),
        )
        rows = await cursor.fetchall()
    return [OperationRecord.model_validate(dict(row)) for row in rows]


async def save_tool_call(item: ToolCallRecord) -> None:
    if not session_store_enabled() or get_pool() is None:
        return
    from psycopg.types.json import Jsonb

    async with get_pool().connection() as conn:
        await conn.execute(
            """
            INSERT INTO agent_tool_calls (
                tool_call_id,operation_id,tool_name,input_hash,input_summary,status,
                result,error,attempts,budget_counted,created_at,updated_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (tool_call_id) DO UPDATE SET
                status=EXCLUDED.status,result=EXCLUDED.result,error=EXCLUDED.error,
                attempts=EXCLUDED.attempts,budget_counted=EXCLUDED.budget_counted,
                updated_at=EXCLUDED.updated_at
            """,
            (
                item.tool_call_id,item.operation_id,item.tool_name,item.input_hash,
                Jsonb(item.input_summary),item.status,
                Jsonb(item.result) if item.result is not None else None,item.error,
                item.attempts,item.budget_counted,item.created_at,item.updated_at,
            ),
        )


async def load_tool_call(tool_call_id: str) -> ToolCallRecord | None:
    if not session_store_enabled() or get_pool() is None:
        return None
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM agent_tool_calls WHERE tool_call_id=%s", (tool_call_id,)
        )
        row = await cursor.fetchone()
    return ToolCallRecord.model_validate(dict(row)) if row else None


async def list_tool_calls(operation_id: str) -> list[ToolCallRecord]:
    """Load the durable tool trace for one operation in execution order."""
    if not session_store_enabled() or get_pool() is None:
        return []
    async with get_pool().connection() as conn:
        cursor = await conn.execute(
            """
            SELECT * FROM agent_tool_calls
            WHERE operation_id=%s
            ORDER BY created_at, tool_call_id
            """,
            (operation_id,),
        )
        rows = await cursor.fetchall()
    return [ToolCallRecord.model_validate(dict(row)) for row in rows]


async def mark_unfinished_operations_retryable() -> int:
    if not session_store_enabled() or get_pool() is None:
        return 0
    async with get_pool().connection() as conn:
        async with conn.transaction():
            cursor = await conn.execute(
                """
                UPDATE agent_operations SET status='failed_retryable',updated_at=NOW(),
                error=COALESCE(error,'execution process stopped before completion'),
                error_details=COALESCE(
                    error_details,
                    '{"type":"ProcessInterrupted","message":"execution process stopped before completion","retryable":true,"upstream_result_unknown":true}'::jsonb
                )
                WHERE status IN ('accepted','running')
                RETURNING operation_id, error, updated_at
                """
            )
            rows = await cursor.fetchall()
            for row in rows:
                await conn.execute(
                    """
                    INSERT INTO agent_operation_events (
                        event_id, operation_id, status, error, created_at
                    ) VALUES (%s,%s,'failed_retryable',%s,%s)
                    """,
                    (
                        f"operation-event:{uuid4().hex}",
                        row["operation_id"],
                        row["error"],
                        row["updated_at"],
                    ),
                )
            return len(rows)
