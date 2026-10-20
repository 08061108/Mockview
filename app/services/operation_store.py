"""Idempotent operation/tool-call ledger used by resumable agent work."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.models.preparation import OperationRecord, ToolCallRecord
from app.services.operation_repository import (
    append_operation_event,
    list_operation_events,
    list_operations_for_case,
    list_tool_calls,
    load_operation,
    load_tool_call,
    mark_unfinished_operations_retryable,
    save_operation,
    save_tool_call,
)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def structured_operation_error(exc: BaseException) -> dict[str, Any]:
    name = type(exc).__name__
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = {type(item).__name__ for item in chain}
    retryable_names = {
        "StructuredOutputTruncatedError",
        "UpstreamResultUnknownError",
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "TimeoutError",
        "APITimeoutError",
        "ReadTimeout",
        "WriteTimeout",
    }
    unknown_names = {
        "UpstreamResultUnknownError",
        "TimeoutError",
        "APITimeoutError",
        "ReadTimeout",
        "WriteTimeout",
    }
    return {
        "type": name,
        "cause_type": next(
            (type(item).__name__ for item in chain[1:] if type(item).__name__ != name),
            None,
        ),
        "message": str(exc),
        "retryable": bool(names & retryable_names),
        "upstream_result_unknown": bool(names & unknown_names),
    }


class OperationStore:
    def __init__(self) -> None:
        self._operations: dict[str, OperationRecord] = {}
        self._tool_calls: dict[str, ToolCallRecord] = {}
        self._operation_events: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def _record_status(self, item: OperationRecord) -> None:
        event = {
            "event_id": f"operation-event:{uuid4().hex}",
            "operation_id": item.operation_id,
            "status": item.status,
            "error": item.error,
            "created_at": item.updated_at,
        }
        self._operation_events.setdefault(item.operation_id, []).append(event)
        await append_operation_event(**event)

    async def accept(
        self,
        *,
        user_id: str,
        kind: str,
        payload: dict[str, Any],
        case_id: str | None = None,
        input_revision: int | None = None,
        operation_id: str | None = None,
        budget: dict[str, int] | None = None,
    ) -> OperationRecord:
        operation_id = operation_id or uuid4().hex
        input_hash = stable_hash(payload)
        async with self._lock:
            existing = self._operations.get(operation_id) or await load_operation(operation_id)
            if existing:
                self._operations[operation_id] = existing
                if (existing.input_hash != input_hash or existing.user_id != user_id
                        or existing.kind != kind or existing.case_id != case_id):
                    raise ValueError("operation_id already exists with different input")
                return existing
            record = OperationRecord(
                operation_id=operation_id,
                user_id=user_id,
                kind=kind,
                case_id=case_id,
                input_revision=input_revision,
                input_hash=input_hash,
                budget=budget or {},
            )
            self._operations[operation_id] = record
            await save_operation(record)
            await self._record_status(record)
            return record

    async def latest_preparation(self, case_id: str, user_id: str, revision: int) -> OperationRecord | None:
        persisted = await list_operations_for_case(case_id)
        known = {item.operation_id: item for item in persisted}
        known.update({key: item for key, item in self._operations.items() if item.case_id == case_id})
        candidates = [item for item in known.values() if item.user_id == user_id
                      and item.kind == "prepare_case" and item.input_revision == revision]
        return max(candidates, key=lambda item: (item.created_at, item.operation_id), default=None)

    async def get(self, operation_id: str, user_id: str | None = None) -> OperationRecord | None:
        record = self._operations.get(operation_id)
        if not record:
            record = await load_operation(operation_id)
            if record:
                self._operations[operation_id] = record
        if record and user_id is not None and record.user_id != user_id:
            return None
        return record

    async def transition(self, operation_id: str, status: str, **updates: Any) -> OperationRecord:
        async with self._lock:
            current = self._operations[operation_id]
            updated = current.model_copy(
                update={"status": status, "updated_at": datetime.now(timezone.utc), **updates}
            )
            self._operations[operation_id] = updated
            await save_operation(updated)
            await self._record_status(updated)
            return updated

    async def consume_budget(self, operation_id: str, key: str, amount: int = 1) -> int:
        """Persist budget consumption before an external/model request."""
        async with self._lock:
            current = self._operations.get(operation_id) or await load_operation(operation_id)
            if not current:
                raise KeyError(operation_id)
            remaining = int(current.budget.get(key, 0))
            if remaining < amount:
                raise RuntimeError(f"operation budget exhausted: {key}")
            budget = dict(current.budget)
            budget[key] = remaining - amount
            updated = current.model_copy(
                update={"budget": budget, "updated_at": datetime.now(timezone.utc)}
            )
            self._operations[operation_id] = updated
            await save_operation(updated)
            return budget[key]

    async def record_model_request_started(
        self,
        operation_id: str,
        *,
        request_kind: str,
        model: str,
        consume_budget: bool = True,
        queue_elapsed_seconds: float = 0.0,
    ) -> OperationRecord | None:
        """Durably count one physical provider request before it is sent."""
        async with self._lock:
            current = self._operations.get(operation_id) or await load_operation(operation_id)
            if current is None:
                return None
            budget = dict(current.budget)
            if consume_budget:
                remaining = int(budget.get("model_requests", 0))
                if remaining <= 0:
                    raise RuntimeError("operation budget exhausted: model_requests")
                budget["model_requests"] = remaining - 1
            metrics = dict(current.metrics)
            metrics["physical_requests"] = int(metrics.get("physical_requests", 0)) + 1
            metrics["model_queue_seconds"] = round(
                float(metrics.get("model_queue_seconds", 0.0))
                + max(0.0, queue_elapsed_seconds),
                6,
            )
            metrics["max_model_queue_seconds"] = round(
                max(
                    float(metrics.get("max_model_queue_seconds", 0.0)),
                    max(0.0, queue_elapsed_seconds),
                ),
                6,
            )
            kinds = dict(metrics.get("request_kinds", {}))
            kinds[request_kind] = int(kinds.get(request_kind, 0)) + 1
            metrics["request_kinds"] = kinds
            models = dict(metrics.get("models", {}))
            models[model] = int(models.get(model, 0)) + 1
            metrics["models"] = models
            updated = current.model_copy(
                update={
                    "budget": budget,
                    "metrics": metrics,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._operations[operation_id] = updated
            await save_operation(updated)
            return updated

    async def record_model_completion(
        self,
        operation_id: str,
        *,
        usage: dict[str, Any] | None,
        elapsed_seconds: float,
    ) -> OperationRecord | None:
        """Aggregate provider-reported usage without inventing unavailable cost."""
        async with self._lock:
            current = self._operations.get(operation_id) or await load_operation(operation_id)
            if current is None:
                return None
            metrics = dict(current.metrics)
            metrics["completed_requests"] = int(metrics.get("completed_requests", 0)) + 1
            metrics["model_elapsed_seconds"] = round(
                float(metrics.get("model_elapsed_seconds", 0.0)) + elapsed_seconds,
                6,
            )
            if usage:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if isinstance(value, (int, float)):
                        metrics[key] = int(metrics.get(key, 0)) + int(value)
            updated = current.model_copy(
                update={"metrics": metrics, "updated_at": datetime.now(timezone.utc)}
            )
            self._operations[operation_id] = updated
            await save_operation(updated)
            return updated

    async def accept_tool_call(
        self,
        *,
        operation_id: str,
        tool_name: str,
        payload: dict[str, Any],
        tool_call_id: str,
    ) -> ToolCallRecord:
        input_hash = stable_hash(payload)
        async with self._lock:
            existing = self._tool_calls.get(tool_call_id) or await load_tool_call(tool_call_id)
            if existing:
                if existing.input_hash != input_hash or existing.tool_name != tool_name:
                    raise ValueError("tool_call_id already exists with different input")
                return existing
            record = ToolCallRecord(
                tool_call_id=tool_call_id,
                operation_id=operation_id,
                tool_name=tool_name,
                input_hash=input_hash,
                input_summary=payload,
            )
            self._tool_calls[tool_call_id] = record
            await save_tool_call(record)
            return record

    async def finish_tool_call(
        self, tool_call_id: str, *, result: dict[str, Any] | None = None, error: str | None = None
    ) -> ToolCallRecord:
        async with self._lock:
            current = self._tool_calls[tool_call_id]
            updated = current.model_copy(
                update={
                    "status": "failed" if error else "succeeded",
                    "result": result,
                    "error": error,
                    "attempts": current.attempts + 1,
                    "budget_counted": True,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._tool_calls[tool_call_id] = updated
            await save_tool_call(updated)
            return updated

    async def tool_calls(
        self, operation_id: str, tool_name: str | None = None
    ) -> list[ToolCallRecord]:
        persisted = await list_tool_calls(operation_id)
        known = {
            item.tool_call_id: item
            for item in self._tool_calls.values()
            if item.operation_id == operation_id
        }
        for item in persisted:
            known[item.tool_call_id] = item
        rows = list(known.values())
        if tool_name is not None:
            rows = [item for item in rows if item.tool_name == tool_name]
        return sorted(rows, key=lambda item: (item.created_at, item.tool_call_id))

    async def events(self, operation_id: str, user_id: str) -> list[dict[str, Any]] | None:
        """Return a replayable, redacted operation timeline backed by durable records."""
        operation = await self.get(operation_id, user_id)
        if not operation:
            return None
        persisted = await self.tool_calls(operation_id)
        persisted_statuses = await list_operation_events(operation_id)
        calls = persisted
        memory_statuses = self._operation_events.get(operation_id, [])
        memory_ids = {item["event_id"] for item in memory_statuses}
        status_events = [
            item for item in persisted_statuses if item["event_id"] not in memory_ids
        ] + list(memory_statuses)
        if not status_events:
            status_events = [{
                "event_id": f"operation:{operation.operation_id}:accepted",
                "operation_id": operation.operation_id,
                "status": "accepted",
                "error": None,
                "created_at": operation.created_at,
            }]
        events: list[dict[str, Any]] = []
        for item in status_events:
            events.append(
                {
                    "event_id": item["event_id"],
                    "type": (
                        "operation_accepted"
                        if item["status"] == "accepted"
                        else "operation_status"
                    ),
                    "status": item["status"],
                    "error": item.get("error"),
                    "created_at": item["created_at"],
                }
            )
        for call in calls:
            events.append(
                {
                    "event_id": f"tool:{call.tool_call_id}:{call.status}:{call.attempts}",
                    "type": "tool_call",
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "status": call.status,
                    "attempts": call.attempts,
                    "error": call.error,
                    "created_at": call.updated_at,
                }
            )
        return events

    async def session_usage_summary(self, session_id: str) -> dict[str, Any]:
        """Aggregate durable model/tool usage for one interview session."""
        known = {
            item.operation_id: item
            for item in self._operations.values()
            if item.case_id == session_id
        }
        for item in await list_operations_for_case(session_id):
            known[item.operation_id] = item
        totals: dict[str, Any] = {
            "operation_count": len(known),
            "physical_requests": 0,
            "completed_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model_queue_seconds": 0.0,
            "max_model_queue_seconds": 0.0,
            "model_elapsed_seconds": 0.0,
            "operation_elapsed_seconds": 0.0,
            "tool_calls": 0,
            "models": {},
            "billing_cost_usd": None,
            "billing_note": "Provider billing cost is unavailable; no price was inferred.",
        }
        for item in known.values():
            metrics = item.metrics or {}
            for key in (
                "physical_requests",
                "completed_requests",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                totals[key] += int(metrics.get(key, 0) or 0)
            totals["model_elapsed_seconds"] += float(
                metrics.get("model_elapsed_seconds", 0.0) or 0.0
            )
            totals["model_queue_seconds"] += float(
                metrics.get("model_queue_seconds", 0.0) or 0.0
            )
            totals["max_model_queue_seconds"] = max(
                totals["max_model_queue_seconds"],
                float(metrics.get("max_model_queue_seconds", 0.0) or 0.0),
            )
            totals["operation_elapsed_seconds"] += max(
                0.0, (item.updated_at - item.created_at).total_seconds()
            )
            for model, count in (metrics.get("models", {}) or {}).items():
                totals["models"][model] = int(totals["models"].get(model, 0)) + int(count)
            totals["tool_calls"] += len(await self.tool_calls(item.operation_id))
        totals["model_elapsed_seconds"] = round(totals["model_elapsed_seconds"], 6)
        totals["model_queue_seconds"] = round(totals["model_queue_seconds"], 6)
        totals["max_model_queue_seconds"] = round(
            totals["max_model_queue_seconds"], 6
        )
        totals["operation_elapsed_seconds"] = round(
            totals["operation_elapsed_seconds"], 6
        )
        return totals

    async def recover_running(self) -> int:
        recovered = 0
        async with self._lock:
            for operation_id, item in list(self._operations.items()):
                if item.status in {"accepted", "running"}:
                    recovered_item = item.model_copy(
                        update={
                            "status": "failed_retryable",
                            "error": item.error
                            or "execution process stopped before completion",
                            "error_details": item.error_details
                            or {
                                "type": "ProcessInterrupted",
                                "message": "execution process stopped before completion",
                                "retryable": True,
                                "upstream_result_unknown": True,
                            },
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                    self._operations[operation_id] = recovered_item
                    await save_operation(recovered_item)
                    await self._record_status(recovered_item)
                    recovered += 1
        recovered += await mark_unfinished_operations_retryable()
        return recovered


_store: OperationStore | None = None


def get_operation_store() -> OperationStore:
    global _store
    if _store is None:
        _store = OperationStore()
    return _store


def reset_operation_store() -> None:
    global _store
    _store = None
