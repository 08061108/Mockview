"""Per-task streaming context used to forward LLM deltas to the transport layer."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar, Token
from typing import Any, Awaitable, Callable
from uuid import uuid4

from app.services.session_event_repository import (
    latest_session_event_sequence,
    save_session_event,
)

StreamSink = Callable[[dict[str, Any]], Awaitable[None]]
logger = logging.getLogger(__name__)


class SessionEventBus:
    """In-process fan-out bus for one interview session.

    The graph publishes events even when no WebSocket is connected. Subscribers
    receive a bounded queue so a slow browser cannot block the graph forever.
    The final LangGraph state remains the source of truth for reconnects.
    """

    def __init__(self, session_id: str = "", max_queue_size: int = 256) -> None:
        self._session_id = session_id
        self._max_queue_size = max_queue_size
        self._sequence = 0
        self._initialized = False
        self._lock = asyncio.Lock()
        self._history: list[dict[str, Any]] = []
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self, after_sequence: int = 0) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._max_queue_size,
        )
        self._subscribers.add(queue)
        for item in self._history:
            if int(item["sequence"]) > after_sequence:
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    break
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def replay(self, after_sequence: int = 0) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self._history
            if int(item["sequence"]) > after_sequence
        ]

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._lock:
            if not self._initialized:
                if self._session_id:
                    try:
                        self._sequence = await latest_session_event_sequence(self._session_id)
                    except Exception:
                        logger.exception("Failed to load event sequence: %s", self._session_id)
                self._initialized = True
            self._sequence += 1
            payload = dict(event)
            event_id = str(payload.get("event_id") or self._event_id(payload))
            item = {
                "sequence": self._sequence,
                "event_id": event_id,
                "event": payload,
            }
            if self._session_id:
                try:
                    if self._should_persist(payload):
                        await save_session_event(
                            self._session_id,
                            self._sequence,
                            event_id,
                            payload,
                        )
                except Exception:
                    logger.exception("Failed to persist session event: %s", event_id)
            self._history.append(item)
            if len(self._history) > self._max_queue_size:
                self._history = self._history[-self._max_queue_size :]
        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                # A final "end" event carries the complete text, so dropping
                # an old delta is recoverable and cannot corrupt checkpoint state.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    pass

    @staticmethod
    def _should_persist(event: dict[str, Any]) -> bool:
        return event.get("type") == "status" or event.get("event") in {"start", "end"}

    @staticmethod
    def _event_id(event: dict[str, Any]) -> str:
        stream_id = event.get("stream_id")
        stream_event = event.get("event")
        if stream_id and stream_event in {"start", "end"}:
            return f"stream:{stream_id}:{stream_event}"
        return f"event:{uuid4().hex}"

_stream_sink: ContextVar[StreamSink | None] = ContextVar(
    "interview_stream_sink",
    default=None,
)


def set_stream_sink(sink: StreamSink | None) -> Token[StreamSink | None]:
    """Install a task-local stream sink for the current graph invocation."""
    return _stream_sink.set(sink)


def reset_stream_sink(token: Token[StreamSink | None]) -> None:
    """Restore the previous task-local stream sink."""
    _stream_sink.reset(token)


def streaming_enabled() -> bool:
    return _stream_sink.get() is not None


async def emit_stream_event(event: dict[str, Any]) -> None:
    """Forward an event when the current invocation has a stream sink."""
    sink = _stream_sink.get()
    if sink is not None:
        await sink(event)


async def emit_status_event(stage: str, message: str) -> None:
    """Publish a user-visible workflow progress update when streaming is active."""
    await emit_stream_event(
        {
            "type": "status",
            "stage": stage,
            "message": message,
        }
    )
