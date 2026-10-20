"""Small foreground-first in-process gate for model requests."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from app.config import settings

_priority: ContextVar[str] = ContextVar("model_request_priority", default="foreground")


@dataclass
class _Waiter:
    task: asyncio.Task
    priority: str
    order: int
    ready: asyncio.Future
    enqueued_at: float


class ModelRequestGate:
    def __init__(self) -> None:
        self.capacity = max(1, settings.model_max_concurrency)
        self.background_limit = max(0, settings.question_prefetch_global_limit)
        self._active = 0
        self._active_background = 0
        self._order = 0
        self._waiters: list[_Waiter] = []
        self._promoted: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def _acquire(self) -> _Waiter:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("model gate requires an asyncio task")
        loop = asyncio.get_running_loop()
        async with self._lock:
            self._order += 1
            priority = "foreground" if task in self._promoted else _priority.get()
            self._promoted.discard(task)
            waiter = _Waiter(
                task,
                priority,
                self._order,
                loop.create_future(),
                time.perf_counter(),
            )
            self._waiters.append(waiter)
            self._dispatch_locked()
        try:
            await waiter.ready
            return waiter
        except BaseException:
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._dispatch_locked()
            raise

    async def _release(self, waiter: _Waiter) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
            if waiter.priority == "background":
                self._active_background = max(0, self._active_background - 1)
            self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        while self._active < self.capacity:
            pending = [item for item in self._waiters if not item.ready.done()]
            if not pending:
                return
            foreground = [item for item in pending if item.priority == "foreground"]
            if foreground:
                chosen = min(foreground, key=lambda item: item.order)
            else:
                background = [item for item in pending if item.priority == "background"]
                if not background or self._active_background >= self.background_limit:
                    return
                chosen = min(background, key=lambda item: item.order)
            self._waiters.remove(chosen)
            self._active += 1
            if chosen.priority == "background":
                self._active_background += 1
            chosen.ready.set_result(None)

    async def promote(self, task: asyncio.Task) -> None:
        """Promote a not-yet-started prefetch without creating another request."""
        async with self._lock:
            for waiter in self._waiters:
                if waiter.task is task and not waiter.ready.done():
                    waiter.priority = "foreground"
                    self._dispatch_locked()
                    return
            self._promoted.add(task)
            task.add_done_callback(self._promoted.discard)

    @asynccontextmanager
    async def slot(self):
        waiter = await self._acquire()
        try:
            yield max(0.0, time.perf_counter() - waiter.enqueued_at)
        finally:
            await self._release(waiter)


_gate: ModelRequestGate | None = None


def get_model_request_gate() -> ModelRequestGate:
    global _gate
    if _gate is None:
        _gate = ModelRequestGate()
    return _gate


@asynccontextmanager
async def model_request_priority(priority: str):
    token = _priority.set(priority)
    try:
        yield
    finally:
        _priority.reset(token)


def reset_model_request_gate() -> None:
    global _gate
    _gate = None
