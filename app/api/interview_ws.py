"""WebSocket endpoint that drives the LangGraph interview in real time."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.cache.rate_limiter import check_rate_limit
from app.cache.websocket_presence import mark_offline, mark_online, refresh_online
from app.config import settings
from app.models.interview import ChatMessage
from app.models.ws_protocol import (
    ServerError,
    ServerFollowUp,
    ServerInterviewEnd,
    ServerQuestion,
    ServerQuestionStream,
    ServerReport,
    ServerStatus,
    ServerTranscriptSnapshot,
)
from app.errors import classify_public_error
from app.security import decode_access_token
from app.services.user_repository import get_user_by_id
from app.services.session_manager import get_session_manager

router = APIRouter()
logger = logging.getLogger(__name__)
WS_WAIT_HEARTBEAT_SECONDS = 15.0


def _supports_stream_sink(method: Any) -> bool:
    """Keep WebSocket tests and legacy manager implementations compatible."""
    try:
        return "stream_sink" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _get_last_interviewer_message(state: dict) -> str | None:
    history = state.get("conversation_history", [])
    for msg in reversed(history):
        if isinstance(msg, ChatMessage) and msg.role == "interviewer":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "interviewer":
            return msg.get("content", "")
    return None


def _get_last_interviewer_record(state: dict) -> ChatMessage | None:
    for raw in reversed(state.get("conversation_history", [])):
        message = raw if isinstance(raw, ChatMessage) else ChatMessage.model_validate(raw)
        if message.role == "interviewer":
            return message
    return None


def _subscribe_events(manager, session_id: str, after_sequence: int = 0):
    method = manager.subscribe_events
    try:
        if "after_sequence" in inspect.signature(method).parameters:
            return method(session_id, after_sequence=after_sequence)
    except (TypeError, ValueError):
        pass
    return method(session_id)


def _stream_delivered_current_turn(state: dict, tracker: dict[str, Any]) -> bool:
    record = _get_last_interviewer_record(state)
    return bool(
        record
        and tracker.get("forwarded_end")
        and tracker.get("last_turn_id") == record.turn_id
    )


def _get_question_meta(state: dict) -> tuple[int, int, list[str], str]:
    """Return (current_index, total, skill_tags, difficulty) for the current question."""
    plan = state.get("question_plan", []) or []
    idx = int(state.get("current_question_index", 0) or 0)
    blueprint = state.get("question_blueprint")
    total = len(plan)
    if blueprint:
        total = int(
            getattr(blueprint, "planned_count", 0)
            or (blueprint.get("planned_count", 0) if isinstance(blueprint, dict) else 0)
            or len(plan)
        )
    if idx < len(plan):
        question = plan[idx]
        skill_tags = getattr(question, "skill_tags", None)
        difficulty = getattr(question, "difficulty", None)
        if isinstance(question, dict):
            skill_tags = question.get("skill_tags", skill_tags)
            difficulty = question.get("difficulty", difficulty)
        return idx, total, skill_tags or [], difficulty or ""
    return idx, total, [], ""


def _report_from_state(session_id: str, state: dict) -> Any:
    mgr = get_session_manager()
    return (
        state.get("practice_report")
        or state.get("professional_report")
        or state.get("final_report")
        or mgr.get_report_for_session(session_id)
    )


async def _send_current_turn(websocket: WebSocket, state: dict) -> bool:
    record = _get_last_interviewer_record(state)
    if not record:
        await _send(websocket, ServerStatus(stage="waiting", message="Waiting for next question."))
        return False

    idx, total, tags, diff = _get_question_meta(state)
    follow_up_count = int(state.get("follow_up_count", 0) or 0)
    if follow_up_count > 0:
        await _send(
            websocket,
            ServerFollowUp(
                question_index=idx + 1,
                follow_up_number=follow_up_count,
                content=record.content,
                question_id=record.question_id or "",
                turn_id=record.turn_id or "",
            ),
        )
    else:
        await _send(
            websocket,
            ServerQuestion(
                question_index=idx + 1,
                total_questions=total,
                content=record.content,
                skill_tags=tags,
                difficulty=diff,
                question_id=record.question_id or "",
                turn_id=record.turn_id or "",
            ),
        )
    return True


async def _send_report_if_ready(websocket: WebSocket, session_id: str, state: dict) -> bool:
    if not state.get("interview_complete"):
        return False
    manager = get_session_manager()
    if hasattr(manager, "get_public_report"):
        report = manager.get_public_report(session_id)
    else:
        from app.services.public_interview import build_public_report
        report = build_public_report(
            _report_from_state(session_id, state), completed=True
        )
    await _send(websocket, ServerInterviewEnd())
    if report:
        await _send(websocket, ServerReport(data=report))
    return True


def _server_error(exc: BaseException) -> ServerError:
    error = classify_public_error(exc)
    return ServerError(
        message=error.message, code=error.code, retryable=error.retryable,
        retry_after_seconds=error.retry_after_seconds,
    )


async def _forward_session_event(
    websocket: WebSocket,
    session_id: str,
    tracker: dict[str, bool],
    event: dict[str, Any],
    *,
    forward: bool = True,
) -> None:
    """Forward one queued event without letting transport failures kill Graph."""
    payload = event.get("event", event)
    if not isinstance(payload, dict):
        return
    sequence = event.get("sequence")
    if sequence is not None:
        previous_sequence = int(tracker.get("last_sequence", 0) or 0)
        if int(sequence) <= previous_sequence:
            return
        tracker["last_sequence"] = int(sequence)

    if payload.get("type") == "status":
        if not forward or tracker.get("transport_failed"):
            return
        try:
            await _send(
                websocket,
                ServerStatus(
                    stage=payload.get("stage", "processing"),
                    message=payload.get("message", ""),
                    sequence=sequence,
                ),
            )
        except Exception:
            tracker["transport_failed"] = True
            logger.warning("Status transport closed for session=%s", session_id)
        return

    if payload.get("event") == "start":
        tracker["started"] = True
    if payload.get("event") == "end":
        tracker["completed"] = True

    if not forward or tracker.get("transport_failed"):
        return

    # An end event without a start event belongs to a previous transport (or
    # was queued after reconnect). Let the checkpoint fallback send one complete
    # question instead of exposing an unrenderable stream fragment.
    if payload.get("event") == "end" and not tracker.get("forwarded_start"):
        return

    try:
        await _send(
            websocket,
            ServerQuestionStream(
                **payload,
                sequence=sequence,
            ),
        )
        if payload.get("event") == "start":
            tracker["forwarded_start"] = True
        if payload.get("event") == "end":
            if tracker.get("forwarded_start"):
                tracker["forwarded_end"] = True
                tracker["last_turn_id"] = payload.get("turn_id")
    except Exception:
        tracker["transport_failed"] = True
        logger.warning("Question stream transport closed for session=%s", session_id)


async def _wait_for_session_task(
    websocket: WebSocket,
    session_id: str,
    manager,
    task: asyncio.Task,
    queue: asyncio.Queue[dict[str, Any]],
    tracker: dict[str, bool],
    *,
    forward_events: bool = True,
    heartbeat_stage: str = "processing",
    heartbeat_message: str = "正在处理当前面试步骤，请稍候。",
) -> dict:
    """Wait for a background operation while forwarding its queued events."""
    event_task: asyncio.Task | None = None
    try:
        while True:
            if task.done():
                # The operation and the event consumer may become ready in the
                # same loop tick. Forward that already-dequeued event before
                # draining the remaining queue.
                if event_task:
                    if not event_task.done():
                        event_task.cancel()
                        await asyncio.gather(event_task, return_exceptions=True)
                    if event_task.done() and not event_task.cancelled():
                        await _forward_session_event(
                            websocket,
                            session_id,
                            tracker,
                            event_task.result(),
                            forward=forward_events,
                        )
                    event_task = None
                while True:
                    try:
                        queued = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await _forward_session_event(
                        websocket,
                        session_id,
                        tracker,
                        queued,
                        forward=forward_events,
                    )
                return task.result()

            # Keep exactly one consumer alive across heartbeat timeouts. Creating
            # a replacement without cancelling the old task lets the orphan
            # consume and hide a later event.
            if event_task is None:
                event_task = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait(
                {task, event_task},
                timeout=WS_WAIT_HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                if forward_events and not tracker.get("transport_failed"):
                    try:
                        await _send(
                            websocket,
                            ServerStatus(
                                stage=heartbeat_stage,
                                message=heartbeat_message,
                            ),
                        )
                    except Exception:
                        tracker["transport_failed"] = True
                        logger.info(
                            "WS heartbeat transport closed for session=%s",
                            session_id,
                        )
                continue
            if event_task in done:
                await _forward_session_event(
                    websocket,
                    session_id,
                    tracker,
                    event_task.result(),
                    forward=forward_events,
                )
                event_task = None
    finally:
        if event_task and not event_task.done():
            event_task.cancel()
            await asyncio.gather(event_task, return_exceptions=True)
        manager.unsubscribe_events(session_id, queue)


async def _ensure_started_background(
    websocket: WebSocket,
    session_id: str,
    stream_tracker: dict[str, bool],
    after_sequence: int = 0,
) -> dict | None:
    """Start or join one background startup operation for a real SessionManager."""
    mgr = get_session_manager()
    state = await mgr.get_last_state(session_id)
    session = mgr.get_session(session_id)
    if session and session.status == "failed":
        await _send(
            websocket,
            ServerError(
                message=mgr.get_error_message(session_id)
                or "Interview processing failed; choose explicit retry or end the interview."
            ),
        )
        return state
    if session and session.status in {"pausing", "paused"}:
        await _send(websocket, ServerStatus(stage=session.status, message="面试已暂停。"))
        return state
    if mgr.has_graph_started(session_id) and state:
        await _send(websocket, ServerStatus(stage="resumed", message="Interview session resumed."))
        return state

    already_running = mgr.has_active_operation(session_id)
    await _send(
        websocket,
        ServerStatus(
            stage="resumed" if already_running else "analyzing_jd",
            message=(
                "Interview preparation is already running; waiting for the current task."
                if already_running
                else "Analyzing JD and preparing questions."
            ),
        ),
    )

    queue = _subscribe_events(mgr, session_id, after_sequence)
    try:
        task = mgr.ensure_start_task(session_id)
        return await _wait_for_session_task(
            websocket,
            session_id,
            mgr,
            task,
            queue,
            stream_tracker,
            forward_events=not already_running,
            heartbeat_stage="preparing_interview",
            heartbeat_message="正在准备材料分析、考察蓝图和首题，请稍候。",
        )
    except Exception as exc:
        logger.exception("Background graph start failed for %s", session_id)
        try:
            await _send(websocket, _server_error(exc))
        except Exception:
            pass
        return None


async def _ensure_started(
    websocket: WebSocket,
    session_id: str,
    stream_tracker: dict[str, bool] | None = None,
    after_sequence: int = 0,
) -> dict | None:
    mgr = get_session_manager()
    if hasattr(mgr, "ensure_start_task") and hasattr(mgr, "subscribe_events"):
        return await _ensure_started_background(
            websocket,
            session_id,
            stream_tracker or {"started": False, "completed": False},
            after_sequence,
        )

    state = await mgr.get_last_state(session_id)
    if mgr.has_graph_started(session_id) and state:
        await _send(websocket, ServerStatus(stage="resumed", message="Interview session resumed."))
        return state

    await _send(
        websocket,
        ServerStatus(stage="analyzing_jd", message="Analyzing JD and preparing questions."),
    )
    tracker = stream_tracker or {"started": False, "completed": False}

    async def stream_sink(event: dict[str, Any]) -> None:
        if event.get("type") == "status":
            if tracker.get("transport_failed"):
                return
            try:
                await _send(
                    websocket,
                    ServerStatus(
                        stage=event.get("stage", "processing"),
                        message=event.get("message", ""),
                    ),
                )
            except Exception:
                tracker["transport_failed"] = True
                logger.warning("Status transport closed for session=%s", session_id)
            return

        tracker["started"] = True
        if event.get("event") == "end":
            tracker["completed"] = True
        if tracker.get("transport_failed"):
            return
        try:
            await _send(websocket, ServerQuestionStream(**event))
            if event.get("event") == "start":
                tracker["forwarded_start"] = True
            if event.get("event") == "end":
                if tracker.get("forwarded_start"):
                    tracker["forwarded_end"] = True
        except Exception:
            tracker["transport_failed"] = True
            logger.warning("Question stream transport closed for session=%s", session_id)

    try:
        if _supports_stream_sink(mgr.start_interview_graph):
            state = await mgr.start_interview_graph(
                session_id,
                stream_sink=stream_sink,
            )
        else:
            state = await mgr.start_interview_graph(session_id)
    except Exception as exc:
        logger.exception("Graph start failed for %s", session_id)
        await _send(websocket, _server_error(exc))
        return None

    if not tracker["completed"]:
        await _send(
            websocket,
            ServerStatus(
                stage="questions_ready",
                message=f"Prepared {len(state.get('question_plan', []) or [])} interview questions.",
            ),
        )
    return state


@router.websocket("/interview/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    """Full interview WebSocket endpoint.

    The endpoint remains backward compatible with the older client: connecting to a
    pending session starts the graph automatically. Reconnecting to an existing
    session resumes from the latest checkpoint instead of starting a duplicate run.
    """
    # Credentials never belong in URLs (access logs, browser history, proxies).
    if "token" in websocket.query_params or "auth_token" in websocket.query_params:
        await websocket.close(code=4401, reason="Use an authentication message")
        return
    await websocket.accept()
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), settings.ws_auth_timeout_seconds)
        if len(raw) > 8192:
            raise ValueError("Authentication message too large")
        auth = json.loads(raw)
        if not isinstance(auth, dict) or auth.get("type") != "auth":
            raise ValueError("Authentication required")
        raw_token = auth.get("token")
        if not isinstance(raw_token, str) or not raw_token or len(raw_token) > 4096:
            raise ValueError("Authentication required")
        payload = decode_access_token(raw_token)
        current_user = await get_user_by_id(str(payload["sub"]))
    except Exception:
        await websocket.close(code=4401, reason="Invalid authentication token")
        return
    if not current_user or not current_user.is_active:
        await websocket.close(code=4401, reason="User not found or inactive")
        return

    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    if session.user_id != current_user.id:
        await websocket.close(code=4403, reason="Session access denied")
        return

    await mark_online(session_id, session.user_id)
    logger.info("WS connected: session=%s", session_id)

    try:
        current_turn_delivered = False
        try:
            after_sequence = max(
                0, int(websocket.query_params.get("after_sequence") or 0)
            )
        except ValueError:
            after_sequence = 0
        startup_stream = {
            "started": False,
            "completed": False,
            "forwarded_end": False,
            "last_sequence": after_sequence,
        }
        recovery_queue = None
        recovery_queue_owned = False
        if hasattr(mgr, "subscribe_events") and hasattr(mgr, "unsubscribe_events"):
            # Subscribe before reading replay/checkpoint state so a completion
            # occurring in that window is queued and later converges by sequence.
            recovery_queue = _subscribe_events(mgr, session_id, after_sequence)
            recovery_queue_owned = True
        try:
            if hasattr(mgr, "get_public_transcript"):
                snapshot = await mgr.get_public_transcript(session_id)
                await _send(websocket, ServerTranscriptSnapshot(data=snapshot))
            if hasattr(mgr, "replay_events"):
                for event in await mgr.replay_events(session_id, after_sequence):
                    await _forward_session_event(
                        websocket, session_id, startup_stream, event
                    )
            state = await _ensure_started(
                websocket,
                session_id,
                startup_stream,
                int(startup_stream.get("last_sequence", after_sequence) or 0),
            )
            if state is None:
                return
            if hasattr(mgr, "ensure_pending_answer_task"):
                pending_task = mgr.ensure_pending_answer_task(session_id)
                if pending_task is not None:
                    if recovery_queue is not None:
                        recovery_queue_owned = False
                        state = await _wait_for_session_task(
                            websocket,
                            session_id,
                            mgr,
                            pending_task,
                            recovery_queue,
                            startup_stream,
                            heartbeat_stage="processing_answer",
                            heartbeat_message="正在继续处理已提交的回答，请稍候。",
                        )
                    else:
                        state = await pending_task
            if recovery_queue is not None and recovery_queue_owned:
                while True:
                    try:
                        event = recovery_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await _forward_session_event(
                        websocket, session_id, startup_stream, event
                    )
                mgr.unsubscribe_events(session_id, recovery_queue)
                recovery_queue_owned = False
        finally:
            if recovery_queue is not None and recovery_queue_owned:
                mgr.unsubscribe_events(session_id, recovery_queue)
        last_sequence = int(startup_stream.get("last_sequence", after_sequence) or 0)
        if await _send_report_if_ready(websocket, session_id, state):
            return
        current_turn_delivered = (
            _stream_delivered_current_turn(state, startup_stream)
            or await _send_current_turn(websocket, state)
        )

        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, ServerError(message="Invalid JSON"))
                continue

            msg_type = msg.get("type", "")

            if msg_type in {"ping"}:
                await refresh_online(session_id)
                await _send(websocket, ServerStatus(stage="pong", message="ok"))
                continue

            if msg_type in {"start", "start_interview"}:
                await refresh_online(session_id)
                if current_turn_delivered:
                    continue
                startup_stream = {
                    "started": False,
                    "completed": False,
                    "forwarded_end": False,
                    "last_sequence": last_sequence,
                }
                state = await _ensure_started(
                    websocket,
                    session_id,
                    startup_stream,
                    last_sequence,
                )
                report_sent = (
                    state is not None
                    and await _send_report_if_ready(websocket, session_id, state)
                )
                last_sequence = int(
                    startup_stream.get("last_sequence", last_sequence) or last_sequence
                )
                if state is not None and not report_sent and not current_turn_delivered:
                    current_turn_delivered = (
                        _stream_delivered_current_turn(state, startup_stream)
                        or await _send_current_turn(websocket, state)
                    )
                continue

            if msg_type in {"stop", "end_interview"}:
                await refresh_online(session_id)
                await _send(
                    websocket,
                    ServerStatus(
                        stage="evaluating",
                        message="Stopping interview and generating report.",
                    ),
                )
                try:
                    state = await (
                        mgr.ensure_end_task(session_id)
                        if hasattr(mgr, "ensure_end_task")
                        else mgr.stop_interview(session_id)
                    )
                except Exception as exc:
                    logger.exception("Stop interview failed for %s", session_id)
                    await _send(websocket, _server_error(exc))
                    continue
                await _send_report_if_ready(websocket, session_id, state)
                break

            if msg_type == "retry":
                retry_id = str(msg.get("retry_id") or "").strip()
                if not retry_id:
                    await _send(websocket, ServerError(message="retry_id is required"))
                    continue
                queue = _subscribe_events(mgr, session_id, last_sequence)
                tracker = {
                    "started": False,
                    "completed": False,
                    "forwarded_end": False,
                    "last_sequence": last_sequence,
                }
                try:
                    retry_kind = (
                        mgr.pending_retry_kind(session_id)
                        if hasattr(mgr, "pending_retry_kind")
                        else None
                    )
                    if retry_kind == "answer" and hasattr(
                        mgr, "retry_pending_answer_task"
                    ):
                        task = await mgr.retry_pending_answer_task(
                            session_id, retry_id
                        )
                    elif mgr.has_graph_started(session_id):
                        task = await mgr.retry_question_task(session_id, retry_id)
                    else:
                        task = await mgr.retry_start_task(session_id, retry_id)
                    state = await _wait_for_session_task(
                        websocket,
                        session_id,
                        mgr,
                        task,
                        queue,
                        tracker,
                        heartbeat_stage="retrying",
                        heartbeat_message="正在重试失败的面试步骤，请稍候。",
                    )
                except Exception as exc:
                    mgr.unsubscribe_events(session_id, queue)
                    logger.exception("Explicit retry failed for %s", session_id)
                    await _send(websocket, _server_error(exc))
                    continue
                last_sequence = int(tracker.get("last_sequence", last_sequence) or last_sequence)
                if await _send_report_if_ready(websocket, session_id, state):
                    break
                current_turn_delivered = (
                    _stream_delivered_current_turn(state, tracker)
                    or await _send_current_turn(websocket, state)
                )
                continue

            if msg_type != "answer":
                await _send(
                    websocket,
                    ServerError(message=f"Expected 'answer' message, got '{msg_type}'"),
                )
                continue

            if not current_turn_delivered:
                state = await mgr.get_last_state(session_id)
                current_turn_delivered = await _send_current_turn(websocket, state)
                if current_turn_delivered:
                    await _send(
                        websocket,
                        ServerError(
                            message=(
                                "The current question was just sent. "
                                "Please answer after it is displayed."
                            )
                        ),
                    )
                    continue
                await _send(
                    websocket,
                    ServerError(message="No interview question is ready yet. Please wait."),
                )
                continue

            answer_text = msg.get("content", "").strip()
            if not answer_text:
                await _send(websocket, ServerError(message="Answer cannot be empty"))
                continue
            await refresh_online(session_id)

            limit_identity = f"session:{session_id}:user:{session.user_id}"
            rate_limit = await check_rate_limit(
                "interview:answer:ws",
                limit_identity,
                settings.rate_limit_answer_per_minute,
                60,
            )
            if not rate_limit.allowed:
                await _send(
                    websocket,
                    ServerError(
                        message=(
                            "Too many answers. "
                            f"Please retry after {rate_limit.retry_after_seconds} seconds."
                        )
                    ),
                )
                continue

            await _send(
                websocket,
                ServerStatus(
                    stage="processing",
                    message="Assessing answer and deciding next step.",
                ),
            )

            current_turn_delivered = False
            stream_tracker = {
                "started": False,
                "completed": False,
                "last_sequence": last_sequence,
            }

            if hasattr(mgr, "ensure_answer_task") and hasattr(mgr, "subscribe_events"):
                answer_id = str(msg.get("answer_id") or uuid4().hex)
                turn_id = str(msg.get("turn_id") or "") or None
                try:
                    await mgr.accept_answer_submission(
                        session_id, answer_text, answer_id, turn_id
                    )
                except ValueError as exc:
                    await _send(websocket, ServerError(message=str(exc)))
                    continue
                already_running = mgr.has_active_operation(session_id)
                queue = _subscribe_events(
                    mgr,
                    session_id,
                    int(stream_tracker.get("last_sequence", 0) or 0),
                )
                try:
                    task = mgr.ensure_answer_task(
                        session_id,
                        answer_text,
                        answer_id,
                    )
                    state = await _wait_for_session_task(
                        websocket,
                        session_id,
                        mgr,
                        task,
                        queue,
                        stream_tracker,
                        forward_events=not already_running,
                        heartbeat_stage="evaluating_answer",
                        heartbeat_message="正在评估回答并准备下一步，请稍候。",
                    )
                    last_sequence = int(
                        stream_tracker.get("last_sequence", last_sequence) or last_sequence
                    )
                except Exception as exc:
                    logger.exception("Background answer processing failed for %s", session_id)
                    await _send(
                        websocket,
                        _server_error(exc),
                    )
                    continue

                if await _send_report_if_ready(websocket, session_id, state):
                    break
                current_turn_delivered = (
                    _stream_delivered_current_turn(state, stream_tracker)
                    or await _send_current_turn(websocket, state)
                )
                continue

            async def stream_sink(event: dict[str, Any]) -> None:
                if event.get("type") == "status":
                    if stream_tracker.get("transport_failed"):
                        return
                    try:
                        await _send(
                            websocket,
                            ServerStatus(
                                stage=event.get("stage", "processing"),
                                message=event.get("message", ""),
                            ),
                        )
                    except Exception:
                        stream_tracker["transport_failed"] = True
                        logger.warning(
                            "Status transport closed for session=%s",
                            session_id,
                        )
                    return

                stream_tracker["started"] = True
                if event.get("event") == "end":
                    stream_tracker["completed"] = True
                if stream_tracker.get("transport_failed"):
                    return
                try:
                    await _send(websocket, ServerQuestionStream(**event))
                except Exception:
                    stream_tracker["transport_failed"] = True
                    logger.warning(
                        "Question stream transport closed for session=%s",
                        session_id,
                    )

            try:
                if _supports_stream_sink(mgr.submit_answer):
                    state = await mgr.submit_answer(
                        session_id,
                        answer_text,
                        stream_sink=stream_sink,
                    )
                else:
                    state = await mgr.submit_answer(session_id, answer_text)
            except Exception as exc:
                logger.exception("Graph resume failed for %s", session_id)
                await _send(websocket, _server_error(exc))
                continue

            if await _send_report_if_ready(websocket, session_id, state):
                break
            current_turn_delivered = (
                stream_tracker["completed"]
                or await _send_current_turn(websocket, state)
            )

    except WebSocketDisconnect:
        logger.info("WS disconnected: session=%s", session_id)
    except RuntimeError as exc:
        if "WebSocket is not connected" in str(exc):
            logger.info("WS disconnected before handler completed: session=%s", session_id)
        else:
            logger.exception("Unexpected runtime error in WS session=%s", session_id)
    except Exception:
        logger.exception("Unexpected error in WS session=%s", session_id)
    finally:
        await mark_offline(session_id)


async def _send(ws: WebSocket, msg) -> None:
    """Send a Pydantic model as JSON over the WebSocket."""
    await ws.send_json(msg.model_dump(mode="json"))
