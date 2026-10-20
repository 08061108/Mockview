"""Unified async LLM client supporting chat / streaming / structured / vision output."""

from __future__ import annotations

import base64
import asyncio
import json
import logging
import os
import re
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar, get_args, get_origin

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import settings
from app.llm.transport import build_model_http_client
from app.llm.scheduler import get_model_request_gate

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
_MODEL_TRACE: ContextVar[dict[str, Any]] = ContextVar("model_trace", default={})


class ToolCallingUnavailableError(RuntimeError):
    """Raised when a configured chat provider does not return native tool calls."""


class UpstreamResultUnknownError(RuntimeError):
    """A request may have reached the provider but no final result was observed."""


class StructuredOutputTruncatedError(ValueError):
    """Provider stopped a structured response because its output budget ended."""


@contextmanager
def model_trace(**fields: Any):
    current = dict(_MODEL_TRACE.get())
    token = _MODEL_TRACE.set({**current, **{key: value for key, value in fields.items() if value is not None}})
    try:
        yield
    finally:
        _MODEL_TRACE.reset(token)


def _network_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.model_connect_timeout_seconds,
        read=settings.model_read_timeout_seconds,
        write=settings.model_write_timeout_seconds,
        pool=settings.model_pool_timeout_seconds,
    )


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    return int(value) if isinstance(value, int) else None


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(raw))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _is_retryable_status(exc: Exception) -> bool:
    return _status_code(exc) in {429, 500, 502, 503, 504}


def _is_timeout_exception(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, (asyncio.TimeoutError, httpx.TimeoutException)):
            return True
        if type(current).__name__ in {"APITimeoutError", "ReadTimeout", "WriteTimeout"}:
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_response_format_unsupported(exc: Exception) -> bool:
    if _status_code(exc) not in {400, 422}:
        return False
    text = str(exc).casefold()
    mentions_format = "response_format" in text or "json_object" in text
    unsupported = any(
        marker in text
        for marker in ("not supported", "unsupported", "does not support", "unknown parameter")
    )
    return mentions_format and unsupported


def is_streaming_unsupported(exc: Exception) -> bool:
    if _status_code(exc) not in {400, 405, 422}:
        return False
    text = str(exc).casefold()
    return "stream" in text and any(
        marker in text
        for marker in ("not supported", "unsupported", "does not support", "unknown parameter")
    )


def _usage_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if isinstance(value, dict):
        return value
    return {
        key: getattr(value, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if getattr(value, key, None) is not None
    } or None


def _append_usage_record(record: dict[str, Any]) -> None:
    log_value = os.environ.get("MODEL_USAGE_LOG_PATH", "").strip()
    if not log_value:
        return
    output = Path(log_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _record_model_request_started(
    *,
    operation: str,
    model: str,
    queue_elapsed_seconds: float,
) -> None:
    trace = dict(_MODEL_TRACE.get())
    operation_id = trace.get("operation_id")
    if operation_id:
        from app.services.operation_store import get_operation_store

        await get_operation_store().record_model_request_started(
            str(operation_id),
            request_kind=operation,
            model=model,
            consume_budget=bool(trace.get("enforce_budget", False)),
            queue_elapsed_seconds=queue_elapsed_seconds,
        )
    _append_usage_record(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event": "physical_request_started",
            "operation": operation,
            "model": model,
            "queue_elapsed_seconds": round(queue_elapsed_seconds, 6),
            **trace,
        }
    )


async def _record_completion_usage(
    *,
    operation: str,
    model: str,
    usage: Any,
    elapsed: float,
    finish_reason: str | None = None,
    response_chars: int | None = None,
) -> None:
    payload = _usage_payload(usage)
    logger.info(
        "LLM usage: operation=%s model=%s elapsed=%.2fs prompt_tokens=%s "
        "completion_tokens=%s total_tokens=%s",
        operation,
        model,
        elapsed,
        (payload or {}).get("prompt_tokens"),
        (payload or {}).get("completion_tokens"),
        (payload or {}).get("total_tokens"),
    )
    trace = dict(_MODEL_TRACE.get())
    operation_id = trace.get("operation_id")
    if operation_id:
        from app.services.operation_store import get_operation_store

        await get_operation_store().record_model_completion(
            str(operation_id),
            usage=payload,
            elapsed_seconds=elapsed,
        )
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event": "completion_usage",
        "operation": operation,
        "model": model,
        "elapsed_seconds": elapsed,
        "usage": payload,
        "finish_reason": finish_reason,
        "response_chars": response_chars,
        **trace,
    }
    _append_usage_record(record)


def _model_gate():
    return get_model_request_gate()


def _build_field_description(model: type[BaseModel], indent: int = 2) -> str:
    """Build a human-readable field list, recursively expanding nested models."""
    prefix = " " * indent
    lines: list[str] = []
    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        desc = field_info.description or ""
        required = field_info.is_required()
        req_tag = "required" if required else "optional"

        inner_model = _get_inner_basemodel(annotation)
        if inner_model:
            is_list = get_origin(annotation) is list or (
                get_origin(annotation) is not None
                and any(get_origin(a) is list for a in get_args(annotation) if a is not type(None))
            )
            if is_list or (get_origin(annotation) is list):
                nested_desc = _build_field_description(inner_model, indent + 4)
                lines.append(f'{prefix}"{name}": [ {nested_desc}, ... ] ({req_tag})')
            else:
                nested_desc = _build_field_description(inner_model, indent + 4)
                lines.append(f'{prefix}"{name}": {nested_desc} ({req_tag})')
        else:
            type_str = _annotation_to_str(annotation)
            line = f'{prefix}"{name}": {type_str} ({req_tag})'
            if desc:
                line += f"  // {desc}"
            lines.append(line)

    return "{\n" + "\n".join(lines) + "\n" + " " * (indent - 2) + "}"


def _get_inner_basemodel(annotation: Any) -> type[BaseModel] | None:
    """If annotation is or contains a BaseModel subclass, return it."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    # Handle Union types (e.g. X | None)
    args = get_args(annotation)
    if args:
        for a in args:
            if isinstance(a, type) and issubclass(a, BaseModel):
                return a
            if get_origin(a) is list:
                inner_args = get_args(a)
                if inner_args and isinstance(inner_args[0], type) and issubclass(inner_args[0], BaseModel):
                    return inner_args[0]
    return None


def _annotation_to_str(annotation: Any) -> str:
    """Convert a type annotation to a readable string like 'list[str]'."""
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        inner = _annotation_to_str(args[0]) if args else "any"
        return f"list[{inner}]"
    if origin is dict:
        args = get_args(annotation)
        key = _annotation_to_str(args[0]) if args else "str"
        value = _annotation_to_str(args[1]) if len(args) > 1 else "any"
        return f"dict[{key}, {value}]"
    if str(origin).endswith("Literal"):
        return " | ".join(json.dumps(item, ensure_ascii=False) for item in get_args(annotation))
    if origin is None:
        if annotation is None:
            return "null"
        name = getattr(annotation, "__name__", str(annotation))
        return name.replace("NoneType", "null")
    # Union types (e.g. str | None)
    args = get_args(annotation)
    if args:
        parts = [_annotation_to_str(a) for a in args]
        return " | ".join(parts)
    return str(annotation)


class LLMClient:
    """Thin wrapper around the OpenAI-compatible async client.

    Supports any provider (OpenAI / DeepSeek / Qwen / Ollama …) via base_url.
    Includes multi-modal vision support via a separate vision client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.llm_api_key
        self._base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=_network_timeout(),
            # Retries are owned by this wrapper so physical request counts are
            # visible and bounded. Never stack SDK retries underneath them.
            max_retries=0,
            http_client=build_model_http_client(),
        )

        self._vision_client = AsyncOpenAI(
            api_key=settings.effective_vision_api_key,
            base_url=settings.effective_vision_base_url,
            timeout=_network_timeout(),
            max_retries=0,
            http_client=build_model_http_client(),
        )
        self.vision_model = settings.vision_model

    async def _request(
        self,
        factory: Callable[[], Awaitable[Any]],
        *,
        operation: str,
        retry_transient: bool = True,
    ) -> Any:
        """Execute a bounded number of visible physical requests.

        Timeouts and stream interruptions are deliberately not replayed: the
        provider may still be computing and a second request could duplicate
        work or cost. Only explicit 429/temporary-service responses are retried.
        """
        attempts = max(1, settings.llm_request_attempts if retry_transient else 1)
        for attempt in range(1, attempts + 1):
            try:
                logger.info(
                    "LLM physical request: operation=%s attempt=%d/%d",
                    operation,
                    attempt,
                    attempts,
                )
                return await factory()
            except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                raise UpstreamResultUnknownError(
                    f"{operation} timed out; upstream result is unknown"
                ) from exc
            except Exception as exc:
                if _is_timeout_exception(exc):
                    raise UpstreamResultUnknownError(
                        f"{operation} timed out; upstream result is unknown"
                    ) from exc
                if not retry_transient or not _is_retryable_status(exc) or attempt >= attempts:
                    raise
                delay = _retry_after_seconds(exc)
                if delay is None:
                    delay = settings.llm_retry_backoff_seconds * (2 ** (attempt - 1))
                delay = min(delay, settings.llm_max_retry_after_seconds)
                logger.warning(
                    "Retrying transient LLM response: operation=%s status=%s delay=%.2fs",
                    operation,
                    _status_code(exc),
                    delay,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

    def _generation_options(self, strategy: str) -> dict[str, Any]:
        """Return provider-compatible bounded generation controls.

        Qwen/OpenAI-compatible endpoints accept provider extensions through
        extra_body. Other providers safely receive only standard max_tokens.
        """
        options: dict[str, Any] = {"max_tokens": settings.llm_max_output_tokens}
        base = str(getattr(self, "_base_url", "")).casefold()
        model = str(getattr(self, "model", settings.llm_model)).casefold()
        if "dashscope" in base or model.startswith("qwen"):
            enabled = (
                settings.llm_simple_enable_thinking
                if strategy == "simple"
                else settings.llm_complex_enable_thinking
            )
            extra: dict[str, Any] = {}
            if enabled is not None:
                extra["enable_thinking"] = enabled
            if enabled and settings.llm_complex_thinking_budget is not None:
                extra["thinking_budget"] = settings.llm_complex_thinking_budget
            if extra:
                options["extra_body"] = extra
        return options

    # ------------------------------------------------------------------
    # Normal chat completion
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        strategy: str = "simple",
    ) -> str:
        started_at = time.perf_counter()

        async def make_request():
            async with _model_gate().slot() as queue_elapsed:
                await _record_model_request_started(
                    operation="chat",
                    model=model or self.model,
                    queue_elapsed_seconds=queue_elapsed,
                )
                return await self._client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    **self._generation_options(strategy),
                )

        response = await self._request(make_request, operation="chat")
        await _record_completion_usage(
            operation="chat",
            model=model or self.model,
            usage=getattr(response, "usage", None),
            elapsed=time.perf_counter() - started_at,
        )
        content = response.choices[0].message.content or ""
        logger.debug("LLM response (%d chars): %s…", len(content), content[:120])
        logger.info(
            "LLM chat completed: model=%s elapsed=%.2fs output_chars=%d",
            model or self.model,
            time.perf_counter() - started_at,
            len(content),
        )
        return content

    async def chat_tool_call(
        self,
        messages: list[dict],
        tools: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Require exactly one provider-native function call."""
        started_at = time.perf_counter()

        async def make_request():
            async with _model_gate().slot() as queue_elapsed:
                await _record_model_request_started(
                    operation="tool_call",
                    model=model or self.model,
                    queue_elapsed_seconds=queue_elapsed,
                )
                return await self._client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    tools=tools,
                    tool_choice="required",
                    **self._generation_options("complex"),
                )

        response = await self._request(make_request, operation="tool_call")
        await _record_completion_usage(
            operation="tool_call",
            model=model or self.model,
            usage=getattr(response, "usage", None),
            elapsed=time.perf_counter() - started_at,
        )
        message = response.choices[0].message
        calls = list(getattr(message, "tool_calls", None) or [])
        if len(calls) != 1:
            raise ToolCallingUnavailableError(
                "configured model must return exactly one native tool call"
            )
        function = calls[0].function
        try:
            arguments = json.loads(function.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("model returned invalid JSON tool arguments") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        logger.info(
            "LLM tool call completed: model=%s tool=%s elapsed=%.2fs",
            model or self.model,
            function.name,
            time.perf_counter() - started_at,
        )
        return {
            "tool_call_id": str(calls[0].id),
            "name": str(function.name),
            "arguments": arguments,
        }

    # ------------------------------------------------------------------
    # Streaming chat completion
    # ------------------------------------------------------------------
    async def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        strategy: str = "simple",
    ) -> AsyncIterator[str]:
        started_at = time.perf_counter()
        async with _model_gate().slot() as queue_elapsed:
            first_attempt = True

            async def make_request():
                nonlocal first_attempt
                await _record_model_request_started(
                    operation="chat_stream",
                    model=model or self.model,
                    queue_elapsed_seconds=queue_elapsed if first_attempt else 0.0,
                )
                first_attempt = False
                return await self._client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    stream=True,
                    **self._generation_options(strategy),
                )

            stream = await self._request(make_request, operation="chat_stream")
            output_chars = 0
            first_token_elapsed: float | None = None
            stream_usage: Any = None
            try:
                async for chunk in stream:
                    if getattr(chunk, "usage", None) is not None:
                        stream_usage = chunk.usage
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        if first_token_elapsed is None:
                            first_token_elapsed = time.perf_counter() - started_at
                        output_chars += len(content)
                        yield content
            except Exception as exc:
                raise UpstreamResultUnknownError(
                    "chat_stream interrupted; upstream result is unknown"
                ) from exc
        logger.info(
            "LLM stream completed: model=%s first_token=%.2fs elapsed=%.2fs output_chars=%d",
            model or self.model,
            first_token_elapsed if first_token_elapsed is not None else -1.0,
            time.perf_counter() - started_at,
            output_chars,
        )
        await _record_completion_usage(
            operation="chat_stream",
            model=model or self.model,
            usage=stream_usage,
            elapsed=time.perf_counter() - started_at,
        )

    # ------------------------------------------------------------------
    # Multi-modal vision completion
    # ------------------------------------------------------------------
    async def chat_vision(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Send an image + text prompt to a vision-capable LLM.

        Encodes the image as base64 and uses the OpenAI vision API format.
        Uses the dedicated vision client/model from config.
        """
        image_path = Path(image_path)
        suffix = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/png")

        image_bytes = image_path.read_bytes()
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"

        logger.info(
            "Vision request: %s (%d bytes, %s)",
            image_path.name, len(image_bytes), model or self.vision_model,
        )

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        })

        started_at = time.perf_counter()

        async def make_request():
            async with _model_gate().slot() as queue_elapsed:
                await _record_model_request_started(
                    operation="vision",
                    model=model or self.vision_model,
                    queue_elapsed_seconds=queue_elapsed,
                )
                return await self._vision_client.chat.completions.create(
                    model=model or self.vision_model,
                    messages=messages,
                    temperature=temperature if temperature is not None else 0.1,
                    max_tokens=settings.llm_max_output_tokens,
                )

        response = await self._request(make_request, operation="vision")
        await _record_completion_usage(
            operation="vision",
            model=model or self.vision_model,
            usage=getattr(response, "usage", None),
            elapsed=time.perf_counter() - started_at,
        )

        content = response.choices[0].message.content or ""
        logger.info("Vision response: %d chars", len(content))
        return content

    # ------------------------------------------------------------------
    # Structured output — force JSON conforming to a Pydantic model
    # ------------------------------------------------------------------
    async def chat_structured(
        self,
        messages: list[dict],
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        strategy: str = "complex",
    ) -> T:
        """Return a validated Pydantic instance parsed from LLM JSON output.

        Retries up to MAX_STRUCTURED_RETRIES times if the LLM returns invalid
        JSON (e.g. echoes back the schema definition instead of real data).
        """
        field_desc = _build_field_description(response_model)

        system_suffix = (
            "\n\n## OUTPUT FORMAT\n"
            "You MUST respond with ONLY a valid JSON object (no markdown, no extra text).\n"
            f"The JSON must have these fields:\n{field_desc}\n\n"
            "LANGUAGE RULE: For every user-visible free-text value in the JSON, "
            "use Simplified Chinese. Keep technical names such as FastAPI, LangGraph, "
            "RAG, WebSocket, StateGraph, React, Redis, Docker and API in English when "
            "appropriate, but write explanations, evidence, summaries, suggestions, "
            "questions and follow-up directions in Chinese.\n\n"
            "IMPORTANT: Fill in actual values based on your analysis. "
            "Do NOT return the schema definition or type descriptions — return real data only."
        )

        last_error: Exception | None = None

        max_attempts = max(1, settings.structured_repair_attempts + 1)
        for attempt in range(1, max_attempts + 1):
            started_at = time.perf_counter()
            patched = list(messages)
            if attempt > 1:
                error_hint = str(last_error or "invalid output")[:1600]
                if isinstance(last_error, StructuredOutputTruncatedError):
                    retry_suffix = (
                        system_suffix
                        + "\n\nWARNING: Your previous JSON response was truncated by the output limit. "
                        "Return the same required object much more concisely: use short factual phrases, "
                        "avoid repeated explanations, keep arrays to the minimum requested by the user, "
                        "and always close every JSON string, array, and object."
                    )
                else:
                    retry_suffix = (
                        system_suffix
                        + "\n\nWARNING: Your previous response failed validation. "
                        f"Validation error:\n{error_hint}\n"
                        "Correct exactly these contract errors and return REAL VALUES. "
                        "Do not repeat invalid enum values, field shapes, nulls, or out-of-range numbers."
                    )
            else:
                retry_suffix = system_suffix

            if patched and patched[0]["role"] == "system":
                patched[0] = {
                    **patched[0],
                    "content": patched[0]["content"] + retry_suffix,
                }
            else:
                patched.insert(0, {"role": "system", "content": retry_suffix})

            async def json_request():
                async with _model_gate().slot() as queue_elapsed:
                    await _record_model_request_started(
                        operation=f"structured:{response_model.__name__}:json",
                        model=model or self.model,
                        queue_elapsed_seconds=queue_elapsed,
                    )
                    return await self._client.chat.completions.create(
                        model=model or self.model,
                        messages=patched,
                        temperature=temperature if temperature is not None else self.temperature,
                        response_format={"type": "json_object"},
                        **self._generation_options(strategy),
                    )

            try:
                response = await self._request(
                    json_request,
                    operation=f"structured:{response_model.__name__}:json",
                )
                raw = response.choices[0].message.content or "{}"
                finish_reason = str(
                    getattr(response.choices[0], "finish_reason", "") or ""
                ).casefold()
                await _record_completion_usage(
                    operation=f"structured:{response_model.__name__}",
                    model=model or self.model,
                    usage=getattr(response, "usage", None),
                    elapsed=time.perf_counter() - started_at,
                    finish_reason=finish_reason or None,
                    response_chars=len(raw),
                )
                if finish_reason in {
                    "length",
                    "max_tokens",
                    "max_completion_tokens",
                }:
                    last_error = StructuredOutputTruncatedError(
                        f"structured {response_model.__name__} output was truncated "
                        f"(finish_reason={finish_reason})"
                    )
                    logger.warning(
                        "Attempt %d/%d: structured output truncated response_model=%s "
                        "finish_reason=%s chars=%d",
                        attempt,
                        max_attempts,
                        response_model.__name__,
                        finish_reason,
                        len(raw),
                    )
                    continue
            except Exception as exc:
                if not _is_response_format_unsupported(exc):
                    raise
                logger.warning(
                    "Provider explicitly rejected response_format; using one plain request"
                )
                raw = await self.chat(
                    patched,
                    model=model,
                    temperature=temperature,
                    strategy=strategy,
                )

            raw = self._clean_json_response(raw)

            if self._looks_like_schema(raw):
                logger.warning(
                    "Attempt %d/%d: LLM returned schema definition instead of data, retrying...",
                    attempt, max_attempts,
                )
                continue

            try:
                result = response_model.model_validate_json(raw)
                if attempt > 1:
                    logger.info("Structured output succeeded on retry #%d", attempt)
                logger.info(
                    "LLM structured output completed: model=%s response_model=%s attempt=%d elapsed=%.2fs",
                    model or self.model,
                    response_model.__name__,
                    attempt,
                    time.perf_counter() - started_at,
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d: validation failed (%s), retrying...",
                    attempt, max_attempts, e,
                )

        raise last_error or ValueError("chat_structured failed after all retries")

    @staticmethod
    def _looks_like_schema(raw: str) -> bool:
        """Detect if the LLM returned a JSON Schema definition instead of data."""
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(obj, dict):
            return False
        schema_keys = {"type", "properties", "description", "$defs", "$schema", "definitions"}
        return bool(schema_keys & set(obj.keys()) and "type" in obj and obj["type"] == "object")

    @staticmethod
    def _clean_json_response(raw: str) -> str:
        """Strip markdown fences and extract the JSON object from LLM output."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        # Some models wrap JSON in extra text; try to extract the outermost { }
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

        return raw


_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
