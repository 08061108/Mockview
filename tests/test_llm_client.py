from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

from app.llm.client import (
    LLMClient,
    StructuredOutputTruncatedError,
    UpstreamResultUnknownError,
    _annotation_to_str,
    model_trace,
)
from app.models.interview import FinalQuestionAssessmentDraft
from app.services.operation_store import structured_operation_error


@pytest.mark.asyncio
async def test_capability_probe_reuses_production_transport_and_one_models_request(monkeypatch):
    import openai
    from app.services import model_capabilities

    sentinel_transport = object()
    calls = {"init": 0, "list": 0, "close": 0}

    class Models:
        async def list(self, *, timeout):
            assert timeout == 10.0
            calls["list"] += 1
            return SimpleNamespace(data=[SimpleNamespace(id="main-model")])

    class Client:
        def __init__(self, **kwargs):
            calls["init"] += 1
            assert kwargs["http_client"] is sentinel_transport
            assert kwargs["max_retries"] == 0
            self.models = Models()

        async def close(self):
            calls["close"] += 1

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    monkeypatch.setattr(model_capabilities, "build_model_http_client", lambda: sentinel_transport)
    monkeypatch.setattr(model_capabilities.settings, "llm_api_key", "same-key")
    monkeypatch.setattr(model_capabilities.settings, "llm_base_url", "https://models.test/v1")
    monkeypatch.setattr(model_capabilities.settings, "llm_model", "main-model")
    monkeypatch.setattr(model_capabilities.settings, "vision_api_key", "same-key")
    monkeypatch.setattr(model_capabilities.settings, "vision_base_url", "https://models.test/v1")
    monkeypatch.setattr(model_capabilities.settings, "vision_model", "vision-model")

    result = await model_capabilities.probe_model_capabilities()

    assert result == {"main": "available", "vision": "unavailable"}
    assert calls == {"init": 1, "list": 1, "close": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (type("Provider401", (Exception,), {"status_code": 401})("bad key"), "unavailable"),
        (type("APIConnectionError", (Exception,), {})("network"), "unknown"),
    ],
)
async def test_capability_probe_classifies_endpoint_failure_once(monkeypatch, failure, expected):
    import openai
    from app.services import model_capabilities

    calls = 0

    class Models:
        async def list(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise failure

    class Client:
        def __init__(self, **_kwargs):
            self.models = Models()

        async def close(self):
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    monkeypatch.setattr(model_capabilities, "build_model_http_client", object)
    monkeypatch.setattr(model_capabilities.settings, "llm_api_key", "same-key")
    monkeypatch.setattr(model_capabilities.settings, "llm_base_url", "https://models.test/v1")
    monkeypatch.setattr(model_capabilities.settings, "llm_model", "main-model")
    monkeypatch.setattr(model_capabilities.settings, "vision_api_key", "same-key")
    monkeypatch.setattr(model_capabilities.settings, "vision_base_url", "https://models.test/v1")
    monkeypatch.setattr(model_capabilities.settings, "vision_model", "vision-model")

    result = await model_capabilities.probe_model_capabilities()

    assert result == {"main": expected, "vision": expected}
    assert calls == 1


@pytest.mark.asyncio
async def test_chat_vision_uses_plain_multimodal_chat_without_response_format(tmp_path: Path):
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="姓名：测试用户"))],
                usage=None,
            )

    client = LLMClient.__new__(LLMClient)
    client._vision_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.vision_model = "qwen-vl-ocr"

    async def direct(factory, **_kwargs):
        return await factory()

    client._request = direct
    image = tmp_path / "resume.png"
    image.write_bytes(b"png-test")
    result = await client.chat_vision(image, "提取文字")

    assert result == "姓名：测试用户"
    assert calls[0]["model"] == "qwen-vl-ocr"
    assert "response_format" not in calls[0]
    content = calls[0]["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]


class _AsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iterator = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **_kwargs):
        return _AsyncStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(chunks),
        )


@pytest.mark.asyncio
async def test_chat_stream_ignores_empty_provider_chunks():
    chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None),
                ),
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="第一段"),
                ),
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="第二段"),
                ),
            ],
        ),
    ]
    client = LLMClient.__new__(LLMClient)
    client._client = _FakeClient(chunks)
    client.model = "test-model"
    client.temperature = 0.7

    result = [part async for part in client.chat_stream(messages=[])]

    assert result == ["第一段", "第二段"]


def test_structured_type_description_preserves_mapping_and_literal_shape():
    assert _annotation_to_str(dict[str, list[str]]) == "dict[str, list[str]]"
    annotation = FinalQuestionAssessmentDraft.model_fields["hint_dependency"].annotation
    rendered = _annotation_to_str(annotation)
    assert '"none"' in rendered
    assert '"clarification"' in rendered


@pytest.mark.asyncio
async def test_structured_normal_stop_parses_without_repair():
    calls = 0

    class Completions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content='{"score":8,"scorable":true,"hint_dependency":"none"}'
                        ),
                    )
                ]
            )

    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client._base_url = "https://example.test/v1"
    client.model = "test-model"
    client.temperature = 0.1

    result = await client.chat_structured([], FinalQuestionAssessmentDraft)
    assert result.score == 8
    assert calls == 1


@pytest.mark.asyncio
async def test_structured_retry_includes_actual_validation_error():
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            content = (
                '{"score":100,"scorable":true,"hint_dependency":null}'
                if len(calls) == 1
                else '{"score":7,"scorable":true,"hint_dependency":"none"}'
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "test-model"
    client.temperature = 0.1
    result = await client.chat_structured(
        [{"role": "system", "content": "evaluate"}],
        FinalQuestionAssessmentDraft,
    )
    assert result.score == 7
    assert "less than or equal to 10" in calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_structured_truncation_uses_compact_repair_without_raising_token_limit():
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(content='{"score":7'),
                        )
                    ]
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content='{"score":7,"scorable":true,"hint_dependency":"none"}'
                        ),
                    )
                ]
            )

    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client._base_url = "https://example.test/v1"
    client.model = "test-model"
    client.temperature = 0.1
    result = await client.chat_structured([], FinalQuestionAssessmentDraft)

    assert result.score == 7
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == calls[1]["max_tokens"]
    assert "previous JSON response was truncated" in calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_structured_repeated_truncation_raises_specific_error():
    class Completions:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content='{"score":7'),
                    )
                ]
            )

    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client._base_url = "https://example.test/v1"
    client.model = "test-model"
    client.temperature = 0.1

    with pytest.raises(StructuredOutputTruncatedError, match="finish_reason=length") as caught:
        await client.chat_structured([], FinalQuestionAssessmentDraft)
    details = structured_operation_error(caught.value)
    assert details["retryable"] is True
    assert details["upstream_result_unknown"] is False


@pytest.mark.asyncio
async def test_structured_timeout_is_not_replayed_or_downgraded():
    calls = 0

    class Completions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("read timeout")

    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "test-model"
    client.temperature = 0.1
    with pytest.raises(UpstreamResultUnknownError, match="upstream result is unknown"):
        await client.chat_structured([], FinalQuestionAssessmentDraft)
    assert calls == 1


@pytest.mark.asyncio
async def test_structured_auth_error_is_not_downgraded_to_plain_chat():
    calls = 0

    class AuthError(RuntimeError):
        status_code = 401

    class Completions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise AuthError("invalid api key")

    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "test-model"
    client.temperature = 0.1
    with pytest.raises(AuthError):
        await client.chat_structured([], FinalQuestionAssessmentDraft)
    assert calls == 1


@pytest.mark.asyncio
async def test_structured_model_not_found_is_not_retried_or_downgraded():
    calls = 0

    class ModelNotFoundError(RuntimeError):
        status_code = 404

    class Completions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise ModelNotFoundError("model not found")

    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "missing-model"
    client.temperature = 0.1
    with pytest.raises(ModelNotFoundError):
        await client.chat_structured([], FinalQuestionAssessmentDraft)
    assert calls == 1


@pytest.mark.asyncio
async def test_structured_explicit_format_rejection_allows_one_plain_request():
    calls = []

    class FormatError(RuntimeError):
        status_code = 400

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("response_format"):
                raise FormatError("response_format json_object is not supported")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=(
                    '{"score":7,"scorable":true,"hint_dependency":"none"}'
                )))]
            )

    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "test-model"
    client.temperature = 0.1
    result = await client.chat_structured([], FinalQuestionAssessmentDraft)
    assert result.score == 7
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


@pytest.mark.asyncio
async def test_transient_rate_limit_has_one_bounded_visible_retry(monkeypatch):
    from app.llm import client as client_module

    calls = 0

    class RateLimitError(RuntimeError):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "0"})

    class Completions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RateLimitError("rate limited")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

    monkeypatch.setattr(client_module.settings, "llm_request_attempts", 2)
    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "test-model"
    client.temperature = 0.1
    result = await client.chat([])
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_transient_service_error_has_one_bounded_visible_retry(monkeypatch):
    from app.llm import client as client_module

    calls = 0

    class ServiceUnavailableError(RuntimeError):
        status_code = 503
        response = SimpleNamespace(headers={})

    class Completions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ServiceUnavailableError("temporarily unavailable")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

    monkeypatch.setattr(client_module.settings, "llm_request_attempts", 2)
    monkeypatch.setattr(client_module.settings, "llm_retry_backoff_seconds", 0)
    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "test-model"
    client.temperature = 0.1
    assert await client.chat([]) == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_physical_request_budget_is_consumed_before_provider_call(monkeypatch):
    from app.llm import client as client_module
    from app.services.operation_store import get_operation_store, reset_operation_store

    calls = 0

    class RateLimitError(RuntimeError):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "0"})

    class Completions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise RateLimitError("rate limited")

    monkeypatch.setattr(client_module.settings, "llm_request_attempts", 2)
    reset_operation_store()
    store = get_operation_store()
    await store.accept(
        user_id="u1",
        kind="budget-test",
        payload={"case": "bounded"},
        operation_id="budget-op",
        budget={"model_requests": 1},
    )
    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "test-model"
    client.temperature = 0.1
    with model_trace(operation_id="budget-op", enforce_budget=True):
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await client.chat([])
    operation = await store.get("budget-op")
    assert operation is not None
    assert calls == 1
    assert operation.budget["model_requests"] == 0
    assert operation.metrics["physical_requests"] == 1


def test_structured_operation_error_preserves_wrapped_unknown_result():
    from app.services.operation_store import structured_operation_error

    try:
        try:
            raise UpstreamResultUnknownError("provider result unknown")
        except UpstreamResultUnknownError as exc:
            raise RuntimeError("question generation failed") from exc
    except RuntimeError as wrapped:
        details = structured_operation_error(wrapped)
    assert details["type"] == "RuntimeError"
    assert details["cause_type"] == "UpstreamResultUnknownError"
    assert details["retryable"] is True
    assert details["upstream_result_unknown"] is True


@pytest.mark.asyncio
async def test_cancelled_queue_wait_does_not_consume_physical_request_budget(monkeypatch):
    from app.llm import scheduler
    from app.llm.scheduler import get_model_request_gate, reset_model_request_gate
    from app.services.operation_store import get_operation_store, reset_operation_store

    monkeypatch.setattr(scheduler.settings, "model_max_concurrency", 1)
    monkeypatch.setattr(scheduler.settings, "question_prefetch_global_limit", 1)
    reset_model_request_gate()
    reset_operation_store()
    gate = get_model_request_gate()
    release = asyncio.Event()
    acquired = asyncio.Event()
    provider_calls = 0

    async def hold_slot():
        async with gate.slot():
            acquired.set()
            await release.wait()

    class Completions:
        async def create(self, **_kwargs):
            nonlocal provider_calls
            provider_calls += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

    holder = asyncio.create_task(hold_slot())
    await acquired.wait()
    store = get_operation_store()
    await store.accept(
        user_id="u1",
        kind="queue-cancel-test",
        payload={},
        operation_id="queue-cancel-operation",
        budget={"model_requests": 1},
    )
    client = LLMClient.__new__(LLMClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client.model = "test-model"
    client.temperature = 0.1

    async def wait_for_model_slot():
        with model_trace(operation_id="queue-cancel-operation", enforce_budget=True):
            await client.chat([])

    queued = asyncio.create_task(wait_for_model_slot())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    operation = await store.get("queue-cancel-operation")
    assert operation is not None
    assert provider_calls == 0
    assert operation.budget["model_requests"] == 1
    assert operation.metrics.get("physical_requests", 0) == 0

    release.set()
    await holder
    reset_model_request_gate()
