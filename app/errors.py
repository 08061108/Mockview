"""Stable public error classification shared by REST and WebSocket APIs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.models.public import PublicError

logger = logging.getLogger(__name__)


@dataclass
class PublicAppError(Exception):
    code: str
    message: str
    status_code: int
    retryable: bool = False
    retry_after_seconds: int | None = None

    def public(self) -> PublicError:
        return PublicError(
            code=self.code, message=self.message, retryable=self.retryable,
            retry_after_seconds=self.retry_after_seconds,
        )


class EvidenceMissingError(PublicAppError, RuntimeError):
    """Domain outcome: retrieval completed, but no evidence can ground a question."""

    def __init__(self) -> None:
        super().__init__(
            code="EVIDENCE_MISSING",
            message="当前知识库没有足够证据支持这些考察点，请调整材料、考察目标或知识库。",
            status_code=422,
            retryable=False,
        )
        self.args = ("No evidence-compatible interview questions were produced",)


def classify_public_error(exc: BaseException) -> PublicAppError:
    if isinstance(exc, PublicAppError):
        return exc
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    headers = getattr(exc, "headers", None) or {}
    request_id = getattr(exc, "request_id", None)
    logger.error("Upstream failure type=%s request_id=%s", type(exc).__name__, request_id)
    if status == 404 or "model_not_found" in text or "model not found" in text:
        return PublicAppError("MODEL_NOT_FOUND", "配置的模型不可用，请联系管理员检查模型配置。", 503)
    if status in {401, 403} or "authentication" in name or "permission" in name:
        return PublicAppError("MODEL_AUTH_FAILED", "模型服务鉴权失败，请联系管理员检查服务配置。", 503)
    if status == 429 or "ratelimit" in name or "rate limit" in text:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        try:
            retry_after = min(60, max(1, int(float(raw)))) if raw else None
        except (TypeError, ValueError):
            retry_after = None
        return PublicAppError("MODEL_RATE_LIMITED", "模型服务繁忙，请稍后重试。", 429, True, retry_after)
    if "timeout" in name or isinstance(exc, TimeoutError):
        return PublicAppError("MODEL_TIMEOUT", "模型处理超时，结果状态未知，请确认后再重试。", 504, False)
    if "unknown" in text and "result" in text:
        return PublicAppError("MODEL_RESULT_UNKNOWN", "模型结果状态未知，请确认后再重试。", 504, False)
    if "connection" in name or "connect" in name:
        return PublicAppError("MODEL_UNAVAILABLE", "模型服务暂时不可用，请稍后重试。", 502, True)
    if isinstance(status, int) and status >= 500:
        return PublicAppError("MODEL_UNAVAILABLE", "模型服务暂时不可用，请稍后重试。", 502, True)
    if isinstance(exc, (ValueError, TypeError)):
        return PublicAppError("INVALID_DOCUMENT", "文档内容无法解析，请检查文件后重试。", 422)
    return PublicAppError("INTERNAL_ERROR", "服务暂时无法完成请求，请稍后重试。", 500, True)


def public_http_exception(exc: BaseException) -> HTTPException:
    public = classify_public_error(exc)
    headers: dict[str, str] = {}
    if public.retry_after_seconds is not None:
        headers["Retry-After"] = str(public.retry_after_seconds)
    return HTTPException(
        status_code=public.status_code,
        detail=public.public().model_dump(mode="json"),
        headers=headers or None,
    )


def document_error(code: str, message: str, status_code: int) -> PublicAppError:
    return PublicAppError(code, message, status_code, False)
