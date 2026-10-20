"""Non-blocking startup capability status for configured model endpoints."""

from __future__ import annotations

import logging
from typing import Literal

from app.config import settings
from app.errors import PublicAppError, classify_public_error
from app.llm.transport import build_model_http_client

logger = logging.getLogger(__name__)
Capability = Literal["available", "unavailable", "unknown"]
_capabilities: dict[str, Capability] = {"main": "unknown", "vision": "unknown"}


async def probe_model_capabilities() -> dict[str, Capability]:
    """Perform one model-list request per distinct endpoint without blocking startup."""
    try:
        from openai import AsyncOpenAI

        endpoints = {
            "main": (settings.llm_api_key, settings.llm_base_url, settings.llm_model),
            "vision": (
                settings.effective_vision_api_key,
                settings.effective_vision_base_url,
                settings.vision_model,
            ),
        }
        cached: dict[tuple[str, str], set[str] | PublicAppError] = {}
        for kind, (key, base_url, model) in endpoints.items():
            if not key:
                _capabilities[kind] = "unavailable"
                continue
            cache_key = (key, base_url)
            if cache_key not in cached:
                try:
                    http_client = build_model_http_client()
                    client = AsyncOpenAI(
                        api_key=key or "missing",
                        base_url=base_url,
                        max_retries=0,
                        http_client=http_client,
                    )
                    try:
                        response = await client.models.list(timeout=10.0)
                        cached[cache_key] = {item.id for item in response.data}
                    finally:
                        await client.close()
                except Exception as exc:
                    cached[cache_key] = classify_public_error(exc)
            result = cached[cache_key]
            if isinstance(result, PublicAppError):
                _capabilities[kind] = (
                    "unavailable"
                    if result.code in {"MODEL_NOT_FOUND", "MODEL_AUTH_FAILED"}
                    else "unknown"
                )
                logger.warning("Model capability probe failed for %s: %s", kind, result.code)
            else:
                _capabilities[kind] = "available" if model in result else "unavailable"
    except Exception:
        logger.exception("Unable to initialize model capability probe")
    return dict(_capabilities)


def get_model_capabilities() -> dict[str, Capability]:
    return dict(_capabilities)


def set_model_capabilities_for_test(**values: Capability) -> None:
    _capabilities.update(values)
