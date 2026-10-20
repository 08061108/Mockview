"""Shared HTTP transport for OpenAI-compatible model providers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import settings


async def _record_model_response(response: httpx.Response) -> None:
    """Append sanitized request/usage metadata when an eval explicitly enables it."""
    log_value = os.environ.get("MODEL_USAGE_LOG_PATH", "").strip()
    if not log_value:
        return
    # Response hooks run as soon as headers arrive. Reading here would consume
    # an entire streaming completion before the caller can observe its first
    # token. Usage is recorded by the LLM wrapper after non-stream responses;
    # this transport hook deliberately records headers/request metadata only.
    try:
        request_body = json.loads(response.request.content or b"{}")
    except Exception:
        request_body = {}
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event": "http_response",
        "method": response.request.method,
        "path": response.request.url.path,
        "status_code": response.status_code,
        "model": request_body.get("model") if isinstance(request_body, dict) else None,
        "usage": None,
        "request_id": response.headers.get("x-request-id"),
    }
    output = Path(log_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_model_http_client() -> httpx.AsyncClient:
    """Build the model client with deterministic proxy and address-family policy.

    Windows can prefer a provider IPv6 edge that closes the TLS handshake while
    the IPv4 edge is healthy.  Bind the local side to IPv4 when configured so
    httpcore resolves/connects using AF_INET.  Model traffic ignores ambient
    proxy variables by default; deployments that intentionally require them can
    set MODEL_TRUST_ENV=true.
    """
    proxy = settings.model_proxy_url.strip() or None
    transport = None
    if settings.model_force_ipv4 and not proxy:
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
    return httpx.AsyncClient(
        transport=transport,
        proxy=proxy,
        timeout=httpx.Timeout(
            connect=settings.model_connect_timeout_seconds,
            read=settings.model_read_timeout_seconds,
            write=settings.model_write_timeout_seconds,
            pool=settings.model_pool_timeout_seconds,
        ),
        # An explicit model-only proxy is authoritative and must not be mixed
        # with ambient proxy variables.
        trust_env=(settings.model_trust_env if proxy is None else False),
        event_hooks={"response": [_record_model_response]},
    )
