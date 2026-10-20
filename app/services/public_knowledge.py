"""Tavily search/extract adapter with explicit opt-in and timeouts."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.models.evidence import EvidenceChunk, SearchSnapshot


class PublicKnowledgeUnavailable(RuntimeError):
    pass


def is_retryable_public_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def sanitize_public_query(query: str) -> str:
    """Remove common contact identifiers before sending technical queries."""
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email]", query)
    value = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[phone]", value)
    value = re.sub(r"(?<!\d)\d{14,17}[0-9Xx](?!\d)", "[identifier]", value)
    value = re.sub(r"https?://\S+", "[url]", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()[:400]


def build_public_technical_query(target, missing_aspects: list[str] | None = None) -> str:
    """Build a target-bound public query without including resume material.

    The caller-provided Agent query is deliberately not used. Assessment
    targets are built from the JD and contain only the public technical scope;
    personal-experience targets are not eligible for public correction.
    """

    if getattr(target, "gap_kind", "technical_knowledge") != "technical_knowledge":
        raise ValueError("public search cannot fill personal experience gaps")
    scope = str(getattr(target, "scope", "") or "").strip()
    goal = str(getattr(target, "verification_goal", "") or "").strip()
    missing = " ".join(str(item).strip() for item in (missing_aspects or []) if str(item).strip())
    query = sanitize_public_query(f"{scope} {goal} {missing} official documentation best practices")
    if not scope or not query:
        raise ValueError("technical assessment target is missing a public scope")
    return query


class PublicKnowledgeClient:
    def _assert_enabled(self) -> None:
        if not settings.public_search_enabled or not settings.tavily_api_key:
            raise PublicKnowledgeUnavailable("public search is not configured")

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        self._assert_enabled()
        timeout = httpx.Timeout(settings.search_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": min(max_results, settings.public_search_max_results),
                    "search_depth": "basic",
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
            return list(response.json().get("results") or [])

    async def extract(self, urls: list[str]) -> list[SearchSnapshot]:
        self._assert_enabled()
        if not urls:
            return []
        timeout = httpx.Timeout(settings.extract_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.tavily.com/extract",
                json={"api_key": settings.tavily_api_key, "urls": urls[:12]},
            )
            response.raise_for_status()
            rows = []
            for item in response.json().get("results") or []:
                content = str(item.get("raw_content") or item.get("content") or "")[
                    : settings.public_snapshot_max_chars
                ]
                rows.append(
                    SearchSnapshot(
                        query="",
                        url=str(item.get("url") or ""),
                        title=str(item.get("title") or item.get("url") or ""),
                        content=content,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    )
                )
            return rows


def snapshots_to_evidence(snapshots: list[SearchSnapshot]) -> list[EvidenceChunk]:
    chunks = []
    remaining = settings.public_snapshot_total_max_chars
    for item in snapshots:
        if remaining <= 0:
            break
        text = item.content[: min(settings.public_snapshot_max_chars, remaining)]
        remaining -= len(text)
        if not text.strip():
            continue
        document_id = hashlib.sha256(item.url.encode("utf-8")).hexdigest()
        chunks.append(
            EvidenceChunk(
                chunk_id=f"web:{document_id}:0",
                document_id=document_id,
                document_version=item.content_hash[:12],
                text=text,
                location="web body",
                source_type="public_web",
                source_url=item.url,
                publication_status="candidate",
                scope="public",
                retrieval_sources=["tavily_extract"],
            )
        )
    return chunks
