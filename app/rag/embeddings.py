"""Embedding service wrapping OpenAI-compatible embedding API."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config import settings
from app.llm.transport import build_model_http_client

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Async embedding client compatible with any OpenAI-format API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.effective_embedding_api_key
        self.base_url = base_url or settings.effective_embedding_base_url
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Create the SDK client only when an embedding call is requested.

        Recent OpenAI SDK versions validate credentials during construction.
        Keeping construction lazy lets storage factories and offline tests load
        without a model credential while still failing clearly before any call.
        """
        if not self.api_key:
            raise RuntimeError(
                "Embedding API key is not configured. Set EMBEDDING_API_KEY "
                "or LLM_API_KEY."
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=settings.model_timeout_seconds,
                max_retries=settings.llm_transport_retries,
                http_client=build_model_http_client(),
            )
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returns list of float vectors."""
        if not texts:
            return []
        response = await self._get_client().embeddings.create(
            model=self.model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        logger.debug("Embedded %d texts, dim=%d", len(texts), len(embeddings[0]))
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self.embed_texts([query])
        return result[0]


_default_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service
