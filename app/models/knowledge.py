"""Knowledge candidate review and immutable release models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeDocument(BaseModel):
    document_id: str
    version: int = 1
    owner_user_id: str | None = None
    scope: Literal["public", "private"] = "public"
    source_type: Literal["question_bank", "public_web", "manual", "private_material"]
    source_url: str | None = None
    title: str
    content: str
    content_hash: str
    status: Literal["candidate", "approved", "rejected", "withdrawn"] = "candidate"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class KnowledgeRelease(BaseModel):
    release_id: str
    status: Literal["building", "ready", "active", "failed", "retired"] = "building"
    document_versions: dict[str, int]
    chroma_collection: str
    bm25_snapshot: str
    embedding_model: str
    embedding_dimension: int | None = None
    tokenization_version: str = "zh-bigram-v1"
    chunking_version: str = "legacy-parent-child-v1"
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    activated_at: datetime | None = None


class KnowledgeCandidateCreate(BaseModel):
    source_type: Literal["public_web", "manual"] = "manual"
    source_url: str | None = None
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
