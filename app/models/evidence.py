"""Evidence and validation contracts shared by Agentic RAG and knowledge review."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EvidenceChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_version: str = "legacy-v1"
    release_id: str | None = None
    text: str
    source_context: str = ""  # Parent title/tags for identity, never rubric support.
    location: str = ""
    source_type: Literal["question_bank", "public_web", "candidate", "private_material"]
    source_url: str | None = None
    record_id: str | None = None
    publication_status: Literal["published", "candidate", "private", "withdrawn"] = "published"
    scope: Literal["public", "private"] = "public"
    retrieval_sources: list[str] = Field(default_factory=list)
    ranking: dict[str, float] = Field(default_factory=dict)

    @property
    def citation_text(self) -> str:
        """Read legacy snapshots without trusting parent text under a child ID."""
        legacy_context = (
            self.source_type == "question_bank"
            and self.text.startswith("Source ID: ")
            and "\nCategory: " in self.text
        )
        if not legacy_context:
            return self.text
        if self.chunk_id != self.document_id:
            # Old snapshots did not retain the actual child body. Require fresh
            # evidence rather than inventing it or relabeling it as a parent.
            return ""
        return "\n".join(
            line for line in self.text.splitlines()
            if line.startswith(("Question or topic: ", "Reference points: ", "Follow-up directions: "))
        )


class EvidenceAssessment(BaseModel):
    target_id: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    supported_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    verdict: Literal["sufficient", "partial", "insufficient", "conflicting"]
    suggested_query: str | None = None
    citations: dict[str, list[str]] = Field(default_factory=dict)


class QuestionValidation(BaseModel):
    question_id: str
    target_match: Literal["supported", "contradicted", "uncertain"]
    rubric_support: Literal["supported", "contradicted", "uncertain"]
    answer_relevance: Literal["supported", "contradicted", "uncertain"]
    version_conflicts: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    unsupported_points: list[str] = Field(default_factory=list)
    citations: dict[str, list[str]] = Field(
        default_factory=dict,
        description="JSON object mapping each exact rubric point to an array of supporting chunk ID strings; never a list or scalar",
    )
    verdict: Literal["supported", "contradicted", "uncertain"]
    revision_reason: str | None = None

    @field_validator("target_match", "rubric_support", "answer_relevance", "verdict", mode="before")
    @classmethod
    def normalize_unsupported_to_uncertain(cls, value):
        return "uncertain" if value == "unsupported" else value

    @field_validator("citations", mode="before")
    @classmethod
    def reject_ambiguous_citations(cls, value):
        return value if isinstance(value, dict) else {}


class QuestionValidationBatch(BaseModel):
    validations: list[QuestionValidation] = Field(default_factory=list)


class SearchSnapshot(BaseModel):
    query: str
    url: str
    title: str
    content: str
    fetched_at: str
    content_hash: str
