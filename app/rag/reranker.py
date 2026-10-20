"""Lightweight metadata-aware reranking for retrieved RAG chunks."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

from app.models.resume import ResumeJDMatch
from app.rag.postprocess import metadata_list, result_category, result_difficulty

if TYPE_CHECKING:
    from app.agents.state import InterviewState


def rerank_results(
    state: InterviewState,
    results: list[Any],
    purpose: str,
    *,
    strategy: str = "legacy",
    retrieval_weight: float = 0.85,
) -> list[Any]:
    """Apply simple business-aware reranking on top of retrieval scores.

    This is intentionally rule-based: vector/BM25/RRF handles broad recall,
    while this layer nudges candidates that better match the interview context.
    """

    if not results:
        return []

    skill_matrix = state["skill_matrix"]
    required_skills = {
        _norm(skill.name)
        for skill in skill_matrix.skills
        if skill.is_required
    }
    all_skills = {_norm(skill.name) for skill in skill_matrix.skills}

    matched_resume_skills: set[str] = set()
    missing_resume_skills: set[str] = set()
    match = state.get("resume_jd_match")
    if isinstance(match, ResumeJDMatch):
        matched_resume_skills = {_norm(skill) for skill in match.matched_skills}
        missing_resume_skills = {_norm(skill) for skill in match.missing_skills}

    reranked = []
    max_score = max(float(getattr(item, "score", 0.0)) for item in results) or 1.0
    for rank, result in enumerate(results, 1):
        metadata = getattr(result, "metadata", {})
        tags = {_norm(tag) for tag in metadata_list(metadata, "skill_tags")}
        text_terms = _norm(
            " ".join([
                str(metadata.get("chunk_text", "")),
                str(metadata.get("content", "")),
                str(metadata.get("category", "")),
            ])
        )

        bonus = 0.0
        if _matches(tags, text_terms, required_skills):
            bonus += 0.15
        if _matches(tags, text_terms, all_skills):
            bonus += 0.08
        if matched_resume_skills and _matches(tags, text_terms, matched_resume_skills):
            bonus += 0.10
        if missing_resume_skills and _matches(tags, text_terms, missing_resume_skills):
            bonus += 0.06
        if metadata.get("chunk_type") in {"question_stem", "question_summary", "generated_query"}:
            bonus += 0.04
        if purpose == "round2" and result_category(result) in {
            "System Design",
            "AI Latest Technologies & Trends",
            "Machine Learning & AI",
        }:
            bonus += 0.10
        if purpose in {"round1", "round2"} and result_difficulty(result) in {"medium", "hard"}:
            bonus += 0.04

        # Slightly favor earlier retrieval ranks while preserving the original score scale.
        rank_bonus = 0.02 / max(rank, 1)
        candidate = copy(result)
        if strategy == "normalized":
            # Context may reorder close candidates, but cannot swamp retrieval
            # scores whose RRF scale is much smaller than the legacy bonuses.
            context_score = min(1.0, bonus / 0.47)
            candidate.score = (
                retrieval_weight * (float(result.score) / max_score)
                + (1.0 - retrieval_weight) * context_score
            )
        else:
            candidate.score = result.score + bonus + rank_bonus
        reranked.append(candidate)

    return sorted(reranked, key=lambda item: item.score, reverse=True)


def _matches(tags: set[str], text: str, targets: set[str]) -> bool:
    if not targets:
        return False
    if tags.intersection(targets):
        return True
    return any(target and target in text for target in targets)


def _norm(value: str) -> str:
    return " ".join(str(value).lower().split())
