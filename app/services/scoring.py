"""Deterministic aggregation for final question scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.models.interview import FinalQuestionAssessment
from app.models.jd import SkillMatrix
from app.domain.skill_identity import resolve_skill_identity


def grade_for_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 8.5:
        return "A"
    if score >= 7.0:
        return "B"
    if score >= 5.5:
        return "C"
    return "D"


@dataclass(frozen=True)
class ScoreSummary:
    overall_score: float | None
    grade: str | None
    skill_scores: dict[str, float]
    covered_skills: list[str]
    uncovered_skills: list[str]
    scored_questions: int
    unscored_questions: int


def aggregate_scores(
    assessments: Iterable[FinalQuestionAssessment],
    skill_matrix: SkillMatrix | None = None,
) -> ScoreSummary:
    assessments = list(assessments)
    valid = [item for item in assessments if item.scorable and item.score is not None]
    unscored = [item for item in assessments if not item.scorable or item.score is None]
    grouped: dict[str, list[int]] = {}
    identities = {}
    for item in valid:
        identity = resolve_skill_identity(item.primary_skill, skill_matrix)
        identities.setdefault(identity.skill_id, identity)
        grouped.setdefault(identity.skill_id, []).append(int(item.score))
    skill_scores = {
        identities[skill].display_name: round(sum(scores) / len(scores), 2)
        for skill, scores in grouped.items()
    }
    active_weights = {
        identity.display_name: max(identity.weight, 0.0)
        for identity in identities.values()
    }
    denominator = sum(active_weights.values())
    overall = None
    if skill_scores:
        if denominator <= 0:
            overall = sum(skill_scores.values()) / len(skill_scores)
        else:
            overall = sum(
                skill_scores[skill] * active_weights[skill] for skill in skill_scores
            ) / denominator
        overall = round(overall, 2)
    all_skills = [item.name for item in skill_matrix.skills] if skill_matrix else list(skill_scores)
    return ScoreSummary(
        overall_score=overall,
        grade=grade_for_score(overall),
        skill_scores=skill_scores,
        covered_skills=list(skill_scores),
        uncovered_skills=[
            skill for skill in all_skills
            if resolve_skill_identity(skill, skill_matrix).skill_id not in grouped
        ],
        scored_questions=len(valid),
        unscored_questions=len(unscored),
    )


def aggregate_professional_rounds(
    round1: float | None,
    round2: float | None,
) -> float | None:
    if round1 is None and round2 is None:
        return None
    if round1 is None:
        return round2
    if round2 is None:
        return round1
    return round(round1 * 0.55 + round2 * 0.45, 2)
