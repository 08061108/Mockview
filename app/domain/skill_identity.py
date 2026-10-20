"""Stable scoring identity, independent of retrieval's fuzzy skill aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.domain.skill_aliases import normalize_skill_text

if TYPE_CHECKING:
    from app.models.jd import SkillMatrix


def canonical_skill_id(value: str) -> str:
    # Keep the existing preparation ID format; do not merge related concepts
    # such as Redis/cache merely because retrieval treats them as aliases.
    return normalize_skill_text(value) or value.strip().casefold() or "unknown"


@dataclass(frozen=True)
class SkillIdentity:
    skill_id: str
    display_name: str
    weight: float


def resolve_skill_identity(value: str, matrix: SkillMatrix | None = None) -> SkillIdentity:
    skill_id = canonical_skill_id(value)
    if matrix:
        for skill in matrix.skills:
            if canonical_skill_id(skill.name) == skill_id:
                return SkillIdentity(skill_id, skill.name, skill.weight)
    # Historical/unlisted dimensions retain their label and default weight.
    return SkillIdentity(skill_id, value, 1.0)
