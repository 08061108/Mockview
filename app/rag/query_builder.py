"""Rule-based query expansion for interview question retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from app.domain.skill_aliases import retrieval_aliases
from app.models.jd import SkillItem, SkillMatrix
from app.models.preparation import AssessmentTarget
from app.models.resume import ResumeJDMatch, ResumeProfile

if TYPE_CHECKING:
    from app.agents.state import InterviewState


RetrievalPurpose = Literal["practice", "round1", "round2"]
QueryStrategy = Literal["legacy", "target_expansion_v1", "target_expansion_v2"]


@dataclass(frozen=True)
class RetrievalQuery:
    """A retrieval query plus a small weight for multi-query fusion."""

    text: str
    purpose: str
    priority: float = 1.0


def build_retrieval_queries(
    state: InterviewState,
    purpose: RetrievalPurpose,
    max_queries: int = 5,
    *,
    strategy: QueryStrategy = "legacy",
    max_expansion_queries: int = 3,
) -> list[RetrievalQuery]:
    """Build a compact set of retrieval queries from interview state.

    This is intentionally deterministic and lightweight. It gives the RAG
    layer enough coverage without adding another LLM call before planning.
    """

    skill_matrix = state["skill_matrix"]
    top_skills = _top_skills(skill_matrix, limit=6)
    required_skills = _top_skills(skill_matrix, limit=5, required_only=True)
    system_skills = _top_skills(skill_matrix, limit=4, categories={"system_design"})
    framework_skills = _top_skills(
        skill_matrix,
        limit=4,
        categories={"framework", "domain"},
    )

    queries: list[RetrievalQuery] = []
    base = _join_skill_names(top_skills)
    if base:
        queries.append(RetrievalQuery(base, "top_skills", 1.0))

    required = _join_skill_names(required_skills)
    if required and required != base:
        queries.append(RetrievalQuery(f"{required} interview deep dive", "required_skills", 0.95))

    if framework_skills:
        queries.append(
            RetrievalQuery(
                f"{_join_skill_names(framework_skills)} project experience pitfalls",
                "framework_or_domain",
                0.85,
            )
        )

    if purpose == "practice":
        queries.append(
            RetrievalQuery(
                f"{base} implementation principles common mistakes",
                "practice_depth",
                0.8,
            )
        )

    if purpose == "round1":
        queries.extend(_resume_aware_queries(state, top_skills))

    if purpose == "round2":
        if system_skills:
            queries.append(
                RetrievalQuery(
                    f"{_join_skill_names(system_skills)} system design scalability reliability",
                    "system_design",
                    1.0,
                )
            )
        queries.append(
            RetrievalQuery(
                f"{base} AI agent RAG LangGraph latest technology architecture",
                "ai_breadth",
                0.95,
            )
        )
        summary_probe = _round1_probe_text(state)
        if summary_probe:
            queries.append(RetrievalQuery(summary_probe, "round1_probe", 0.9))

    weak_skills = _state_list(state, "weak_skills")
    if weak_skills:
        queries.append(
            RetrievalQuery(
                f"{' '.join(weak_skills[:5])} weakness follow-up fundamentals",
                "weak_skills",
                0.9,
            )
        )

    base_queries = _dedupe_queries(queries, max_queries=max_queries)
    if strategy == "legacy":
        return base_queries
    if strategy not in {"target_expansion_v1", "target_expansion_v2"}:
        raise ValueError(f"unsupported RAG query strategy: {strategy}")
    return _dedupe_queries(
        [
            *base_queries,
            *build_target_expansion_queries(
                state,
                strategy=strategy,
                max_queries=max_expansion_queries,
            ),
        ],
        max_queries=max_queries + max_expansion_queries,
    )


def build_target_expansion_queries(
    state: InterviewState,
    *,
    strategy: Literal[
        "target_expansion_v1", "target_expansion_v2"
    ] = "target_expansion_v1",
    max_queries: int = 3,
) -> list[RetrievalQuery]:
    """Build stable production candidate queries from frozen assessment targets."""

    if max_queries <= 0:
        return []
    targets = [
        item if isinstance(item, AssessmentTarget) else AssessmentTarget.model_validate(item)
        for item in (state.get("assessment_targets", []) or [])
    ]
    targets.sort(key=lambda item: (-item.importance, item.target_id))
    expanded: list[RetrievalQuery] = []
    for target in targets:
        if strategy == "target_expansion_v1":
            text = f"{target.scope} {target.verification_goal}"
            purpose = "assessment_target"
        else:
            aliases = retrieval_aliases([target.scope, target.skill_id], limit=8)
            text = " ".join(
                [
                    target.scope,
                    *aliases,
                    "原理 场景 工程边界 implementation pitfalls interview",
                ]
            )
            purpose = "assessment_target_aliases"
        expanded.append(
            RetrievalQuery(text, purpose, max(0.6, target.importance))
        )
    if not expanded:
        expanded = [
            RetrievalQuery(
                f"{skill.name} 原理 项目实践 工程边界 面试题",
                "target_skill_expansion",
                max(0.6, skill.weight),
            )
            for skill in _top_skills(state["skill_matrix"], limit=max_queries)
        ]
    return _dedupe_queries(expanded, max_queries=max_queries)


def build_target_recovery_queries(
    target: AssessmentTarget,
    requested_query: str,
    *,
    max_queries: int = 3,
) -> list[RetrievalQuery]:
    """Build bounded corrective queries without mixing broad and narrow aliases."""

    if max_queries <= 0:
        return []
    internal_markers = {
        "technical_knowledge",
        "personal_experience",
        "evidence_missing",
        "evidence_partial",
    }
    requested = " ".join(
        token
        for token in str(requested_query).split()
        if token.casefold() not in internal_markers
    )
    queries = [RetrievalQuery(requested, "agent_requested", 1.0)]
    queries.append(
        RetrievalQuery(
            f"{target.scope} {target.verification_goal}",
            "target_verification",
            1.0,
        )
    )
    # The scope can also match a broad family such as RAG.  Expanding only the
    # frozen skill identifier keeps the corrective alias query specific.
    aliases = retrieval_aliases([target.skill_id], limit=6)
    alias_text = " ".join(
        item
        for item in aliases
        if item.casefold() != target.skill_id.casefold()
    )
    if alias_text:
        broad_prefix = target.scope.split()[0] if target.scope.split() else ""
        queries.append(
            RetrievalQuery(
                f"{broad_prefix} {alias_text}".strip(),
                "target_skill_aliases",
                0.95,
            )
        )
    return _dedupe_queries(queries, max_queries=max_queries)


def _resume_aware_queries(
    state: InterviewState,
    top_skills: list[SkillItem],
) -> list[RetrievalQuery]:
    queries: list[RetrievalQuery] = []
    profile = state.get("resume_profile")
    match = state.get("resume_jd_match")

    if isinstance(match, ResumeJDMatch):
        if match.matched_skills:
            queries.append(
                RetrievalQuery(
                    f"{' '.join(match.matched_skills[:6])} resume project deep dive",
                    "resume_matched_skills",
                    1.0,
                )
            )
        if match.missing_skills:
            queries.append(
                RetrievalQuery(
                    f"{' '.join(match.missing_skills[:5])} interview fundamentals",
                    "resume_missing_skills",
                    0.8,
                )
            )
        if match.interview_focus:
            queries.append(
                RetrievalQuery(
                    " ".join(match.interview_focus[:3]),
                    "resume_interview_focus",
                    0.9,
                )
            )

    if isinstance(profile, ResumeProfile):
        project_terms = _project_terms(profile)
        if project_terms:
            queries.append(
                RetrievalQuery(
                    f"{' '.join(project_terms)} {' '.join(s.name for s in top_skills[:3])} project architecture",
                    "resume_projects",
                    0.95,
                )
            )

    return queries


def _top_skills(
    skill_matrix: SkillMatrix,
    limit: int,
    required_only: bool = False,
    categories: set[str] | None = None,
) -> list[SkillItem]:
    skills = skill_matrix.skills
    if required_only:
        skills = [s for s in skills if s.is_required]
    if categories:
        skills = [s for s in skills if s.category in categories]
    return sorted(skills, key=lambda s: (s.is_required, s.weight), reverse=True)[:limit]


def _join_skill_names(skills: list[SkillItem]) -> str:
    return " ".join(s.name for s in skills if s.name).strip()


def _project_terms(profile: ResumeProfile) -> list[str]:
    terms: list[str] = []
    for project in profile.projects[:3]:
        terms.append(project.name)
        terms.extend(project.tech_stack[:4])
    return _unique_terms(terms)[:10]


def _round1_probe_text(state: InterviewState) -> str:
    raw = state.get("round1_summary_text", "")
    if not raw:
        return ""
    text = " ".join(str(raw).split())
    return text[:300]


def _state_list(state: InterviewState, key: str) -> list[str]:
    value = state.get(key)  # type: ignore[literal-required]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _unique_terms(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        normalized = str(item).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _dedupe_queries(
    queries: list[RetrievalQuery],
    max_queries: int,
) -> list[RetrievalQuery]:
    if max_queries <= 0:
        return []
    seen: set[str] = set()
    unique: list[RetrievalQuery] = []
    for query in queries:
        text = " ".join(query.text.split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        unique.append(RetrievalQuery(text=text, purpose=query.purpose, priority=query.priority))
        if len(unique) >= max_queries:
            break
    return unique
