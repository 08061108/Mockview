"""Preparation cases and explainable assessment-target generation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.agents.jd_analyst import analyze_jd
from app.agents.resume_analyst import analyze_resume
from app.domain.skill_aliases import find_skill_mention, normalize_skill_text
from app.domain.skill_identity import canonical_skill_id
from app.models.jd import SkillMatrix
from app.models.preparation import (
    AssessmentTarget,
    ClarificationRequest,
    MaterialCitation,
    PreparationCase,
    PreparationCreate,
    PreparationUpdate,
    RetrainingRequest,
)
from app.models.resume import ResumeProfile
from app.resume.matcher import match_resume_to_jd
from app.memory.models import SkillMemory
from app.memory.service import get_memory_service
from app.services.preparation_repository import load_preparation_case, save_preparation_case

logger = logging.getLogger(__name__)


def _citation(kind: str, revision: int, source: str, needle: str) -> MaterialCitation:
    lowered = source.casefold()
    start = lowered.find(needle.casefold()) if needle else -1
    if start < 0:
        start = None
        end = None
        quote = source[:240]
    else:
        end = start + len(needle)
        quote = source[max(0, start - 80) : min(len(source), end + 120)]
    return MaterialCitation(
        material_type=kind,
        revision=revision,
        quote=quote,
        start=start,
        end=end,
    )


def build_assessment_targets(
    case: PreparationCase,
    matrix: SkillMatrix,
    profile: ResumeProfile | None = None,
) -> list[AssessmentTarget]:
    resume = case.resume_text
    project = case.project_notes
    resume_match = match_resume_to_jd(profile, matrix) if profile else None
    targets: list[AssessmentTarget] = []
    for index, skill in enumerate(matrix.skills, 1):
        normalized = normalize_skill_text(skill.name)
        citations = [_citation("jd", case.material_revision, case.jd_text, skill.name)]
        sources: list[str] = ["job_requirement"]
        resume_mention = find_skill_mention(resume, skill.name)
        project_mention = find_skill_mention(project, skill.name)
        profile_match = bool(
            resume_match and skill.name in resume_match.matched_skills
        )
        has_claim = bool(resume_mention or project_mention or profile_match)
        if has_claim:
            source_text = case.project_notes if project_mention else case.resume_text
            material_type = "project_note" if project_mention else "resume"
            mention = project_mention or resume_mention or skill.name
            citations.append(_citation(material_type, case.material_revision, source_text, mention))
            sources.append("project_claim")
        requirement_id = f"requirement:{index}:{normalized}"
        target_id = uuid5(
            NAMESPACE_URL,
            f"mockview:{case.case_id}:{case.material_revision}:{requirement_id}",
        ).hex
        verification = (
            f"验证候选人是否能结合真实项目说明 {skill.name} 的设计、边界与取舍"
            if has_claim
            else f"验证候选人对岗位要求 {skill.name} 的原理、场景与工程边界的掌握"
        )
        targets.append(
            AssessmentTarget(
                target_id=target_id,
                case_id=case.case_id,
                material_revision=case.material_revision,
                requirement_id=requirement_id,
                skill_id=canonical_skill_id(skill.name),
                scope=skill.name,
                importance=skill.weight,
                sources=sources,  # type: ignore[arg-type]
                citations=citations,
                verification_goal=verification,
                constraints=["不得将未写明的经历推断为不会", "资料不足时请求澄清或缩小范围"],
                gap_kind="personal_experience" if has_claim else "technical_knowledge",
                resume_signal="claimed" if has_claim else "unmentioned",
                status="pending",
            )
        )
    return targets


def enrich_targets_with_memory_and_preferences(
    case: PreparationCase,
    targets: list[AssessmentTarget],
    weak_skills: list[SkillMemory],
) -> list[AssessmentTarget]:
    focus = {
        str(item).casefold()
        for item in (
            case.preferences.get("focus_skills")
            or case.preferences.get("selected_skills")
            or []
        )
        if str(item).strip()
    }
    memories = {item.skill_name.casefold(): item for item in weak_skills}
    enriched: list[AssessmentTarget] = []
    for target in targets:
        normalized = target.scope.casefold()
        memory = memories.get(normalized)
        if memory is None:
            memory = next(
                (
                    item
                    for key, item in memories.items()
                    if key in normalized or normalized in key
                ),
                None,
            )
        updates: dict = {}
        if memory:
            weak_points = [item for item in memory.weak_points[:3] if item]
            updates["sources"] = list(dict.fromkeys([*target.sources, "historical_observation"]))
            updates["citations"] = [
                *target.citations,
                MaterialCitation(
                    material_type="observation",
                    revision=case.material_revision,
                    source_id=(memory.evidence_memory_ids[0] if memory.evidence_memory_ids else memory.id),
                    quote="；".join(weak_points) or f"历史观察显示 {memory.skill_name} 需要复练",
                ),
            ]
            if weak_points:
                updates["verification_goal"] = (
                    target.verification_goal
                    + "；复核历史薄弱点："
                    + "、".join(weak_points)
                )
            updates["importance"] = min(
                1.0, max(target.importance, memory.next_practice_priority)
            )
        if normalized in focus:
            updates["importance"] = 1.0
            updates["constraints"] = list(
                dict.fromkeys([*target.constraints, "优先覆盖用户本次指定的练习能力"])
            )
        enriched.append(target.model_copy(update=updates) if updates else target)
    return enriched


class PreparationService:
    def __init__(self) -> None:
        self._cases: dict[str, PreparationCase] = {}
        self._lock = asyncio.Lock()

    async def create(self, user_id: str, request: PreparationCreate) -> PreparationCase:
        case = PreparationCase(case_id=uuid4().hex, user_id=user_id, **request.model_dump())
        async with self._lock:
            self._cases[case.case_id] = case
        await save_preparation_case(case)
        return case

    async def get(self, case_id: str, user_id: str) -> PreparationCase | None:
        case = self._cases.get(case_id)
        if not case:
            case = await load_preparation_case(case_id, user_id)
            if case:
                self._cases[case_id] = case
        return case if case and case.user_id == user_id else None

    async def update(
        self, case_id: str, user_id: str, request: PreparationUpdate
    ) -> PreparationCase:
        existing = await self.get(case_id, user_id)
        if not existing:
            raise KeyError(case_id)
        async with self._lock:
            case = self._cases.get(case_id) or existing
            if not case or case.user_id != user_id:
                raise KeyError(case_id)
            changes = request.model_dump(exclude_none=True)
            if request.resume_text is not None:
                changes["resume_parse_warnings"] = []
            updated = case.model_copy(
                update={
                    **changes,
                    "material_revision": case.material_revision + 1,
                    "knowledge_release_id": None,
                    "status": "draft",
                    "targets": [],
                    "clarifications": [],
                    "evidence_chunks": {},
                    "evidence_assessments": {},
                    "skill_matrix": None,
                    "resume_profile": None,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._cases[case_id] = updated
            await save_preparation_case(updated)
            return updated

    async def analyze(self, case_id: str, user_id: str) -> PreparationCase:
        case = await self.get(case_id, user_id)
        if not case:
            raise KeyError(case_id)
        if (
            case.skill_matrix is not None
            and case.targets
            and case.analysis_strategy_version == "material-analysis-v1"
        ):
            logger.info(
                "Reusing frozen material analysis: case=%s revision=%d",
                case.case_id,
                case.material_revision,
            )
            return case
        revision = case.material_revision
        analyzing = case.model_copy(
            update={"status": "analyzing", "updated_at": datetime.now(timezone.utc)}
        )
        self._cases[case_id] = analyzing
        await save_preparation_case(analyzing)
        jd_result = await analyze_jd({"jd_text": case.jd_text})
        profile = None
        if case.resume_text.strip():
            resume_result = await analyze_resume({"resume_text": case.resume_text})
            profile = resume_result.get("resume_profile")
        current = await self.get(case_id, user_id)
        if not current or current.material_revision != revision:
            raise RuntimeError("materials changed while analysis was running")
        targets = build_assessment_targets(current, jd_result["skill_matrix"], profile)
        try:
            memory_context = await get_memory_service().abuild_structured_context(user_id)
            targets = enrich_targets_with_memory_and_preferences(
                current, targets, memory_context.weak_skills
            )
        except Exception:
            logger.exception("Unable to enrich preparation targets with memory: %s", case_id)
        clarifications: list[ClarificationRequest] = []
        concerns = list(getattr(profile, "concerns", []) or [])
        warnings = list(getattr(profile, "parse_warnings", []) or [])
        for index, text in enumerate([*concerns, *warnings][:3], 1):
            clarifications.append(
                ClarificationRequest(
                    clarification_id=uuid5(
                        NAMESPACE_URL, f"mockview:{case_id}:{revision}:clarification:{index}:{text}"
                    ).hex,
                    case_id=case_id,
                    material_revision=revision,
                    question=f"请补充说明：{text}",
                    missing_information=text,
                )
            )
        status = "waiting_input" if clarifications else "ready"
        updated = current.model_copy(
            update={
                "targets": targets,
                "skill_matrix": jd_result["skill_matrix"],
                "resume_profile": profile,
                "clarifications": clarifications,
                "status": status,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._cases[case_id] = updated
        await save_preparation_case(updated)
        return updated

    async def answer_clarification(
        self, case_id: str, clarification_id: str, user_id: str, answer: str
    ) -> PreparationCase:
        case = await self.get(case_id, user_id)
        if not case:
            raise KeyError(case_id)
        updated_rows = []
        found = False
        for item in case.clarifications:
            if item.clarification_id == clarification_id:
                if item.status == "answered":
                    if item.answer == answer:
                        return case
                    raise ValueError("Clarification already answered; update materials for a different answer")
                updated_rows.append(item.model_copy(update={"status": "answered", "answer": answer}))
                found = True
            else:
                updated_rows.append(item)
        if not found:
            raise KeyError(clarification_id)
        all_answered = all(item.status == "answered" for item in updated_rows)
        updated = case.model_copy(
            update={
                "clarifications": updated_rows,
                "project_notes": (case.project_notes + "\n" + answer).strip(),
                "status": "ready" if all_answered else "waiting_input",
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._cases[case_id] = updated
        await save_preparation_case(updated)
        return updated

    async def create_repractice(
        self, case_id: str, user_id: str, request: RetrainingRequest,
        *, source_question_ids: list[str], source_revision: int,
    ) -> PreparationCase:
        source = await self.get(case_id, user_id)
        if not source:
            raise KeyError(case_id)
        if source.material_revision != source_revision:
            raise ValueError("Source materials changed since interview; repractice requires the original revision")
        selected = {canonical_skill_id(item) for item in request.selected_skills}
        new_case_id = uuid5(NAMESPACE_URL, f"mockview:repractice:{user_id}:{request.source_session_id}:{source_revision}:{','.join(sorted(selected))}").hex
        existing = await self.get(new_case_id, user_id)
        if existing:
            return existing
        targets = []
        for target in source.targets:
            if canonical_skill_id(target.scope) not in selected and canonical_skill_id(target.skill_id) not in selected:
                continue
            targets.append(
                target.model_copy(
                    update={
                        "target_id": uuid5(
                            NAMESPACE_URL,
                            f"mockview:{new_case_id}:repractice:{target.target_id}",
                        ).hex,
                        "case_id": new_case_id,
                        "status": "pending",
                        "verification_goal": (
                            target.verification_goal
                            + "；使用与上次不同的题面或场景验证迁移能力"
                        ),
                    }
                )
            )
        if not targets:
            raise ValueError("Selected weak skills have no matching preparation targets")
        case = PreparationCase(
            case_id=new_case_id,
            user_id=user_id,
            parent_case_id=source.case_id,
            material_revision=source.material_revision,
            jd_text=source.jd_text,
            resume_text=source.resume_text,
            resume_parse_warnings=source.resume_parse_warnings,
            project_notes=source.project_notes,
            preferences={
                **{key: value for key, value in source.preferences.items() if key != "avoid_question_ids"},
                **{key: value for key, value in request.preferences.items() if key != "avoid_question_ids"},
                "repractice": True,
                "selected_skills": request.selected_skills,
                "source_session_id": request.source_session_id,
                "source_material_revision": source_revision,
                "source_question_ids": source_question_ids,
            },
            status="ready",
            targets=targets,
            evidence_chunks={},
            evidence_assessments={},
            skill_matrix=source.skill_matrix,
            resume_profile=source.resume_profile,
            analysis_strategy_version=source.analysis_strategy_version,
        )
        self._cases[case.case_id] = case
        await save_preparation_case(case)
        return case


_service: PreparationService | None = None


def get_preparation_service() -> PreparationService:
    global _service
    if _service is None:
        _service = PreparationService()
    return _service


def reset_preparation_service() -> None:
    global _service
    _service = None
