"""Long-term memory service for interview personalization."""

from __future__ import annotations

import hashlib
import logging
from uuid import uuid4

from app.memory.models import (
    MasteryLevel,
    MemoryContext,
    MemoryItem,
    MemoryType,
    SkillMemory,
    utc_now,
)
from app.memory.store import MemoryStore, get_memory_store
from app.memory.vector_store import MemoryVectorStore, get_memory_vector_store
from app.models.interview import AnswerAssessment, FinalQuestionAssessment
from app.models.question import QuestionItem
from app.models.report import InterviewReport, PracticeReport, ProfessionalReport
from app.models.resume import ResumeProfile

logger = logging.getLogger(__name__)


def _stable_id(*parts: object) -> str:
    raw = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"mem_{digest}"


def _dedupe_append(existing: list[str], new_values: list[str], limit: int = 12) -> list[str]:
    result = list(existing)
    seen = {v.strip().lower() for v in result}
    for value in new_values:
        clean = str(value).strip()
        if clean and clean.lower() not in seen:
            result.append(clean)
            seen.add(clean.lower())
    return result[-limit:]


def _mastery(score: float) -> str:
    if score >= 8:
        return MasteryLevel.STRONG.value
    if score >= 6:
        return MasteryLevel.NORMAL.value
    return MasteryLevel.WEAK.value


def _priority(avg_score: float, recent_score: float, attempts: int) -> float:
    weakness = max(0.0, 10.0 - (recent_score * 0.65 + avg_score * 0.35)) / 10.0
    evidence_boost = min(attempts, 5) * 0.04
    return max(0.0, min(1.0, weakness + evidence_boost))


def _is_valid(item: MemoryItem) -> bool:
    return item.structured.get("validity", "valid") == "valid"


class MemoryService:
    """Coordinates memory extraction, consolidation, and recall."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        vector_store: MemoryVectorStore | None = None,
    ) -> None:
        self.store = store or get_memory_store()
        self.vector_store = vector_store

    def build_context(self, user_id: str, tags: list[str] | None = None) -> MemoryContext:
        """Recall compact memory context for question planning."""
        profile_items = self.store.list_memory_items(
            user_id,
            memory_types=[MemoryType.PROFILE.value, MemoryType.PREFERENCE.value],
            limit=4,
        )
        resume_items = self.store.list_memory_items(
            user_id,
            memory_types=[MemoryType.RESUME_PROJECT.value],
            limit=5,
        )
        recent_reflections = self.store.list_memory_items(
            user_id,
            memory_types=[MemoryType.SESSION_REFLECTION.value],
            limit=3,
        )
        weak_skills = [
            memory
            for memory in self.store.list_skill_memories(user_id, limit=6)
            if (
                memory.mastery_level == MasteryLevel.WEAK.value
                or memory.next_practice_priority >= 0.45
            )
        ]
        relevant_episodes = []
        if tags:
            relevant_episodes = self.store.list_memory_items(
                user_id,
                memory_types=[MemoryType.INTERVIEW_EPISODE.value],
                tags=tags,
                limit=5,
            )
        return MemoryContext(
            user_id=user_id,
            profile_items=[item for item in profile_items if _is_valid(item)],
            resume_items=[item for item in resume_items if _is_valid(item)],
            recent_reflections=[item for item in recent_reflections if _is_valid(item)],
            weak_skills=weak_skills,
            relevant_episodes=[item for item in relevant_episodes if _is_valid(item)],
        )

    async def abuild_context(
        self,
        user_id: str,
        semantic_query: str = "",
        tags: list[str] | None = None,
    ) -> MemoryContext:
        """Recall structured memory plus semantic memories relevant to the query."""
        context = await self.abuild_structured_context(user_id=user_id, tags=tags)
        if not semantic_query.strip():
            return context

        try:
            vector_store = self.vector_store or get_memory_vector_store()
            results = await vector_store.semantic_search(
                user_id=user_id,
                query=semantic_query,
                memory_types=[
                    MemoryType.RESUME_PROJECT.value,
                    MemoryType.INTERVIEW_EPISODE.value,
                    MemoryType.SESSION_REFLECTION.value,
                    MemoryType.PROFILE.value,
                    MemoryType.STRATEGY.value,
                    MemoryType.PREFERENCE.value,
                ],
                limit=5,
            )
        except Exception:
            logger.exception("Semantic memory recall failed for user=%s", user_id)
            return context

        seen_ids = {
            item.id
            for bucket in [
                context.profile_items,
                context.resume_items,
                context.recent_reflections,
                context.relevant_episodes,
            ]
            for item in bucket
        }
        semantic_memories: list[MemoryItem] = []
        for result in results:
            item = await self._aget_memory_item(result["id"])
            if not item or item.id in seen_ids or not _is_valid(item):
                continue
            semantic_memories.append(
                item.model_copy(
                    update={
                        "structured": {
                            **item.structured,
                            "_semantic_score": round(float(result.get("score", 0.0)), 4),
                        }
                    }
                )
            )
            seen_ids.add(item.id)

        context.semantic_memories = semantic_memories
        return context

    async def abuild_structured_context(
        self,
        user_id: str,
        tags: list[str] | None = None,
    ) -> MemoryContext:
        if not hasattr(self.store, "alist_memory_items"):
            return self.build_context(user_id=user_id, tags=tags)

        profile_items = await self.store.alist_memory_items(
            user_id,
            memory_types=[MemoryType.PROFILE.value, MemoryType.PREFERENCE.value],
            limit=4,
        )
        resume_items = await self.store.alist_memory_items(
            user_id,
            memory_types=[MemoryType.RESUME_PROJECT.value],
            limit=5,
        )
        recent_reflections = await self.store.alist_memory_items(
            user_id,
            memory_types=[MemoryType.SESSION_REFLECTION.value],
            limit=3,
        )
        weak_skills = [
            memory
            for memory in await self.store.alist_skill_memories(user_id, limit=6)
            if (
                memory.mastery_level == MasteryLevel.WEAK.value
                or memory.next_practice_priority >= 0.45
            )
        ]
        relevant_episodes = []
        if tags:
            relevant_episodes = await self.store.alist_memory_items(
                user_id,
                memory_types=[MemoryType.INTERVIEW_EPISODE.value],
                tags=tags,
                limit=5,
            )
        return MemoryContext(
            user_id=user_id,
            profile_items=[item for item in profile_items if _is_valid(item)],
            resume_items=[item for item in resume_items if _is_valid(item)],
            recent_reflections=[item for item in recent_reflections if _is_valid(item)],
            weak_skills=weak_skills,
            relevant_episodes=[item for item in relevant_episodes if _is_valid(item)],
        )

    def save_final_observation(
        self,
        user_id: str,
        session_id: str,
        final: FinalQuestionAssessment,
        question: QuestionItem | None,
        transcript: list[dict] | None = None,
    ) -> MemoryItem:
        skills = [final.primary_skill]
        item = MemoryItem(
            id=final.observation_id,
            user_id=user_id,
            memory_type=MemoryType.INTERVIEW_EPISODE,
            content=(
                f"Final observation for {final.primary_skill}: "
                f"score={final.score if final.score is not None else 'unscored'}/10; "
                f"covered={'; '.join(final.covered_points[:4]) or 'none'}; "
                f"missed={'; '.join(final.missed_points[:4]) or 'none'}."
            ),
            structured={
                "observation_id": final.observation_id,
                "session_id": session_id,
                "question_id": final.question_id,
                "display_id": final.display_id,
                "question": question.content if question else "",
                "score": final.score,
                "scorable": final.scorable,
                "covered_points": final.covered_points,
                "missed_points": final.missed_points,
                "answer_evidence": final.answer_evidence,
                "hint_dependency": final.hint_dependency,
                "skill_tags": skills,
                "primary_skill": final.primary_skill,
                "validity": "valid",
                "projection_status": "pending",
                "transcript": transcript or [],
            },
            tags=["episode", "final_observation", *skills],
            source="final_question_assessment",
            source_id=session_id,
            importance=0.8 if final.score is not None and final.score < 7 else 0.6,
            confidence=0.9 if final.scorable else 0.5,
        )
        saved = self.store.upsert_memory_item(item)
        self.recompute_skill_memory(user_id, final.primary_skill)
        return saved

    async def asave_final_observation(
        self,
        user_id: str,
        session_id: str,
        final: FinalQuestionAssessment,
        question: QuestionItem | None,
        transcript: list[dict] | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            id=final.observation_id,
            user_id=user_id,
            memory_type=MemoryType.INTERVIEW_EPISODE,
            content=(
                f"Final observation for {final.primary_skill}: "
                f"score={final.score if final.score is not None else 'unscored'}/10; "
                f"covered={'; '.join(final.covered_points[:4]) or 'none'}; "
                f"missed={'; '.join(final.missed_points[:4]) or 'none'}."
            ),
            structured={
                "observation_id": final.observation_id,
                "session_id": session_id,
                "question_id": final.question_id,
                "display_id": final.display_id,
                "question": question.content if question else "",
                "score": final.score,
                "scorable": final.scorable,
                "covered_points": final.covered_points,
                "missed_points": final.missed_points,
                "answer_evidence": final.answer_evidence,
                "hint_dependency": final.hint_dependency,
                "skill_tags": [final.primary_skill],
                "primary_skill": final.primary_skill,
                "validity": "valid",
                "projection_status": "pending",
                "transcript": transcript or [],
            },
            tags=["episode", "final_observation", final.primary_skill],
            source="final_question_assessment",
            source_id=session_id,
            importance=0.8 if final.score is not None and final.score < 7 else 0.6,
            confidence=0.9 if final.scorable else 0.5,
        )
        if hasattr(self.store, "aupsert_memory_item"):
            saved = await self.store.aupsert_memory_item(item)
        else:
            saved = self.store.upsert_memory_item(item)
        await self.arecompute_skill_memory(user_id, final.primary_skill)
        return saved

    def recompute_skill_memory(self, user_id: str, skill_name: str) -> SkillMemory:
        observations = self.store.list_memory_items(
            user_id, memory_types=[MemoryType.INTERVIEW_EPISODE.value], limit=1000
        )
        valid = [
            item for item in observations
            if _is_valid(item)
            and item.structured.get("primary_skill") == skill_name
            and item.structured.get("scorable", True)
            and item.structured.get("score") is not None
        ]
        scores = [float(item.structured["score"]) for item in valid]
        recent = scores[0] if scores else 0.0
        average = sum(scores) / len(scores) if scores else 0.0
        covered = [point for item in valid for point in item.structured.get("covered_points", [])]
        missed = [point for item in valid for point in item.structured.get("missed_points", [])]
        existing = self.store.get_skill_memory(user_id, skill_name)
        memory = SkillMemory(
            id=(existing.id if existing else _stable_id(user_id, "skill", skill_name)),
            user_id=user_id,
            skill_name=skill_name,
            category=(existing.category if existing else "unknown"),
            attempts=len(valid),
            avg_score=round(average, 2),
            recent_score=recent,
            mastery_level=_mastery(recent) if scores else MasteryLevel.UNKNOWN.value,
            strengths=_dedupe_append([], covered),
            weak_points=_dedupe_append([], missed),
            evidence_memory_ids=[item.id for item in valid],
            next_practice_priority=round(_priority(average, recent, len(valid)), 3) if scores else 0.5,
            created_at=(existing.created_at if existing else utc_now()),
            updated_at=utc_now(),
        )
        return self.store.upsert_skill_memory(memory)

    async def arecompute_skill_memory(self, user_id: str, skill_name: str) -> SkillMemory:
        if not hasattr(self.store, "alist_memory_items"):
            return self.recompute_skill_memory(user_id, skill_name)
        observations = await self.store.alist_memory_items(
            user_id, memory_types=[MemoryType.INTERVIEW_EPISODE.value], limit=1000
        )
        valid = [
            item for item in observations
            if _is_valid(item)
            and item.structured.get("primary_skill") == skill_name
            and item.structured.get("scorable", True)
            and item.structured.get("score") is not None
        ]
        scores = [float(item.structured["score"]) for item in valid]
        recent = scores[0] if scores else 0.0
        average = sum(scores) / len(scores) if scores else 0.0
        existing = await self.store.aget_skill_memory(user_id, skill_name)
        memory = SkillMemory(
            id=(existing.id if existing else _stable_id(user_id, "skill", skill_name)),
            user_id=user_id,
            skill_name=skill_name,
            category=(existing.category if existing else "unknown"),
            attempts=len(valid),
            avg_score=round(average, 2),
            recent_score=recent,
            mastery_level=_mastery(recent) if scores else MasteryLevel.UNKNOWN.value,
            strengths=_dedupe_append([], [point for item in valid for point in item.structured.get("covered_points", [])]),
            weak_points=_dedupe_append([], [point for item in valid for point in item.structured.get("missed_points", [])]),
            evidence_memory_ids=[item.id for item in valid],
            next_practice_priority=round(_priority(average, recent, len(valid)), 3) if scores else 0.5,
            created_at=(existing.created_at if existing else utc_now()),
            updated_at=utc_now(),
        )
        return await self.store.aupsert_skill_memory(memory)

    async def dispute_observation(self, user_id: str, observation_id: str) -> MemoryItem:
        item = await self._aget_memory_item(observation_id)
        if not item or item.user_id != user_id:
            raise KeyError(observation_id)
        updated = item.model_copy(
            update={
                "structured": {**item.structured, "validity": "disputed"},
                "updated_at": utc_now(),
            }
        )
        if hasattr(self.store, "aupsert_memory_item"):
            updated = await self.store.aupsert_memory_item(updated)
        else:
            updated = self.store.upsert_memory_item(updated)
        skill = str(updated.structured.get("primary_skill") or "")
        if skill:
            await self.arecompute_skill_memory(user_id, skill)
        return updated

    async def correct_fact(
        self,
        user_id: str,
        content: str,
        replaces_memory_id: str | None = None,
    ) -> MemoryItem:
        replacement_id = f"fact:{user_id}:{uuid4().hex}"
        replaced_skill = ""
        if replaces_memory_id:
            prior = await self._aget_memory_item(replaces_memory_id)
            if not prior or prior.user_id != user_id:
                raise KeyError(replaces_memory_id)
            replaced = prior.model_copy(
                update={
                    "structured": {
                        **prior.structured,
                        "validity": "replaced",
                        "replaced_by": replacement_id,
                    },
                    "updated_at": utc_now(),
                }
            )
            if hasattr(self.store, "aupsert_memory_item"):
                await self.store.aupsert_memory_item(replaced)
            else:
                self.store.upsert_memory_item(replaced)
            replaced_skill = str(prior.structured.get("primary_skill") or "")
        item = MemoryItem(
            id=replacement_id,
            user_id=user_id,
            memory_type=MemoryType.PROFILE,
            content=content,
            structured={
                "validity": "valid",
                "projection_status": "pending",
                "replaces_memory_id": replaces_memory_id,
                "user_confirmed": True,
            },
            tags=["profile", "user_correction"],
            source="user_correction",
            source_id=replaces_memory_id,
            confidence=1.0,
            importance=0.9,
        )
        if hasattr(self.store, "aupsert_memory_item"):
            saved = await self.store.aupsert_memory_item(item)
        else:
            saved = self.store.upsert_memory_item(item)
        if replaced_skill:
            await self.arecompute_skill_memory(user_id, replaced_skill)
        await self.aindex_memory_item(saved)
        return saved

    async def _aget_memory_item(self, memory_id: str) -> MemoryItem | None:
        if hasattr(self.store, "aget_memory_item"):
            return await self.store.aget_memory_item(memory_id)
        return self.store.get_memory_item(memory_id)

    def format_context(self, context: MemoryContext) -> str:
        if context.is_empty():
            return "No long-term memory is available for this user yet."

        lines = ["## Long-term Memory Context"]
        if context.profile_items:
            lines.append("\nUser Profile / Preferences:")
            for item in context.profile_items[:4]:
                lines.append(f"- {item.content}")

        if context.resume_items:
            lines.append("\nResume / Project Memory:")
            for item in context.resume_items[:5]:
                lines.append(f"- {item.content}")

        if context.weak_skills:
            lines.append("\nWeak Skills To Prioritize:")
            for skill in context.weak_skills[:6]:
                weak = "; ".join(skill.weak_points[:3]) or "No concrete missed point recorded."
                lines.append(
                    f"- {skill.skill_name}: recent={skill.recent_score:.1f}, "
                    f"avg={skill.avg_score:.1f}, priority={skill.next_practice_priority:.2f}; "
                    f"missed={weak}"
                )

        if context.recent_reflections:
            lines.append("\nRecent Session Reflections:")
            for item in context.recent_reflections[:3]:
                lines.append(f"- {item.content}")

        if context.relevant_episodes:
            lines.append("\nRelevant Past Interview Episodes:")
            for item in context.relevant_episodes[:3]:
                lines.append(f"- {item.content}")

        if context.semantic_memories:
            lines.append("\nSemantically Relevant Memories:")
            for item in context.semantic_memories[:5]:
                score = item.structured.get("_semantic_score")
                score_text = f" score={score:.2f};" if isinstance(score, float) else ""
                lines.append(f"- [{item.memory_type}]{score_text} {item.content}")

        lines.append(
            "\nPlanning guidance: use these memories to personalize coverage, "
            "especially by probing weak skills and resume project risks. Do not reveal "
            "the memory text verbatim to the candidate."
        )
        return "\n".join(lines)

    def save_resume_profile(
        self,
        user_id: str,
        session_id: str,
        profile: ResumeProfile | None,
    ) -> list[MemoryItem]:
        saved = [
            self.store.upsert_memory_item(item)
            for item in self._build_resume_profile_items(user_id, session_id, profile)
        ]
        logger.info("Saved %d resume/profile memories for user=%s", len(saved), user_id)
        return saved

    async def asave_resume_profile(
        self,
        user_id: str,
        session_id: str,
        profile: ResumeProfile | None,
    ) -> list[MemoryItem]:
        if not hasattr(self.store, "aupsert_memory_item"):
            return self.save_resume_profile(user_id, session_id, profile)

        async_saved: list[MemoryItem] = []
        for item in self._build_resume_profile_items(user_id, session_id, profile):
            async_saved.append(await self.store.aupsert_memory_item(item))
        logger.info("Saved %d resume/profile memories for user=%s", len(async_saved), user_id)
        return async_saved

    def _build_resume_profile_items(
        self,
        user_id: str,
        session_id: str,
        profile: ResumeProfile | None,
    ) -> list[MemoryItem]:
        if not profile:
            return []

        items: list[MemoryItem] = []
        if profile.skills or profile.summary:
            content = (
                f"Candidate profile summary: {profile.summary or 'No summary'}; "
                f"skills: {', '.join(profile.skills[:12]) or 'not specified'}."
            )
            items.append(
                MemoryItem(
                    id=_stable_id(user_id, "profile", session_id),
                    user_id=user_id,
                    memory_type=MemoryType.PROFILE,
                    content=content,
                    structured=profile.model_dump(mode="json"),
                    tags=["profile", "resume", *profile.skills[:12]],
                    source="resume_profile",
                    source_id=session_id,
                    importance=0.7,
                    confidence=0.8,
                )
            )

        for project in profile.projects:
            tech = ", ".join(project.tech_stack)
            dives = "; ".join(project.potential_deep_dive_points[:4])
            content = (
                f"Resume project [{project.name}] uses {tech or 'unspecified tech'}; "
                f"description: {project.description}; deep-dive points: {dives or 'not extracted'}."
            )
            items.append(
                MemoryItem(
                    id=_stable_id(
                        user_id,
                        "resume_project",
                        project.name,
                        project.description[:80],
                    ),
                    user_id=user_id,
                    memory_type=MemoryType.RESUME_PROJECT,
                    content=content,
                    structured=project.model_dump(mode="json"),
                    tags=["resume", "project", project.name, *project.tech_stack],
                    source="resume_profile",
                    source_id=session_id,
                    importance=0.85,
                    confidence=0.8,
                )
            )
        return items

    async def aindex_memory_item(self, item: MemoryItem) -> bool:
        """Index one MemoryItem for semantic recall. Failures should not block interviews."""
        try:
            vector_store = self.vector_store or get_memory_vector_store()
            await vector_store.upsert_item(item)
            if item.structured.get("projection_status") == "pending":
                completed = item.model_copy(
                    update={
                        "structured": {
                            **item.structured,
                            "projection_status": "completed",
                        },
                        "updated_at": utc_now(),
                    }
                )
                if hasattr(self.store, "aupsert_memory_item"):
                    await self.store.aupsert_memory_item(completed)
                else:
                    self.store.upsert_memory_item(completed)
            return True
        except Exception:
            logger.exception("Failed to index memory item: id=%s", item.id)
            return False

    async def aindex_memory_items(self, items: list[MemoryItem]) -> None:
        for item in items:
            await self.aindex_memory_item(item)

    def save_assessment_episode(
        self,
        user_id: str,
        session_id: str,
        assessment: AnswerAssessment,
        question: QuestionItem | None,
        answer: str,
        mode: str,
    ) -> MemoryItem:
        item = self._build_assessment_episode_item(
            user_id=user_id,
            session_id=session_id,
            assessment=assessment,
            question=question,
            answer=answer,
            mode=mode,
        )
        item = self.store.upsert_memory_item(item)
        self.update_skill_memories(user_id, assessment, question, item.id)
        return item

    async def asave_assessment_episode(
        self,
        user_id: str,
        session_id: str,
        assessment: AnswerAssessment,
        question: QuestionItem | None,
        answer: str,
        mode: str,
    ) -> MemoryItem:
        if not hasattr(self.store, "aupsert_memory_item"):
            return self.save_assessment_episode(user_id, session_id, assessment, question, answer, mode)

        item = self._build_assessment_episode_item(
            user_id=user_id,
            session_id=session_id,
            assessment=assessment,
            question=question,
            answer=answer,
            mode=mode,
        )
        item = await self.store.aupsert_memory_item(item)
        await self.aupdate_skill_memories(user_id, assessment, question, item.id)
        return item

    def _build_assessment_episode_item(
        self,
        user_id: str,
        session_id: str,
        assessment: AnswerAssessment,
        question: QuestionItem | None,
        answer: str,
        mode: str,
    ) -> MemoryItem:
        skills = question.skill_tags if question else []
        question_text = question.content if question else f"Question #{assessment.question_id}"
        missed = "; ".join(assessment.missed_points[:4]) or "No major missed points."
        covered = "; ".join(assessment.covered_points[:4]) or "No clear covered points."
        content = (
            f"Interview episode for Q{assessment.question_id} "
            f"({', '.join(skills) or 'unknown skill'}): "
            f"score={assessment.score}/10; covered={covered}; missed={missed}."
        )
        item = MemoryItem(
            id=_stable_id(
                user_id,
                session_id,
                assessment.question_id,
                len(answer),
                utc_now().isoformat(),
            ),
            user_id=user_id,
            memory_type=MemoryType.INTERVIEW_EPISODE,
            content=content,
            structured={
                "session_id": session_id,
                "mode": mode,
                "question_id": assessment.question_id,
                "question": question_text,
                "answer": answer,
                "score": assessment.score,
                "covered_points": assessment.covered_points,
                "missed_points": assessment.missed_points,
                "should_follow_up": assessment.should_follow_up,
                "skill_tags": skills,
            },
            tags=["episode", mode, *skills],
            source="answer_assessment",
            source_id=session_id,
            importance=0.75 if assessment.score < 7 else 0.55,
            confidence=0.85,
        )
        return item

    def update_skill_memories(
        self,
        user_id: str,
        assessment: AnswerAssessment,
        question: QuestionItem | None,
        evidence_memory_id: str,
    ) -> list[SkillMemory]:
        if not question:
            return []

        updated: list[SkillMemory] = []
        for skill_name in question.skill_tags:
            existing = self.store.get_skill_memory(user_id, skill_name)
            if existing is None:
                existing = SkillMemory(
                    id=_stable_id(user_id, "skill", skill_name),
                    user_id=user_id,
                    skill_name=skill_name,
                    category="unknown",
                    created_at=utc_now(),
                )

            attempts = existing.attempts + 1
            avg_score = ((existing.avg_score * existing.attempts) + assessment.score) / attempts
            recent_score = float(assessment.score)
            memory = existing.model_copy(
                update={
                    "attempts": attempts,
                    "avg_score": round(avg_score, 2),
                    "recent_score": recent_score,
                    "mastery_level": _mastery(recent_score),
                    "strengths": _dedupe_append(existing.strengths, assessment.covered_points),
                    "weak_points": _dedupe_append(existing.weak_points, assessment.missed_points),
                    "evidence_memory_ids": _dedupe_append(
                        existing.evidence_memory_ids,
                        [evidence_memory_id],
                        limit=20,
                    ),
                    "next_practice_priority": round(
                        _priority(avg_score, recent_score, attempts),
                        3,
                    ),
                    "updated_at": utc_now(),
                }
            )
            updated.append(self.store.upsert_skill_memory(memory))
        return updated

    async def aupdate_skill_memories(
        self,
        user_id: str,
        assessment: AnswerAssessment,
        question: QuestionItem | None,
        evidence_memory_id: str,
    ) -> list[SkillMemory]:
        if not hasattr(self.store, "aget_skill_memory") or not hasattr(
            self.store, "aupsert_skill_memory"
        ):
            return self.update_skill_memories(user_id, assessment, question, evidence_memory_id)
        if not question:
            return []

        updated: list[SkillMemory] = []
        for skill_name in question.skill_tags:
            existing = await self.store.aget_skill_memory(user_id, skill_name)
            if existing is None:
                existing = SkillMemory(
                    id=_stable_id(user_id, "skill", skill_name),
                    user_id=user_id,
                    skill_name=skill_name,
                    category="unknown",
                    created_at=utc_now(),
                )

            attempts = existing.attempts + 1
            avg_score = ((existing.avg_score * existing.attempts) + assessment.score) / attempts
            recent_score = float(assessment.score)
            memory = existing.model_copy(
                update={
                    "attempts": attempts,
                    "avg_score": round(avg_score, 2),
                    "recent_score": recent_score,
                    "mastery_level": _mastery(recent_score),
                    "strengths": _dedupe_append(existing.strengths, assessment.covered_points),
                    "weak_points": _dedupe_append(existing.weak_points, assessment.missed_points),
                    "evidence_memory_ids": _dedupe_append(
                        existing.evidence_memory_ids,
                        [evidence_memory_id],
                        limit=20,
                    ),
                    "next_practice_priority": round(
                        _priority(avg_score, recent_score, attempts),
                        3,
                    ),
                    "updated_at": utc_now(),
                }
            )
            updated.append(await self.store.aupsert_skill_memory(memory))
        return updated

    def save_session_reflection(
        self,
        user_id: str,
        session_id: str,
        mode: str,
        report: PracticeReport | InterviewReport | ProfessionalReport | None,
        assessments: list[AnswerAssessment],
    ) -> MemoryItem | None:
        item = self._build_session_reflection_item(user_id, session_id, mode, report, assessments)
        return self.store.upsert_memory_item(item) if item else None

    async def asave_session_reflection(
        self,
        user_id: str,
        session_id: str,
        mode: str,
        report: PracticeReport | InterviewReport | ProfessionalReport | None,
        assessments: list[AnswerAssessment],
    ) -> MemoryItem | None:
        if not hasattr(self.store, "aupsert_memory_item"):
            return self.save_session_reflection(user_id, session_id, mode, report, assessments)

        item = self._build_session_reflection_item(user_id, session_id, mode, report, assessments)
        return await self.store.aupsert_memory_item(item) if item else None

    def _build_session_reflection_item(
        self,
        user_id: str,
        session_id: str,
        mode: str,
        report: PracticeReport | InterviewReport | ProfessionalReport | None,
        assessments: list[AnswerAssessment],
    ) -> MemoryItem | None:
        if not report and not assessments:
            return None

        overall = getattr(report, "overall_score", None)
        overall_text = getattr(report, "overall_assessment", "")
        skill_scores = getattr(report, "skill_scores", []) or []
        weak_skills = [s.skill_name for s in skill_scores if getattr(s, "score", 10) < 7]
        missed_points: list[str] = []
        for assessment in assessments:
            missed_points.extend(assessment.missed_points[:3])
        missed_points = _dedupe_append([], missed_points, limit=8)

        content = (
            f"Session reflection ({mode}): overall={overall if overall is not None else 'n/a'}/10. "
            f"{overall_text or 'No final report text.'} "
            f"Weak skills: {', '.join(weak_skills[:5]) or 'not identified'}. "
            f"Recurring missed points: {'; '.join(missed_points[:5]) or 'not identified'}."
        )
        item = MemoryItem(
            id=_stable_id(user_id, "reflection", session_id),
            user_id=user_id,
            memory_type=MemoryType.SESSION_REFLECTION,
            content=content,
            structured={
                "session_id": session_id,
                "mode": mode,
                "overall_score": overall,
                "weak_skills": weak_skills,
                "missed_points": missed_points,
            },
            tags=["reflection", mode, *weak_skills[:8]],
            source="final_report",
            source_id=session_id,
            importance=0.8,
            confidence=0.8,
        )
        return item


_memory_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
