"""Evaluator Agents — produce structured reports for both professional and practice modes.

Also includes the round1_summary node for intermediate feedback in dual-round mode.
"""

from __future__ import annotations

from datetime import datetime, timezone

import logging

from app.agents.state import InterviewState
from app.llm.client import get_llm_client
from app.llm.prompts import (
    EVALUATOR_SYSTEM,
    PRACTICE_EVALUATOR_SYSTEM,
    PROFESSIONAL_EVALUATOR_SYSTEM,
    ROUND1_SUMMARY_SYSTEM,
)
from app.models.interview import ChatMessage
from app.models.report import (
    InterviewReport,
    PracticeReport,
    ProfessionalReport,
    Round1Summary,
    RoundScore,
    SkillScore,
    MissedKnowledge,
)
from app.models.interview import FinalQuestionAssessment
from app.domain.skill_identity import canonical_skill_id
from app.services.scoring import aggregate_professional_rounds, aggregate_scores, grade_for_score

logger = logging.getLogger(__name__)


def _final_items(raw: dict | None) -> list[FinalQuestionAssessment]:
    return [
        item if isinstance(item, FinalQuestionAssessment) else FinalQuestionAssessment.model_validate(item)
        for item in (raw or {}).values()
    ]


def _coverage(summary) -> dict:
    return {
        "covered_skills": summary.covered_skills,
        "uncovered_skills": summary.uncovered_skills,
        "scored_questions": summary.scored_questions,
        "unscored_questions": summary.unscored_questions,
    }


def _deterministic_skill_scores(summary, items: list[FinalQuestionAssessment]) -> list[SkillScore]:
    rows = []
    for skill, score in summary.skill_scores.items():
        evidence = []
        for item in items:
            if canonical_skill_id(item.primary_skill) == canonical_skill_id(skill):
                evidence.extend(item.answer_evidence[:2])
        rows.append(
            SkillScore(
                skill_name=skill,
                score=score,
                evidence="；".join(evidence[:3]) or "来自最终单题评价。",
            )
        )
    return rows


def _deterministic_round_scores(report: ProfessionalReport, r1_summary, r2_summary) -> list[RoundScore]:
    """Discard model-authored numbers and keep only rounds with valid final scores."""
    existing = list(report.round_scores)
    rows: list[RoundScore] = []
    for index, (name, summary) in enumerate(
        (("一面（技术深度）", r1_summary), ("二面（技术广度）", r2_summary))
    ):
        if summary.overall_score is None or summary.grade is None:
            continue
        narrative = existing[index].summary if index < len(existing) else "基于本轮有效单题评价汇总。"
        rows.append(
            RoundScore(
                round_name=name,
                score=summary.overall_score,
                grade=summary.grade,
                summary=narrative,
            )
        )
    return rows


# ── Shared formatting helpers ────────────────────────────────────────


def _format_transcript(state: InterviewState) -> str:
    lines = []
    for msg in state.get("conversation_history", []):
        if isinstance(msg, ChatMessage):
            role, content = msg.role, msg.content
        elif isinstance(msg, dict):
            role, content = msg.get("role", "?"), msg.get("content", "")
        else:
            continue
        if role == "interviewer":
            lines.append(f"[Interviewer]: {content}")
        elif role == "candidate":
            lines.append(f"[Candidate]: {content}")
    return "\n\n".join(lines)


def _format_assessments(state: InterviewState) -> str:
    lines = []
    for a in _final_items(state.get("final_question_assessments")):
        lines.append(
            f"question_id={a.question_id}, Q{a.display_id}: final_score={a.score if a.scorable else None}, "
            f"covered={a.covered_points}, missed={a.missed_points}"
        )
    summary = aggregate_scores(_final_items(state.get("final_question_assessments")), state.get("skill_matrix"))
    return ("以下最终单题评价与汇总是唯一权威数值；null 表示未评分。文字叙述只解释证据与建议，"
            "不要复述或自行推算分数、等级、题目数或覆盖率。不要从过程追问评分再次计算。\n"
            f"Authoritative summary: {summary}\n" + "\n".join(lines))


def _deterministic_missed_knowledge(state, items, generated):
    questions = {}
    for key in ("round1_question_plan", "round2_question_plan", "question_plan"):
        for raw in state.get(key, []) or []:
            q = raw if isinstance(raw, dict) else raw.model_dump()
            if q.get("question_id"):
                questions[q["question_id"]] = q["content"]
    rows = []
    for item in items:
        scored = item.scorable and item.score is not None
        if scored and item.score >= 8 and not item.missed_points:
            continue
        content = questions.get(item.question_id, f"问题 {item.display_id}")
        matches = [row for row in generated if row.question_id == item.question_id or
                   (not row.question_id and row.question == content and
                    list(questions.values()).count(content) == 1)]
        narrative = matches[0].reference_answer if len(matches) == 1 else "暂无可可靠关联的参考答案，请结合本题反馈复习。"
        rows.append(MissedKnowledge(question=content, question_id=item.question_id,
                    score=item.score if scored else None, scorable=scored,
                    missed_points=item.missed_points if scored else [],
                    reference_answer=narrative if scored else "本题缺少可靠评分依据，未计入技能分和综合分。"))
    return rows


def _format_skill_matrix(state: InterviewState) -> str:
    sm = state.get("skill_matrix")
    if not sm:
        return "Skill matrix not available."
    lines = [f"Position: {sm.position_title} ({sm.experience_level})"]
    for s in sm.skills:
        lines.append(f"  - {s.name}: weight={s.weight:.1f}, required={s.is_required}")
    return "\n".join(lines)


def _format_question_plan(state: InterviewState) -> str:
    """Include question content + reference points so the evaluator can write answers."""
    plan = state.get("question_plan", [])
    if not plan:
        return "No question plan available."
    lines = []
    for q in plan:
        lines.append(
            f"question_id={q.question_id}, Q{q.id} [{q.difficulty}]: {q.content}\n"
            f"  Reference points: {', '.join(q.reference_points)}"
        )
    return "\n".join(lines)


# ── Professional Mode Evaluator ──────────────────────────────────────


def _format_round1_data(state: InterviewState) -> str:
    """Format Round 1 assessments for the final evaluator."""
    finals = state.get("round1_final_question_assessments") or {}
    if not finals:
        return ""
    return "## Round 1 Final Assessments\n" + _format_assessments({
        "final_question_assessments": finals, "skill_matrix": state.get("skill_matrix")})


async def evaluate_interview(state: InterviewState) -> dict:
    """LangGraph node: produce a professional-mode evaluation report.

    In dual-round mode, uses ProfessionalReport for a comprehensive dual-round report.
    Falls back to InterviewReport for single-round or early-stop scenarios.
    """
    transcript = _format_transcript(state)
    assessments_text = _format_assessments(state)
    skill_context = _format_skill_matrix(state)
    round1_data = _format_round1_data(state)

    current_round = state.get("current_round", 1)
    is_dual_round = bool(round1_data) and current_round >= 2

    logger.info(
        "Evaluating interview (professional, round=%d, dual=%s): %d messages, %d assessments",
        current_round,
        is_dual_round,
        len(state.get("conversation_history", [])),
        len(state.get("assessments", [])),
    )

    user_content = f"## Skill Matrix\n{skill_context}\n\n"

    if is_dual_round:
        user_content += f"{round1_data}\n\n"
        user_content += "## All Final Assessments\n" + assessments_text + "\n\n"
        r1 = aggregate_scores(_final_items(state.get("round1_final_question_assessments")), state.get("skill_matrix"))
        r1_ids = set(state.get("round1_final_question_assessments") or {})
        r2 = aggregate_scores([item for item in _final_items(state.get("final_question_assessments"))
                               if item.question_id not in r1_ids], state.get("skill_matrix"))
        user_content += f"\nAuthoritative professional overall (55/45): {aggregate_professional_rounds(r1.overall_score, r2.overall_score)}\n"

        user_content += (
            f"## Full Interview Transcript (Both Rounds)\n{transcript}\n\n"
            f"请基于以上两轮面试信息生成完整的双轮面试评价报告。"
            f"报告中所有候选人可见文本必须使用简体中文。"
        )

        llm = get_llm_client()
        report = await llm.chat_structured(
            messages=[
                {"role": "system", "content": PROFESSIONAL_EVALUATOR_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_model=ProfessionalReport,
            temperature=0.3,
        )

        r1_items = _final_items(state.get("round1_final_question_assessments"))
        all_items = _final_items(state.get("final_question_assessments"))
        r1_ids = {item.question_id for item in r1_items}
        r2_items = [item for item in all_items if item.question_id not in r1_ids]
        r1_summary = aggregate_scores(r1_items, state.get("skill_matrix"))
        r2_summary = aggregate_scores(r2_items, state.get("skill_matrix"))
        combined = aggregate_scores(all_items, state.get("skill_matrix"))
        overall = aggregate_professional_rounds(r1_summary.overall_score, r2_summary.overall_score)
        report = report.model_copy(
            update={
                "overall_score": overall,
                "grade": grade_for_score(overall),
                "technical_depth_score": r1_summary.overall_score,
                "technical_breadth_score": r2_summary.overall_score,
                "round_scores": _deterministic_round_scores(report, r1_summary, r2_summary),
                "skill_scores": _deterministic_skill_scores(combined, all_items),
                "coverage": _coverage(combined),
                "total_questions": combined.scored_questions + combined.unscored_questions,
            }
        )

        logger.info(
            "Professional evaluation complete: overall=%s (%s), depth=%s, breadth=%s, rec=%s",
            report.overall_score, report.grade,
            report.technical_depth_score, report.technical_breadth_score,
            report.hiring_recommendation,
        )
        return {"professional_report": report, "interview_complete": True}

    user_content += f"## Per-Question Assessments\n{assessments_text}\n\n"
    user_content += (
        f"## Full Interview Transcript\n{transcript}\n\n"
        f"请基于以上内容生成完整面试评价报告。"
        f"报告中所有候选人可见文本必须使用简体中文。"
    )

    llm = get_llm_client()
    report = await llm.chat_structured(
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=InterviewReport,
        temperature=0.3,
    )

    final_items = _final_items(state.get("final_question_assessments"))
    summary = aggregate_scores(final_items, state.get("skill_matrix"))
    report = report.model_copy(
        update={
            "overall_score": summary.overall_score,
            "grade": summary.grade,
            "skill_scores": _deterministic_skill_scores(summary, final_items),
            "coverage": _coverage(summary),
            "total_questions": summary.scored_questions + summary.unscored_questions,
        }
    )

    logger.info(
        "Evaluation complete: overall=%s (%s), %d skill scores",
        report.overall_score, report.grade, len(report.skill_scores),
    )
    return {"final_report": report, "interview_complete": True}


# ── Round 1 Summary (intermediate node in dual-round professional mode) ──


async def summarize_round1(state: InterviewState) -> dict:
    """LangGraph node: produce an intermediate summary after Round 1.

    Saves Round 1 data (question plan + assessments) and generates a
    summary that guides Round 2 question planning.
    """
    transcript = _format_transcript(state)
    assessments_text = _format_assessments(state)
    skill_context = _format_skill_matrix(state)
    question_context = _format_question_plan(state)

    num_assessed = len(state.get("final_question_assessments", {}))
    logger.info("Summarizing Round 1: %d assessments", num_assessed)

    user_content = (
        f"## Skill Matrix\n{skill_context}\n\n"
        f"## Round 1 Question Plan\n{question_context}\n\n"
        f"## Per-Question Assessments\n{assessments_text}\n\n"
        f"## Full Round 1 Transcript\n{transcript}\n\n"
        f"Total questions asked in Round 1: {num_assessed}\n\n"
        f"请生成一面阶段总结。所有候选人可见文本必须使用简体中文。"
    )

    llm = get_llm_client()
    summary = await llm.chat_structured(
        messages=[
            {"role": "system", "content": ROUND1_SUMMARY_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=Round1Summary,
        temperature=0.3,
    )

    final_items = _final_items(state.get("final_question_assessments"))
    score_summary = aggregate_scores(final_items, state.get("skill_matrix"))
    summary = summary.model_copy(
        update={
            "round1_score": score_summary.overall_score,
            "round1_grade": score_summary.grade,
        }
    )

    logger.info(
        "Round 1 summary: score=%s (%s), proceed=%s",
        summary.round1_score, summary.round1_grade, summary.proceed_to_round2,
    )

    return {
        "round1_question_plan": list(state.get("question_plan", [])),
        "round1_assessments": list(state.get("assessments", [])),
        "round1_final_question_assessments": dict(
            state.get("final_question_assessments", {})
        ),
        "round1_summary_text": summary.model_dump_json(),
        "current_round": 2,
        "current_question_index": 0,
        "follow_up_count": 0,
        "conversation_history": [
            ChatMessage(
                role="interviewer",
                content=(
                    f"---\n\n**一面结束 | Round 1 Complete**\n\n"
                    f"{summary.round1_feedback}\n\n"
                    + (
                        f"得分: {summary.round1_score:.1f}/10 ({summary.round1_grade})\n\n"
                        if summary.round1_score is not None
                        else "得分: 暂无可汇总分数\n\n"
                    )
                    + f"---\n\n现在进入二面（技术广度），准备好了吗？"
                ),
                created_at=datetime.now(timezone.utc),
            )
        ],
    }


# ── Practice Mode Evaluator ──────────────────────────────────────────


async def evaluate_practice(state: InterviewState) -> dict:
    """LangGraph node: produce a practice-mode report with reference answers."""
    transcript = _format_transcript(state)
    assessments_text = _format_assessments(state)
    skill_context = _format_skill_matrix(state)
    question_context = _format_question_plan(state)

    num_answered = len(state.get("final_question_assessments", {}))
    logger.info(
        "Evaluating practice session: %d questions answered, %d assessments",
        num_answered, len(state.get("assessments", [])),
    )

    user_content = (
        f"## Skill Matrix\n{skill_context}\n\n"
        f"## Question Plan (with reference points)\n{question_context}\n\n"
        f"## Per-Question Assessments\n{assessments_text}\n\n"
        f"## Full Interview Transcript\n{transcript}\n\n"
        f"Total questions answered: {num_answered}\n\n"
        f"请基于以上内容生成练习模式评估报告。"
        f"所有候选人可见文本必须使用简体中文。"
        f"对于得分低于 8 分的问题，请给出覆盖遗漏点的中文参考答案。"
    )

    llm = get_llm_client()
    report = await llm.chat_structured(
        messages=[
            {"role": "system", "content": PRACTICE_EVALUATOR_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=PracticeReport,
        temperature=0.3,
    )

    final_items = _final_items(state.get("final_question_assessments"))
    summary = aggregate_scores(final_items, state.get("skill_matrix"))
    report = report.model_copy(
        update={
            "overall_score": summary.overall_score,
            "grade": summary.grade,
            "skill_scores": _deterministic_skill_scores(summary, final_items),
            "coverage": _coverage(summary),
            "missed_knowledge": _deterministic_missed_knowledge(state, final_items, report.missed_knowledge),
            "total_questions": summary.scored_questions + summary.unscored_questions,
        }
    )

    logger.info(
        "Practice evaluation complete: overall=%s (%s), %d missed knowledge items",
        report.overall_score, report.grade, len(report.missed_knowledge),
    )
    return {"practice_report": report, "interview_complete": True}
