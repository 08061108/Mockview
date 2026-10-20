"""Interviewer Agent — conducts the interview: ask, follow-up, assess each answer."""

from __future__ import annotations

import logging
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from app.agents.state import InterviewState
from app.llm.client import get_llm_client, is_streaming_unsupported
from app.llm.prompts import ANSWER_ASSESSOR_SYSTEM, INTERVIEWER_SYSTEM
from app.models.interview import (
    AnswerAssessment,
    ChatMessage,
    FinalQuestionAssessment,
    FinalQuestionAssessmentDraft,
)
from app.models.question import QuestionBlueprint, QuestionItem
from app.domain.skill_identity import canonical_skill_id
from app.streaming import emit_stream_event, streaming_enabled

logger = logging.getLogger(__name__)


def _current_question(state: InterviewState):
    idx = state.get("current_question_index", 0)
    plan = state.get("question_plan", [])
    if idx < len(plan):
        raw = plan[idx]
        return raw if isinstance(raw, QuestionItem) else QuestionItem.model_validate(raw)
    return None


def _planned_question_count(state: InterviewState) -> int:
    raw = state.get("question_blueprint")
    if raw:
        blueprint = raw if isinstance(raw, QuestionBlueprint) else QuestionBlueprint.model_validate(raw)
        return blueprint.planned_count
    return len(state.get("question_plan", []) or [])


def _stable_question_id(state: InterviewState, question) -> str:
    if question.question_id:
        return question.question_id
    namespace = f"{state.get('interview_mode', 'practice')}:{state.get('current_round', 1)}"
    return question.ensure_stable_id(namespace).question_id


def _same_question_messages(state: InterviewState, question_id: str) -> list[ChatMessage]:
    result: list[ChatMessage] = []
    for raw in state.get("conversation_history", []):
        message = raw if isinstance(raw, ChatMessage) else ChatMessage.model_validate(raw)
        if message.question_id == question_id:
            result.append(message)
    return result


def _latest_assessment(state: InterviewState, question_id: str) -> AnswerAssessment | None:
    matches = []
    for raw in state.get("assessments", []):
        item = raw if isinstance(raw, AnswerAssessment) else AnswerAssessment.model_validate(raw)
        if str(item.question_id) == question_id:
            matches.append(item)
    return matches[-1] if matches else None


def _build_conversation_context(state: InterviewState) -> list[dict]:
    """Convert conversation history into OpenAI message format."""
    msgs: list[dict] = []
    for m in state.get("conversation_history", []):
        if isinstance(m, ChatMessage):
            role = "assistant" if m.role == "interviewer" else "user"
            msgs.append({"role": role, "content": m.content})
        elif isinstance(m, dict):
            role = "assistant" if m.get("role") == "interviewer" else "user"
            msgs.append({"role": role, "content": m.get("content", "")})
    return msgs


async def _generate_interviewer_text(
    *,
    messages: list[dict],
    temperature: float,
    kind: str,
    question_index: int,
    total_questions: int,
    follow_up_number: int | None = None,
    skill_tags: list[str] | None = None,
    difficulty: str = "",
    question_id: str = "",
    turn_id: str = "",
) -> str:
    """Generate interviewer text and optionally publish incremental chunks.

    The graph node still returns one complete string. Streaming is only a
    transport concern and is enabled by the WebSocket request through the
    task-local stream sink.
    """
    llm = get_llm_client()
    if not streaming_enabled():
        return await llm.chat(messages=messages, temperature=temperature)

    stream_id = uuid4().hex
    metadata = {
        "stream_id": stream_id,
        "kind": kind,
        "question_index": question_index,
        "total_questions": total_questions,
        "follow_up_number": follow_up_number,
        "skill_tags": skill_tags or [],
        "difficulty": difficulty,
        "question_id": question_id,
        "turn_id": turn_id,
    }
    await emit_stream_event({"event": "start", **metadata})

    chunks: list[str] = []
    try:
        async for chunk in llm.chat_stream(messages=messages, temperature=temperature):
            chunks.append(chunk)
            await emit_stream_event(
                {
                    "event": "delta",
                    **metadata,
                    "chunk": chunk,
                    "done": False,
                }
            )
    except Exception as exc:
        if chunks:
            logger.exception("Streaming interviewer response failed after partial output")
            raise
        if not is_streaming_unsupported(exc):
            logger.exception("Streaming interviewer response failed; result is unknown")
            raise
        logger.warning("Provider explicitly rejected streaming; using one chat request")
        response = await llm.chat(messages=messages, temperature=temperature)
        if response:
            chunks.append(response)
            await emit_stream_event(
                {
                    "event": "delta",
                    **metadata,
                    "chunk": response,
                    "done": False,
                }
            )

    if not chunks:
        logger.warning("Streaming interviewer response was empty; using chat fallback")
        response = await llm.chat(messages=messages, temperature=temperature)
        if not response.strip():
            raise RuntimeError("The model returned an empty interview question")
        chunks.append(response)
        await emit_stream_event(
            {
                "event": "delta",
                **metadata,
                "chunk": response,
                "done": False,
            }
        )

    response = "".join(chunks)
    await emit_stream_event(
        {
            "event": "end",
            **metadata,
            "content": response,
            "done": True,
        }
    )
    return response


# ─────────────────────────────────────────────────────────────────────
# Node 1: Ask a question
# ─────────────────────────────────────────────────────────────────────
async def ask_question(state: InterviewState) -> dict:
    """LangGraph node: ask the current question to the candidate.

    Reads:  question_plan, current_question_index, conversation_history
    Writes: conversation_history (append interviewer message)
    """
    question = _current_question(state)
    if not question:
        if state.get("question_blueprint"):
            raise RuntimeError("The next blueprint slot is not ready; retry question generation")
        return {"interview_complete": True}

    question_id = _stable_question_id(state, question)
    turn_id = f"{question_id}:main"
    response = question.content.strip()
    if not question.question_id:
        max_follow_ups = state.get("max_follow_ups", 2)
        system_prompt = INTERVIEWER_SYSTEM.format(max_follow_ups=max_follow_ups)
        conversation = _build_conversation_context(state)
        conversation.append(
            {
                "role": "user",
                "content": (
                    f"[内部指令，不要展示给候选人] 现在请用简体中文提问第 {question.id} 题："
                    f"{question.content}\n考察技能：{', '.join(question.skill_tags)}\n"
                    f"难度：{question.difficulty}"
                ),
            }
        )
        response = await _generate_interviewer_text(
            messages=[{"role": "system", "content": system_prompt}] + conversation,
            temperature=0.7,
            kind="question",
            question_index=state.get("current_question_index", 0) + 1,
            total_questions=_planned_question_count(state),
            skill_tags=list(question.skill_tags),
            difficulty=question.difficulty,
            question_id=question_id,
            turn_id=turn_id,
        )
    elif streaming_enabled():
        stream_id = uuid4().hex
        metadata = {
            "stream_id": stream_id,
            "kind": "question",
            "question_index": state.get("current_question_index", 0) + 1,
            "total_questions": _planned_question_count(state),
            "follow_up_number": None,
            "skill_tags": list(question.skill_tags),
            "difficulty": question.difficulty,
            "question_id": question_id,
            "turn_id": turn_id,
        }
        await emit_stream_event({"event": "start", **metadata})
        await emit_stream_event({"event": "delta", **metadata, "chunk": response, "done": False})
        await emit_stream_event({"event": "end", **metadata, "content": response, "done": True})

    logger.info("Interviewer asks Q%d: %s", question.id, response[:80])

    return {
        "conversation_history": [
            ChatMessage(
                message_id=f"{turn_id}:interviewer",
                role="interviewer",
                content=response,
                question_id=question_id,
                turn_id=turn_id,
                kind="main",
                created_at=datetime.now(timezone.utc),
            )
        ],
        "current_question_id": question_id,
        "current_turn_id": turn_id,
        "follow_up_count": 0,
    }


# ─────────────────────────────────────────────────────────────────────
# Node 2: Ask a follow-up question
# ─────────────────────────────────────────────────────────────────────
async def ask_follow_up(state: InterviewState) -> dict:
    """LangGraph node: ask a targeted follow-up based on the latest assessment.

    Reads:  question_plan, current_question_index, assessments, conversation_history
    Writes: conversation_history (append follow-up), follow_up_count
    """
    question = _current_question(state)
    question_id = _stable_question_id(state, question)
    latest_assessment = _latest_assessment(state, question_id)

    max_follow_ups = state.get("max_follow_ups", 2)
    system_prompt = INTERVIEWER_SYSTEM.format(max_follow_ups=max_follow_ups)

    missed_hint = ""
    if latest_assessment and latest_assessment.missed_points:
        missed_hint = (
            f"\n[内部指令] 候选人遗漏了这些点："
            f"{', '.join(latest_assessment.missed_points[:3])}。"
            f"追问原因：{latest_assessment.follow_up_reason or '需要进一步深入'}。"
            f"请用简体中文自然追问，引导候选人补充这些方面。"
        )

    follow_up_dirs = ""
    if question and question.follow_up_directions:
        follow_up_dirs = (
            f"\n[内部指令] 可选追问方向："
            f"{', '.join(question.follow_up_directions)}"
        )

    conversation = _build_conversation_context(state)
    conversation.append({
        "role": "user",
        "content": (
            f"[内部指令] 候选人的回答还不完整。"
            f"请用简体中文提出一个有针对性的追问，继续深入考察。"
            f"{missed_hint}{follow_up_dirs}"
        ),
    })

    next_follow_up = state.get("follow_up_count", 0) + 1
    turn_id = f"{question_id}:followup:{next_follow_up}"
    response = await _generate_interviewer_text(
        messages=[{"role": "system", "content": system_prompt}] + conversation,
        temperature=0.7,
        kind="follow_up",
        question_index=state.get("current_question_index", 0) + 1,
        total_questions=_planned_question_count(state),
        follow_up_number=next_follow_up,
        question_id=question_id,
        turn_id=turn_id,
    )

    logger.info("Interviewer follow-up #%d: %s", state.get("follow_up_count", 0) + 1, response[:80])

    follow_up_kind = (
        latest_assessment.follow_up_kind if latest_assessment and latest_assessment.follow_up_kind else "depth"
    )
    return {
        "conversation_history": [
            ChatMessage(
                message_id=f"{turn_id}:interviewer",
                role="interviewer",
                content=response,
                question_id=question_id,
                turn_id=turn_id,
                kind=follow_up_kind,
                created_at=datetime.now(timezone.utc),
            )
        ],
        "current_turn_id": turn_id,
        "follow_up_count": state.get("follow_up_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────
# Node 3: Assess the candidate's answer (internal, not shown to user)
# ─────────────────────────────────────────────────────────────────────
async def assess_answer(state: InterviewState) -> dict:
    """LangGraph node: evaluate the candidate's latest answer.

    Reads:  current_candidate_answer, question_plan, current_question_index,
            follow_up_count, max_follow_ups
    Writes: assessments (append), conversation_history (append candidate msg)
    """
    question = _current_question(state)
    answer = state.get("current_candidate_answer", "")

    if not question:
        return {"interview_complete": True}

    max_follow_ups = state.get("max_follow_ups", 2)
    follow_up_count = state.get("follow_up_count", 0)
    question_id = _stable_question_id(state, question)
    turn_id = state.get("current_turn_id") or f"{question_id}:main"
    answer_id = state.get("current_answer_id") or hashlib.sha256(
        f"{question_id}:{turn_id}:{answer}".encode("utf-8")
    ).hexdigest()

    llm = get_llm_client()
    system_prompt = ANSWER_ASSESSOR_SYSTEM.format(max_follow_ups=max_follow_ups)

    prior_dialogue = []
    for message in _same_question_messages(state, question_id):
        prior_dialogue.append(f"{message.role}: {message.content}")
    user_content = (
        f"Question (#{question.id}, {question.difficulty}):\n{question.content}\n\n"
        f"Reference answer key points:\n"
        + "\n".join(f"- {p}" for p in question.reference_points)
        + "\n\nDialogue for this question so far:\n"
        + ("\n".join(prior_dialogue) or "(no prior dialogue)")
        + f"\n\nCandidate's latest answer:\n{answer}\n\n"
        f"Current follow-up count for this question: {follow_up_count}/{max_follow_ups}\n\n"
        f"Produce a structured assessment. All free-text fields must be Simplified Chinese."
    )

    assessment = await llm.chat_structured(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_model=AnswerAssessment,
        temperature=0.2,
    )
    assessment.question_id = question_id
    assessment.turn_id = turn_id
    assessment.assessment_id = f"{question_id}:{turn_id}:process"

    # Override follow-up decision if we've hit the limit
    if follow_up_count >= max_follow_ups:
        assessment.should_follow_up = False
        assessment.follow_up_reason = None

    logger.info(
        "Assessment Q%d: score=%d, follow_up=%s (covered=%d, missed=%d)",
        question.id,
        assessment.score,
        assessment.should_follow_up,
        len(assessment.covered_points),
        len(assessment.missed_points),
    )

    return {
        "assessments": [assessment],
        "conversation_history": [
            ChatMessage(
                message_id=f"{answer_id}:candidate",
                role="candidate",
                content=answer,
                question_id=question_id,
                turn_id=turn_id,
                kind=("main" if follow_up_count == 0 else None),
                answer_id=answer_id,
                created_at=datetime.now(timezone.utc),
            )
        ],
    }


async def finalize_question_assessment(state: InterviewState) -> dict:
    """Score the complete main-answer/follow-up trajectory exactly once."""
    question = _current_question(state)
    if not question:
        return {}
    question_id = _stable_question_id(state, question)
    existing = state.get("final_question_assessments", {}).get(question_id)
    if existing:
        return {"final_question_assessments": {question_id: existing}}

    messages = _same_question_messages(state, question_id)
    transcript = "\n".join(f"[{item.kind or item.role}] {item.role}: {item.content}" for item in messages)
    process_rows = []
    for raw in state.get("assessments", []):
        item = raw if isinstance(raw, AnswerAssessment) else AnswerAssessment.model_validate(raw)
        if str(item.question_id) == question_id:
            process_rows.append(
                f"turn={item.turn_id} score={item.score} covered={item.covered_points} missed={item.missed_points}"
            )
    llm = get_llm_client()
    draft = await llm.chat_structured(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是最终单题评价器。综合同一题的全部问答，而不是只看最后一句。"
                    "允许合理替代方案；区分澄清追问、深入追问和提示性追问。"
                    "若题面缺少判断所需材料，scorable=false 且 score=null，不能把资料缺失算作低分。"
                    "answer_evidence 必须引用候选人的实际回答片段。"
                    "score 必须是 1 到 10 的整数；仅 scorable=false 时为 null。"
                    "hint_dependency 只能是 none、clarification、substantial 之一。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题目：{question.content}\n"
                    f"主要技能：{question.primary_skill or (question.skill_tags[0] if question.skill_tags else 'unknown')}\n"
                    f"评分要点：{question.reference_points}\n"
                    f"可接受替代思路：{question.acceptable_alternatives}\n\n"
                    f"完整同题对话：\n{transcript}\n\n"
                    f"过程评价（仅作参考）：\n{chr(10).join(process_rows)}"
                ),
            },
        ],
        response_model=FinalQuestionAssessmentDraft,
        temperature=0.2,
    )
    if not draft.scorable:
        draft.score = None
    elif draft.score is None:
        draft.scorable = False
        draft.uncertainty = [*draft.uncertainty, "最终评价未返回有效分数。"]
    final = FinalQuestionAssessment(
        observation_id=f"observation:{question_id}",
        question_id=question_id,
        display_id=question.id,
        primary_skill=canonical_skill_id(
            question.primary_skill or (question.skill_tags[0] if question.skill_tags else "unknown")
        ),
        rubric_version=state.get("rubric_version", "p1-v1"),
        **draft.model_dump(),
    )
    return {"final_question_assessments": {question_id: final}}


async def finalize_answered_question_on_end(state: InterviewState) -> dict:
    """Settle an answered current question, or explicitly retain it as unscored."""
    question = _current_question(state)
    if not question:
        return {}
    question_id = _stable_question_id(state, question)
    if question_id in (state.get("final_question_assessments") or {}):
        return {}
    messages = _same_question_messages(state, question_id)
    answered = any(item.role == "candidate" and item.content.strip() for item in messages)
    assessment = _latest_assessment(state, question_id)
    if assessment is None and not question.question_id:
        # Pre-stable-ID snapshots used display numbers. A process assessment
        # proves this question was answered, but cannot reconstruct its text.
        rows = list(state.get("assessments", []))
        if state.get("current_round", 1) >= 2:
            rows = rows[len(state.get("round1_assessments", [])):]
        assessment = _latest_assessment({"assessments": rows}, str(question.id))
    if not answered and assessment is None:
        return {}

    reason = "已记录回答评价，但缺少可恢复的同题回答正文，未评分。"
    if answered:
        try:
            return await finalize_question_assessment(state)
        except Exception as exc:
            # Failure is not a low score and must not erase the answered question.
            logger.warning("End settlement unavailable question=%s type=%s", question_id, type(exc).__name__)
            reason = "提前结束时最终评价未完成，本题保留为未评分。"
    final = FinalQuestionAssessment(
        observation_id=f"observation:{question_id}",
        question_id=question_id,
        display_id=question.id,
        primary_skill=canonical_skill_id(
            question.primary_skill or (question.skill_tags[0] if question.skill_tags else "unknown")
        ),
        scorable=False,
        score=None,
        uncertainty=[reason],
        rubric_version=state.get("rubric_version", "p1-v1"),
    )
    return {"final_question_assessments": {question_id: final}}


# ─────────────────────────────────────────────────────────────────────
# Routing function: decide what happens after assessment
# ─────────────────────────────────────────────────────────────────────
def route_after_assessment(state: InterviewState) -> str:
    """Conditional edge: decide whether to follow up, move to next question, or finish.

    For practice mode, returns: "ask_follow_up", "ask_question", or "evaluate".
    For professional mode with dual rounds, returns:
      - "ask_follow_up", "ask_question" during a round
      - "summarize_round1" when Round 1 ends
      - "evaluate" when Round 2 ends
    """
    assessments = state.get("assessments", [])
    if not assessments:
        return "finalize_question"

    question = _current_question(state)
    if not question:
        return "finalize_question"
    latest = _latest_assessment(state, _stable_question_id(state, question))
    if latest is None:
        return "finalize_question"
    if latest.should_follow_up:
        logger.info("Route → follow up (score=%d, reason=%s)", latest.score, latest.follow_up_reason)
        return "ask_follow_up"

    return "finalize_question"


def route_after_final_assessment(state: InterviewState) -> str:
    current_idx = state.get("current_question_index", 0)
    planned_count = _planned_question_count(state)
    if current_idx < planned_count - 1:
        logger.info("Route → next question (Q%d done, moving to Q%d)", current_idx + 1, current_idx + 2)
        return "ask_question"

    mode = state.get("interview_mode", "practice")
    current_round = state.get("current_round", 1)

    if mode == "professional" and current_round == 1:
        logger.info("Route → summarize_round1 (Round 1 complete, %d questions)", planned_count)
        return "summarize_round1"

    logger.info("Route → evaluate (all questions done, round=%d)", current_round)
    return "evaluate"


# ─────────────────────────────────────────────────────────────────────
# Helper node: advance to next question
# ─────────────────────────────────────────────────────────────────────
async def advance_question(state: InterviewState) -> dict:
    """Increment current_question_index and reset follow_up_count."""
    new_idx = state.get("current_question_index", 0) + 1
    return {
        "current_question_index": new_idx,
        "follow_up_count": 0,
    }
