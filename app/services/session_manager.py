"""Interview session manager bridges frontend with LangGraph execution.

Supports two modes:
  - practice:     Quick practice with reference answers in the report
  - professional: Dual-round interview (Round 1: depth, Round 2: breadth)
                  with resume analysis and detailed evaluation
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

from app.cache.locks import session_answer_lock
from app.cache.session_cache import save_session_report, save_session_snapshot
from app.config import settings
from app.agents.evaluator import evaluate_interview, evaluate_practice
from app.agents.graph import build_practice_graph, build_professional_graph
from app.agents.question_planner import generate_current_question
from app.agents.interviewer import (
    _current_question,
    _stable_question_id,
    finalize_answered_question_on_end,
)
from app.agents.state import InterviewState
from app.memory.service import get_memory_service
from app.models.interview import (
    AnswerAssessment,
    ChatMessage,
    FinalQuestionAssessment,
    InterviewSession,
)
from app.models.question import QuestionItem
from app.models.jd import SkillMatrix
from app.models.report import InterviewReport, PracticeReport, ProfessionalReport
from app.models.public import PublicSessionOverview, PublicSessionState, PublicTranscript
from app.models.resume import ResumeParseResult, ResumeProfile
from app.models.preparation import AssessmentTarget
from app.models.evidence import EvidenceAssessment, EvidenceChunk
from app.services.checkpoint import get_checkpointer
from app.services.session_repository import (
    PersistedSession,
    list_session_records_for_user,
    load_session_record,
    save_session_record,
)
from app.services.session_event_repository import list_session_events
from app.services.operation_store import get_operation_store, structured_operation_error
from app.services.public_interview import (
    build_public_report,
    build_public_transcript,
    current_public_turn,
)
from app.services.question_artifact_store import get_question_artifact_store
from app.llm.scheduler import model_request_priority
from app.llm.client import model_trace
from app.streaming import (
    SessionEventBus,
    reset_stream_sink,
    set_stream_sink,
)

logger = logging.getLogger(__name__)

StreamSink = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class _SessionData:
    """Internal bookkeeping for a live interview session."""

    session: InterviewSession
    mode: Literal["practice", "professional"] = "practice"
    resume_text: str = ""
    resume_parse_result: ResumeParseResult | None = None
    report: InterviewReport | None = None
    professional_report: ProfessionalReport | None = None
    practice_report: PracticeReport | None = None
    user_id: str = "local-user"
    preparation_case_id: str | None = None
    assessment_targets: list[AssessmentTarget] = field(default_factory=list)
    evidence_chunks: dict[str, list[EvidenceChunk]] = field(default_factory=dict)
    evidence_assessments: dict[str, EvidenceAssessment] = field(default_factory=dict)
    skill_matrix: SkillMatrix | None = None
    resume_profile: ResumeProfile | None = None
    material_revision: int | None = None
    analysis_strategy_version: str = "material-analysis-v1"
    knowledge_release_id: str | None = None
    pending_answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    persisted_assessment_count: int = 0
    final_memory_saved: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    graph_started: bool = False
    last_state: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    event_bus: SessionEventBus = field(default_factory=SessionEventBus)
    start_task: asyncio.Task | None = None
    answer_task: asyncio.Task | None = None
    question_task: asyncio.Task | None = None
    question_retry_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    prefetch_task: asyncio.Task | None = None
    active_operation: Literal["start", "answer"] | None = None
    active_answer_id: str | None = None
    active_answer_content: str | None = None
    last_answer_id: str | None = None
    last_answer_content: str | None = None
    last_answer_result: dict[str, Any] | None = None
    start_attempt: int = 1
    start_retry_commands: dict[str, int] = field(default_factory=dict)
    answer_retry_commands: dict[str, str] = field(default_factory=dict)
    answer_retry_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    completion_reason: Literal["normal", "early", "no_assessment"] | None = None
    paused_at: datetime | None = None
    pause_requested: bool = False
    end_requested: bool = False
    end_task: asyncio.Task | None = None


class SessionManager:
    """Manages interview sessions and drives the LangGraph execution.

    Each session gets its own LangGraph thread_id so checkpoint state
    is isolated. Separate graphs are compiled for practice and professional modes.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionData] = {}
        self._checkpointer = get_checkpointer()
        self._practice_graph = self._build_graph("practice")
        self._professional_graph = self._build_graph("professional")

    # Graph construction

    def _build_graph(self, mode: str):
        graph = (
            build_professional_graph()
            if mode == "professional"
            else build_practice_graph()
        )
        compiled = graph.compile(
            interrupt_before=["assess_answer"],
            checkpointer=self._checkpointer,
        )
        logger.info("SessionManager: %s graph compiled", mode)
        return compiled

    def _get_graph(self, mode: str):
        return self._practice_graph if mode == "practice" else self._professional_graph

    # Session CRUD

    def create_session(
        self,
        session_id: str,
        jd_text: str,
        max_follow_ups: int = 2,
        mode: Literal["practice", "professional"] = "practice",
        resume_text: str = "",
        resume_parse_result: ResumeParseResult | None = None,
        user_id: str = "local-user",
        preparation_case_id: str | None = None,
        assessment_targets: list[AssessmentTarget] | None = None,
        evidence_chunks: dict[str, list[EvidenceChunk]] | None = None,
        evidence_assessments: dict[str, EvidenceAssessment] | None = None,
        skill_matrix: SkillMatrix | None = None,
        resume_profile: ResumeProfile | None = None,
        material_revision: int | None = None,
        analysis_strategy_version: str = "material-analysis-v1",
        knowledge_release_id: str | None = None,
    ) -> InterviewSession:
        session = InterviewSession(
            session_id=session_id,
            user_id=user_id,
            preparation_case_id=preparation_case_id,
            jd_text=jd_text,
            max_follow_ups=max_follow_ups,
        )
        data = _SessionData(
            session=session,
            mode=mode,
            resume_text=resume_text,
            resume_parse_result=resume_parse_result,
            user_id=user_id,
            preparation_case_id=preparation_case_id,
            assessment_targets=list(assessment_targets or []),
            evidence_chunks=dict(evidence_chunks or {}),
            evidence_assessments=dict(evidence_assessments or {}),
            skill_matrix=skill_matrix,
            resume_profile=resume_profile,
            material_revision=material_revision,
            analysis_strategy_version=analysis_strategy_version,
            knowledge_release_id=knowledge_release_id,
            event_bus=SessionEventBus(session_id),
        )
        self._sessions[session_id] = data
        logger.info(
            "Session created: %s (mode=%s, resume=%d chars)",
            session_id,
            mode,
            len(resume_text),
        )
        return session

    def get_session(self, session_id: str) -> InterviewSession | None:
        data = self._sessions.get(session_id)
        return data.session if data else None

    async def ensure_session_loaded(self, session_id: str) -> InterviewSession | None:
        """Load session metadata from persistent storage when memory is empty."""
        data = self._sessions.get(session_id)
        if data:
            return data.session

        persisted = await load_session_record(session_id)
        if not persisted:
            return None

        self._sessions[session_id] = _SessionData(
            session=persisted.session,
            mode=persisted.mode,  # type: ignore[arg-type]
            resume_text=persisted.resume_text,
            resume_parse_result=persisted.resume_parse_result,
            user_id=persisted.session.user_id,
            preparation_case_id=persisted.preparation_case_id,
            assessment_targets=list(persisted.assessment_targets or []),
            evidence_chunks=dict(persisted.evidence_chunks or {}),
            evidence_assessments=dict(persisted.evidence_assessments or {}),
            pending_answers=dict(persisted.pending_answers or {}),
            skill_matrix=(
                SkillMatrix.model_validate(persisted.analysis_snapshot["skill_matrix"])
                if persisted.analysis_snapshot and persisted.analysis_snapshot.get("skill_matrix")
                else None
            ),
            resume_profile=(
                ResumeProfile.model_validate(persisted.analysis_snapshot["resume_profile"])
                if persisted.analysis_snapshot and persisted.analysis_snapshot.get("resume_profile")
                else None
            ),
            material_revision=persisted.material_revision,
            analysis_strategy_version=str(
                (persisted.analysis_snapshot or {}).get(
                    "analysis_strategy_version", "material-analysis-v1"
                )
            ),
            knowledge_release_id=persisted.knowledge_release_id,
            start_attempt=int((persisted.operation_state or {}).get("start_attempt", 1)),
            start_retry_commands=dict(
                (persisted.operation_state or {}).get("start_retry_commands", {})
            ),
            answer_retry_commands=dict(
                (persisted.operation_state or {}).get("answer_retry_commands", {})
            ),
            completion_reason=persisted.completion_reason,
            paused_at=persisted.paused_at,
            pause_requested=bool((persisted.operation_state or {}).get("pause_requested")),
            end_requested=bool((persisted.operation_state or {}).get("end_requested")),
            graph_started=persisted.graph_started,
            last_state=persisted.last_state or {},
            persisted_assessment_count=persisted.persisted_assessment_count,
            final_memory_saved=persisted.final_memory_saved,
            error_message=persisted.error_message,
            event_bus=SessionEventBus(session_id),
        )
        restored = self._sessions[session_id]
        if restored.session.status == "pausing" and not self.has_active_operation(session_id):
            # A process restart invalidates an in-memory running task. Preserve the
            # accepted answer and require an explicit retry unless its checkpoint
            # already contains that answer.
            for answer_id, record in list(restored.pending_answers.items()):
                operation = await get_operation_store().get(
                    f"interview-answer:{session_id}:{answer_id}", restored.user_id
                )
                if (
                    self._answer_from_state(restored.last_state, answer_id) is not None
                    or (operation is not None and operation.status == "succeeded")
                ):
                    restored.pending_answers[answer_id] = {**record, "status": "succeeded"}
                elif record.get("status") in {"accepted", "running"}:
                    restored.pending_answers[answer_id] = {**record, "status": "failed_retryable"}
            restored.session.status = "paused"
            restored.pause_requested = False
            restored.paused_at = restored.paused_at or datetime.now(timezone.utc)
            await self._persist_session(session_id)
        logger.info("Restored session metadata from Postgres: %s", session_id)
        return persisted.session

    async def list_user_sessions(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return lightweight session summaries for one user."""
        summaries_by_id: dict[str, dict[str, Any]] = {}

        persisted_rows = await list_session_records_for_user(
            user_id,
            limit=limit + offset,
            offset=0,
        )
        for persisted in persisted_rows:
            summaries_by_id[persisted.session.session_id] = self._session_summary_from_persisted(
                persisted
            )

        for data in self._sessions.values():
            if data.session.user_id != user_id:
                continue
            summaries_by_id[data.session.session_id] = self._session_summary_from_data(data)

        summaries = list(summaries_by_id.values())
        summaries.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
        return summaries[offset : offset + limit]

    def _session_summary_from_persisted(self, persisted: PersistedSession) -> dict[str, Any]:
        session = persisted.session
        last_state = persisted.last_state or {}
        return self._build_session_summary(
            session=session,
            mode=persisted.mode,
            graph_started=persisted.graph_started,
            last_state=last_state,
            error_message=persisted.error_message,
            created_at=persisted.created_at,
            updated_at=persisted.updated_at,
            completed_at=persisted.completed_at,
            completion_reason=persisted.completion_reason,
            paused_at=persisted.paused_at,
            pending_answers=persisted.pending_answers or {},
        )

    def _session_summary_from_data(self, data: _SessionData) -> dict[str, Any]:
        return self._build_session_summary(
            session=data.session,
            mode=data.mode,
            graph_started=data.graph_started,
            last_state=data.last_state,
            error_message=data.error_message,
            created_at=None,
            updated_at=None,
            completed_at=None,
            completion_reason=data.completion_reason,
            paused_at=data.paused_at,
            pending_answers=data.pending_answers,
        )

    def _build_session_summary(
        self,
        *,
        session: InterviewSession,
        mode: str,
        graph_started: bool,
        last_state: dict[str, Any],
        error_message: str | None,
        created_at: Any | None,
        updated_at: Any | None,
        completed_at: Any | None,
        completion_reason: str | None,
        paused_at: Any | None,
        pending_answers: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        report = (
            last_state.get("practice_report")
            or last_state.get("professional_report")
            or last_state.get("final_report")
        )
        question_count = self._count_questions(last_state)
        assessment_count = len(last_state.get("assessments", []) or session.assessments or [])
        final_count = self._report_value(report, "total_questions")
        if final_count is not None:
            question_count = int(final_count)
            assessment_count = int(final_count)
        title = self._session_title(session.jd_text)
        return {
            "session_id": session.session_id,
            "preparation_case_id": session.preparation_case_id,
            "title": title,
            "mode": mode,
            "status": session.status,
            "graph_started": graph_started,
            "has_report": bool(report),
            "question_count": question_count,
            "assessment_count": assessment_count,
            "current_question_index": session.current_question_index,
            "max_follow_ups": session.max_follow_ups,
            "jd_preview": self._text_preview(session.jd_text, 180),
            "overall_score": self._report_value(report, "overall_score"),
            "grade": self._report_value(report, "grade"),
            "error_message": error_message,
            "created_at": self._iso_or_none(created_at),
            "updated_at": self._iso_or_none(updated_at),
            "completed_at": self._iso_or_none(completed_at),
            "completion_reason": completion_reason,
            "paused_at": self._iso_or_none(paused_at),
            "pending_action": (
                "answer_retry_required"
                if any(item.get("status") == "failed_retryable" for item in pending_answers.values())
                else None
            ),
        }

    def _count_questions(self, state: dict[str, Any]) -> int:
        plans = [
            state.get("question_plan", []),
            state.get("round1_question_plan", []),
            state.get("round2_question_plan", []),
        ]
        return sum(len(plan or []) for plan in plans)

    def _session_title(self, jd_text: str) -> str:
        for line in jd_text.splitlines():
            text = line.strip(" #：:")
            if text:
                return self._text_preview(text, 36)
        return "未命名面试"

    def _text_preview(self, text: str, length: int) -> str:
        compact = " ".join(text.split())
        if len(compact) <= length:
            return compact
        return compact[: length - 1].rstrip() + "..."

    def _report_value(self, report: Any, key: str) -> Any:
        if not report:
            return None
        if isinstance(report, dict):
            return report.get(key)
        return getattr(report, key, None)

    def _iso_or_none(self, value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def get_session_mode(self, session_id: str) -> str:
        data = self._sessions.get(session_id)
        return data.mode if data else "practice"

    def get_public_overview(self, session_id: str) -> PublicSessionOverview:
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        return PublicSessionOverview.model_validate(self._session_summary_from_data(data))

    def preparation_revision(self, session_id: str) -> int | None:
        data = self._sessions.get(session_id)
        return data.material_revision if data else None

    async def get_public_state(self, session_id: str) -> PublicSessionState:
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        state = await self.get_last_state(session_id)
        overview = self.get_public_overview(session_id)
        return PublicSessionState(
            session=overview,
            graph_started=data.graph_started,
            current_turn=current_public_turn(state),
            pending_action=overview.pending_action,
            has_report=overview.has_report,
            interview_complete=data.session.status == "completed",
        )

    async def get_public_transcript(self, session_id: str) -> PublicTranscript:
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        state = await self.get_last_state(session_id)
        return build_public_transcript(
            session=data.session,
            overview=self.get_public_overview(session_id),
            state=state,
            pending_answers=data.pending_answers,
            report=self.get_report_for_session(session_id),
        )

    def get_public_report(self, session_id: str):
        data = self._sessions.get(session_id)
        if not data:
            return None
        return build_public_report(
            self.get_report_for_session(session_id),
            completed=data.session.status == "completed",
        )

    def update_session_status(self, session_id: str, status: str) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.session.status = status
            self._schedule_persist(session_id)

    def set_resume(
        self,
        session_id: str,
        resume_text: str,
        resume_parse_result: ResumeParseResult | None = None,
    ) -> None:
        """Attach parsed resume content before the graph starts."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        if data.graph_started:
            raise RuntimeError("Resume cannot be changed after the interview has started")
        data.resume_text = resume_text
        data.resume_parse_result = resume_parse_result
        self._schedule_persist(session_id)

    def get_resume_parse_result(self, session_id: str) -> ResumeParseResult | None:
        data = self._sessions.get(session_id)
        return data.resume_parse_result if data else None

    def has_graph_started(self, session_id: str) -> bool:
        data = self._sessions.get(session_id)
        return bool(data and data.graph_started)

    def get_error_message(self, session_id: str) -> str | None:
        data = self._sessions.get(session_id)
        return data.error_message if data else None

    def save_report(self, session_id: str, report: InterviewReport) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.report = report
            data.session.status = "completed"
            self._schedule_persist(session_id)

    def save_professional_report(self, session_id: str, report: ProfessionalReport) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.professional_report = report
            data.session.status = "completed"
            self._schedule_persist(session_id)

    def save_practice_report(self, session_id: str, report: PracticeReport) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.practice_report = report
            data.session.status = "completed"
            self._schedule_persist(session_id)

    def get_report(self, session_id: str) -> InterviewReport | None:
        data = self._sessions.get(session_id)
        return data.report if data else None

    def get_professional_report(self, session_id: str) -> ProfessionalReport | None:
        data = self._sessions.get(session_id)
        return data.professional_report if data else None

    def get_practice_report(self, session_id: str) -> PracticeReport | None:
        data = self._sessions.get(session_id)
        return data.practice_report if data else None

    def get_report_for_session(
        self,
        session_id: str,
    ) -> PracticeReport | ProfessionalReport | InterviewReport | dict | None:
        data = self._sessions.get(session_id)
        if not data:
            return None
        if data.mode == "practice":
            return (
                data.practice_report
                or self._report_from_last_state(session_id, "practice_report", PracticeReport)
                or data.report
                or self._report_from_last_state(session_id, "final_report", InterviewReport)
            )
        return (
            data.professional_report
            or self._report_from_last_state(session_id, "professional_report", ProfessionalReport)
            or data.report
            or self._report_from_last_state(session_id, "final_report", InterviewReport)
        )

    def _report_from_last_state(self, session_id: str, key: str, model_cls: type) -> Any:
        data = self._sessions.get(session_id)
        if not data:
            return None
        raw_report = (data.last_state or {}).get(key)
        if not raw_report:
            return None
        if isinstance(raw_report, model_cls):
            return raw_report
        try:
            restored = model_cls.model_validate(raw_report)
        except Exception:
            logger.debug(
                "Returning raw report from last_state: session=%s key=%s",
                session_id,
                key,
            )
            return raw_report

        if key == "practice_report":
            data.practice_report = restored
        elif key == "professional_report":
            data.professional_report = restored
        elif key == "final_report":
            data.report = restored
        return restored

    async def get_last_state(self, session_id: str) -> dict[str, Any]:
        data = self._sessions.get(session_id)
        if not data:
            return {}
        graph_state = await self.get_graph_state(session_id)
        if graph_state:
            self._sync_session_from_state(session_id, graph_state)
            await self._cache_runtime_state(session_id, graph_state)
            return graph_state
        return data.last_state

    def _sync_session_from_state(self, session_id: str, state: dict | None) -> None:
        """Mirror LangGraph state into lightweight session metadata."""
        if not state:
            return
        data = self._sessions.get(session_id)
        if not data:
            return

        state_dict = dict(state)
        data.last_state = state_dict

        session = data.session
        session.current_question_index = int(
            state_dict.get("current_question_index", session.current_question_index) or 0
        )
        session.follow_up_count = int(
            state_dict.get("follow_up_count", session.follow_up_count) or 0
        )

        history = []
        for raw_msg in state_dict.get("conversation_history", []) or []:
            try:
                if isinstance(raw_msg, ChatMessage):
                    history.append(raw_msg)
                else:
                    history.append(ChatMessage.model_validate(raw_msg))
            except Exception:
                logger.debug("Skipped invalid chat message in session=%s: %r", session_id, raw_msg)
        session.conversation_history = history

        assessments = []
        for raw_assessment in state_dict.get("assessments", []) or []:
            try:
                assessments.append(
                    raw_assessment
                    if isinstance(raw_assessment, AnswerAssessment)
                    else AnswerAssessment.model_validate(raw_assessment)
                )
            except Exception:
                logger.debug(
                    "Skipped invalid assessment in session=%s: %r",
                    session_id,
                    raw_assessment,
                )
        session.assessments = assessments

        if state_dict.get("interview_complete"):
            session.status = "completed"
            if data.completion_reason is None:
                data.completion_reason = "normal"

    async def _cache_runtime_state(self, session_id: str, state: dict | None) -> None:
        if not state:
            return
        data = self._sessions.get(session_id)
        if not data:
            return
        await self._persist_session(session_id)
        await save_session_snapshot(
            data.session,
            dict(state),
            mode=data.mode,
            graph_started=data.graph_started,
        )
        report = (
            state.get("practice_report")
            or state.get("professional_report")
            or state.get("final_report")
            or self.get_report_for_session(session_id)
        )
        if report:
            await save_session_report(session_id, report)

    # LangGraph execution helpers

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def subscribe_events(
        self, session_id: str, after_sequence: int = 0
    ) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to live events for a session."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        return data.event_bus.subscribe(after_sequence)

    async def replay_events(
        self, session_id: str, after_sequence: int = 0
    ) -> list[dict[str, Any]]:
        """Replay durable non-token events after a client-confirmed sequence."""
        persisted = await list_session_events(
            session_id, after_sequence=max(after_sequence, 0), limit=256
        )
        data = self._sessions.get(session_id)
        live = data.event_bus.replay(after_sequence) if data else []
        by_sequence = {int(item["sequence"]): item for item in persisted}
        for item in live:
            by_sequence[int(item["sequence"])] = item
        return [by_sequence[key] for key in sorted(by_sequence)]

    def unsubscribe_events(
        self,
        session_id: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.event_bus.unsubscribe(queue)

    def has_active_operation(self, session_id: str) -> bool:
        data = self._sessions.get(session_id)
        if not data:
            return False
        return bool(
            (data.start_task and not data.start_task.done())
            or (data.answer_task and not data.answer_task.done())
            or (data.question_task and not data.question_task.done())
        )

    def _active_task(self, data: _SessionData) -> asyncio.Task | None:
        for task in (data.answer_task, data.start_task, data.question_task):
            if task is not None and not task.done():
                return task
        return None

    def _stream_sink_for(self, data: _SessionData) -> StreamSink:
        async def publish(event: dict[str, Any]) -> None:
            if (data.pause_requested or data.end_requested) and event.get("type") != "status":
                return
            await data.event_bus.publish(event)
        return publish

    async def _complete_deferred_pause(self, session_id: str) -> None:
        data = self._sessions.get(session_id)
        if not data or not data.pause_requested or data.end_requested:
            return
        if self._active_task(data) is not None:
            return
        data.pause_requested = False
        data.session.status = "paused"
        data.paused_at = datetime.now(timezone.utc)
        await self._persist_session(session_id)
        await data.event_bus.publish({
            "type": "status", "stage": "paused", "message": "面试已暂停。"
        })

    async def request_pause(self, session_id: str) -> PublicSessionState:
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        if data.session.status in {"completed", "failed"}:
            raise RuntimeError("Terminal interview cannot be paused")
        if data.session.status == "paused":
            return await self.get_public_state(session_id)
        data.pause_requested = True
        active = self._active_task(data)
        if active is None:
            await self._complete_deferred_pause(session_id)
        else:
            data.session.status = "pausing"
            await self._persist_session(session_id)
            active.add_done_callback(
                lambda _task: asyncio.create_task(self._complete_deferred_pause(session_id))
            )
        return await self.get_public_state(session_id)

    async def resume_interview(self, session_id: str) -> PublicSessionState:
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        if data.session.status in {"completed", "failed"}:
            raise RuntimeError("Terminal interview cannot be resumed")
        if data.session.status == "pausing" and self._active_task(data) is not None:
            raise RuntimeError("Pause is still waiting for the current operation")
        data.pause_requested = False
        data.end_requested = False
        data.paused_at = None
        data.session.status = "interviewing" if data.graph_started else "pending"
        await self._persist_session(session_id)
        return await self.get_public_state(session_id)

    def ensure_end_task(self, session_id: str) -> asyncio.Task:
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        if data.end_task is not None:
            return data.end_task
        if data.session.status == "completed":
            async def completed() -> dict[str, Any]:
                return data.last_state
            data.end_task = asyncio.create_task(completed())
            return data.end_task
        if data.session.status == "failed":
            raise RuntimeError("Failed interview cannot be ended again")
        data.end_requested = True
        data.pause_requested = False

        async def run() -> dict[str, Any]:
            active = self._active_task(data)
            if active is not None and active is not asyncio.current_task():
                await asyncio.gather(asyncio.shield(active), return_exceptions=True)
            return await self.stop_interview(session_id)

        data.end_task = asyncio.create_task(run(), name=f"interview-end:{session_id}")
        return data.end_task

    def ensure_start_task(self, session_id: str) -> asyncio.Task:
        """Start one background graph run for WebSocket consumers."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        if data.graph_started:
            raise RuntimeError("Interview graph has already started")
        if data.start_task is not None:
            return data.start_task

        if data.session.status == "failed":
            async def return_terminal_failure() -> dict:
                raise RuntimeError(
                    data.error_message or "Interview startup failed; explicit retry required"
                )

            data.start_task = asyncio.create_task(
                return_terminal_failure(), name=f"interview-start-failed:{session_id}"
            )
            return data.start_task

        data.active_operation = "start"
        attempt = data.start_attempt
        operation_id = f"interview-start:{session_id}:attempt:{attempt}"

        async def run() -> dict:
            operations = get_operation_store()
            operation = await operations.accept(
                user_id=data.user_id,
                kind="interview_start",
                case_id=session_id,
                input_revision=attempt,
                operation_id=operation_id,
                payload={
                    "session_id": session_id,
                    "attempt": attempt,
                    "mode": data.mode,
                    "material_revision": data.material_revision,
                },
                budget={
                    "model_requests": settings.interview_start_model_request_budget
                },
            )
            if operation.status == "succeeded" and data.last_state:
                return data.last_state
            if operation.status in {"failed", "failed_retryable", "cancelled"}:
                checkpoint = await self.get_graph_state(session_id) or data.last_state
                safe_resume = False
                if checkpoint:
                    index = int(checkpoint.get("current_question_index", 0) or 0)
                    if index < len(checkpoint.get("question_plan", []) or []):
                        safe_resume = True
                    else:
                        blueprint = checkpoint.get("question_blueprint")
                        if blueprint:
                            from app.models.question import QuestionBlueprint

                            parsed = (
                                blueprint
                                if isinstance(blueprint, QuestionBlueprint)
                                else QuestionBlueprint.model_validate(blueprint)
                            )
                            if index < parsed.planned_count:
                                artifact = await get_question_artifact_store().get(
                                    session_id,
                                    parsed.blueprint_id,
                                    parsed.slots[index].slot_id,
                                )
                                safe_resume = bool(
                                    artifact
                                    and artifact.status == "ready"
                                    and not artifact.terminated
                                )
                if not safe_resume:
                    message = operation.error or "startup failed; explicit retry required"
                    data.error_message = message
                    data.session.status = "failed"
                    await self._persist_session(session_id)
                    raise RuntimeError(message)
            await operations.transition(
                operation_id, "running", error=None, error_details=None
            )
            try:
                async with asyncio.timeout(settings.interview_start_timeout_seconds):
                    result = await self.start_interview_graph(
                        session_id,
                        stream_sink=self._stream_sink_for(data),
                    )
                await operations.transition(
                    operation_id,
                    "succeeded",
                    result={"session_id": session_id, "graph_started": True},
                    error=None,
                    error_details=None,
                )
                return result
            except Exception as exc:
                await operations.transition(
                    operation_id,
                    "failed_retryable",
                    error=str(exc),
                    error_details=structured_operation_error(exc),
                )
                raise
            finally:
                if data.active_operation == "start":
                    data.active_operation = None

        task = asyncio.create_task(run(), name=f"interview-start:{session_id}")
        def clear_start_operation(_task: asyncio.Task) -> None:
            if data.active_operation == "start":
                data.active_operation = None

        task.add_done_callback(clear_start_operation)
        data.start_task = task
        return task

    async def retry_start_task(self, session_id: str, retry_id: str) -> asyncio.Task:
        """Create one explicit startup attempt; duplicate retry commands are idempotent."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        if data.session.status in {"pausing", "paused", "completed"}:
            raise RuntimeError("Interview is not startable in its current state")
        if data.graph_started:
            raise RuntimeError("Interview graph has already started")
        normalized = retry_id.strip()
        if not normalized:
            raise ValueError("retry_id is required")
        previous = data.start_retry_commands.get(normalized)
        if previous is not None:
            if data.start_task is not None and data.start_attempt == previous:
                return data.start_task
            operation_id = f"interview-start:{session_id}:attempt:{previous}"
            operation = await get_operation_store().get(operation_id, data.user_id)
            if operation is None or operation.status in {
                "accepted",
                "succeeded",
                "failed",
                "failed_retryable",
                "cancelled",
            }:
                # The command was durably recorded but the process may have
                # stopped before its task began. Resume the same attempt; do
                # not allocate another budget or attempt number.
                data.start_attempt = previous
                data.start_task = None
                data.error_message = None
                data.session.status = "pending"
                return self.ensure_start_task(session_id)
            raise RuntimeError("This retry command has already been consumed")
        if data.start_task and not data.start_task.done():
            return data.start_task
        if data.start_attempt >= settings.interview_start_max_attempts:
            raise RuntimeError("Interview startup retry budget is exhausted")
        data.start_attempt += 1
        data.start_retry_commands[normalized] = data.start_attempt
        data.start_task = None
        data.error_message = None
        data.session.status = "pending"
        await self._persist_session(session_id)
        return self.ensure_start_task(session_id)

    def ensure_answer_task(
        self,
        session_id: str,
        answer: str,
        answer_id: str,
    ) -> asyncio.Task:
        """Submit one answer exactly once within the current process."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")

        if data.answer_task and not data.answer_task.done():
            if data.active_answer_id == answer_id:
                if data.active_answer_content != answer:
                    raise ValueError("answer_id already exists with different content")
                return data.answer_task
            raise RuntimeError("Another answer is already being processed")

        persisted_answer = self._answer_from_state(data.last_state, answer_id)
        if persisted_answer is not None:
            if persisted_answer != answer:
                raise ValueError("answer_id already exists with different content")
            async def return_persisted() -> dict[str, Any]:
                return data.last_state or {}

            return asyncio.create_task(
                return_persisted(),
                name=f"interview-answer-persisted:{session_id}",
            )

        if data.last_answer_id == answer_id and data.last_answer_result is not None:
            if data.last_answer_content != answer:
                raise ValueError("answer_id already exists with different content")
            async def return_cached() -> dict[str, Any]:
                return data.last_answer_result or {}

            return asyncio.create_task(
                return_cached(),
                name=f"interview-answer-cached:{session_id}",
            )

        data.active_operation = "answer"
        data.active_answer_id = answer_id
        data.active_answer_content = answer

        async def run() -> dict:
            operations = get_operation_store()
            operation_id = f"interview-answer:{session_id}:{answer_id}"
            operation = await operations.accept(
                user_id=data.user_id,
                kind="interview_answer",
                case_id=session_id,
                operation_id=operation_id,
                payload={
                    "session_id": session_id,
                    "answer_id": answer_id,
                    "answer_hash": __import__("hashlib").sha256(
                        answer.encode("utf-8")
                    ).hexdigest(),
                },
                budget={
                    "model_requests": settings.answer_processing_model_request_budget
                },
            )
            if operation.status == "succeeded":
                if data.last_state:
                    return data.last_state
                raise RuntimeError(
                    "answer operation succeeded but its checkpoint is unavailable"
                )
            await operations.transition(
                operation_id, "running", error=None, error_details=None
            )
            try:
                import inspect

                parameters = inspect.signature(self.submit_answer).parameters
                kwargs = {"stream_sink": self._stream_sink_for(data)}
                if "answer_id" in parameters:
                    kwargs["answer_id"] = answer_id
                with model_trace(
                    session_id=session_id,
                    operation_id=operation_id,
                    enforce_budget=True,
                ):
                    async with asyncio.timeout(
                        settings.answer_processing_timeout_seconds
                    ):
                        result = await self.submit_answer(session_id, answer, **kwargs)
                data.last_answer_id = answer_id
                data.last_answer_content = answer
                data.last_answer_result = result
                if answer_id in data.pending_answers:
                    data.pending_answers[answer_id] = {
                        **data.pending_answers[answer_id],
                        "status": "succeeded",
                    }
                    await self._persist_session(session_id)
                await operations.transition(
                    operation_id,
                    "succeeded",
                    result={
                        "session_id": session_id,
                        "answer_id": answer_id,
                        "interview_complete": bool(result.get("interview_complete")),
                    },
                    error=None,
                    error_details=None,
                )
                return result
            except Exception as exc:
                await operations.transition(
                    operation_id,
                    "failed_retryable",
                    error=str(exc),
                    error_details=structured_operation_error(exc),
                )
                if answer_id in data.pending_answers:
                    data.pending_answers[answer_id] = {
                        **data.pending_answers[answer_id],
                        "status": "failed_retryable",
                    }
                    await self._persist_session(session_id)
                raise
            finally:
                data.active_answer_id = None
                data.active_answer_content = None
                if data.active_operation == "answer":
                    data.active_operation = None

        task = asyncio.create_task(run(), name=f"interview-answer:{session_id}")
        data.answer_task = task
        return task

    async def accept_answer_submission(
        self,
        session_id: str,
        answer: str,
        answer_id: str,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        if data.session.status in {"pausing", "paused", "completed"}:
            raise RuntimeError("Interview is not accepting answers in its current state")
        existing = data.pending_answers.get(answer_id)
        if existing:
            if existing.get("content") != answer or existing.get("turn_id") != turn_id:
                raise ValueError("answer_id already exists with different content")
            return existing
        record = {
            "answer_id": answer_id,
            "turn_id": turn_id,
            "content": answer,
            "status": "accepted",
            "accepted_at": time.time(),
        }
        data.pending_answers[answer_id] = record
        await self._persist_session(session_id)
        return record

    def ensure_pending_answer_task(self, session_id: str) -> asyncio.Task | None:
        """Attach to an already-running accepted answer, never replay it on reconnect."""
        data = self._sessions.get(session_id)
        if not data:
            return None
        if data.answer_task is not None and not data.answer_task.done():
            return data.answer_task
        return None

    def pending_retry_kind(self, session_id: str) -> str | None:
        """Classify an explicit retry without issuing any work."""
        data = self._sessions.get(session_id)
        if not data:
            return None
        pending = any(
            item.get("status") in {"accepted", "failed_retryable"} for item in data.pending_answers.values()
        )
        if not pending:
            return None
        state = data.last_state or {}
        index = int(state.get("current_question_index", 0) or 0)
        return "answer" if index < len(state.get("question_plan", []) or []) else "question"

    async def retry_pending_answer_task(
        self, session_id: str, retry_id: str
    ) -> asyncio.Task:
        """Explicitly resume one accepted answer using its remaining durable budget."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        normalized = retry_id.strip()
        if not normalized:
            raise ValueError("retry_id is required")
        previous_task = data.answer_retry_tasks.get(normalized)
        if previous_task is not None:
            return previous_task
        pending = next(
            (
                item
                for item in data.pending_answers.values()
                if item.get("status") in {"accepted", "failed_retryable"}
            ),
            None,
        )
        if pending is None:
            raise RuntimeError("No accepted answer is available to retry")
        previous_answer_id = data.answer_retry_commands.get(normalized)
        if previous_answer_id is not None:
            raise RuntimeError(
                "This retry command is already consumed; submit a new retry_id"
            )
        answer_id = str(pending["answer_id"])
        data.answer_retry_commands[normalized] = answer_id
        await self._persist_session(session_id)
        task = self.ensure_answer_task(
            session_id,
            str(pending["content"]),
            answer_id,
        )
        data.answer_retry_tasks[normalized] = task
        return task

    def _schedule_question_prefetch(self, session_id: str, state: dict[str, Any]) -> None:
        """Save at most one validated next-slot candidate without advancing Graph."""
        if not settings.question_prefetch_enabled or state.get("interview_complete"):
            return
        data = self._sessions.get(session_id)
        if not data:
            return
        if data.pause_requested or data.end_requested or data.session.status in {"pausing", "paused"}:
            return
        if data.prefetch_task is not None and not data.prefetch_task.done():
            return
        blueprint = state.get("question_blueprint")
        if not blueprint:
            return
        planned = int(
            getattr(blueprint, "planned_count", 0)
            or (blueprint.get("planned_count", 0) if isinstance(blueprint, dict) else 0)
        )
        next_index = int(state.get("current_question_index", 0) or 0) + 1
        if next_index >= planned or next_index < len(state.get("question_plan", []) or []):
            return
        snapshot = dict(state)
        snapshot["current_question_index"] = next_index
        snapshot["question_generation_background"] = True

        async def run() -> None:
            try:
                async with model_request_priority("background"):
                    await generate_current_question(snapshot)  # candidate stays outside checkpoint
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Next-question prefetch failed: session=%s index=%d",
                    session_id,
                    next_index,
                )

        data.prefetch_task = asyncio.create_task(
            run(), name=f"interview-prefetch:{session_id}:{next_index}"
        )

    async def retry_question_task(self, session_id: str, retry_id: str) -> asyncio.Task:
        """Retry only the failed current slot and keep the accepted-answer checkpoint."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        normalized = retry_id.strip()
        if not normalized:
            raise ValueError("retry_id is required")
        previous_task = data.question_retry_tasks.get(normalized)
        if previous_task is not None:
            return previous_task
        if data.question_task is not None and not data.question_task.done():
            return data.question_task
        state = await self.get_graph_state(session_id) or data.last_state
        blueprint_raw = state.get("question_blueprint") if state else None
        if not blueprint_raw:
            raise RuntimeError("Current session has no progressive question blueprint")
        from app.models.question import QuestionBlueprint

        blueprint = (
            blueprint_raw
            if isinstance(blueprint_raw, QuestionBlueprint)
            else QuestionBlueprint.model_validate(blueprint_raw)
        )
        index = int(state.get("current_question_index", 0) or 0)
        if index >= blueprint.planned_count:
            raise RuntimeError("No failed question slot is available to retry")
        slot = blueprint.slots[index]
        attempts = dict(state.get("question_generation_attempts", {}) or {})
        commands = dict(state.get("question_retry_commands", {}) or {})
        duplicate_command = normalized in commands
        if duplicate_command:
            attempt = int(commands[normalized])
            if index < len(state.get("question_plan", []) or []):
                async def return_retried_state() -> dict[str, Any]:
                    return state

                task = asyncio.create_task(
                    return_retried_state(),
                    name=(
                        f"interview-question-retry-complete:{session_id}:"
                        f"{slot.slot_id}:{attempt}"
                    ),
                )
                data.question_task = task
                data.question_retry_tasks[normalized] = task
                return task
            artifact = await get_question_artifact_store().get(
                session_id,
                blueprint.blueprint_id,
                slot.slot_id,
            )
            operation_id = (
                f"interview-question:{session_id}:{blueprint.blueprint_id}:"
                f"{slot.slot_id}:attempt:{attempt}"
            )
            operation = await get_operation_store().get(operation_id, data.user_id)
            if not (artifact and artifact.status == "ready") and operation is not None:
                if operation.status in {
                    "running",
                    "failed",
                    "failed_retryable",
                    "cancelled",
                }:
                    raise RuntimeError(
                        "This retry command is already consumed; submit a new retry_id"
                    )
        else:
            current_attempt = max(1, int(attempts.get(slot.slot_id, 1)))
            if current_attempt >= settings.question_generation_max_attempts:
                raise RuntimeError("Question-generation retry budget is exhausted")
            attempt = current_attempt + 1
            commands[normalized] = attempt
            attempts[slot.slot_id] = attempt
            graph = self._get_graph(data.mode)
            update = {
                "question_generation_attempts": attempts,
                "question_retry_commands": commands,
                "question_generation_status": "idle",
                "question_generation_error": "",
            }
            if hasattr(graph, "aupdate_state"):
                await graph.aupdate_state(self._config(session_id), update)
            else:
                graph.update_state(self._config(session_id), update)

        async def run() -> dict:
            graph = self._get_graph(data.mode)
            token = set_stream_sink(self._stream_sink_for(data))
            try:
                async with data.lock:
                    result = await graph.ainvoke(None, self._config(session_id))
                    data.error_message = None
                    self._sync_session_from_state(session_id, result)
                    if not result.get("interview_complete"):
                        self.update_session_status(session_id, "interviewing")
                    for answer_id, record in list(data.pending_answers.items()):
                        if self._answer_from_state(result, answer_id) is not None:
                            data.pending_answers[answer_id] = {**record, "status": "succeeded"}
                    await self._cache_runtime_state(session_id, result)
                    self._schedule_question_prefetch(session_id, result)
                    return result
            except Exception as exc:
                data.error_message = str(exc)
                self.update_session_status(session_id, "failed")
                await self._persist_session(session_id)
                raise
            finally:
                reset_stream_sink(token)

        data.question_task = asyncio.create_task(
            run(), name=f"interview-question-retry:{session_id}:{slot.slot_id}:{attempt}"
        )
        data.question_retry_tasks[normalized] = data.question_task
        return data.question_task

    def _answer_from_state(self, state: dict[str, Any], answer_id: str) -> str | None:
        for raw in state.get("conversation_history", []) or []:
            try:
                message = raw if isinstance(raw, ChatMessage) else ChatMessage.model_validate(raw)
            except Exception:
                continue
            if message.answer_id == answer_id:
                return message.content
        return None

    async def _persist_session(self, session_id: str) -> None:
        data = self._sessions.get(session_id)
        if not data:
            return
        try:
            await save_session_record(
                session=data.session,
                mode=data.mode,
                resume_text=data.resume_text,
                resume_parse_result=data.resume_parse_result,
                graph_started=data.graph_started,
                last_state=data.last_state,
                persisted_assessment_count=data.persisted_assessment_count,
                final_memory_saved=data.final_memory_saved,
                error_message=data.error_message,
                preparation_case_id=data.preparation_case_id,
                assessment_targets=data.assessment_targets,
                evidence_chunks=data.evidence_chunks,
                evidence_assessments=data.evidence_assessments,
                pending_answers=data.pending_answers,
                analysis_snapshot={
                    "skill_matrix": data.skill_matrix,
                    "resume_profile": data.resume_profile,
                    "analysis_strategy_version": data.analysis_strategy_version,
                },
                material_revision=data.material_revision,
                knowledge_release_id=data.knowledge_release_id,
                operation_state={
                    "start_attempt": data.start_attempt,
                    "start_retry_commands": data.start_retry_commands,
                    "answer_retry_commands": data.answer_retry_commands,
                    "pause_requested": data.pause_requested,
                    "end_requested": data.end_requested,
                },
                completion_reason=data.completion_reason,
                paused_at=data.paused_at,
            )
        except Exception:
            logger.exception("Failed to persist session metadata: %s", session_id)
            raise

    async def persist_session(self, session_id: str) -> None:
        """Persist lightweight session metadata through the configured store."""
        await self._persist_session(session_id)

    def _schedule_persist(self, session_id: str) -> None:
        try:
            asyncio.get_running_loop().create_task(self._persist_session(session_id))
        except RuntimeError:
            logger.debug("No running loop; skipped async persist scheduling for %s", session_id)

    async def start_interview_graph(
        self,
        session_id: str,
        *,
        stream_sink: StreamSink | None = None,
    ) -> dict:
        """Kick off the graph.

        Practice:     analyze_jd -> plan_questions -> ask_question -> interrupt
        Professional: analyze_resume -> analyze_jd -> plan_round1 -> ask_question -> interrupt
        """
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")

        started_at = time.perf_counter()
        async with data.lock:
            if data.graph_started:
                state = await self.get_graph_state(session_id) or data.last_state
                self._sync_session_from_state(session_id, state)
                await self._cache_runtime_state(session_id, state)
                return state

            session = data.session
            mode = data.mode
            graph = self._get_graph(mode)
            existing_snapshot = None
            try:
                if hasattr(graph, "aget_state"):
                    existing_snapshot = await graph.aget_state(self._config(session_id))
                else:
                    existing_snapshot = graph.get_state(self._config(session_id))
            except Exception:
                logger.exception("Unable to inspect startup checkpoint: %s", session_id)
            existing_values = (
                dict(existing_snapshot.values)
                if existing_snapshot is not None and getattr(existing_snapshot, "values", None)
                else {}
            )
            existing_next = tuple(getattr(existing_snapshot, "next", ()) or ())
            if existing_values and not existing_next:
                data.graph_started = bool(existing_values.get("conversation_history"))
                self._sync_session_from_state(session_id, existing_values)
                await self._cache_runtime_state(session_id, existing_values)
                return existing_values
            memory = get_memory_service()
            if existing_values and existing_next:
                # A checkpoint already froze the material and long-term-memory
                # snapshot for this interview. Resume its pending node without
                # reading newer mutable memory or rebuilding initial state.
                memory_context_text = str(existing_values.get("memory_context", ""))
                retrieved_memories = dict(
                    existing_values.get("retrieved_memories", {}) or {}
                )
            else:
                semantic_query = self._build_memory_query(session.jd_text, data.resume_text)
                memory_started_at = time.perf_counter()
                memory_context = await memory.abuild_context(
                    data.user_id,
                    semantic_query=semantic_query,
                )
                logger.info(
                    "Interview preparation memory context ready: session=%s elapsed=%.2fs items=%d",
                    session_id,
                    time.perf_counter() - memory_started_at,
                    sum(
                        len(items)
                        for items in (
                            memory_context.profile_items,
                            memory_context.resume_items,
                            memory_context.recent_reflections,
                            memory_context.weak_skills,
                            memory_context.relevant_episodes,
                            memory_context.semantic_memories,
                        )
                    ),
                )
                memory_context_text = memory.format_context(memory_context)
                retrieved_memories = memory_context.model_dump(mode="json")

            initial_state: InterviewState = {
                "interview_mode": mode,
                "session_id": session_id,
                "user_id": data.user_id,
                "jd_text": session.jd_text,
                "memory_context": memory_context_text,
                "retrieved_memories": retrieved_memories,
                "max_follow_ups": session.max_follow_ups,
                "current_question_index": 0,
                "follow_up_count": 0,
                "conversation_history": [],
                "assessments": [],
                "final_question_assessments": {},
                "interview_complete": False,
                "current_round": 1,
                "graph_version": "interview-p6-v1",
                "state_version": "interview-state-p6-v1",
                "rubric_version": "p1-v1",
                "assessment_targets": list(data.assessment_targets),
                "evidence_chunks": dict(data.evidence_chunks),
                "evidence_assessments": dict(data.evidence_assessments),
                "preparation_case_id": data.preparation_case_id,
                "material_revision": data.material_revision,
                "analysis_strategy_version": data.analysis_strategy_version,
                "knowledge_release_id": data.knowledge_release_id,
            }
            if data.skill_matrix:
                initial_state["skill_matrix"] = data.skill_matrix
            if data.resume_profile:
                initial_state["resume_profile"] = data.resume_profile

            if mode == "professional" and data.resume_text:
                initial_state["resume_text"] = data.resume_text
            if mode == "professional" and data.resume_parse_result:
                initial_state["resume_parse_result"] = data.resume_parse_result

            self.update_session_status(session_id, "analyzing")
            stream_token = set_stream_sink(stream_sink)
            try:
                graph_started_at = time.perf_counter()
                with model_trace(
                    session_id=session_id,
                    operation_id=f"interview-start:{session_id}:attempt:{data.start_attempt}",
                    attempt=data.start_attempt,
                    enforce_budget=True,
                ):
                    async with asyncio.timeout(settings.interview_start_timeout_seconds):
                        result = await graph.ainvoke(
                            None if existing_values and existing_next else initial_state,
                            self._config(session_id),
                        )
                logger.info(
                    "Interview preparation graph ready: session=%s mode=%s elapsed=%.2fs total=%.2fs questions=%d",
                    session_id,
                    mode,
                    time.perf_counter() - graph_started_at,
                    time.perf_counter() - started_at,
                    len(result.get("question_plan", []) or []),
                )
                data.graph_started = True
                if result.get("resume_profile"):
                    try:
                        saved = await memory.asave_resume_profile(
                            data.user_id,
                            session_id,
                            result.get("resume_profile"),
                        )
                        await memory.aindex_memory_items(saved)
                    except Exception:
                        logger.exception("Failed to save resume memory for session=%s", session_id)
                self._sync_session_from_state(session_id, result)
                if not result.get("interview_complete"):
                    self.update_session_status(session_id, "interviewing")
                await self._cache_runtime_state(session_id, result)
                self._schedule_question_prefetch(session_id, result)
                return result
            except Exception as exc:
                data.error_message = str(exc)
                try:
                    failed_state = await self.get_graph_state(session_id)
                    if failed_state and failed_state.get("question_blueprint"):
                        data.graph_started = True
                        self._sync_session_from_state(session_id, failed_state)
                except Exception:
                    logger.exception("Unable to snapshot failed startup: %s", session_id)
                self.update_session_status(session_id, "failed")
                await self._persist_session(session_id)
                raise
            finally:
                reset_stream_sink(stream_token)

    async def submit_answer(
        self,
        session_id: str,
        answer: str,
        *,
        answer_id: str | None = None,
        stream_sink: StreamSink | None = None,
    ) -> dict:
        """Inject candidate answer and resume the graph."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")

        async with session_answer_lock(session_id):
            async with data.lock:
                if not data.graph_started:
                    raise RuntimeError("Interview graph has not started")

                mode = data.mode
                graph = self._get_graph(mode)
                config = self._config(session_id)

                self.update_session_status(session_id, "evaluating")
                stream_token = set_stream_sink(stream_sink)
                try:
                    stable_answer_id = answer_id or __import__("uuid").uuid4().hex
                    update = {"current_candidate_answer": answer}
                    if answer_id is not None:
                        update["current_answer_id"] = stable_answer_id
                    if hasattr(graph, "aupdate_state"):
                        await graph.aupdate_state(config, update)
                    else:
                        graph.update_state(config, update)
                    with model_trace(
                        session_id=session_id,
                        operation_id=f"interview-answer:{session_id}:{stable_answer_id}",
                    ):
                        result = await graph.ainvoke(None, config)
                    await self._persist_new_assessment_memories(session_id, result, answer)

                    if result.get("interview_complete"):
                        if mode == "practice":
                            practice_report = result.get("practice_report")
                            if practice_report:
                                self.save_practice_report(session_id, practice_report)
                        else:
                            pro_report = result.get("professional_report")
                            if pro_report:
                                self.save_professional_report(session_id, pro_report)
                            else:
                                report = result.get("final_report")
                                if report:
                                    self.save_report(session_id, report)
                        await self._persist_final_memory(session_id, result)

                    self._sync_session_from_state(session_id, result)
                    if not result.get("interview_complete"):
                        self.update_session_status(session_id, "interviewing")
                    await self._cache_runtime_state(session_id, result)
                    self._schedule_question_prefetch(session_id, result)
                    return result
                except Exception as exc:
                    data.error_message = str(exc)
                    try:
                        failed_state = await self.get_graph_state(session_id)
                        if failed_state:
                            self._sync_session_from_state(session_id, failed_state)
                            data.last_state = failed_state
                    except Exception:
                        logger.exception(
                            "Unable to snapshot failed answer processing: %s",
                            session_id,
                        )
                    self.update_session_status(session_id, "failed")
                    await self._persist_session(session_id)
                    raise
                finally:
                    reset_stream_sink(stream_token)

    async def stop_interview(self, session_id: str) -> dict:
        """Stop the interview early and generate a report with whatever has been answered.

        Works by directly calling the appropriate evaluator with the current state.
        """
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")

        if data.prefetch_task and not data.prefetch_task.done():
            data.prefetch_task.cancel()
            await asyncio.gather(data.prefetch_task, return_exceptions=True)
        await get_question_artifact_store().terminate_session(session_id)

        async with data.lock:
            if data.session.status == "completed":
                return data.last_state
            mode = data.mode
            graph = self._get_graph(mode)
            config = self._config(session_id)

            end_node = "evaluate_practice" if mode == "practice" else "evaluate_interview"

            snapshot = None
            try:
                if hasattr(graph, "aget_state"):
                    snapshot = await graph.aget_state(config)
                else:
                    snapshot = graph.get_state(config)
                current_state = snapshot.values if snapshot and snapshot.values else data.last_state
            except Exception:
                current_state = data.last_state

            current_state = dict(current_state or {})
            # Business snapshots are JSON; checkpoint snapshots may contain
            # model instances. Both paths must support the same end evaluator.
            for field, model in (
                ("question_plan", QuestionItem), ("round1_question_plan", QuestionItem),
                ("assessments", AnswerAssessment), ("round1_assessments", AnswerAssessment),
                ("conversation_history", ChatMessage),
            ):
                current_state[field] = [
                    raw if isinstance(raw, model) else model.model_validate(raw)
                    for raw in current_state.get(field, []) or []
                ]
            if isinstance(current_state.get("skill_matrix"), dict):
                current_state["skill_matrix"] = SkillMatrix.model_validate(current_state["skill_matrix"])
            if current_state.get("interview_complete"):
                self._sync_session_from_state(session_id, current_state)
                data.end_requested = False
                await self._cache_runtime_state(session_id, current_state)
                return current_state

            # A report failure/restart must not repeat an already durable final
            # assessment, even when only the business snapshot was recovered.
            finals = dict(data.last_state.get("final_question_assessments") or {})
            finals.update(current_state.get("final_question_assessments") or {})
            current_state["final_question_assessments"] = finals
            recovered_messages = self._accepted_messages_for_end(data, current_state)
            if recovered_messages:
                current_state["conversation_history"] = [
                    *(current_state.get("conversation_history") or []), *recovered_messages
                ]
            settlement = await finalize_answered_question_on_end(current_state)
            if settlement or recovered_messages or not (snapshot and snapshot.values):
                finals.update(settlement.get("final_question_assessments") or {})
                current_state["final_question_assessments"] = finals
                # Reducers merge stable IDs. Save the full snapshot so recovery
                # without a checkpoint does not lose the question or transcript.
                if hasattr(graph, "aupdate_state"):
                    await graph.aupdate_state(config, current_state, as_node=end_node)
                elif hasattr(graph, "update_state"):
                    graph.update_state(config, current_state, as_node=end_node)
                self._sync_session_from_state(session_id, current_state)
                # Commit before the report call: a report retry reuses this result.
                await self._persist_session(session_id)

            if not current_state.get("assessments") and not finals:
                has_r1 = bool(current_state.get("round1_assessments"))
                if not has_r1:
                    result = {
                        "interview_complete": True,
                        "practice_report": None,
                        "professional_report": None,
                        "final_report": None,
                    }
                    data.completion_reason = "no_assessment"
                    if hasattr(graph, "aupdate_state"):
                        await graph.aupdate_state(config, result, as_node=end_node)
                    elif hasattr(graph, "update_state"):
                        graph.update_state(config, result, as_node=end_node)
                    data.end_requested = False
                    self.update_session_status(session_id, "completed")
                    self._sync_session_from_state(session_id, result)
                    await self._cache_runtime_state(session_id, result)
                    return result

            logger.info(
                "Stopping interview early: %s (mode=%s, round=%d, %d assessments so far)",
                session_id,
                mode,
                current_state.get("current_round", 1),
                len(current_state.get("assessments", [])),
            )

            self.update_session_status(session_id, "evaluating")
            try:
                data.completion_reason = "early"
                if mode == "practice":
                    result = await evaluate_practice(current_state)
                    practice_report = result.get("practice_report")
                    if practice_report:
                        self.save_practice_report(session_id, practice_report)
                else:
                    result = await evaluate_interview(current_state)
                    pro_report = result.get("professional_report")
                    if pro_report:
                        self.save_professional_report(session_id, pro_report)
                    else:
                        report = result.get("final_report")
                        if report:
                            self.save_report(session_id, report)

                result["interview_complete"] = True
                merged_state = dict(current_state)
                merged_state.update(result)
                await self._persist_new_assessment_memories(session_id, merged_state, "")
                await self._persist_final_memory(session_id, merged_state)
                if hasattr(graph, "aupdate_state"):
                    await graph.aupdate_state(config, merged_state, as_node=end_node)
                elif hasattr(graph, "update_state"):
                    graph.update_state(config, merged_state, as_node=end_node)
                self._sync_session_from_state(session_id, merged_state)
                self.update_session_status(session_id, "completed")
                data.end_requested = False
                await self._cache_runtime_state(session_id, merged_state)
                return merged_state
            except Exception as exc:
                data.error_message = str(exc)
                self.update_session_status(session_id, "failed")
                raise

    @staticmethod
    def _accepted_messages_for_end(data: _SessionData, state: dict) -> list[ChatMessage]:
        """Recover only accepted answers whose turn belongs to the current question."""
        question = _current_question(state)
        if not question:
            return []
        question_id = _stable_question_id(state, question)
        history = [
            raw if isinstance(raw, ChatMessage) else ChatMessage.model_validate(raw)
            for raw in state.get("conversation_history", [])
        ]
        known_answers = {item.answer_id for item in history if item.answer_id}
        turns = {
            item.turn_id for item in history
            if item.question_id == question_id and item.turn_id
        }
        if state.get("current_turn_id"):
            turns.add(state["current_turn_id"])
        recovered = []
        for answer_id, record in data.pending_answers.items():
            turn_id = record.get("turn_id")
            if answer_id in known_answers or not str(record.get("content") or "").strip():
                continue
            if turn_id not in turns and answer_id != state.get("current_answer_id"):
                continue
            recovered.append(ChatMessage(
                message_id=f"{answer_id}:candidate", role="candidate",
                content=record["content"], question_id=question_id,
                turn_id=turn_id or state.get("current_turn_id"), answer_id=answer_id,
            ))
        return recovered

    async def get_graph_state(self, session_id: str) -> dict | None:
        mode = self.get_session_mode(session_id)
        graph = self._get_graph(mode)
        try:
            if hasattr(graph, "aget_state"):
                snapshot = await graph.aget_state(self._config(session_id))
            else:
                snapshot = graph.get_state(self._config(session_id))
            return snapshot.values if snapshot else None
        except Exception:
            return None

    # Long-term memory helpers

    def _build_memory_query(self, jd_text: str, resume_text: str = "") -> str:
        parts = [
            "Interview personalization query.",
            "Job description:",
            jd_text[:1200],
        ]
        if resume_text:
            parts.extend(["Candidate resume:", resume_text[:1000]])
        return "\n".join(parts)

    def _candidate_question(self, state: dict, question_id: int | str) -> QuestionItem | None:
        plans = [
            state.get("question_plan", []),
            state.get("round1_question_plan", []),
            state.get("round2_question_plan", []),
        ]
        for plan in plans:
            for raw_question in plan or []:
                question = (
                    raw_question
                    if isinstance(raw_question, QuestionItem)
                    else QuestionItem.model_validate(raw_question)
                )
                if question.question_id == str(question_id) or question.id == question_id:
                    return question
        return None

    async def _persist_new_assessment_memories(
        self,
        session_id: str,
        state: dict,
        latest_answer: str,
    ) -> None:
        data = self._sessions.get(session_id)
        if not data:
            return

        final_rows = state.get("final_question_assessments", {}) or {}
        memory = get_memory_service()
        for raw_final in final_rows.values():
            try:
                final = (
                    raw_final
                    if isinstance(raw_final, FinalQuestionAssessment)
                    else FinalQuestionAssessment.model_validate(raw_final)
                )
                question = self._candidate_question(state, final.question_id)
                transcript = []
                for raw_message in state.get("conversation_history", []) or []:
                    message = raw_message if isinstance(raw_message, ChatMessage) else ChatMessage.model_validate(raw_message)
                    if message.question_id == final.question_id:
                        transcript.append(message.model_dump(mode="json"))
                item = await memory.asave_final_observation(
                    user_id=data.user_id,
                    session_id=session_id,
                    final=final,
                    question=question,
                    transcript=transcript,
                )
                await memory.aindex_memory_item(item)
                data.persisted_assessment_count = len(final_rows)
            except Exception:
                logger.exception(
                    "Failed to persist final observation: session=%s index=%d",
                    session_id,
                    data.persisted_assessment_count,
                )

    async def _persist_final_memory(self, session_id: str, state: dict) -> None:
        data = self._sessions.get(session_id)
        if not data or data.final_memory_saved:
            return

        report = (
            state.get("practice_report")
            or state.get("professional_report")
            or state.get("final_report")
        )
        try:
            memory = get_memory_service()
            item = await memory.asave_session_reflection(
                user_id=data.user_id,
                session_id=session_id,
                mode=data.mode,
                report=report,
                assessments=state.get("assessments", []) or [],
            )
            if item:
                await memory.aindex_memory_item(item)
            data.final_memory_saved = True
        except Exception:
            logger.exception("Failed to persist final memory: session=%s", session_id)


# Module-level singleton
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


def reset_session_manager() -> None:
    """Reset the singleton manager so a newly initialized checkpointer is used."""
    global _manager
    _manager = None
