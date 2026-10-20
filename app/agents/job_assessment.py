"""Bounded ReAct-style job assessment agent and server-owned tool registry."""

from __future__ import annotations

import asyncio
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agents.evidence_pipeline import assess_evidence, retrieved_to_evidence
from app.config import settings
from app.llm.client import get_llm_client, model_trace
from app.memory.service import get_memory_service
from app.models.evidence import EvidenceChunk
from app.models.preparation import AgentDecision, AssessmentTarget, ClarificationRequest
from app.rag.query_builder import build_target_recovery_queries
from app.rag.retriever import get_retriever
from app.services.operation_store import OperationStore, get_operation_store, stable_hash
from app.services.preparation_service import PreparationService, get_preparation_service
from app.services.preparation_repository import save_preparation_case
from app.services.public_knowledge import (
    PublicKnowledgeClient,
    build_public_technical_query,
    is_retryable_public_error,
    sanitize_public_query,
    snapshots_to_evidence,
)
from app.services.knowledge_service import get_knowledge_service
from app.services.knowledge_repository import save_document
from app.models.knowledge import KnowledgeCandidateCreate


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ReadMaterialArguments(_ToolArguments):
    material: Literal["jd", "resume", "project_notes"]
    start: int = Field(default=0, ge=0)
    length: int = Field(default=1200, ge=1, le=2400)


class _ObservationArguments(_ToolArguments):
    skill_id: str = Field(min_length=1)


class _RetrievalArguments(_ToolArguments):
    query: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=12)
    query_mode: Literal["legacy", "target_recovery"] = "legacy"


class _PublicSearchArguments(_ToolArguments):
    query: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


class _ClarificationArguments(_ToolArguments):
    target_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    missing_information: str = Field(min_length=1)


class _FinishArguments(_ToolArguments):
    target_ids: list[str] = Field(default_factory=list)
    summary: str = ""


TOOL_ARGUMENT_MODELS: dict[str, type[_ToolArguments]] = {
    "read_material": _ReadMaterialArguments,
    "get_observations": _ObservationArguments,
    "retrieve_knowledge": _RetrievalArguments,
    "search_public_knowledge": _PublicSearchArguments,
    "request_clarification": _ClarificationArguments,
    "finish_assessment": _FinishArguments,
}


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    **properties,
                    "reason_summary": {
                        "type": "string",
                        "description": "简短说明本次选择依据，不输出内部思维过程",
                    },
                },
                "required": [*required, "reason_summary"],
                "additionalProperties": False,
            },
        },
    }


ASSESSMENT_TOOL_SCHEMAS = [
    _tool_schema(
        "read_material",
        "读取当前用户准备任务中指定材料的有限片段。",
        {
            "material": {"type": "string", "enum": ["jd", "resume", "project_notes"]},
            "start": {"type": "integer", "minimum": 0},
            "length": {"type": "integer", "minimum": 1, "maximum": 2400},
        },
        ["material"],
    ),
    _tool_schema(
        "get_observations",
        "读取当前用户指定能力的有效历史观察。",
        {"skill_id": {"type": "string"}},
        ["skill_id"],
    ),
    _tool_schema(
        "retrieve_knowledge",
        "检索正式本地题库和当前准备任务固定的知识发布版本。",
        {
            "query": {"type": "string"},
            "target_id": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 12},
        },
        ["query", "target_id"],
    ),
    _tool_schema(
        "search_public_knowledge",
        "本地证据仍不足时进行受预算约束的公开技术资料补查。",
        {"query": {"type": "string"}, "target_id": {"type": "string"}},
        ["query", "target_id"],
    ),
    _tool_schema(
        "request_clarification",
        "个人经历材料不足时向用户提出一个简短澄清问题。",
        {
            "target_id": {"type": "string"},
            "question": {"type": "string"},
            "missing_information": {"type": "string"},
        },
        ["target_id", "question", "missing_information"],
    ),
    _tool_schema(
        "finish_assessment",
        "材料充分、没有信息增益或预算应停止时完成本轮评估。",
        {
            "target_ids": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        ["target_ids", "summary"],
    ),
]


@dataclass
class ToolContext:
    operation_id: str
    user_id: str
    case_id: str
    material_revision: int
    targets: list[AssessmentTarget]


class AssessmentTools:
    def __init__(
        self,
        context: ToolContext,
        preparation: PreparationService | None = None,
        public: PublicKnowledgeClient | None = None,
        operations: OperationStore | None = None,
    ) -> None:
        self.context = context
        self.preparation = preparation or get_preparation_service()
        self.public = public or PublicKnowledgeClient()
        self.operations = operations or get_operation_store()

    async def _public_request(self, budget_key: str, request):
        attempts = max(1, settings.public_transport_retries + 1)
        for attempt in range(attempts):
            await self.operations.consume_budget(self.context.operation_id, budget_key)
            try:
                return await request()
            except Exception as exc:
                if attempt + 1 >= attempts or not is_retryable_public_error(exc):
                    raise
                await asyncio.sleep(0.25 * (2**attempt))
        raise RuntimeError("unreachable public request retry state")

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, ToolHandler] = {
            "read_material": self.read_material,
            "get_observations": self.get_observations,
            "retrieve_knowledge": self.retrieve_knowledge,
            "search_public_knowledge": self.search_public_knowledge,
            "request_clarification": self.request_clarification,
            "finish_assessment": self.finish_assessment,
        }
        if name not in handlers:
            raise ValueError(f"unsupported tool: {name}")
        validated = TOOL_ARGUMENT_MODELS[name].model_validate(arguments)
        return await handlers[name](validated.model_dump())

    async def _case(self):
        operation = await self.operations.get(
            self.context.operation_id, self.context.user_id
        )
        if operation and operation.status == "cancelled":
            raise asyncio.CancelledError()
        case = await self.preparation.get(self.context.case_id, self.context.user_id)
        if not case:
            raise PermissionError("preparation case not found or not owned by user")
        if case.material_revision != self.context.material_revision:
            raise RuntimeError("material revision changed")
        return case

    async def read_material(self, args: dict[str, Any]) -> dict[str, Any]:
        case = await self._case()
        material = str(args.get("material") or "jd")
        value = {
            "jd": case.jd_text,
            "resume": case.resume_text,
            "project_notes": case.project_notes,
        }.get(material)
        if value is None:
            raise ValueError("material must be jd, resume, or project_notes")
        start = max(0, int(args.get("start") or 0))
        length = min(max(int(args.get("length") or 1200), 1), 2400)
        return {"material": material, "start": start, "text": value[start : start + length]}

    async def get_observations(self, args: dict[str, Any]) -> dict[str, Any]:
        memory = get_memory_service()
        context = await memory.abuild_context(
            self.context.user_id,
            tags=[str(args.get("skill_id") or "")],
        )
        return {
            "observations": [
                item.model_dump(mode="json") for item in context.relevant_episodes[:5]
            ],
            "skill_memories": [
                item.model_dump(mode="json") for item in context.weak_skills[:5]
            ],
        }

    async def seed_material_evidence(self, target: AssessmentTarget) -> None:
        if target.gap_kind != "personal_experience":
            return
        case = await self._case()
        existing = list(case.evidence_chunks.get(target.target_id, []))
        known = {item.chunk_id for item in existing}
        for index, citation in enumerate(target.citations):
            if citation.material_type not in {"resume", "project_note", "observation"}:
                continue
            chunk_id = f"material:{target.target_id}:{index}"
            if chunk_id in known or not citation.quote.strip():
                continue
            existing.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=case.case_id,
                    document_version=str(case.material_revision),
                    text=citation.quote,
                    location=citation.material_type,
                    source_type="private_material",
                    record_id=citation.source_id,
                    publication_status="private",
                    scope="private",
                    retrieval_sources=["prepared_material"],
                )
            )
        clarification_text = "\n".join(
            item.answer or ""
            for item in case.clarifications
            if item.status == "answered"
            and item.answer
            and item.clarification_id.endswith(target.target_id)
        ).strip()
        if clarification_text:
            chunk_id = f"material:{target.target_id}:clarification"
            if chunk_id not in known:
                existing.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
                        document_id=case.case_id,
                        document_version=str(case.material_revision),
                        text=f"{target.scope} {clarification_text}",
                        location="clarification",
                        source_type="private_material",
                        publication_status="private",
                        scope="private",
                        retrieval_sources=["user_clarification"],
                    )
                )
        case.evidence_chunks[target.target_id] = existing
        await save_preparation_case(case)

    async def retrieve_knowledge(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        top_k = min(int(args.get("top_k") or 8), 12)
        target_id = str(args.get("target_id") or "")
        target = next(
            (item for item in self.context.targets if item.target_id == target_id), None
        )
        if not target:
            raise ValueError("target_id is not part of the current preparation case")
        query_mode = str(args.get("query_mode") or "legacy")
        retrieval_queries = (
            build_target_recovery_queries(target, query)
            if query_mode == "target_recovery"
            else [query]
        )
        effective_queries = [
            item.text if hasattr(item, "text") else str(item)
            for item in retrieval_queries
        ]
        retrieval_id = (
            f"{self.context.operation_id}:local-retrieval:"
            f"{hashlib.sha256(json.dumps({'target_id': target_id, 'queries': effective_queries, 'top_k': top_k, 'material_revision': self.context.material_revision}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]}"
        )
        prior_retrievals = await self.operations.tool_calls(
            self.context.operation_id, "local_knowledge_retrieval"
        )
        target_retrieval_ids = {
            item.tool_call_id
            for item in prior_retrievals
            if item.input_summary.get("target_id") == target_id
        }
        if retrieval_id not in target_retrieval_ids and len(target_retrieval_ids) >= 2:
            raise RuntimeError("local query rewrite budget exhausted for target")
        record = await self.operations.accept_tool_call(
            operation_id=self.context.operation_id,
            tool_name="local_knowledge_retrieval",
            payload={
                "query": query,
                "queries": effective_queries,
                "target_id": target_id,
                "top_k": top_k,
                "material_revision": self.context.material_revision,
            },
            tool_call_id=retrieval_id,
        )
        if record.status == "succeeded" and record.result is not None:
            chunks = [
                EvidenceChunk.model_validate(item)
                for item in record.result.get("chunks", [])
            ]
        else:
            retriever = get_retriever()
            if hasattr(retriever, "retrieve_multi_outcome"):
                outcome = await retriever.retrieve_multi_outcome(
                    retrieval_queries,
                    top_k_per_query=top_k,
                    final_top_k=(top_k * 2 if query_mode == "target_recovery" else top_k),
                    fingerprint_context={
                        "purpose": (
                            "agent_target_recovery"
                            if query_mode == "target_recovery"
                            else "deterministic_initial_retrieval"
                        ),
                        "target_id": target.target_id,
                        "target_version": target.version,
                        "material_revision": self.context.material_revision,
                    },
                )
                results = outcome.items
            else:
                # Compatibility for narrow test doubles and older adapters.
                results = await retriever.retrieve(query, top_k=top_k)
            chunks = [retrieved_to_evidence(item) for item in results]
            knowledge = get_knowledge_service()
            case = await self._case()
            await knowledge.hydrate()
            if case.knowledge_release_id is None:
                active_release = knowledge.active_release()
                if active_release:
                    case.knowledge_release_id = active_release.release_id
                    await save_preparation_case(case)
            released = (
                await knowledge.search_release(
                    case.knowledge_release_id, query, top_k=top_k
                )
                if case.knowledge_release_id
                else []
            )
            known_chunk_ids = {item.chunk_id for item in chunks}
            chunks.extend(item for item in released if item.chunk_id not in known_chunk_ids)
            await self.operations.finish_tool_call(
                retrieval_id,
                result={"chunks": [item.model_dump(mode="json") for item in chunks]},
            )
        case = await self._case()
        existing = list(case.evidence_chunks.get(target.target_id, []))
        known_chunk_ids = {item.chunk_id for item in existing}
        combined = [
            *existing,
            *[item for item in chunks if item.chunk_id not in known_chunk_ids],
        ]
        assessment = await self._assess_evidence(target, combined)
        if assessment:
            case.evidence_chunks[target.target_id] = combined
            case.evidence_assessments[target.target_id] = assessment
            case.targets = [
                item.model_copy(
                    update={
                        "status": (
                            "ready" if assessment.verdict == "sufficient"
                            else "partial" if assessment.verdict == "partial"
                            else "missing_evidence"
                        )
                    }
                ) if item.target_id == target.target_id else item
                for item in case.targets
            ]
            await save_preparation_case(case)
        return {
            "query": query,
            "queries": effective_queries,
            "query_mode": query_mode,
            "chunks": [item.model_dump(mode="json") for item in combined],
            "evidence_assessment": assessment.model_dump(mode="json") if assessment else None,
        }

    async def _assess_evidence(
        self,
        target: AssessmentTarget,
        chunks: list[EvidenceChunk],
    ):
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "target_id": target.target_id,
                    "target_version": target.version,
                    "chunk_ids": sorted(item.chunk_id for item in chunks),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        assessment_id = (
            f"{self.context.operation_id}:evidence-assessment:{target.target_id}:{fingerprint}"
        )
        record = await self.operations.accept_tool_call(
            operation_id=self.context.operation_id,
            tool_name="assess_evidence",
            payload={
                "target_id": target.target_id,
                "target_version": target.version,
                "chunk_ids": sorted(item.chunk_id for item in chunks),
            },
            tool_call_id=assessment_id,
        )
        if record.status == "succeeded" and record.result is not None:
            from app.models.evidence import EvidenceAssessment

            return EvidenceAssessment.model_validate(record.result["assessment"])
        operation = await self.operations.get(
            self.context.operation_id, self.context.user_id
        )
        semantic = bool(
            settings.assessment_semantic_enabled
            and operation
            and operation.budget.get("semantic_checks", 0) > 0
        )
        if semantic:
            await self.operations.consume_budget(
                self.context.operation_id, "semantic_checks"
            )
        assessment = await assess_evidence(target, chunks, semantic=semantic)
        await self.operations.finish_tool_call(
            assessment_id,
            result={
                "assessment": assessment.model_dump(mode="json"),
                "validation_mode": "semantic" if semantic else "deterministic",
            },
        )
        return assessment

    async def search_public_knowledge(self, args: dict[str, Any]) -> dict[str, Any]:
        target_id = str(args.get("target_id") or "")
        target = next(
            (item for item in self.context.targets if item.target_id == target_id), None
        )
        if not target:
            raise ValueError("target_id is not part of the current preparation case")
        case = await self._case()
        assessment = case.evidence_assessments.get(target_id)
        missing_aspects = list(assessment.missing_aspects) if assessment else []
        query = build_public_technical_query(target, missing_aspects)
        search_id = (
            f"{self.context.operation_id}:search:"
            f"{hashlib.sha256(f'{target_id}:{query}'.encode('utf-8')).hexdigest()[:16]}"
        )
        prior_searches = await self.operations.tool_calls(
            self.context.operation_id, "public_search_request"
        )
        target_search_ids = {
            item.tool_call_id
            for item in prior_searches
            if item.input_summary.get("target_id") == target_id
        }
        if search_id not in target_search_ids and len(target_search_ids) >= 2:
            raise RuntimeError("public search budget exhausted for target")
        search_record = await self.operations.accept_tool_call(
            operation_id=self.context.operation_id,
            tool_name="public_search_request",
            payload={
                "query": query,
                "target_id": target_id,
                "max_results": settings.public_search_max_results,
                "query_source": "target_bound",
            },
            tool_call_id=search_id,
        )
        if search_record.status == "succeeded" and search_record.result is not None:
            results = list(search_record.result.get("results") or [])
        else:
            results = await self._public_request(
                "searches",
                lambda: self.public.search(
                    query, max_results=settings.public_search_max_results
                ),
            )
            await self.operations.finish_tool_call(search_id, result={"results": results})
        urls = [str(item.get("url") or "") for item in results if item.get("url")]
        snapshots = []
        for url in urls[: settings.public_extract_max_urls]:
            extract_id = (
                f"{self.context.operation_id}:extract:"
                f"{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}"
            )
            extract_record = await self.operations.accept_tool_call(
                operation_id=self.context.operation_id,
                tool_name="public_extract_request",
                payload={"url": url},
                tool_call_id=extract_id,
            )
            if extract_record.status == "succeeded" and extract_record.result is not None:
                from app.models.evidence import SearchSnapshot

                snapshots.extend(
                    SearchSnapshot.model_validate(item)
                    for item in extract_record.result.get("snapshots", [])
                )
                continue
            extracted = await self._public_request(
                "extracts", lambda url=url: self.public.extract([url])
            )
            await self.operations.finish_tool_call(
                extract_id,
                result={
                    "snapshots": [item.model_dump(mode="json") for item in extracted]
                },
            )
            snapshots.extend(extracted)
        chunks = snapshots_to_evidence(snapshots)
        knowledge = get_knowledge_service()
        await knowledge.hydrate()
        for snapshot in snapshots:
            candidate = knowledge.add_candidate(
                KnowledgeCandidateCreate(
                    source_type="public_web",
                    source_url=snapshot.url,
                    title=snapshot.title,
                    content=snapshot.content,
                    metadata={
                        "query": query,
                        "fetched_at": snapshot.fetched_at,
                        "content_hash": snapshot.content_hash,
                    },
                )
            )
            await save_document(candidate)
        # Public results remain review candidates. They are intentionally not
        # merged into formal evidence or assessed until publication approval.
        return {
            "query": query,
            "search_results": results,
            "chunks": [item.model_dump(mode="json") for item in chunks],
            "evidence_assessment": None,
            "review_status": "pending_review",
        }

    async def request_clarification(self, args: dict[str, Any]) -> dict[str, Any]:
        case = await self._case()
        target_id = str(args.get("target_id") or "")
        target = next(
            (item for item in self.context.targets if item.target_id == target_id), None
        )
        if target is None:
            raise ValueError("target_id is not part of the current preparation case")
        if target.gap_kind != "personal_experience":
            raise ValueError("clarification is only allowed for personal experience gaps")
        question = str(args.get("question") or "").strip()
        missing = str(args.get("missing_information") or question).strip()
        if not question:
            raise ValueError("question is required")
        clarification = ClarificationRequest(
            clarification_id=(
                f"{self.context.operation_id}:clarification:{target_id}"
            ),
            case_id=case.case_id,
            material_revision=case.material_revision,
            question=question,
            missing_information=missing,
        )
        if not any(item.clarification_id == clarification.clarification_id for item in case.clarifications):
            case.clarifications.append(clarification)
        case.status = "waiting_input"
        await save_preparation_case(case)
        return {"target_id": target_id, "clarification": clarification.model_dump(mode="json")}

    async def finish_assessment(self, args: dict[str, Any]) -> dict[str, Any]:
        target_ids = {item.target_id for item in self.context.targets}
        cited = [str(item) for item in args.get("target_ids", []) if str(item) in target_ids]
        if not cited:
            raise ValueError("finish_assessment requires explicit target_ids")
        case = await self._case()
        complete: list[str] = []
        unresolved: list[str] = []
        for target_id in cited:
            assessment = case.evidence_assessments.get(target_id)
            if assessment and assessment.verdict == "sufficient":
                complete.append(target_id)
            else:
                unresolved.append(target_id)
        return {
            "target_ids": cited,
            "completed_target_ids": complete,
            "unresolved_target_ids": unresolved,
            "complete": not unresolved,
            "summary": str(args.get("summary") or ""),
        }


class JobAssessmentAgent:
    def __init__(self, operations: OperationStore | None = None) -> None:
        self.operations = operations or get_operation_store()

    @staticmethod
    def _decision_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep decision-relevant facts without sending retrieved documents back to the LLM."""

        compact: list[dict[str, Any]] = []
        for item in trajectory[-6:]:
            row = {
                "target_id": item.get("target_id"),
                "tool": item.get("tool"),
                "error": item.get("error"),
                "reused": item.get("reused"),
            }
            result = item.get("result")
            if isinstance(result, dict):
                summary: dict[str, Any] = {}
                if result.get("query"):
                    summary["query"] = str(result["query"])[:300]
                if isinstance(result.get("evidence_assessment"), dict):
                    assessment = result["evidence_assessment"]
                    summary["evidence_assessment"] = {
                        key: assessment.get(key)
                        for key in (
                            "target_id",
                            "verdict",
                            "supported_aspects",
                            "missing_aspects",
                            "conflicts",
                            "suggested_query",
                        )
                    }
                if isinstance(result.get("chunks"), list):
                    summary["chunk_count"] = len(result["chunks"])
                    summary["chunk_ids"] = [
                        str(chunk.get("chunk_id") or "")
                        for chunk in result["chunks"][:8]
                        if isinstance(chunk, dict)
                    ]
                if result.get("material"):
                    summary.update(
                        {
                            "material": result.get("material"),
                            "start": result.get("start"),
                            "text": str(result.get("text") or "")[:800],
                        }
                    )
                if isinstance(result.get("observations"), list):
                    summary["observation_count"] = len(result["observations"])
                if isinstance(result.get("skill_memories"), list):
                    summary["skill_memory_count"] = len(result["skill_memories"])
                for key in (
                    "review_status",
                    "complete",
                    "completed_target_ids",
                    "unresolved_target_ids",
                    "clarification",
                ):
                    if key in result:
                        summary[key] = result[key]
                row["result"] = summary
            compact.append(row)
        return compact

    @staticmethod
    def _guardrail_decision(
        context: ToolContext,
        trajectory: list[dict[str, Any]],
    ) -> AgentDecision | None:
        """Apply deterministic stop/HITL invariants before ambiguous model choice."""
        if not trajectory:
            return None
        latest = trajectory[-1]
        latest_result = latest.get("result") if isinstance(latest.get("result"), dict) else {}
        assessment = latest_result.get("evidence_assessment")
        verdict = str(
            assessment.get("verdict")
            if isinstance(assessment, dict)
            else latest_result.get("verdict") or ""
        )
        target_ids = [item.target_id for item in context.targets]
        if verdict == "sufficient":
            return AgentDecision(
                tool="finish_assessment",
                arguments={"target_ids": target_ids, "summary": "本地证据已经充分。"},
                reason_summary="本地证据充分，结束评估。",
            )
        # Public correction is not a safety invariant. When the autonomous
        # loop is enabled, the model must explicitly choose it and the
        # target/gap guardrails below validate that choice.
        if len(trajectory) >= 2:
            prior = trajectory[-2]
            same_failed_tool = (
                latest.get("tool") == prior.get("tool")
                and bool(latest.get("error"))
                and bool(prior.get("error"))
            )
            if same_failed_tool:
                return AgentDecision(
                    tool="finish_assessment",
                    arguments={"target_ids": target_ids, "summary": "重复工具失败且没有新增信息。"},
                    reason_summary="连续失败没有信息增益，按停止规则结束。",
                )
        if latest.get("tool") == "read_material":
            text = json.dumps(latest_result, ensure_ascii=False).casefold()
            missing_markers = ("缺少", "缺失", "为空", "未说明", "missing", "empty")
            has_personal_claim = any(
                "project_claim" in item.sources or "resume_claim" in item.sources
                for item in context.targets
            )
            if has_personal_claim and any(marker in text for marker in missing_markers):
                return AgentDecision(
                    tool="request_clarification",
                    arguments={
                        "question": "请补充这段个人经历的具体实现、你的职责和关键取舍。",
                        "missing_information": "个人经历的实现细节、职责与取舍",
                        "target_id": next(
                            (
                                item.target_id
                                for item in context.targets
                                if item.gap_kind == "personal_experience"
                                or "project_claim" in item.sources
                            ),
                            context.targets[0].target_id,
                        ),
                    },
                    reason_summary="个人经历材料缺少可验证细节，需要用户澄清。",
                )
        return None

    async def _decide(
        self,
        context: ToolContext,
        trajectory: list[dict[str, Any]],
    ) -> AgentDecision:
        guarded = self._guardrail_decision(context, trajectory)
        if guarded is not None:
            return guarded
        llm = get_llm_client()
        target = context.targets[0]
        used_tools = {str(item.get("tool") or "") for item in trajectory}
        if target.gap_kind == "personal_experience":
            allowed_tools = {"request_clarification", "finish_assessment"}
            if "read_material" not in used_tools:
                allowed_tools.add("read_material")
            if (
                "historical_observation" in target.sources
                and "get_observations" not in used_tools
            ):
                allowed_tools.add("get_observations")
        else:
            allowed_tools = {"retrieve_knowledge", "finish_assessment"}
            if settings.public_search_enabled:
                allowed_tools.add("search_public_knowledge")
        tool_schemas = [
            item
            for item in ASSESSMENT_TOOL_SCHEMAS
            if item["function"]["name"] in allowed_tools
        ]
        call = await llm.chat_tool_call(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是岗位评估 Agent。每次只选择一个已注册工具。个人经历不清楚时澄清；"
                        "技术事实不清楚时先本地检索；本地不足且配置允许时才公开补查；"
                        "材料足够就 finish_assessment；没有信息增益或达到预算时合理停止。"
                        "若 trajectory 为空且目标是技术事实，先 retrieve_knowledge，不要重复读取已归纳的材料。"
                        "只能从本次提供的工具中选择；已执行工具的结果已在 trajectory 中，"
                        "必须利用结果决定下一步，禁止用公网资料推断个人经历。"
                        "reason_summary 只记录简短决策依据，不输出内部思维过程。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "case_id": context.case_id,
                            "material_revision": context.material_revision,
                            "targets": [item.model_dump(mode="json") for item in context.targets],
                            "trajectory": self._decision_trajectory(trajectory),
                            "allowed_tools": sorted(allowed_tools),
                            "public_search_enabled": settings.public_search_enabled,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            tools=tool_schemas,
            temperature=0.1,
        )
        arguments = dict(call["arguments"])
        reason = str(arguments.pop("reason_summary", "")).strip()
        if not reason:
            raise ValueError("native tool call is missing reason_summary")
        return AgentDecision(
            tool=call["name"],
            arguments=arguments,
            reason_summary=reason,
        )

    async def _execute_decision(
        self,
        context: ToolContext,
        tools: AssessmentTools,
        decision: AgentDecision,
    ) -> tuple[dict[str, Any], bool]:
        """Execute one idempotent bounded decision and report whether it was reused."""

        signature = stable_hash(
            {"tool": decision.tool, "arguments": decision.arguments}
        )
        target_id = str(decision.arguments.get("target_id") or "global")
        tool_call_id = (
            f"{context.operation_id}:decision:{target_id}:{decision.tool}:{signature[:16]}"
        )
        record = await self.operations.accept_tool_call(
            operation_id=context.operation_id,
            tool_name=decision.tool,
            payload=decision.arguments,
            tool_call_id=tool_call_id,
        )
        if record.status == "succeeded" and record.result is not None:
            # Re-enter the idempotent tool so persisted results can rebuild any
            # case projection that was not committed before an interruption.
            return await tools.execute(decision.tool, decision.arguments), True
        await self.operations.consume_budget(context.operation_id, "decisions")
        try:
            result = await tools.execute(decision.tool, decision.arguments)
        except Exception as exc:
            await self.operations.finish_tool_call(
                tool_call_id, error=f"{type(exc).__name__}: {exc}"
            )
            raise
        await self.operations.finish_tool_call(tool_call_id, result=result)
        return result, False

    @staticmethod
    def _validate_autonomous_decision(
        decision: AgentDecision,
        target: AssessmentTarget,
    ) -> None:
        """Enforce target ownership and gap-specific tool permissions."""

        target_id = str(decision.arguments.get("target_id") or "")
        target_tools = {
            "retrieve_knowledge",
            "search_public_knowledge",
            "request_clarification",
        }
        if decision.tool in target_tools and target_id != target.target_id:
            raise ValueError("autonomous decision target_id does not match current target")
        if (
            target.gap_kind == "personal_experience"
            and decision.tool == "search_public_knowledge"
        ):
            raise ValueError("public search cannot fill personal experience gaps")
        if (
            target.gap_kind == "technical_knowledge"
            and decision.tool == "request_clarification"
        ):
            raise ValueError("technical knowledge gaps cannot request personal clarification")
        if decision.tool == "finish_assessment":
            target_ids = [str(item) for item in decision.arguments.get("target_ids", [])]
            if target_ids != [target.target_id]:
                raise ValueError("finish_assessment must address exactly the current target")
        if decision.tool == "search_public_knowledge" and not settings.public_search_enabled:
            raise ValueError("public search is disabled")

    async def _run_autonomous_gap(
        self,
        context: ToolContext,
        tools: AssessmentTools,
        target: AssessmentTarget,
        trajectory: list[dict[str, Any]],
    ) -> tuple[Any, str | None]:
        """Let the LLM choose only bounded corrective actions for one unresolved target."""

        target_context = ToolContext(
            operation_id=context.operation_id,
            user_id=context.user_id,
            case_id=context.case_id,
            material_revision=context.material_revision,
            targets=[target],
        )
        target_trajectory = [
            item for item in trajectory if item.get("target_id") == target.target_id
        ]
        for _step in range(settings.assessment_max_decisions):
            case = await tools._case()
            assessment = case.evidence_assessments.get(target.target_id)
            if assessment and assessment.verdict == "sufficient":
                return assessment, None
            try:
                decision, decision_reused = await self._persisted_decision(
                    target_context,
                    target_trajectory,
                )
                self._validate_autonomous_decision(decision, target)
                if decision.tool == "retrieve_knowledge":
                    decision.arguments["query_mode"] = "target_recovery"
            except RuntimeError as exc:
                if "budget exhausted" in str(exc):
                    return assessment, "budget_exhausted"
                raise
            except ValueError as exc:
                target_trajectory.append(
                    {
                        "target_id": target.target_id,
                        "tool": "invalid_decision",
                        "error": str(exc),
                    }
                )
                trajectory.append(target_trajectory[-1])
                continue

            try:
                result, reused = await self._execute_decision(
                    context, tools, decision
                )
                row = {
                    "target_id": target.target_id,
                    "tool": decision.tool,
                    "reason": decision.reason_summary,
                    "decision_source": "llm_tool_call",
                    "decision_reused": decision_reused,
                    "reused": reused,
                    "result": result,
                }
            except RuntimeError as exc:
                row = {
                    "target_id": target.target_id,
                    "tool": decision.tool,
                    "reason": decision.reason_summary,
                    "decision_source": "llm_tool_call",
                    "decision_reused": decision_reused,
                    "reused": False,
                    "error": str(exc),
                }
                trajectory.append(row)
                target_trajectory.append(row)
                if "budget exhausted" in str(exc):
                    return assessment, "budget_exhausted"
                continue
            trajectory.append(row)
            target_trajectory.append(row)

            if decision.tool == "request_clarification":
                case = await tools._case()
                return case.evidence_assessments.get(target.target_id), "user_clarification_required"
            if decision.tool == "finish_assessment":
                case = await tools._case()
                assessment = case.evidence_assessments.get(target.target_id)
                return assessment, (
                    None
                    if assessment and assessment.verdict == "sufficient"
                    else "agent_stopped_unresolved"
                )

        case = await tools._case()
        return case.evidence_assessments.get(target.target_id), "decision_limit_exhausted"

    async def _persisted_decision(
        self,
        context: ToolContext,
        trajectory: list[dict[str, Any]],
    ) -> tuple[AgentDecision, bool]:
        """Persist successful model decisions so recovery does not call the LLM again."""

        target_id = context.targets[0].target_id
        trajectory_fingerprint = stable_hash(
            [
                {
                    "target_id": item.get("target_id"),
                    "tool": item.get("tool"),
                    "error": item.get("error"),
                    "result_hash": stable_hash(item.get("result"))
                    if item.get("result") is not None
                    else None,
                }
                for item in trajectory
            ]
        )
        decision_id = (
            f"{context.operation_id}:agent-decision:{target_id}:"
            f"{trajectory_fingerprint[:16]}"
        )
        record = await self.operations.accept_tool_call(
            operation_id=context.operation_id,
            tool_name="agent_decision",
            payload={
                "target_id": target_id,
                "trajectory_fingerprint": trajectory_fingerprint,
            },
            tool_call_id=decision_id,
        )
        if record.status == "succeeded" and record.result is not None:
            return AgentDecision.model_validate(record.result["decision"]), True
        try:
            with model_trace(
                operation_id=context.operation_id,
                evaluation="bounded-agentic-rag",
                target_id=target_id,
                enforce_budget=True,
            ):
                decision = await self._decide(context, trajectory)
        except Exception as exc:
            await self.operations.finish_tool_call(
                decision_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        await self.operations.finish_tool_call(
            decision_id,
            result={"decision": decision.model_dump(mode="json")},
        )
        return decision, False

    @staticmethod
    def _target_result(target: AssessmentTarget, assessment) -> dict[str, Any]:
        verdict = assessment.verdict if assessment else "not_performed"
        return {
            "target_id": target.target_id,
            "gap_kind": target.gap_kind,
            "resume_signal": target.resume_signal,
            "verdict": verdict,
            "complete": verdict == "sufficient",
        }

    async def run(self, context: ToolContext) -> dict[str, Any]:
        """Run deterministic retrieval first, then bounded gap-specific actions."""

        operation = await self.operations.get(context.operation_id, context.user_id)
        if not operation:
            raise KeyError(context.operation_id)
        tools = AssessmentTools(context, operations=self.operations)
        trajectory: list[dict[str, Any]] = []
        target_results: list[dict[str, Any]] = []
        for target in context.targets:
            current = await self.operations.get(context.operation_id, context.user_id)
            if current and current.status == "cancelled":
                return {
                    "status": "cancelled",
                    "trajectory": trajectory,
                    "targets": target_results,
                }

            await tools.seed_material_evidence(target)
            case = await tools._case()
            assessment = case.evidence_assessments.get(target.target_id)
            if not assessment or assessment.verdict != "sufficient":
                query = f"{target.scope} {target.verification_goal}"
                decision = AgentDecision(
                    tool="retrieve_knowledge",
                    arguments={
                        "query": query,
                        "target_id": target.target_id,
                        "top_k": 8,
                        "query_mode": "legacy",
                    },
                    reason_summary="先执行确定性本地混合检索。",
                )
                try:
                    result, reused = await self._execute_decision(
                        context, tools, decision
                    )
                    trajectory.append(
                        {
                            "target_id": target.target_id,
                            "tool": decision.tool,
                            "reason": decision.reason_summary,
                            "reused": reused,
                            "result": result,
                        }
                    )
                except RuntimeError as exc:
                    if "budget exhausted" in str(exc):
                        target_results.append(
                            {
                                **self._target_result(target, assessment),
                                "stop_reason": "budget_exhausted",
                            }
                        )
                        continue
                    raise
                case = await tools._case()
                assessment = case.evidence_assessments.get(target.target_id)

            if assessment and assessment.verdict == "sufficient":
                target_results.append(self._target_result(target, assessment))
                continue

            if settings.assessment_autonomous_loop_enabled:
                assessment, stop_reason = await self._run_autonomous_gap(
                    context, tools, target, trajectory
                )
                if stop_reason == "user_clarification_required":
                    target_results.append(
                        {
                            **self._target_result(target, assessment),
                            "stop_reason": stop_reason,
                        }
                    )
                    return {
                        "status": "waiting_input",
                        "trajectory": trajectory,
                        "targets": target_results,
                        "complete": False,
                    }
                target_results.append(
                    {
                        **self._target_result(target, assessment),
                        "stop_reason": stop_reason,
                    }
                )
                continue

            if target.gap_kind == "personal_experience":
                pending = next(
                    (
                        item
                        for item in case.clarifications
                        if item.status == "pending"
                        and item.clarification_id.endswith(target.target_id)
                    ),
                    None,
                )
                if pending is None:
                    decision = AgentDecision(
                        tool="request_clarification",
                        arguments={
                            "target_id": target.target_id,
                            "question": (
                                f"请补充你在 {target.scope} 相关项目中的具体实现、"
                                "个人职责和关键取舍。"
                            ),
                            "missing_information": "个人实现、职责与工程取舍",
                        },
                        reason_summary="个人经历缺口只能由用户澄清。",
                    )
                    result, reused = await self._execute_decision(
                        context, tools, decision
                    )
                    trajectory.append(
                        {
                            "target_id": target.target_id,
                            "tool": decision.tool,
                            "reason": decision.reason_summary,
                            "reused": reused,
                            "result": result,
                        }
                    )
                target_results.append(
                    {
                        **self._target_result(target, assessment),
                        "stop_reason": "user_clarification_required",
                    }
                )
                return {
                    "status": "waiting_input",
                    "trajectory": trajectory,
                    "targets": target_results,
                    "complete": False,
                }

            if assessment and assessment.suggested_query:
                decision = AgentDecision(
                    tool="retrieve_knowledge",
                    arguments={
                        "query": assessment.suggested_query,
                        "target_id": target.target_id,
                        "top_k": 8,
                        "query_mode": "target_recovery",
                    },
                    reason_summary="技术证据不足，执行一次有界本地查询改写。",
                )
                try:
                    result, reused = await self._execute_decision(
                        context, tools, decision
                    )
                    trajectory.append(
                        {
                            "target_id": target.target_id,
                            "tool": decision.tool,
                            "reason": decision.reason_summary,
                            "reused": reused,
                            "result": result,
                        }
                    )
                except RuntimeError as exc:
                    if "budget exhausted" not in str(exc):
                        raise
                case = await tools._case()
                assessment = case.evidence_assessments.get(target.target_id)

            target_results.append(
                {
                    **self._target_result(target, assessment),
                    "stop_reason": (
                        None
                        if assessment and assessment.verdict == "sufficient"
                        else "local_evidence_insufficient"
                    ),
                }
            )

        complete = bool(target_results) and all(
            item["complete"] for item in target_results
        )
        return {
            "status": "succeeded",
            "trajectory": trajectory,
            "targets": target_results,
            "complete": complete,
            "result": {
                "completed_target_ids": [
                    item["target_id"] for item in target_results if item["complete"]
                ],
                "unresolved_target_ids": [
                    item["target_id"] for item in target_results if not item["complete"]
                ],
            },
        }
