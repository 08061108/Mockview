"""Evidence sufficiency, bounded correction, and question validation."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable

from app.llm.client import get_llm_client
from app.models.evidence import (
    EvidenceAssessment,
    EvidenceChunk,
    QuestionValidation,
    QuestionValidationBatch,
)
from app.models.preparation import AssessmentTarget
from app.models.question import QuestionItem
from app.rag.postprocess import metadata_list, result_parent_id


def retrieved_to_evidence(result) -> EvidenceChunk:
    metadata = getattr(result, "metadata", {}) or {}
    parent_id = result_parent_id(result)
    chunk_id = str(getattr(result, "id", "") or parent_id)
    parent_title = str(metadata.get("content") or "")
    source_context = "\n".join(
        [parent_title, "; ".join(metadata_list(metadata, "skill_tags"))]
    ).strip()
    if chunk_id != parent_id:
        # Retrieval documents contain embedding headers; only chunk_text is
        # the source body. Missing legacy metadata cannot prove a child claim.
        body = str(metadata.get("chunk_text") or "")
        location = str(metadata.get("location") or metadata.get("chunk_type") or "child")
    else:
        # hydrate_parent_results explicitly returns the real parent ID.
        body = "\n".join([
            parent_title or str(getattr(result, "content", "") or ""),
            *metadata_list(metadata, "reference_points"),
            *metadata_list(metadata, "follow_up_directions"),
        ]).strip()
        location = "parent_question"
    return EvidenceChunk(
        chunk_id=chunk_id,
        document_id=parent_id,
        document_version=str(metadata.get("document_version") or "legacy-v1"),
        text=body,
        source_context=source_context,
        location=location,
        source_type="question_bank",
        record_id=parent_id,
        publication_status="published",
        scope="public",
        retrieval_sources=[str(getattr(result, "source", "") or "unknown")],
        ranking={"score": float(getattr(result, "score", 0.0) or 0.0)},
    )


_GENERIC_IDENTITY_TOKENS = {
    "ai", "ml", "rag", "python", "java", "system", "design", "engineering",
    "engineer", "project", "experience", "technical", "technology", "knowledge",
    "skill", "skills", "application", "applications",
}
_GENERIC_CLAIM_TOKENS = {
    *_GENERIC_IDENTITY_TOKENS,
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "use", "used", "using", "what", "when", "which", "why", "with",
}
_GENERIC_HAN_PHRASES = {
    "人工智能", "机器学习", "系统设计", "项目经历", "技术知识", "工程能力",
    "岗位要求", "相关项目", "个人职责", "实现细节", "关键取舍",
}
_GENERIC_HAN_GRAMS = {
    "如何", "什么", "相关", "项目", "技术", "系统", "设计", "实现", "原理",
    "场景", "工程", "边界", "掌握", "说明", "问题", "影响", "能力", "知识",
    "进行", "可以", "需要", "以及", "通过", "验证", "候选", "岗位", "要求",
}

# A small terminology bridge is safer than pretending arbitrary cross-language
# paraphrases are verified.  These are domain concepts, never corpus IDs or
# frozen-evaluation labels.  Unknown translations remain unsupported.
_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "child_retrieval_unit": (
        "子块", "小块检索", "检索单元", "child chunk", "small retrieval unit",
    ),
    "parent_context": (
        "父块", "父级上下文", "完整上下文", "parent document", "parent context",
        "preserve full context", "larger generation unit",
    ),
    "recall": ("召回", "recall"),
    "parent_id": ("parent_id", "父级标识", "父子关联"),
    "reciprocal_rank": ("倒数排名", "排名位置", "reciprocal rank", "rank position"),
    "raw_score_comparability": (
        "原始分数", "原始分数可比", "分数可比性", "raw score", "raw scores",
        "scores comparable",
    ),
    "multiple_ranked_lists": (
        "多列表", "多个排序列表", "multiple ranked lists", "multiple lists",
    ),
    "rrf_k_parameter": (
        "k 参数", "顶部排名", "k parameter", "top ranks dominate",
    ),
    "query_rewriting": ("查询改写", "改写查询", "query rewriting", "rewrite query"),
    "query_transformation": (
        "转换为检索", "转换成检索", "检索友好", "transform the query",
        "transforms the query", "retrieval-friendly query",
    ),
    "abbreviation_expansion": ("缩写扩展", "expand abbreviations", "abbreviation expansion"),
    "domain_terms": ("领域术语", "domain terms"),
    "subquery": ("子查询", "subquery", "subqueries", "split a broad query"),
    "multi_query": ("多查询", "multi-query", "multiple queries"),
    "query_drift": ("查询漂移", "query drift"),
    "fusion": ("融合", "fusion"),
    "deduplication": ("去重", "deduplication", "deduplicate"),
    "evidence_boundary": (
        "证据边界", "评分边界", "结论边界", "证据不足时停止", "evidence boundary",
        "scoring boundary", "stop when evidence is insufficient",
    ),
}
_IDENTITY_CONCEPTS = {"query_rewriting"}
_TARGET_DETAIL_CONCEPTS: dict[str, set[str]] = {
    "query_rewriting": {
        "query_transformation",
        "abbreviation_expansion",
        "domain_terms",
        "subquery",
        "multi_query",
        "query_drift",
        "fusion",
        "deduplication",
    },
    "parent_child": {
        "child_retrieval_unit",
        "parent_context",
        "recall",
        "parent_id",
    },
    "rrf": {
        "reciprocal_rank",
        "raw_score_comparability",
        "multiple_ranked_lists",
        "rrf_k_parameter",
    },
}


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9_+#./\u4e00-\u9fff]+", "", _normalized_text(value))


def _english_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_+.#/]*", _normalized_text(value)))


def _han_sequences(value: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]+", _normalized_text(value))


def _han_ngrams(value: str, size: int = 3) -> set[str]:
    grams: set[str] = set()
    for sequence in _han_sequences(value):
        if len(sequence) < size:
            continue
        grams.update(
            sequence[index : index + size]
            for index in range(len(sequence) - size + 1)
        )
    return grams


def _contains_alias(text: str, alias: str) -> bool:
    normalized = _normalized_text(text)
    alias_normalized = _normalized_text(alias).strip()
    if not alias_normalized:
        return False
    if re.search(r"[\u4e00-\u9fff]", alias_normalized):
        return _compact_text(alias_normalized) in _compact_text(normalized)
    # Match an English phrase contiguously.  A bag-of-words check incorrectly
    # treated distant terms such as "multi-hop ... Query Rewriting" as the
    # concept "multi-query".
    alias_phrase = " ".join(re.findall(r"[a-z0-9]+", alias_normalized))
    text_phrase = " ".join(re.findall(r"[a-z0-9]+", normalized))
    return bool(alias_phrase) and f" {alias_phrase} " in f" {text_phrase} "


def _concepts(value: str) -> set[str]:
    return {
        concept
        for concept, aliases in _CONCEPT_ALIASES.items()
        if any(_contains_alias(value, alias) for alias in aliases)
    }


def _identity_parts(target: AssessmentTarget) -> tuple[set[str], list[set[str]], set[str]]:
    """Return Chinese phrases, English compound alternatives and all identity tokens."""

    chinese = {
        sequence
        for sequence in _han_sequences(target.scope)
        if len(sequence) >= 3 and sequence not in _GENERIC_HAN_PHRASES
    }
    specific_alternatives: list[set[str]] = []
    fallback_alternatives: list[set[str]] = []
    all_tokens: set[str] = set()
    for raw in (target.scope, target.skill_id.replace("-", " ").replace("_", " ")):
        tokens = _english_tokens(raw)
        if not tokens:
            continue
        all_tokens.update(tokens)
        specific = tokens - _GENERIC_IDENTITY_TOKENS
        (specific_alternatives if specific else fallback_alternatives).append(
            specific or tokens
        )
    alternatives = specific_alternatives or ([] if chinese else fallback_alternatives)
    unique_alternatives: list[set[str]] = []
    for item in alternatives:
        if item and item not in unique_alternatives:
            unique_alternatives.append(item)
    return chinese, unique_alternatives, all_tokens


def _matches_target_identity(
    target: AssessmentTarget,
    text: str,
    *,
    allow_broad_personal_match: bool = False,
) -> bool:
    """Require a target phrase or compound technical identifier, not one generic word."""

    target_concepts = _concepts(
        target.scope + " " + target.skill_id.replace("-", " ")
    ).intersection(_IDENTITY_CONCEPTS)
    if target_concepts.intersection(_concepts(text)):
        return True
    chinese, alternatives, all_identity_tokens = _identity_parts(target)
    if all_identity_tokens == {"rag"} and _concepts(text):
        return True
    compact = _compact_text(text)
    if any(_compact_text(phrase) in compact for phrase in chinese):
        return True
    text_tokens = _english_tokens(text)
    if any(parts.issubset(text_tokens) for parts in alternatives):
        return True
    if allow_broad_personal_match:
        if all_identity_tokens.intersection(text_tokens):
            return True
        scope_bigrams = {
            sequence[index : index + 2]
            for sequence in _han_sequences(target.scope)
            for index in range(max(0, len(sequence) - 1))
            if sequence[index : index + 2] not in _GENERIC_HAN_GRAMS
        }
        text_bigrams = {
            sequence[index : index + 2]
            for sequence in _han_sequences(text)
            for index in range(max(0, len(sequence) - 1))
        }
        return bool(scope_bigrams.intersection(text_bigrams))
    return False


def _target_concept_family(target: AssessmentTarget) -> str | None:
    identity = target.scope + " " + target.skill_id.replace("-", " ").replace("_", " ")
    concepts = _concepts(identity)
    tokens = _english_tokens(identity)
    compact = _compact_text(identity)
    if "query_rewriting" in concepts:
        return "query_rewriting"
    if "rrf" in tokens or "reciprocalrankfusion" in compact or "倒数排名融合" in compact:
        return "rrf"
    if (
        {"parent", "child"}.issubset(tokens)
        or "父子分块" in compact
        or "父子文档" in compact
    ):
        return "parent_child"
    return None


def _chunk_has_substantive_target_support(
    target: AssessmentTarget,
    chunk: EvidenceChunk,
) -> bool:
    """Distinguish an explanatory claim from a bare target-name mention."""

    body = chunk.citation_text
    if not body or not _matches_target_identity(
        target,
        body + "\n" + chunk.source_context,
        allow_broad_personal_match=(
            target.gap_kind == "personal_experience"
            and chunk.source_type == "private_material"
        ),
    ):
        return False
    family = _target_concept_family(target)
    if family:
        return bool(_TARGET_DETAIL_CONCEPTS[family].intersection(_concepts(body)))

    target_concepts = _concepts(
        target.scope + " " + target.skill_id.replace("-", " ")
    )
    if _concepts(body) - target_concepts:
        return True

    identity_tokens = _identity_parts(target)[2]
    specific_tokens = _english_tokens(body) - identity_tokens - _GENERIC_CLAIM_TOKENS
    if any(len(token) >= 3 for token in specific_tokens):
        return True

    target_han = _han_ngrams(target.scope)
    claim_han = {
        gram
        for gram in _han_ngrams(body) - target_han
        if gram not in _GENERIC_HAN_GRAMS
    }
    action_markers = (
        "通过", "采用", "基于", "避免", "导致", "提升", "降低", "用于", "依赖",
        "区别", "保证", "计算", "限制", "处理", "实现", "优化", "验证", "停止",
        "because", "therefore", "using", "requires", "prevents", "improves",
        "reduces", "computes", "limits", "ensures",
    )
    normalized = _normalized_text(body)
    return bool(claim_han) and any(marker in normalized for marker in action_markers)


def _chunk_supports_point(
    point: str,
    chunk: EvidenceChunk,
    target: AssessmentTarget | None,
) -> bool:
    """Conservatively prove one rubric point from one exact chunk."""

    body = chunk.citation_text
    if not body:
        return False
    if target and not _matches_target_identity(
        target,
        body + "\n" + chunk.source_context,
        allow_broad_personal_match=(
            target.gap_kind == "personal_experience"
            and chunk.source_type == "private_material"
        ),
    ):
        return False

    shared_concepts = _concepts(point).intersection(_concepts(body))
    if shared_concepts:
        return True

    point_tokens = _english_tokens(point)
    chunk_tokens = _english_tokens(body)
    identity_tokens = _identity_parts(target)[2] if target else set()
    claim_tokens = point_tokens - identity_tokens - _GENERIC_CLAIM_TOKENS
    strong_overlap = {
        token
        for token in claim_tokens.intersection(chunk_tokens)
        if len(token) >= 2
    }
    if strong_overlap:
        return True

    target_han = _han_ngrams(target.scope) if target else set()
    point_han = {
        gram
        for gram in _han_ngrams(point) - target_han
        if gram not in _GENERIC_HAN_GRAMS
    }
    if point_han.intersection(_han_ngrams(body)):
        return True

    # A rubric that is only the target concept itself is still traceable when
    # the exact target identity was established above.
    remaining_han = {
        gram for gram in _han_ngrams(point) if gram not in _GENERIC_HAN_GRAMS
    } - target_han
    if target and not claim_tokens and not remaining_han:
        return _matches_target_identity(target, point)
    return False


def assess_evidence_deterministic(
    target: AssessmentTarget,
    chunks: Iterable[EvidenceChunk],
) -> EvidenceAssessment:
    chunks = [
        item for item in chunks
        if item.citation_text and evidence_allowed_for_assessment(item, target)
    ]
    relevant: list[EvidenceChunk] = []
    for item in chunks:
        if _matches_target_identity(
            target,
            item.citation_text + "\n" + item.source_context,
            allow_broad_personal_match=(
                target.gap_kind == "personal_experience"
                and item.source_type == "private_material"
            ),
        ):
            relevant.append(item)
    substantive_published = [
        item
        for item in relevant
        if item.publication_status == "published"
        and item.source_type in {"question_bank", "public_web"}
        and _chunk_has_substantive_target_support(target, item)
    ]
    private = [item for item in relevant if item.source_type == "private_material"]
    detailed_private = [item for item in private if _has_personal_implementation_detail(item.text)]
    supported: list[str] = []
    missing: list[str] = []
    if substantive_published:
        supported.append("technical_knowledge")
    else:
        missing.append("technical_knowledge")
    if target.gap_kind == "personal_experience":
        if detailed_private:
            supported.append("personal_experience")
        else:
            missing.append("personal_experience")
    if not relevant:
        verdict = "insufficient"
    elif not missing:
        verdict = "sufficient"
    else:
        verdict = "partial"
    return EvidenceAssessment(
        target_id=target.target_id,
        relevant_chunk_ids=[item.chunk_id for item in relevant],
        supported_aspects=supported,
        missing_aspects=missing,
        conflicts=[],
        verdict=verdict,
        suggested_query=(
            " ".join(
                part.strip()
                for part in (target.scope, target.verification_goal)
                if part and part.strip()
            )
            if missing
            else None
        ),
        citations={
            aspect: [
                item.chunk_id
                for item in (
                    substantive_published
                    if aspect == "technical_knowledge"
                    else detailed_private
                )[:3]
            ]
            for aspect in supported
        },
    )


def _has_personal_implementation_detail(text: str) -> bool:
    """Require an attributable implementation detail, not a bare skill mention."""

    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    if len(normalized) < 24:
        return False
    action_markers = (
        "负责", "实现", "设计", "优化", "排查", "部署", "重构", "选型", "取舍",
        "指标", "延迟", "吞吐", "准确率", "召回率", "我的职责", "i built",
        "implemented", "designed", "optimized", "trade-off", "latency",
    )
    return any(marker in normalized for marker in action_markers)


def evidence_allowed_for_assessment(
    chunk: EvidenceChunk,
    target: AssessmentTarget | None = None,
) -> bool:
    """Keep unreviewed external evidence out of formal targets and rubrics."""

    if chunk.publication_status == "withdrawn":
        return False
    if chunk.source_type in {"question_bank", "public_web"}:
        return chunk.publication_status == "published"
    if chunk.source_type == "candidate":
        return False
    if chunk.source_type == "private_material":
        return bool(target and target.gap_kind == "personal_experience")
    return False


async def assess_evidence(
    target: AssessmentTarget,
    chunks: list[EvidenceChunk],
    *,
    semantic: bool = True,
) -> EvidenceAssessment:
    allowed_chunks = [
        item for item in chunks
        if item.citation_text and evidence_allowed_for_assessment(item, target)
    ]
    deterministic = assess_evidence_deterministic(target, allowed_chunks)
    if not semantic or not allowed_chunks:
        return deterministic
    llm = get_llm_client()
    result = await llm.chat_structured(
        messages=[
            {
                "role": "system",
                "content": (
                    "判断给定证据是否足以支持具体面试考察目标。每个 supported_aspect 必须携带实际 chunk_id；"
                    "不要把检索分数当成事实正确率。区分 partial、insufficient 和 conflicting。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"target_id={target.target_id}\n目标={target.verification_goal}\n"
                    f"约束={target.constraints}\n\n证据：\n"
                    + "\n\n".join(
                        f"[{item.chunk_id}] {item.citation_text[:1800]}" for item in allowed_chunks
                    )
                ),
            },
        ],
        response_model=EvidenceAssessment,
        temperature=0.1,
    )
    valid_ids = {item.chunk_id for item in allowed_chunks}
    result.target_id = target.target_id
    result.relevant_chunk_ids = [item for item in result.relevant_chunk_ids if item in valid_ids]
    result.citations = {
        aspect: [chunk_id for chunk_id in ids if chunk_id in valid_ids]
        for aspect, ids in result.citations.items()
    }
    return result


def validate_question_deterministic(
    question: QuestionItem,
    target: AssessmentTarget | None,
    chunks: list[EvidenceChunk],
    asked_questions: list[QuestionItem] | None = None,
    *,
    chunk_owners: dict[str, set[str]] | None = None,
) -> QuestionValidation:
    selected = resolve_question_evidence(
        question,
        chunks,
        target_id=target.target_id if target else None,
        chunk_owners=chunk_owners,
    )
    citations = {
        point: [
            item.chunk_id
            for item in selected
            if _chunk_supports_point(point, item, target)
        ][:3]
        for point in question.reference_points
    }
    unsupported = [
        point for point in question.reference_points if not citations.get(point)
    ]
    citations = {
        point: ids for point, ids in citations.items() if ids
    }
    target_match = "supported"
    if target and not _matches_target_identity(
        target,
        question.content + " " + " ".join(question.skill_tags),
    ):
        target_match = "uncertain"
    duplicate_of = None
    normalized = re.sub(r"\s+", "", question.content).casefold()
    for prior in asked_questions or []:
        if re.sub(r"\s+", "", prior.content).casefold() == normalized:
            duplicate_of = prior.question_id or str(prior.id)
            break
    rubric = "supported" if selected and not unsupported else "uncertain"
    verdict = "supported" if target_match == "supported" and rubric == "supported" and not duplicate_of else "uncertain"
    return QuestionValidation(
        question_id=question.question_id,
        target_match=target_match,
        rubric_support=rubric,
        answer_relevance="supported" if question.reference_points else "uncertain",
        duplicate_of=duplicate_of,
        unsupported_points=unsupported,
        citations=citations,
        verdict=verdict,
        revision_reason=("缺少可定位证据或与已有题重复" if verdict != "supported" else None),
    )


def resolve_question_evidence(
    question: QuestionItem,
    chunks: list[EvidenceChunk],
    *,
    target_id: str | None,
    chunk_owners: dict[str, set[str]] | None = None,
    allow_unreviewed_external: bool = False,
) -> list[EvidenceChunk]:
    """Resolve exact, target-compatible chunk IDs without ambiguous aliases."""
    evidence_by_chunk: dict[str, EvidenceChunk] = {}
    conflicting_ids: set[str] = set()
    for item in chunks:
        existing = evidence_by_chunk.get(item.chunk_id)
        if existing is not None and existing.text != item.text:
            conflicting_ids.add(item.chunk_id)
            evidence_by_chunk.pop(item.chunk_id, None)
            continue
        if item.chunk_id not in conflicting_ids:
            evidence_by_chunk[item.chunk_id] = item

    selected: list[EvidenceChunk] = []
    selected_ids: set[str] = set()
    for item_id in [*question.evidence_chunk_ids, *question.source_ids]:
        item = evidence_by_chunk.get(str(item_id))
        if item is None or item.chunk_id in selected_ids or not item.citation_text:
            continue
        if (
            item.source_type in {"question_bank", "public_web"}
            and item.publication_status != "published"
            and not (allow_unreviewed_external and item.source_type == "public_web")
        ):
            continue
        if (
            item.source_type == "candidate"
            and not allow_unreviewed_external
        ) or item.publication_status == "withdrawn":
            continue
        owners = (chunk_owners or {}).get(item.chunk_id, set())
        if owners and (target_id is None or target_id not in owners):
            continue
        selected.append(item)
        selected_ids.add(item.chunk_id)
    return selected


async def validate_questions_semantic(
    questions: list[QuestionItem],
    targets: list[AssessmentTarget],
    chunks: list[EvidenceChunk],
    *,
    chunk_owners: dict[str, set[str]] | None = None,
    include_unreviewed_external: bool = False,
) -> dict[str, QuestionValidation]:
    """Validate one generated plan in a single bounded structured-model call."""
    if not questions or not chunks:
        return {}
    target_by_id = {item.target_id: item for item in targets}
    chunk_by_id = {item.chunk_id: item for item in chunks}
    payload = []
    selected_chunk_ids: set[str] = set()
    for question in questions:
        target = target_by_id.get(question.target_id or "")
        selected = resolve_question_evidence(
            question,
            chunks,
            target_id=target.target_id if target else None,
            chunk_owners=chunk_owners,
            allow_unreviewed_external=include_unreviewed_external,
        )
        ids = [item.chunk_id for item in selected]
        selected_chunk_ids.update(ids)
        payload.append(
            {
                "question_id": question.question_id,
                "target": target.model_dump(mode="json") if target else None,
                "question": question.model_dump(mode="json"),
                "selected_chunk_ids": ids,
            }
        )
    evidence = [
        {"chunk_id": item_id, "text": chunk_by_id[item_id].citation_text[:1800]}
        for item_id in sorted(selected_chunk_ids)
    ]
    if not evidence:
        return {}
    result = await get_llm_client().chat_structured(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是面试题规划校验器。逐题检查目标匹配、事实性评分点证据、"
                    "参考要点是否回答题面、版本冲突和重复。只能依据提供的 chunk；"
                    "开放设计题允许多种合理方案，但事实性要求仍须有证据。"
                    "unsupported_points 必须原样返回没有证据支持的评分点；"
                    "citations 必须是 JSON 对象，键是评分点原文，值是支持它的 chunk_id 字符串数组；"
                    "不能把 citations 输出为数组。没有引用时输出空对象 {}。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "待校验题目：\n"
                    + json.dumps(payload, ensure_ascii=False)
                    + "\n\n可用证据：\n"
                    + json.dumps(evidence, ensure_ascii=False)
                ),
            },
        ],
        response_model=QuestionValidationBatch,
        temperature=0.1,
    )
    valid_ids = {item.question_id for item in questions}
    selected_ids_by_question = {
        item["question_id"]: set(item["selected_chunk_ids"])
        for item in payload
    }
    validated: dict[str, QuestionValidation] = {}
    question_by_id = {item.question_id: item for item in questions}
    for item in result.validations:
        if item.question_id not in valid_ids:
            continue
        allowed_ids = selected_ids_by_question.get(item.question_id, set())
        citations: dict[str, list[str]] = {}
        for point, ids in item.citations.items():
            deduped: list[str] = []
            for chunk_id in ids:
                if chunk_id in allowed_ids and chunk_id not in deduped:
                    deduped.append(chunk_id)
            citations[point] = deduped
        question = question_by_id[item.question_id]
        missing_citations = [
            point
            for point in question.reference_points
            if not citations.get(point)
        ]
        unsupported_points = list(
            dict.fromkeys([*item.unsupported_points, *missing_citations])
        )
        verdict = item.verdict
        revision_reason = item.revision_reason
        if missing_citations and verdict == "supported":
            verdict = "uncertain"
            revision_reason = revision_reason or "部分评分点缺少可验证引用。"
        validated[item.question_id] = item.model_copy(
            update={
                "citations": citations,
                "unsupported_points": unsupported_points,
                "verdict": verdict,
                "revision_reason": revision_reason,
            }
        )
    return validated
