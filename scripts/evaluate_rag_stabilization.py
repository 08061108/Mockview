"""One controlled retrieval-only comparison for RAG stabilization.

This script calls only the configured Embedding endpoint. It never invokes an
LLM judge, question generator, scorer, OCR, or public search.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.models.jd import SkillItem, SkillMatrix  # noqa: E402
from app.rag.postprocess import hydrate_parent_results, result_parent_id  # noqa: E402
from app.rag.reranker import rerank_results  # noqa: E402
from app.rag.retriever import RetrievedQuestion, get_retriever  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

FROZEN = ROOT / "data" / "eval" / "retrieval_golden.json"
CHINESE = ROOT / "data" / "eval" / "retrieval_zh_golden.json"
RESULTS = ROOT / "data" / "eval" / "results" / "rag_stabilization"
TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    for item in frozen:
        item["id"] = item.get("id") or item["name"]
        item["split"] = "frozen"
        item["expanded_queries"] = item.get("queries", [])[1:]
    chinese = json.loads(CHINESE.read_text(encoding="utf-8"))
    return (
        frozen,
        [item for item in chinese if item["split"] == "dev"],
        [item for item in chinese if item["split"] == "holdout"],
    )


def unique_parent_ids(items: list[RetrievedQuestion], top_k: int = 10) -> list[str]:
    values: list[str] = []
    for item in items:
        parent = result_parent_id(item)
        if parent and parent not in values:
            values.append(parent)
        if len(values) >= top_k:
            break
    return values


def metrics(ids: list[str], expected_ids: list[str]) -> dict[str, float]:
    expected = set(expected_ids)
    top5 = ids[:5]
    top10 = ids[:10]
    hit = 1.0 if expected.intersection(top5) else 0.0
    recall = len(expected.intersection(top5)) / max(len(expected), 1)
    mrr = next((1.0 / rank for rank, value in enumerate(top10, 1) if value in expected), 0.0)
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, value in enumerate(top10, 1)
        if value in expected
    )
    ideal_count = min(len(expected), 10)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {"hit@5": hit, "recall@5": recall, "mrr@10": mrr, "ndcg@10": dcg / ideal if ideal else 0.0}


def query_state(query: str) -> dict[str, Any]:
    known = re.findall(
        r"(?i)Python|GIL|asyncio|RAG|BM25|RRF|Embedding|WebSocket|Docker|TCP|HTTPS|DNS|epoll|MySQL|MVCC|Redis|Kafka|LangGraph|Agent|消息队列|缓存|分布式锁",
        query,
    )
    skills = list(dict.fromkeys(known)) or [query[:40]]
    return {
        "skill_matrix": SkillMatrix(
            position_title="Retrieval evaluation",
            experience_level="mid",
            skills=[
                SkillItem(name=item, category="domain", weight=1.0, is_required=True)
                for item in skills[:6]
            ],
        )
    }


def apply_variant(name: str, raw: list[RetrievedQuestion], query: str, top_k: int) -> list[RetrievedQuestion]:
    state = query_state(query)
    if name == "A":
        return hydrate_parent_results(
            rerank_results(state, raw, purpose="practice"), top_k=top_k
        )
    bounded = hydrate_parent_results(raw, top_k=max(top_k, 16), strategy="bounded")
    if name == "B":
        return bounded[:top_k]
    if name == "C":
        return rerank_results(
            state, bounded, purpose="practice", strategy="normalized"
        )[:top_k]
    raise ValueError(name)


async def retrieve_case(case: dict[str, Any], queries: list[str]) -> tuple[list[RetrievedQuestion], int]:
    retriever = get_retriever()
    started = time.perf_counter()
    outcome = await retriever.retrieve_multi_outcome(
        queries,
        top_k_per_query=8,
        final_top_k=50,
        fingerprint_context={"eval_case": case["id"], "query_variant": queries},
    )
    return outcome.items, round((time.perf_counter() - started) * 1000)


async def evaluate_variants(cases: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        base_queries = case.get("queries") or [case["query"]]
        raw, elapsed = await retrieve_case(case, base_queries)
        for name in names:
            ranked = apply_variant(name, raw, case["query"], 10)
            ids = unique_parent_ids(ranked)
            rows.append({
                "split": case["split"],
                "case_id": case["id"],
                "variant": name,
                "query_count": len(base_queries),
                "elapsed_ms": elapsed,
                "expected_ids": case["expected_ids"],
                "retrieved_ids": ids,
                **metrics(ids, case["expected_ids"]),
            })
    return rows


def summarize(rows: list[dict[str, Any]], variant: str, split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant and row["split"] == split]
    elapsed = sorted(row["elapsed_ms"] for row in selected)
    p95_index = max(0, math.ceil(len(elapsed) * 0.95) - 1)
    return {
        "variant": variant,
        "split": split,
        "cases": len(selected),
        **{
            metric: mean(row[metric] for row in selected) if selected else 0.0
            for metric in ("hit@5", "recall@5", "mrr@10", "ndcg@10")
        },
        "p95_ms": elapsed[p95_index] if elapsed else 0,
    }


def quality_compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> int:
    for metric in ("recall@5", "mrr@10", "ndcg@10"):
        delta = candidate[metric] - baseline[metric]
        if abs(delta) > TOLERANCE:
            return 1 if delta > 0 else -1
    return 0


def actual_case_improvement(rows: list[dict[str, Any]], candidate: str, baseline: str, split: str) -> bool:
    by_key = {(row["case_id"], row["variant"]): row for row in rows if row["split"] == split}
    ids = {case_id for case_id, variant in by_key if variant == baseline}
    return any(
        quality_compare(by_key[(case_id, candidate)], by_key[(case_id, baseline)]) > 0
        for case_id in ids
    )


def no_regression(candidate: dict[str, Any], baseline: dict[str, Any], *, allowance: float) -> bool:
    if candidate["hit@5"] + TOLERANCE < baseline["hit@5"]:
        return False
    return all(
        candidate[metric] + allowance + TOLERANCE >= baseline[metric]
        for metric in ("recall@5", "mrr@10", "ndcg@10")
    ) and candidate["p95_ms"] <= baseline["p95_ms"] * 1.2 + 1


def no_new_hit_miss(rows: list[dict[str, Any]], candidate: str, baseline: str, split: str) -> bool:
    by_key = {(row["case_id"], row["variant"]): row for row in rows if row["split"] == split}
    ids = {case_id for case_id, variant in by_key if variant == baseline}
    return all(
        not (
            by_key[(case_id, baseline)]["hit@5"] == 1.0
            and by_key[(case_id, candidate)]["hit@5"] == 0.0
        )
        for case_id in ids
    )


async def main_async() -> Path:
    args = parse_args()
    frozen, dev, holdout = load_cases()
    retriever = get_retriever()
    before_batches = retriever.embedding_batch_calls

    # Generate every unique query vector once, in bounded provider batches.
    # A hybrid comparison is invalid when this step cannot complete; do not
    # silently turn a quality experiment into a BM25-only comparison.
    all_queries: list[str] = []
    seen_queries: set[str] = set()
    for case in [*frozen, *dev, *holdout]:
        values = [*(case.get("queries") or [case["query"]]), *case.get("expanded_queries", [])]
        for value in values:
            normalized = " ".join(str(value).split())
            key = normalized.casefold()
            if normalized and key not in seen_queries:
                seen_queries.add(key)
                all_queries.append(normalized)
    prewarmed = await retriever._embed_queries_isolated(all_queries)
    failures = [item for item in prewarmed if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(
            "controlled embedding evaluation aborted: "
            f"{len(failures)}/{len(prewarmed)} query embeddings failed "
            f"({type(failures[0]).__name__})"
        )

    dev_rows = await evaluate_variants(dev, ["A", "B", "C"])
    dev_summaries = {name: summarize(dev_rows, name, "dev") for name in ("A", "B", "C")}
    ranking_winner = "A"
    for name in ("B", "C"):
        if (
            quality_compare(dev_summaries[name], dev_summaries[ranking_winner]) > 0
            and actual_case_improvement(dev_rows, name, ranking_winner, "dev")
        ):
            ranking_winner = name

    validation_rows = await evaluate_variants(
        [*frozen, *holdout], list(dict.fromkeys(["A", ranking_winner]))
    )
    rows = [*dev_rows, *validation_rows]
    summaries = [
        summarize(rows, name, split)
        for split in ("dev", "frozen", "holdout")
        for name in sorted({row["variant"] for row in rows if row["split"] == split})
    ]
    summary_map = {(item["variant"], item["split"]): item for item in summaries}
    ranking_enabled = ranking_winner != "A"
    if ranking_enabled:
        ranking_enabled = (
            no_regression(summary_map[(ranking_winner, "frozen")], summary_map[("A", "frozen")], allowance=0.0)
            and no_regression(summary_map[(ranking_winner, "holdout")], summary_map[("A", "holdout")], allowance=0.01)
            and no_new_hit_miss(rows, ranking_winner, "A", "frozen")
            and no_new_hit_miss(rows, ranking_winner, "A", "holdout")
        )
    selected_ranking = ranking_winner if ranking_enabled else "A"

    d_rows: list[dict[str, Any]] = []
    for case in dev:
        base_queries = case.get("queries") or [case["query"]]
        queries = [*base_queries, *case.get("expanded_queries", [])]
        raw, elapsed = await retrieve_case(case, queries)
        ranked = apply_variant(selected_ranking, raw, case["query"], 10)
        ids = unique_parent_ids(ranked)
        d_rows.append({
            "split": "dev", "case_id": case["id"], "variant": "D",
            "query_count": len(queries), "elapsed_ms": elapsed,
            "expected_ids": case["expected_ids"], "retrieved_ids": ids,
            **metrics(ids, case["expected_ids"]),
        })
    rows.extend(d_rows)
    d_dev = summarize(rows, "D", "dev")
    r_dev = summary_map[(selected_ranking, "dev")]
    d_candidate = quality_compare(d_dev, r_dev) > 0 and actual_case_improvement(rows, "D", selected_ranking, "dev")

    d_validation: list[dict[str, Any]] = []
    if d_candidate:
        for case in [*frozen, *holdout]:
            base_queries = case.get("queries") or [case["query"]]
            queries = [*base_queries, *case.get("expanded_queries", [])]
            raw, elapsed = await retrieve_case(case, queries)
            ranked = apply_variant(selected_ranking, raw, case["query"], 10)
            ids = unique_parent_ids(ranked)
            d_validation.append({
                "split": case["split"], "case_id": case["id"], "variant": "D",
                "query_count": len(queries), "elapsed_ms": elapsed,
                "expected_ids": case["expected_ids"], "retrieved_ids": ids,
                **metrics(ids, case["expected_ids"]),
            })
        rows.extend(d_validation)
    summaries = [
        summarize(rows, name, split)
        for split in ("dev", "frozen", "holdout")
        for name in sorted({row["variant"] for row in rows if row["split"] == split})
    ]
    summary_map = {(item["variant"], item["split"]): item for item in summaries}
    d_enabled = bool(d_candidate and d_validation)
    if d_enabled:
        d_enabled = (
            no_regression(summary_map[("D", "frozen")], summary_map[(selected_ranking, "frozen")], allowance=0.0)
            and no_regression(summary_map[("D", "holdout")], summary_map[(selected_ranking, "holdout")], allowance=0.01)
            and no_new_hit_miss(rows, "D", selected_ranking, "frozen")
            and no_new_hit_miss(rows, "D", selected_ranking, "holdout")
        )

    payload = {
        "schema_version": 1,
        "valid": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": {
            "frozen": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
            "chinese": hashlib.sha256(CHINESE.read_bytes()).hexdigest(),
        },
        "index_manifest": get_vector_store().index_manifest,
        "embedding_model": settings.embedding_model,
        "top_k": args.top_k,
        "selection_order": ["recall@5", "mrr@10", "ndcg@10"],
        "tolerance": TOLERANCE,
        "ranking_winner_on_dev": ranking_winner,
        "ranking_enabled": ranking_enabled,
        "selected_ranking": selected_ranking,
        "query_expansion_candidate": d_candidate,
        "query_expansion_enabled": d_enabled,
        "recommended_strategy": "expanded" if d_enabled else (
            "normalized" if selected_ranking == "C" else "bounded" if selected_ranking == "B" else "legacy"
        ),
        "embedding_batch_calls": retriever.embedding_batch_calls - before_batches,
        "summaries": summaries,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"comparison_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "ranking_winner_on_dev", "ranking_enabled", "selected_ranking",
        "query_expansion_candidate", "query_expansion_enabled", "recommended_strategy",
        "embedding_batch_calls",
    )}, ensure_ascii=False, indent=2))
    print(path)
    return path


if __name__ == "__main__":
    asyncio.run(main_async())
