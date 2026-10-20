"""Offline P3 comparison of the frozen legacy and current BM25 tokenizers.

This reads the existing Chroma collection but never writes or rebuilds it.  The
output is append-only and explicitly reports BM25-only metrics; it must not be
presented as a successful hybrid/embedding evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable

from rank_bm25 import BM25Okapi

from app.rag.postprocess import result_parent_id
from app.rag.retriever import RetrievedQuestion, _tokenize
from app.rag.vector_store import get_vector_store
from scripts.evaluate_retrieval import load_cases, metric_row

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "eval" / "results" / "p3"


def legacy_tokenize(text: str) -> list[str]:
    return [item.lower() for item in re.split(r"\W+", text) if item]


def _evaluate(
    docs: dict,
    tokenizer: Callable[[str], list[str]],
    cases: list[dict],
) -> tuple[list[dict], dict[str, float]]:
    corpus = [tokenizer(text) for text in docs["documents"]]
    index = BM25Okapi(corpus)
    rows = []
    for case in cases:
        scores = index.get_scores(tokenizer(case["query"]))
        ranked = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)
        results = []
        for position in ranked:
            if scores[position] <= 0:
                break
            results.append(
                RetrievedQuestion(
                    id=docs["ids"][position],
                    content=docs["documents"][position],
                    metadata=docs["metadatas"][position],
                    score=float(scores[position]),
                    source="bm25",
                )
            )
            if len(results) >= 10:
                break
        metrics = metric_row(results, case["expected_ids"])
        rows.append(
            {
                "id": case.get("id") or case["name"],
                "expected_ids": case["expected_ids"],
                "retrieved_parent_ids": [result_parent_id(item) for item in results],
                **metrics,
            }
        )
    summary = {
        metric: round(mean(row[metric] for row in rows), 4)
        for metric in ("hit@5", "recall@5", "mrr@10")
    }
    return rows, summary


def _evaluate_current(docs: dict, cases: list[dict]) -> tuple[list[dict], dict[str, float]]:
    """Mirror production's legacy-English/enhanced-Chinese index selection."""
    enhanced = BM25Okapi([_tokenize(text) for text in docs["documents"]])
    legacy = BM25Okapi([legacy_tokenize(text) for text in docs["documents"]])
    rows = []
    for case in cases:
        is_chinese = bool(re.search(r"[\u4e00-\u9fff]", case["query"]))
        index = enhanced if is_chinese else legacy
        tokenizer = _tokenize if is_chinese else legacy_tokenize
        scores = index.get_scores(tokenizer(case["query"]))
        ranked = sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)
        results = []
        for position in ranked:
            if scores[position] <= 0:
                break
            results.append(
                RetrievedQuestion(
                    id=docs["ids"][position],
                    content=docs["documents"][position],
                    metadata=docs["metadatas"][position],
                    score=float(scores[position]),
                    source="bm25",
                )
            )
            if len(results) >= 10:
                break
        metrics = metric_row(results, case["expected_ids"])
        rows.append(
            {
                "id": case.get("id") or case["name"],
                "query_index": "enhanced_chinese" if is_chinese else "frozen_legacy_non_chinese",
                "expected_ids": case["expected_ids"],
                "retrieved_parent_ids": [result_parent_id(item) for item in results],
                **metrics,
            }
        )
    summary = {
        metric: round(mean(row[metric] for row in rows), 4)
        for metric in ("hit@5", "recall@5", "mrr@10")
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    args = parser.parse_args()
    cases = load_cases()
    docs = get_vector_store().get_all_documents()
    legacy_rows, legacy = _evaluate(docs, legacy_tokenize, cases)
    current_rows, current = _evaluate_current(docs, cases)
    payload = {
        "schema_version": 1,
        "kind": "p3_offline_bm25_tokenization_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(docs["ids"]),
        "case_count": len(cases),
        "golden_sha256": hashlib.sha256(
            (ROOT / "data" / "eval" / "retrieval_golden.json").read_bytes()
        ).hexdigest(),
        "network_requests": 0,
        "embedding_requests": 0,
        "scope_note": "BM25-only offline comparison; not the hybrid retrieval baseline",
        "legacy": {"summary": legacy, "rows": legacy_rows},
        "current": {"summary": current, "rows": current_rows},
        "delta": {
            metric: round(current[metric] - legacy[metric], 4)
            for metric in ("hit@5", "recall@5", "mrr@10")
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"bm25_tokenization_{args.stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))
    print(json.dumps({"legacy": legacy, "current": current, "delta": payload["delta"]}))


if __name__ == "__main__":
    main()
