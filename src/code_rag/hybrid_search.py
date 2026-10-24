from __future__ import annotations

import re
from typing import Any

from code_rag.keyword_search import BM25KeywordSearch
from code_rag.search import search_chunks


def chunk_key(chunk: dict[str, Any]) -> tuple[Any, ...]:
    """Identify a chunk consistently across retrieval methods."""
    return (
        chunk.get("file_path"),
        chunk.get("chunk_type"),
        chunk.get("symbol_name"),
        chunk.get("start_line"),
        chunk.get("end_line"),
    )


def reciprocal_rank_fusion(
    *result_lists: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 5,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked results, favoring exact symbols and smaller chunks."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_tokens = set(re.findall(r"[a-zA-Z_]\w*|\d+", query or ""))
    query_tokens = {token.lower() for token in query_tokens}

    fused: dict[tuple[Any, ...], dict[str, Any]] = {}

    for results in result_lists:
        seen_in_list: set[tuple[Any, ...]] = set()

        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]
            key = chunk_key(chunk)

            if key in seen_in_list:
                continue
            seen_in_list.add(key)

            if key not in fused:
                fused[key] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "sources": [],
                }

            fused[key]["rrf_score"] += 1.0 / (k + rank)

            source = result.get("source")
            if source and source not in fused[key]["sources"]:
                fused[key]["sources"].append(source)

    for item in fused.values():
        chunk = item["chunk"]
        symbol = chunk.get("symbol_name") or ""
        normalized_symbol = re.sub(r"[^a-z0-9]", "", symbol.lower())

        # Add one rank-1 RRF contribution for an exact symbol mention.
        if normalized_symbol and normalized_symbol in query_tokens:
            item["rrf_score"] += 1.0 / (k + 1)

        # Reduce the dominance of broad chunks without discarding them.
        if chunk.get("oversized", False):
            item["rrf_score"] *= 0.75

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )
    return ranked[:top_k]


def hybrid_search(
    query: str,
    query_vector,
    dense_index,
    chunks: list[dict[str, Any]],
    keyword_search: BM25KeywordSearch,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[dict[str, Any]]:
    """Combine dense and keyword retrieval using RRF."""
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1.")

    dense_results = search_chunks(
        query_vector=query_vector,
        index=dense_index,
        chunks=chunks,
        top_k=candidate_k,
    )
    for result in dense_results:
        result["source"] = "dense"

    keyword_results = keyword_search.search(
        query,
        top_k=candidate_k,
    )
    for result in keyword_results:
        result["source"] = "bm25"

    return reciprocal_rank_fusion(
        dense_results,
        keyword_results,
        k=60,
        top_k=top_k,
        query=query,
    )
