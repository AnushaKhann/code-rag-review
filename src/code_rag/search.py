from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from code_rag.embeddings import EmbeddingModel


def load_chunks(chunks_path: str | Path) -> list[dict[str, Any]]:
    """Load chunks in the same order used to build the vector index."""
    with Path(chunks_path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def search_chunks(
    query_vector: np.ndarray,
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve the top matching chunks for a query embedding."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index has {index.ntotal} vectors but {len(chunks)} chunks were provided."
        )

    if index.ntotal == 0:
        return []

    query_vector = np.asarray(query_vector, dtype=np.float32)

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    if query_vector.shape != (1, index.d):
        raise ValueError(
            f"Expected query shape (1, {index.d}), received {query_vector.shape}."
        )

    if not np.isfinite(query_vector).all():
        raise ValueError("Query vector contains non-finite values.")

    query_vector = np.ascontiguousarray(query_vector)
    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal),
    )

    results = []
    for score, chunk_index in zip(scores[0], indices[0], strict=True):
        if chunk_index < 0:
            continue

        results.append(
            {
                "score": float(score),
                "chunk": chunks[int(chunk_index)],
            }
        )

    return results


def search_repository(
    query: str,
    model: EmbeddingModel,
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Embed a natural-language question and search repository chunks."""
    query_vector = model.encode_query(query)

    return search_chunks(
        query_vector=query_vector,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )
