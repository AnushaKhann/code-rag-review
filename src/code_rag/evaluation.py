from __future__ import annotations

from typing import Any


def is_relevant(
    chunk: dict[str, Any],
    question: dict[str, Any],
) -> bool:
    """Check whether a chunk matches an expected symbol and file."""
    expected_symbols = set(question["relevant_symbols"])
    expected_files = set(question["relevant_files"])

    return (
        chunk.get("symbol_name") in expected_symbols
        and chunk.get("file_path") in expected_files
    )


def compute_metrics(
    results: list[dict[str, Any]],
    question: dict[str, Any],
    top_k: int = 5,
) -> dict[str, float]:
    """Calculate Recall@k, Precision@k, and reciprocal rank."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    expected_symbols = set(question["relevant_symbols"])
    expected_files = set(question["relevant_files"])

    if not expected_symbols or not expected_files:
        raise ValueError("Expected symbols and files cannot be empty.")

    expected_pairs = {
        (symbol, file_path)
        for symbol in expected_symbols
        for file_path in expected_files
    }

    top_results = results[:top_k]
    found_pairs: set[tuple[str, str]] = set()
    relevant_count = 0
    reciprocal_rank = 0.0

    for rank, result in enumerate(top_results, start=1):
        chunk = result["chunk"]

        if is_relevant(chunk, question):
            relevant_count += 1
            pair = (chunk["symbol_name"], chunk["file_path"])
            found_pairs.add(pair)

            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank

    return {
        f"recall@{top_k}": len(found_pairs & expected_pairs) / len(expected_pairs),
        f"precision@{top_k}": relevant_count / top_k,
        "reciprocal_rank": reciprocal_rank,
    }


def aggregate_metrics(
    question_metrics: list[dict[str, float]],
    top_k: int = 5,
) -> dict[str, float]:
    """Calculate macro-averaged metrics across evaluation questions."""
    if not question_metrics:
        raise ValueError("Cannot aggregate an empty evaluation.")

    return {
        f"recall@{top_k}": sum(item[f"recall@{top_k}"] for item in question_metrics)
        / len(question_metrics),
        f"precision@{top_k}": sum(
            item[f"precision@{top_k}"] for item in question_metrics
        )
        / len(question_metrics),
        "mrr": sum(item["reciprocal_rank"] for item in question_metrics)
        / len(question_metrics),
    }
