import pytest

from code_rag.evaluation import (
    aggregate_metrics,
    compute_metrics,
    is_relevant,
)

QUESTION = {
    "id": "Q1",
    "question": "Where is Example defined?",
    "relevant_symbols": ["Example"],
    "relevant_files": ["src/example.py"],
}


def make_chunk(file_path, symbol_name):
    return {
        "file_path": file_path,
        "symbol_name": symbol_name,
    }


def make_result(file_path, symbol_name):
    return {"chunk": make_chunk(file_path, symbol_name)}


def test_is_relevant_requires_matching_symbol_and_file():
    assert is_relevant(make_chunk("src/example.py", "Example"), QUESTION)
    assert not is_relevant(make_chunk("src/other.py", "Example"), QUESTION)
    assert not is_relevant(make_chunk("src/example.py", "Other"), QUESTION)


def test_metrics_when_relevant_result_is_rank_two():
    results = [
        make_result("src/other.py", "Other"),
        make_result("src/example.py", "Example"),
    ]

    metrics = compute_metrics(results, QUESTION, top_k=5)

    assert metrics["recall@5"] == 1.0
    assert metrics["precision@5"] == pytest.approx(0.2)
    assert metrics["reciprocal_rank"] == 0.5


def test_metrics_when_relevant_result_is_missing():
    results = [make_result("src/other.py", "Other")]

    metrics = compute_metrics(results, QUESTION, top_k=5)

    assert metrics["recall@5"] == 0.0
    assert metrics["precision@5"] == 0.0
    assert metrics["reciprocal_rank"] == 0.0


def test_compute_metrics_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        compute_metrics([], QUESTION, top_k=0)


def test_aggregate_metrics_calculates_means():
    rows = [
        {
            "recall@5": 1.0,
            "precision@5": 0.2,
            "reciprocal_rank": 1.0,
        },
        {
            "recall@5": 0.0,
            "precision@5": 0.0,
            "reciprocal_rank": 0.0,
        },
    ]

    summary = aggregate_metrics(rows, top_k=5)

    assert summary["recall@5"] == 0.5
    assert summary["precision@5"] == pytest.approx(0.1)
    assert summary["mrr"] == 0.5


def test_aggregate_metrics_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_metrics([])
