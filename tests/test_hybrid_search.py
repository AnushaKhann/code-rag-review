import pytest

from code_rag.hybrid_search import reciprocal_rank_fusion


def test_rrf_promotes_chunk_found_by_both_search_methods() -> None:
    shared_chunk = {
        "file_path": "src/a.py",
        "symbol_name": "connect",
        "start_line": 10,
        "end_line": 20,
    }
    dense_only = {
        "file_path": "src/b.py",
        "symbol_name": "send",
        "start_line": 1,
        "end_line": 5,
    }
    keyword_only = {
        "file_path": "src/c.py",
        "symbol_name": "ConnectTimeout",
        "start_line": 2,
        "end_line": 8,
    }

    results = reciprocal_rank_fusion(
        [{"chunk": shared_chunk}, {"chunk": dense_only}],
        [{"chunk": shared_chunk}, {"chunk": keyword_only}],
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["chunk"]["symbol_name"] == "connect"
    assert results[0]["rrf_score"] == pytest.approx(2 / 61)


def test_rrf_deduplicates_chunks_and_respects_top_k() -> None:
    chunk = {
        "file_path": "src/a.py",
        "symbol_name": "connect",
        "start_line": 10,
        "end_line": 20,
    }

    results = reciprocal_rank_fusion(
        [{"chunk": chunk}, {"chunk": chunk}],
        top_k=1,
    )

    assert len(results) == 1


def test_rrf_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion(k=0)

    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion(top_k=0)


def test_rrf_boosts_exact_symbol_match() -> None:
    broad_chunk = {
        "file_path": "src/adapters.py",
        "symbol_name": "HTTPAdapter",
        "start_line": 10,
        "end_line": 500,
        "oversized": True,
    }
    exact_chunk = {
        "file_path": "src/exceptions.py",
        "symbol_name": "ConnectTimeout",
        "start_line": 20,
        "end_line": 25,
        "oversized": False,
    }

    results = reciprocal_rank_fusion(
        [{"chunk": broad_chunk}, {"chunk": exact_chunk}],
        query="Where is ConnectTimeout defined?",
        top_k=2,
    )

    assert results[0]["chunk"]["symbol_name"] == "ConnectTimeout"


def test_rrf_penalizes_oversized_chunks() -> None:
    normal_chunk = {
        "file_path": "src/normal.py",
        "symbol_name": "normal_function",
        "start_line": 1,
        "end_line": 10,
        "oversized": False,
    }
    oversized_chunk = {
        "file_path": "src/large.py",
        "symbol_name": "large_class",
        "start_line": 1,
        "end_line": 500,
        "oversized": True,
    }

    results = reciprocal_rank_fusion(
        [{"chunk": oversized_chunk}, {"chunk": normal_chunk}],
        top_k=2,
    )

    assert results[0]["chunk"]["symbol_name"] == "normal_function"
