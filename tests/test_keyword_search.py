import pytest

from code_rag.keyword_search import BM25KeywordSearch, tokenize


def test_tokenize_splits_code_identifiers() -> None:
    tokens = tokenize("ConnectTimeout HTTPAdapter prepare_headers")

    assert "connect" in tokens
    assert "timeout" in tokens
    assert "http" in tokens
    assert "adapter" in tokens
    assert "prepare" in tokens
    assert "headers" in tokens


def test_bm25_ranks_exact_symbol_match() -> None:
    chunks = [
        {
            "file_path": "src/requests/exceptions.py",
            "symbol_name": "ConnectTimeout",
            "content": "The request timed out while connecting.",
        },
        {
            "file_path": "src/requests/auth.py",
            "symbol_name": "HTTPBasicAuth",
            "content": "Attaches basic authentication to a request.",
        },
        {
            "file_path": "src/requests/models.py",
            "symbol_name": "PreparedRequest.prepare_headers",
            "content": "Prepares the request headers.",
        },
    ]

    search = BM25KeywordSearch(chunks)
    results = search.search("ConnectTimeout", top_k=3)

    assert results[0]["chunk"]["symbol_name"] == "ConnectTimeout"


def test_bm25_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="no chunks"):
        BM25KeywordSearch([])


def test_bm25_rejects_invalid_top_k() -> None:
    search = BM25KeywordSearch([{"file_path": "a.py", "content": "example code"}])

    with pytest.raises(ValueError, match="top_k"):
        search.search("example", top_k=0)
