import faiss
import numpy as np
import pytest

from code_rag.search import search_chunks


def test_search_returns_most_similar_chunk_first() -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(3)
    index.add(vectors)

    chunks = [
        {"file_path": "a.py", "content": "connection handling"},
        {"file_path": "b.py", "content": "authentication"},
        {"file_path": "c.py", "content": "request encoding"},
    ]

    results = search_chunks(
        query_vector=np.array([0.9, 0.1, 0.0], dtype=np.float32),
        index=index,
        chunks=chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk"]["file_path"] == "a.py"
    assert results[0]["score"] > results[1]["score"]


def test_search_rejects_mismatched_chunk_count() -> None:
    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="vectors"):
        search_chunks(
            query_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            index=index,
            chunks=[],
        )
