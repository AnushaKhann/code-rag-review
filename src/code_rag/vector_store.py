from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_vector_index(
    embeddings_path: str | Path,
    index_path: str | Path,
) -> faiss.IndexFlatIP:
    """Build and save a FAISS inner-product index."""
    embeddings_path = Path(embeddings_path)
    index_path = Path(index_path)

    vectors = np.load(embeddings_path)

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values.")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    return index


def load_vector_index(index_path: str | Path) -> faiss.Index:
    """Load a previously saved FAISS index."""
    return faiss.read_index(str(index_path))
