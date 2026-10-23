from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_IMPORTS = 12


def build_embedding_text(chunk: dict[str, Any]) -> str:
    """
    Build contextual text for embedding a code or documentation chunk.

    Preserve the source code while adding identifiers that help
    distinguish similar implementations in different files.
    """
    parts = [
        f"File: {chunk['file_path']}",
        f"Chunk type: {chunk['chunk_type']}",
    ]

    symbol_name = chunk.get("symbol_name")
    if symbol_name:
        parts.append(f"Symbol: {symbol_name}")

    docstring = chunk.get("docstring")
    if docstring:
        parts.append(f"Docstring: {docstring}")

    imports = chunk.get("imports") or []
    if imports:
        import_text = "\n".join(imports[:MAX_IMPORTS])
        parts.append(f"Module imports:\n{import_text}")

    parts.append(f"Content:\n{chunk['content']}")

    return "\n\n".join(parts)


class EmbeddingModel:
    """Generate normalized embeddings for code and documentation."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Embed documents in batches and return float32 vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single natural-language query."""
        vectors = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(vectors[0], dtype=np.float32)
