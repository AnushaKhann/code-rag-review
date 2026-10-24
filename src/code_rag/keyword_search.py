from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Preserve identifiers and split compound names without duplicate tokens."""
    identifiers = re.findall(r"[A-Za-z_]\w*|\d+", text)
    tokens: list[str] = []

    for identifier in identifiers:
        full_token = identifier.lower()
        tokens.append(full_token)

        # Split camelCase, PascalCase, and acronym boundaries.
        split_identifier = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", identifier)
        split_identifier = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", split_identifier)

        parts = re.findall(r"[A-Za-z]+|\d+", split_identifier)

        # Add components only when the identifier is compound.
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)

    return tokens


def chunk_to_search_text(chunk: dict[str, Any]) -> str:
    """Build lexical search text without repeated module imports."""
    parts = [
        chunk.get("file_path", ""),
        chunk.get("symbol_name") or "",
        chunk.get("docstring") or "",
        chunk.get("content", ""),
    ]
    return "\n".join(parts)


class BM25KeywordSearch:
    """Keyword search over repository chunks using BM25."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 search with no chunks.")

        self.chunks = chunks
        self.tokenized_chunks = [
            tokenize(chunk_to_search_text(chunk)) for chunk in chunks
        ]
        self.index = BM25Okapi(self.tokenized_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return chunks ranked by BM25 relevance."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:top_k]

        return [
            {
                "score": float(scores[i]),
                "chunk": self.chunks[i],
            }
            for i in ranked_indices
        ]
