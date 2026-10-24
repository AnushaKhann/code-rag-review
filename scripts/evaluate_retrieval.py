from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from code_rag.embeddings import EmbeddingModel
from code_rag.evaluation import aggregate_metrics, compute_metrics
from code_rag.hybrid_search import hybrid_search
from code_rag.keyword_search import BM25KeywordSearch
from code_rag.search import load_chunks, search_chunks
from code_rag.vector_store import load_vector_index


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation questions from JSONL."""
    with Path(path).open(encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]

    if not questions:
        raise ValueError("Evaluation dataset contains no questions.")

    return questions


def validate_questions(
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """Verify every expected symbol/file pair exists in the corpus."""
    available_pairs = {
        (chunk.get("symbol_name"), chunk.get("file_path")) for chunk in chunks
    }

    for question in questions:
        expected_symbols = question.get("relevant_symbols", [])
        expected_files = question.get("relevant_files", [])

        if not expected_symbols or not expected_files:
            raise ValueError(
                f"{question.get('id', '?')}: expected symbols/files cannot be empty."
            )

        for symbol in expected_symbols:
            for file_path in expected_files:
                if (symbol, file_path) not in available_pairs:
                    raise ValueError(
                        f"{question.get('id', '?')}: expected pair "
                        f"{symbol!r}, {file_path!r} was not found."
                    )


def serialize_results(
    results: list[dict[str, Any]],
    question: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Save ranked results without duplicating full source code."""
    serialized = []

    for rank, result in enumerate(results[:top_k], start=1):
        chunk = result["chunk"]
        serialized.append(
            {
                "rank": rank,
                "score": result.get("rrf_score", result.get("score")),
                "file_path": chunk.get("file_path"),
                "symbol_name": chunk.get("symbol_name"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "relevant": (
                    chunk.get("symbol_name") in question["relevant_symbols"]
                    and chunk.get("file_path") in question["relevant_files"]
                ),
                "sources": result.get("sources", []),
            }
        )

    return serialized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dense, BM25, and hybrid code retrieval."
    )
    parser.add_argument(
        "--questions",
        default="data/evaluation/requests_questions.jsonl",
    )
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--index", default="data/indexes/requests.faiss")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--output",
        default="results/retrieval_evaluation.json",
    )
    args = parser.parse_args()

    if args.top_k < 1 or args.candidate_k < 1:
        parser.error("--top-k and --candidate-k must be at least 1.")

    questions = load_questions(args.questions)
    chunks = load_chunks(args.chunks)
    index = load_vector_index(args.index)

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index has {index.ntotal} vectors but {len(chunks)} chunks were loaded."
        )

    validate_questions(questions, chunks)

    model = EmbeddingModel()
    keyword_search = BM25KeywordSearch(chunks)

    method_metrics: dict[str, list[dict[str, float]]] = {
        "dense": [],
        "bm25": [],
        "hybrid": [],
    }
    per_question = []

    for number, question in enumerate(questions, start=1):
        query = question["question"]
        print(f"[{number}/{len(questions)}] {query}")

        query_vector = model.encode_query(query)

        retrieval_results = {
            "dense": search_chunks(query_vector, index, chunks, top_k=args.top_k),
            "bm25": keyword_search.search(query, top_k=args.top_k),
            "hybrid": hybrid_search(
                query=query,
                query_vector=query_vector,
                dense_index=index,
                chunks=chunks,
                keyword_search=keyword_search,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
            ),
        }


        question_record: dict[str, Any] = {
            "id": question["id"],
            "question": query,
            "methods": {},
        }

        for method, results in retrieval_results.items():
            metrics = compute_metrics(results, question, top_k=args.top_k)
            method_metrics[method].append(metrics)

            question_record["methods"][method] = {
                **metrics,
                "results": serialize_results(results, question, args.top_k),
            }

        per_question.append(question_record)

    summary = {
        method: aggregate_metrics(metrics, top_k=args.top_k)
        for method, metrics in method_metrics.items()
    }

    report = {
        "dataset": str(args.questions),
        "num_questions": len(questions),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "metrics": summary,
        "per_question": per_question,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Retrieval Evaluation ===")
    print(f"Questions: {len(questions)}")
    print(f"Results saved to: {output_path}")
    print(
        f"\n{'Method':<12} {'Recall@' + str(args.top_k):>12} "
        f"{'MRR':>10} {'Precision@' + str(args.top_k):>15}"
    )

    for method, metrics in summary.items():
        print(
            f"{method:<12} "
            f"{metrics[f'recall@{args.top_k}']:>12.3f} "
            f"{metrics['mrr']:>10.3f} "
            f"{metrics[f'precision@{args.top_k}']:>15.3f}"
        )


if __name__ == "__main__":
    main()
