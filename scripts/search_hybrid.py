import argparse

from code_rag.embeddings import EmbeddingModel
from code_rag.hybrid_search import hybrid_search
from code_rag.keyword_search import BM25KeywordSearch
from code_rag.search import load_chunks
from code_rag.vector_store import load_vector_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search repository code using hybrid retrieval."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--index", default="data/indexes/requests.faiss")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    dense_index = load_vector_index(args.index)
    model = EmbeddingModel()
    keyword_search = BM25KeywordSearch(chunks)
    query_vector = model.encode_query(args.question)

    results = hybrid_search(
        query=args.question,
        query_vector=query_vector,
        dense_index=dense_index,
        chunks=chunks,
        keyword_search=keyword_search,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    if not results:
        print("No matching chunks found.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n{'=' * 70}")
        print(f"Rank: {rank} | RRF score: {result['rrf_score']:.5f}")
        print(f"Retrieved by: {', '.join(result['sources']) or 'unknown'}")
        print(f"File: {chunk['file_path']}")
        print(f"Symbol: {chunk.get('symbol_name') or '(none)'}")
        print(f"Lines: {chunk['start_line']}-{chunk['end_line']}")
        print("-" * 70)
        print(chunk["content"][:1200])


if __name__ == "__main__":
    main()
