import argparse

from code_rag.embeddings import EmbeddingModel
from code_rag.search import load_chunks, search_repository
from code_rag.vector_store import load_vector_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a repository using semantic similarity."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Natural-language question about the repository",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--index", default="data/indexes/requests.faiss")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    index = load_vector_index(args.index)
    model = EmbeddingModel()

    results = search_repository(
        query=args.question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=args.top_k,
    )

    if not results:
        print("No matching chunks found.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n{'=' * 70}")
        print(f"Rank: {rank} | Similarity: {result['score']:.4f}")
        print(f"File: {chunk['file_path']}")
        print(f"Symbol: {chunk.get('symbol_name') or '(none)'}")
        print(f"Lines: {chunk['start_line']}-{chunk['end_line']}")
        print(f"Type: {chunk['chunk_type']}")
        print("-" * 70)
        print(chunk["content"][:1500])


if __name__ == "__main__":
    main()
