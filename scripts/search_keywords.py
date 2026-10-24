import argparse

from code_rag.keyword_search import BM25KeywordSearch
from code_rag.search import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search repository chunks using BM25 keywords."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    search = BM25KeywordSearch(chunks)
    results = search.search(args.question, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n{'=' * 70}")
        print(f"Rank: {rank} | BM25 score: {result['score']:.4f}")
        print(f"File: {chunk['file_path']}")
        print(f"Symbol: {chunk.get('symbol_name') or '(none)'}")
        print(f"Lines: {chunk['start_line']}-{chunk['end_line']}")
        print("-" * 70)
        print(chunk["content"][:1200])


if __name__ == "__main__":
    main()
