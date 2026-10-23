import argparse

from code_rag.vector_store import build_vector_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from saved embeddings."
    )
    parser.add_argument(
        "--embeddings",
        default="data/indexes/requests_embeddings.npy",
    )
    parser.add_argument(
        "--output",
        default="data/indexes/requests.faiss",
    )
    args = parser.parse_args()

    index = build_vector_index(args.embeddings, args.output)

    print(f"Vectors indexed: {index.ntotal}")
    print(f"Vector dimensions: {index.d}")
    print(f"Index type: {type(index).__name__}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
