import argparse
import json
from pathlib import Path

import numpy as np

from code_rag.embeddings import EmbeddingModel, build_embedding_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed previously ingested repository chunks."
    )
    parser.add_argument(
        "--input",
        default="data/chunks.jsonl",
        help="Input JSONL chunks",
    )
    parser.add_argument(
        "--output",
        default="data/indexes/requests_embeddings.npy",
        help="Output NumPy vectors",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open(encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if not chunks:
        raise SystemExit("No chunks found in input file.")

    texts = [build_embedding_text(chunk) for chunk in chunks]

    model = EmbeddingModel()
    vectors = model.encode_documents(
        texts,
        batch_size=args.batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, vectors)

    print(f"Chunks: {len(chunks)}")
    print(f"Vector shape: {vectors.shape}")
    print(f"Vector dtype: {vectors.dtype}")
    print(f"Output: {output_path}")
    print(
        "First vector norm:",
        round(float(np.linalg.norm(vectors[0])), 5),
    )


if __name__ == "__main__":
    main()
