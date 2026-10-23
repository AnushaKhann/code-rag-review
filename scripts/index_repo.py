import argparse
from pathlib import Path

from code_rag.ingestion import (
    clone_repository,
    ingest_repository,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a Python Git repository into JSONL chunks."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Path to an existing Git checkout")
    source.add_argument("--url", help="GitHub URL to clone")

    parser.add_argument(
        "--clone-dir",
        default="data/repos/target",
        help="Destination for a cloned repository",
    )
    parser.add_argument(
        "--output",
        default="data/chunks.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if args.url:
        repo = clone_repository(args.url, Path(args.clone_dir))
    else:
        repo = Path(args.repo).expanduser().resolve()

    chunks = ingest_repository(repo)
    output = Path(args.output).expanduser().resolve()
    write_jsonl(chunks, output)

    counts: dict[str, int] = {}
    for chunk in chunks:
        kind = chunk["chunk_type"]
        counts[kind] = counts.get(kind, 0) + 1

    print(f"Repository: {repo}")
    print(f"Chunks written: {len(chunks)}")
    print(f"Output: {output}")
    print("Chunk counts:")
    for kind, count in sorted(counts.items()):
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
