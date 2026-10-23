# Code-aware RAG & PR Review Agent

A Python-first code intelligence system for repository Q&A with source citations and context-aware pull request reviews.

## Project Status

**Current phase:** Repository ingestion and AST-aware chunking.

The current pipeline extracts Python classes, functions, and methods using Tree-sitter and indexes README and documentation files as separate chunk types. Each chunk includes source metadata and original line ranges.

## Features

- Git-aware repository file discovery that respects ignore rules.
- Tree-sitter-based Python symbol extraction.
- Metadata for file paths, symbol names, line ranges, docstrings, and imports.
- Separate README and documentation chunks.
- JSONL output for inspection and downstream processing.
- Automated tests for symbol extraction, ignore rules, and source-line preservation.

## Tech Stack

- Python 3.11
- Tree-sitter
- pytest
- Ruff

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run Tests

```bash
pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
```

## Ingest a Repository

```bash
python scripts/index_repo.py \
  --url https://github.com/psf/requests.git \
  --clone-dir data/repos/requests \
  --output data/chunks.jsonl
```

Generated repository copies and chunk files are excluded from version control.

## Roadmap

1. Repository ingestion and AST-aware chunking.
2. Baseline dense retrieval and cited codebase Q&A.
3. Hybrid retrieval, reranking, graph expansion, and evaluation.
4. Context-aware PR review and verification.
5. API, UI, deployment, and benchmark documentation.

## Evaluation

Benchmark results will be published after the evaluation dataset, metrics, and experiment configurations are established. No performance improvements are claimed at this stage.

## Limitations

The initial implementation targets Python. Large symbols are flagged rather than automatically split, and import metadata is currently collected at module level.

## License

To be determined.