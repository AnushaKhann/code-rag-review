import subprocess
from pathlib import Path

from code_rag.ingestion import (
    ingest_repository,
    parse_python_file,
)


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fixture"
    repo.mkdir()

    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )

    (repo / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (repo / "example.py").write_text(
        '"""Example module."""\n'
        "import os\n\n"
        "def outer(value):\n"
        '    """Return a normalized value."""\n'
        "    def inner():\n"
        "        return value\n"
        "    return inner().strip()\n\n"
        "class Client:\n"
        '    """Example client."""\n'
        "    def connect(self):\n"
        "        return True\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# Fixture\nTest repository.\n")
    (repo / "ignored.py").write_text("def should_not_appear(): pass\n")

    subprocess.run(
        ["git", "-C", str(repo), "add", ".gitignore", "example.py", "README.md"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add fixtures"],
        check=True,
        capture_output=True,
    )

    return repo


def test_extracts_symbols_and_metadata(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    chunks = parse_python_file(repo / "example.py", repo)

    symbols = {chunk["symbol_name"]: chunk for chunk in chunks}

    assert "outer" in symbols
    assert "outer.inner" in symbols
    assert "Client" in symbols
    assert "Client.connect" in symbols

    assert symbols["outer"]["docstring"] == '"""Return a normalized value."""'
    assert symbols["Client.connect"]["chunk_type"] == "method"
    assert symbols["outer"]["start_line"] == 4
    assert symbols["outer"]["end_line"] == 8
    assert "import os" in symbols["outer"]["imports"]
    assert "return inner().strip()" in symbols["outer"]["content"]


def test_respects_gitignore_and_indexes_readme(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    chunks = ingest_repository(repo)

    paths = {chunk["file_path"] for chunk in chunks}
    kinds = {chunk["chunk_type"] for chunk in chunks}

    assert "ignored.py" not in paths
    assert "README.md" in paths
    assert "readme" in kinds
    assert "function" in kinds
    assert "class" in kinds
    assert "method" in kinds


def test_source_chunks_preserve_original_lines(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    chunks = parse_python_file(repo / "example.py", repo)

    outer = next(c for c in chunks if c["symbol_name"] == "outer")
    source_lines = (repo / "example.py").read_text().splitlines()
    actual = "\n".join(source_lines[outer["start_line"] - 1 : outer["end_line"]])

    assert actual == outer["content"]
