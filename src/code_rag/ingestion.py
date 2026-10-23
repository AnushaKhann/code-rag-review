from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import tree_sitter_python
from tree_sitter import Language, Parser

PYTHON_LANGUAGE = Language(tree_sitter_python.language())
PYTHON_PARSER = Parser(PYTHON_LANGUAGE)

DOC_SUFFIXES = {".md", ".rst", ".txt"}
README_NAMES = {"readme", "readme.md", "readme.rst", "readme.txt"}


def run_git(repo: Path, *args: str) -> str:
    """Run Git safely and return stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def clone_repository(url: str, destination: Path) -> Path:
    """Clone a repository into a new destination."""
    destination = destination.expanduser().resolve()

    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)],
        check=True,
    )

    return destination


def list_repository_files(repo: Path) -> list[Path]:
    """
    List tracked and non-ignored untracked files using Git.

    Git handles .gitignore, nested ignore rules, and global
    excludes. Paths returned by Git are relative to the repo.
    """
    repo = repo.resolve()

    output = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    ).stdout

    paths = {
        item.decode("utf-8", errors="surrogateescape")
        for item in output.split(b"\0")
        if item
    }

    files = []
    for relative_path in sorted(paths):
        path = repo / relative_path

        # Ignore missing paths and symlinks in this initial version.
        if not path.is_file() or path.is_symlink():
            continue

        files.append(path)

    return files


def relative_path(path: Path, repo: Path) -> str:
    """Return a stable, repository-relative POSIX path."""
    return path.relative_to(repo).as_posix()


def node_text(source: bytes, node: Any) -> str:
    """Extract the exact source represented by a Tree-sitter node."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def extract_imports(root: Any, source: bytes) -> list[str]:
    """Extract Python import statements from a whole module."""
    imports: list[str] = []
    stack = [root]

    while stack:
        node = stack.pop()

        if node.type in {"import_statement", "import_from_statement"}:
            imports.append(node_text(source, node).strip())

        stack.extend(reversed(node.named_children))

    return list(dict.fromkeys(imports))


def extract_docstring(node: Any, source: bytes) -> str | None:
    """Extract a leading Python string expression, if present."""
    body = node.child_by_field_name("body")
    if body is None:
        return None

    for statement in body.named_children:
        if statement.type != "expression_statement":
            break

        children = statement.named_children
        if not children:
            break

        expression = children[0]

        if expression.type in {"string", "concatenated_string"}:
            return node_text(source, expression)

        break

    return None


def qualified_symbol_name(node: Any, source: bytes) -> tuple[str, str]:
    """
    Return a qualified symbol name and its kind.

    Examples:
      parse_config
      Client.connect
      Client.connect._prepare
    """
    ancestors = []
    parent = node.parent

    while parent is not None:
        if parent.type in {"class_definition", "function_definition"}:
            ancestors.append(parent)
        parent = parent.parent

    ancestors.reverse()
    ancestors.append(node)

    names = [ancestor.child_by_field_name("name") for ancestor in ancestors]
    parts = [node_text(source, name) for name in names if name is not None]

    enclosing_class = any(
        ancestor.type == "class_definition" for ancestor in ancestors[:-1]
    )

    if node.type == "class_definition":
        kind = "class"
    elif enclosing_class:
        kind = "method"
    else:
        kind = "function"

    return ".".join(parts), kind


def parse_python_file(
    path: Path,
    repo: Path,
    max_chunk_chars: int = 12000,
) -> list[dict[str, Any]]:
    """Parse a Python file into symbol-level source chunks."""
    source = path.read_bytes()
    tree = PYTHON_PARSER.parse(source)
    imports = extract_imports(tree.root_node, source)
    chunks: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if node.type in {"function_definition", "class_definition"}:
            symbol_name, kind = qualified_symbol_name(node, source)
            content = node_text(source, node)

            chunks.append(
                {
                    "chunk_id": (
                        f"{relative_path(path, repo)}:"
                        f"{node.start_point.row + 1}:"
                        f"{node.end_point.row + 1}:{symbol_name}"
                    ),
                    "chunk_type": kind,
                    "file_path": relative_path(path, repo),
                    "language": "python",
                    "symbol_name": symbol_name,
                    "start_line": node.start_point.row + 1,
                    "end_line": node.end_point.row + 1,
                    "docstring": extract_docstring(node, source),
                    "imports": imports,
                    "content": content,
                    "oversized": len(content) > max_chunk_chars,
                }
            )

        for child in node.named_children:
            visit(child)

    visit(tree.root_node)

    # Record parse errors so we can audit unsupported or malformed files.
    if tree.root_node.has_error:
        chunks.append(
            {
                "chunk_id": f"{relative_path(path, repo)}:parse-warning",
                "chunk_type": "parse_warning",
                "file_path": relative_path(path, repo),
                "language": "python",
                "symbol_name": None,
                "start_line": 1,
                "end_line": source.count(b"\n") + 1,
                "docstring": None,
                "imports": imports,
                "content": ("Tree-sitter reported syntax errors or missing nodes."),
                "oversized": False,
            }
        )

    return chunks


def is_documentation(path: Path) -> bool:
    """Identify README and supported text documentation files."""
    if path.name.lower() in README_NAMES:
        return True

    return path.suffix.lower() in DOC_SUFFIXES and (
        "docs" in path.parts or "doc" in path.parts
    )


def parse_document(path: Path, repo: Path) -> dict[str, Any]:
    """Represent a documentation file as a separate chunk."""
    content = path.read_text(encoding="utf-8", errors="replace")
    line_count = max(1, content.count("\n") + (not content.endswith("\n")))

    return {
        "chunk_id": f"{relative_path(path, repo)}:document",
        "chunk_type": (
            "readme" if path.name.lower().startswith("readme") else "documentation"
        ),
        "file_path": relative_path(path, repo),
        "language": "markdown"
        if path.suffix.lower() == ".md"
        else path.suffix.lower().lstrip("."),
        "symbol_name": None,
        "start_line": 1,
        "end_line": line_count,
        "docstring": None,
        "imports": [],
        "content": content,
        "oversized": len(content) > 12000,
    }


def ingest_repository(repo: Path) -> list[dict[str, Any]]:
    """Ingest supported source and documentation files."""
    repo = repo.expanduser().resolve()

    if not (repo / ".git").exists():
        raise ValueError(f"Not a Git checkout: {repo}")

    chunks: list[dict[str, Any]] = []

    for path in list_repository_files(repo):
        if path.suffix.lower() == ".py":
            try:
                chunks.extend(parse_python_file(path, repo))
            except (OSError, UnicodeError) as exc:
                chunks.append(
                    {
                        "chunk_id": f"{relative_path(path, repo)}:read-error",
                        "chunk_type": "ingestion_warning",
                        "file_path": relative_path(path, repo),
                        "language": "python",
                        "symbol_name": None,
                        "start_line": 1,
                        "end_line": 1,
                        "docstring": None,
                        "imports": [],
                        "content": f"Could not read file: {exc}",
                        "oversized": False,
                    }
                )

        elif is_documentation(path):
            try:
                chunks.append(parse_document(path, repo))
            except OSError as exc:
                print(f"Warning: could not read {path}: {exc}")

    return chunks


def write_jsonl(chunks: list[dict[str, Any]], output: Path) -> None:
    """Write one JSON object per line for reproducible inspection."""
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
