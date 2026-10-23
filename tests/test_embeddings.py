from code_rag.embeddings import build_embedding_text


def test_embedding_text_includes_code_context() -> None:
    chunk = {
        "file_path": "src/client.py",
        "chunk_type": "method",
        "symbol_name": "Client.connect",
        "docstring": "Open a connection.",
        "imports": ["import socket"],
        "content": "def connect(self):\n    return socket.create_connection()",
    }

    text = build_embedding_text(chunk)

    assert "src/client.py" in text
    assert "Client.connect" in text
    assert "Open a connection." in text
    assert "import socket" in text
    assert "socket.create_connection()" in text


def test_embedding_text_handles_missing_optional_metadata() -> None:
    chunk = {
        "file_path": "README.md",
        "chunk_type": "readme",
        "symbol_name": None,
        "docstring": None,
        "imports": [],
        "content": "# Example project",
    }

    text = build_embedding_text(chunk)

    assert "README.md" in text
    assert "# Example project" in text
    assert "Symbol:" not in text
    assert "Docstring:" not in text
    assert "Module imports:" not in text
