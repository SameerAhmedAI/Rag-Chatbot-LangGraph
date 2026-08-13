"""
Basic tests for the ingestion loaders.
Run with: pytest tests/test_ingestion.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.ingestion.loader_factory import load_document, UnsupportedFileTypeError
from app.ingestion.txt_loader import load_txt


def test_txt_loader(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("This is a test document about RAG systems.")

    docs = load_txt(str(sample_file), "sample.txt")

    assert len(docs) == 1
    assert "RAG systems" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "txt"


def test_empty_txt_file_returns_no_documents(tmp_path):
    sample_file = tmp_path / "empty.txt"
    sample_file.write_text("")

    docs = load_txt(str(sample_file), "empty.txt")
    assert docs == []


def test_unsupported_file_type_raises(tmp_path):
    sample_file = tmp_path / "sample.xyz"
    sample_file.write_text("data")

    try:
        load_document(str(sample_file))
        assert False, "Expected UnsupportedFileTypeError"
    except UnsupportedFileTypeError:
        pass


def test_loader_factory_routes_txt(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Routing test content.")

    docs = load_document(str(sample_file))
    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == "txt"