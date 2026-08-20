"""
Tests for the ingestion loaders (Strategy pattern) and session memory.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.ingestion.loader_factory import LoaderFactory, UnsupportedFileTypeError
from app.ingestion.txt_loader import TxtLoader
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.excel_loader import ExcelLoader
from app.chains.memory import SessionMemory


# ---------------------------------------------------------------------------
# Loader strategies
# ---------------------------------------------------------------------------

def test_txt_loader(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("This is a test document about RAG systems.")

    docs = TxtLoader().load(str(sample_file), "sample.txt")

    assert len(docs) == 1
    assert "RAG systems" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "txt"


def test_empty_txt_file_returns_no_documents(tmp_path):
    sample_file = tmp_path / "empty.txt"
    sample_file.write_text("")

    docs = TxtLoader().load(str(sample_file), "empty.txt")
    assert docs == []


def test_unsupported_file_type_raises(tmp_path):
    sample_file = tmp_path / "sample.xyz"
    sample_file.write_text("data")

    try:
        LoaderFactory.load_document(str(sample_file))
        assert False, "Expected UnsupportedFileTypeError"
    except UnsupportedFileTypeError:
        pass


def test_loader_factory_routes_txt(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Routing test content.")

    docs = LoaderFactory.load_document(str(sample_file))
    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == "txt"


def test_loader_factory_routes_by_extension_for_every_supported_type():
    """
    Verifies LoaderFactory.get_loader() returns the correct concrete
    strategy for each registered extension — the core behavior the
    Strategy pattern is meant to provide.
    """
    assert isinstance(LoaderFactory.get_loader(".pdf"), PDFLoader)
    assert isinstance(LoaderFactory.get_loader(".docx"), DocxLoader)
    assert isinstance(LoaderFactory.get_loader(".txt"), TxtLoader)
    assert isinstance(LoaderFactory.get_loader(".xlsx"), ExcelLoader)
    assert isinstance(LoaderFactory.get_loader(".xls"), ExcelLoader)


def test_loader_factory_extension_matching_is_case_insensitive(tmp_path):
    sample_file = tmp_path / "sample.TXT"
    sample_file.write_text("Uppercase extension test.")

    docs = LoaderFactory.load_document(str(sample_file))
    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == "txt"


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------

def test_session_memory_isolates_sessions():
    memory = SessionMemory()
    memory.add_turn("session_a", "hello", "hi there")
    memory.add_turn("session_b", "different question", "different answer")

    assert len(memory.get_history("session_a")) == 2
    assert len(memory.get_history("session_b")) == 2
    assert memory.get_history("session_a")[0].content == "hello"


def test_session_memory_trims_to_max_history():
    memory = SessionMemory()
    for i in range(10):
        memory.add_turn("session_a", f"question {i}", f"answer {i}")

    history = memory.get_history("session_a")
    assert len(history) == SessionMemory.MAX_HISTORY_MESSAGES


def test_session_memory_clear_history():
    memory = SessionMemory()
    memory.add_turn("session_a", "hello", "hi there")
    memory.clear_history("session_a")

    assert memory.get_history("session_a") == []


def test_session_memory_format_history_for_prompt_empty():
    memory = SessionMemory()
    assert memory.format_history_for_prompt("nonexistent") == "No prior conversation."