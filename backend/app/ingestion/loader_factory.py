"""
Loader factory: routes an uploaded file to the correct format-specific loader
based on file extension. This is the single entry point the rest of the app
calls — nothing else needs to know how each format is parsed.
"""

from pathlib import Path
from langchain_core.documents import Document

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.txt_loader import load_txt
from app.ingestion.excel_loader import load_excel


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx", ".xls"}


class UnsupportedFileTypeError(Exception):
    pass


def load_document(file_path: str) -> list[Document]:
    """
    Route file_path to the correct loader based on extension.
    Returns a list of LangChain Document objects ready for chunking.
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    source_name = path.name

    if extension == ".pdf":
        return load_pdf(file_path, source_name)
    elif extension == ".docx":
        return load_docx(file_path, source_name)
    elif extension == ".txt":
        return load_txt(file_path, source_name)
    elif extension in (".xlsx", ".xls"):
        return load_excel(file_path, source_name)
    else:
        raise UnsupportedFileTypeError(
            f"File type '{extension}' is not supported. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )