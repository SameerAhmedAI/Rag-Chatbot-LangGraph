"""
Loader factory: selects the correct DocumentLoader strategy for an
uploaded file based on its extension. This is the single entry point
the rest of the app calls — nothing else needs to know how each format
is parsed, or that new formats can be added here without touching
any calling code (upload route, tests, etc.).
"""

from pathlib import Path
from langchain_core.documents import Document

from app.ingestion.base_loader import DocumentLoader
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.txt_loader import TxtLoader
from app.ingestion.excel_loader import ExcelLoader


class UnsupportedFileTypeError(Exception):
    pass


class LoaderFactory:
    """
    Maps file extensions to DocumentLoader strategy instances and
    dispatches loading to the correct one.

    Extending support for a new format means adding one entry to
    _REGISTRY — no other code in the app changes.
    """

    _REGISTRY: dict[str, DocumentLoader] = {
        ".pdf": PDFLoader(),
        ".docx": DocxLoader(),
        ".txt": TxtLoader(),
        ".xlsx": ExcelLoader(),
        ".xls": ExcelLoader(),
    }

    @classmethod
    def supported_extensions(cls) -> set[str]:
        return set(cls._REGISTRY.keys())

    @classmethod
    def get_loader(cls, extension: str) -> DocumentLoader:
        loader = cls._REGISTRY.get(extension.lower())
        if loader is None:
            raise UnsupportedFileTypeError(
                f"File type '{extension}' is not supported. "
                f"Supported types: {', '.join(sorted(cls.supported_extensions()))}"
            )
        return loader

    @classmethod
    def load_document(cls, file_path: str) -> list[Document]:
        """
        Route file_path to the correct loader strategy based on extension
        and return the parsed LangChain Document objects, ready for chunking.
        """
        path = Path(file_path)
        loader = cls.get_loader(path.suffix)
        return loader.load(file_path, path.name)


# Module-level convenience wrapper so callers that only need the common
# case don't have to import the class — keeps the public call site
# (`load_document(path)`) identical to the pre-refactor version, so
# routes_upload.py and tests only need an import path change.
def load_document(file_path: str) -> list[Document]:
    return LoaderFactory.load_document(file_path)