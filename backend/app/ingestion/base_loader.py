"""
Base loader interface (Strategy pattern).

Every format-specific loader (PDF, DOCX, TXT, Excel) implements this
interface, so LoaderFactory can select and invoke any of them
interchangeably at runtime without knowing the parsing details of any
specific format. Adding a new supported format later means writing one
new class here and registering it in LoaderFactory — nothing else in
the app changes.
"""

from abc import ABC, abstractmethod
from langchain_core.documents import Document


class DocumentLoader(ABC):
    """
    Strategy interface for document loaders.

    Concrete strategies: PDFLoader, DocxLoader, TxtLoader, ExcelLoader.
    """

    @abstractmethod
    def load(self, file_path: str, source_name: str) -> list[Document]:
        """
        Parse the file at file_path and return a list of LangChain
        Document objects, each carrying metadata needed for citation
        (at minimum: source filename and file_type).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def file_type(self) -> str:
        """Short label used in Document metadata, e.g. 'pdf', 'docx'."""
        raise NotImplementedError