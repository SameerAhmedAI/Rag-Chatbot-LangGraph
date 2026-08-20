"""
PDF document loader (concrete Strategy).
Extracts text page-by-page and attaches page-level metadata for citation.
"""

from langchain_core.documents import Document
from pypdf import PdfReader

from app.ingestion.base_loader import DocumentLoader


class PDFLoader(DocumentLoader):
    """Loads .pdf files, one Document per page (blank/scanned pages skipped)."""

    @property
    def file_type(self) -> str:
        return "pdf"

    def load(self, file_path: str, source_name: str) -> list[Document]:
        reader = PdfReader(file_path)
        documents: list[Document] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                # skip blank/scanned pages with no extractable text
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "file_type": self.file_type,
                    },
                )
            )

        return documents