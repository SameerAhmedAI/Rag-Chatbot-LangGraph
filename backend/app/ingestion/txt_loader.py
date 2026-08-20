"""
Plain text document loader (concrete Strategy).
"""

from langchain_core.documents import Document

from app.ingestion.base_loader import DocumentLoader


class TxtLoader(DocumentLoader):
    """Loads .txt files. Tries utf-8 first, falls back to latin-1 for odd encodings."""

    @property
    def file_type(self) -> str:
        return "txt"

    def load(self, file_path: str, source_name: str) -> list[Document]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()

        text = text.strip()
        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "file_type": self.file_type,
                },
            )
        ]