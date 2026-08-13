"""
PDF document loader.
Extracts text page-by-page and attaches page-level metadata for citation.
"""

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(file_path: str, source_name: str) -> list[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects,
    one per page, with metadata (source filename, page number, file type).
    """
    reader = PdfReader(file_path)
    documents: list[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            # skip blank/scanned pages with no extractable text
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "page": page_num,
                    "file_type": "pdf",
                },
            )
        )

    return documents