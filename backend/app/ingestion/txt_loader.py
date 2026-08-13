"""
Plain text document loader.
"""

from langchain_core.documents import Document


def load_txt(file_path: str, source_name: str) -> list[Document]:
    """
    Load a .txt file and return a single LangChain Document.
    Tries utf-8 first, falls back to latin-1 for odd encodings.
    """
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
                "file_type": "txt",
            },
        )
    ]