"""
ChromaDB vector store wrapper.
Handles chunking of raw Documents and persistence of embeddings via LangChain's
Chroma integration. This is the single point of contact for indexing and
querying vectors — retrieval.py builds on top of this.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from functools import lru_cache

from app.config import settings
from app.embeddings.embedder import get_embedder


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Returns a cached Chroma vector store instance, persisted to disk so
    the index survives server restarts.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedder(),
        persist_directory=settings.chroma_persist_dir,
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split raw loaded documents into smaller overlapping chunks for
    better retrieval granularity.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def add_documents(documents: list[Document]) -> int:
    """
    Chunk and add documents to the vector store.
    Returns the number of chunks added.
    """
    if not documents:
        return 0

    chunks = chunk_documents(documents)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    """
    Run a similarity search against the vector store for the given query.
    """
    vectorstore = get_vectorstore()
    top_k = k or settings.top_k_results
    return vectorstore.similarity_search(query, k=top_k)


def get_document_count() -> int:
    """
    Returns the total number of chunks currently stored (useful for
    a health/status endpoint).
    """
    vectorstore = get_vectorstore()
    return vectorstore._collection.count()