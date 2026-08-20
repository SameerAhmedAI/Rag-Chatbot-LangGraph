"""
ChromaDB vector store repository.

This is a Repository pattern, not a Strategy pattern: there's one
persistence backend (Chroma) and this class's job is to hide the
details of talking to it — chunking, indexing, querying — behind a
clean interface, not to offer interchangeable algorithms. Forcing a
Strategy shape here (as with ingestion/base_loader.py) would be the
same mistake as leaving everything as loose functions, just relabeled.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.embeddings.embedder import Embedder


class VectorStoreRepository:
    """
    Single point of contact for indexing and querying vectors.
    retrieval/retriever.py builds on top of this.
    """

    _instance: Chroma | None = None

    @classmethod
    def _get_store(cls) -> Chroma:
        if cls._instance is None:
            cls._instance = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=Embedder.get(),
                persist_directory=settings.chroma_persist_dir,
            )
        return cls._instance

    @staticmethod
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

    @classmethod
    def add_documents(cls, documents: list[Document]) -> int:
        """
        Chunk and add documents to the vector store.
        Returns the number of chunks added.
        """
        if not documents:
            return 0

        chunks = cls.chunk_documents(documents)
        store = cls._get_store()
        store.add_documents(chunks)
        return len(chunks)

    @classmethod
    def similarity_search(cls, query: str, k: int | None = None) -> list[Document]:
        """Run a similarity search against the vector store for the given query."""
        store = cls._get_store()
        top_k = k or settings.top_k_results
        return store.similarity_search(query, k=top_k)

    @classmethod
    def as_retriever(cls, k: int | None = None):
        """Returns a LangChain retriever object backed by this store."""
        store = cls._get_store()
        top_k = k or settings.top_k_results
        return store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    @classmethod
    def get_document_count(cls) -> int:
        """Returns the total number of chunks currently stored."""
        store = cls._get_store()
        return store._collection.count()


# Module-level convenience wrappers so existing call sites don't all
# need to change their import/call shape in the same commit.
def get_vectorstore() -> Chroma:
    return VectorStoreRepository._get_store()


def add_documents(documents: list[Document]) -> int:
    return VectorStoreRepository.add_documents(documents)


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    return VectorStoreRepository.similarity_search(query, k=k)


def get_document_count() -> int:
    return VectorStoreRepository.get_document_count()