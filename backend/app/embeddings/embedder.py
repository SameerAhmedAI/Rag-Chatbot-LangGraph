"""
Embedding model wrapper.

This is a plain class rather than a Strategy pattern: there is currently
only one embedding backend (sentence-transformers, local, CPU) so a formal
interface with interchangeable strategies would be premature abstraction
for no real benefit. If a second embedding backend is added later (e.g. an
API-based one), promoting this to an ABC + concrete strategies mirrors
exactly what was done in ingestion/base_loader.py.

Wrapping it in a class (instead of a cached free function) keeps the
loaded model instance and its config together as one object with a clear
lifecycle, and makes the singleton behavior explicit rather than implicit
in a decorator.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings


class Embedder:
    """
    Lazily loads and caches a single HuggingFaceEmbeddings instance so the
    model is only loaded into memory once per process, not on every request.
    """

    _instance: HuggingFaceEmbeddings | None = None

    @classmethod
    def get(cls) -> HuggingFaceEmbeddings:
        if cls._instance is None:
            cls._instance = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return cls._instance


# Module-level convenience wrapper so existing call sites
# (`get_embedder()`) don't all need to change to `Embedder.get()`.
def get_embedder() -> HuggingFaceEmbeddings:
    return Embedder.get()