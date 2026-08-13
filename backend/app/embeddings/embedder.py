"""
Embedding model wrapper.
Uses sentence-transformers locally (no API cost, no rate limits) via
LangChain's HuggingFaceEmbeddings interface so it plugs directly into Chroma.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings instance so the model is only
    loaded into memory once per process, not on every request.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )