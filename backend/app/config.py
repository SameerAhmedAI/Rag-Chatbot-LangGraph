"""
Central configuration for the RAG chatbot.
Loads from .env and exposes a single `settings` object used across the app.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    llm_model: str = "openai/gpt-oss-120b"  # verify exact id from Groq's /models endpoint

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "../data/chroma_db"
    chroma_collection_name: str = "rag_documents"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    top_k_results: int = 4

    # Uploads
    upload_dir: str = "../data/uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure runtime directories exist
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)