"""
FastAPI application entrypoint.
Wires together the upload, chat (LangChain), and agent-chat (LangGraph) routes.

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_upload, routes_chat, routes_graph
from app.vectorstore.chroma_store import VectorStoreRepository

app = FastAPI(
    title="RAG Chatbot with LangGraph",
    description=(
        "Intern Task 3 — RAG chatbot with LangChain + ChromaDB, "
        "extended with a LangGraph multi-agent workflow."
    ),
    version="1.0.0",
)

# Allow local frontend dev servers to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_upload.router, tags=["Ingestion"])
app.include_router(routes_chat.router, tags=["Chat - LangChain"])
app.include_router(routes_graph.router, tags=["Chat - LangGraph Agent"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "RAG Chatbot with LangGraph"}


@app.get("/health")
async def health():
    try:
        count = VectorStoreRepository.get_document_count()
    except Exception:
        count = 0
    return {"status": "healthy", "indexed_chunks": count}