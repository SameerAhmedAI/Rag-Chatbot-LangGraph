"""
Chat endpoint — Intermediate level deliverable.
Straight LangChain conversational RAG chain (no agent routing).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.chains.qa_chain import QAChain

router = APIRouter()

# Shared instance for the process (matches the singleton-style lifecycle
# used by VectorStoreRepository / Embedder — one chain, reused across requests).
_qa_chain = QAChain()


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    rewritten_query: str  # shows what retrieval actually searched for, for debugging/verification


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = _qa_chain.answer(request.question, request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        rewritten_query=result["rewritten_query"],
    )