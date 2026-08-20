"""
Agent chat endpoint — Advanced level deliverable.
Routes the question through the LangGraph multi-node agent workflow
(router -> retrieve -> generate -> critique).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.graph import agent_graph

router = APIRouter()


class AgentChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class AgentChatResponse(BaseModel):
    answer: str
    route: str | None
    sources: list[dict]
    was_refined: bool


@router.post("/agent-chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = agent_graph.run(request.question, request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow failed: {e}")

    return AgentChatResponse(**result)