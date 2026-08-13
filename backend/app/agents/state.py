"""
Shared state definition for the LangGraph agent workflow.
Every node reads from and writes to this TypedDict as it flows through the graph.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    question: str                     # original user question
    session_id: str                   # for chat history lookup
    route: Optional[str]              # "rag" | "general" — decided by router_node
    context: Optional[str]            # formatted retrieved context
    sources: Optional[list[dict]]     # source metadata for citations
    draft_answer: Optional[str]       # answer before critique
    final_answer: Optional[str]       # answer after critique/refinement
    needs_refinement: Optional[bool]  # flag set by critique_node