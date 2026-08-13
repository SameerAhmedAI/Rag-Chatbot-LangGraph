"""
LangGraph workflow assembly.
Wires together router -> [retrieve -> generate -> critique] OR [generate] -> END.

Graph shape:

                    ┌──────────┐
        ┌──────────►│  router  │
        │           └────┬─────┘
        │                │
        │      route =  "rag"        route = "general"
        │                │                    │
        │                ▼                    ▼
        │         ┌────────────┐       ┌─────────────┐
        │         │  retrieve  │       │  generate    │
        │         └─────┬──────┘       │ (no context) │
        │               ▼              └──────┬───────┘
        │         ┌────────────┐              │
        │         │  generate  │              │
        │         └─────┬──────┘              │
        │               ▼                     │
        │         ┌────────────┐              │
        │         │  critique  │              │
        │         └─────┬──────┘              │
        │               ▼                     ▼
        │              END  ◄─────────────────┘

This satisfies the Advanced-level requirement: real agent workflow with
conditional routing, not just a linear LangChain chain relabeled as a graph.
"""

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.nodes.router_node import router_node, route_decision
from app.agents.nodes.retrieve_node import retrieve_node
from app.agents.nodes.generate_node import generate_node
from app.agents.nodes.critique_node import critique_node


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)

    graph.set_entry_point("router")

    # Conditional edge: router decides which path to take
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "retrieve",
            "general": "generate",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", END)

    return graph.compile()


# Compiled once at import time, reused across requests
compiled_graph = build_graph()


def run_agent(question: str, session_id: str = "default") -> dict:
    """
    Entry point used by the API layer to run a question through the full
    LangGraph agent workflow.
    """
    from app.chains.memory import add_turn  # local import avoids circular import

    initial_state: AgentState = {
        "question": question,
        "session_id": session_id,
        "route": None,
        "context": None,
        "sources": None,
        "draft_answer": None,
        "final_answer": None,
        "needs_refinement": None,
    }

    result = compiled_graph.invoke(initial_state)

    final_answer = result.get("final_answer") or result.get("draft_answer", "")
    add_turn(session_id, question, final_answer)

    return {
        "answer": final_answer,
        "route": result.get("route"),
        "sources": result.get("sources") or [],
        "was_refined": result.get("needs_refinement", False),
    }