"""
Retrieve node.
Pulls relevant chunks from the vector store for the question and
attaches formatted context + source metadata to the graph state.

Uses a rewritten, standalone version of the question for the similarity
search (not the raw question) so follow-up questions with pronouns
("why did that happen?") resolve correctly against conversation history
before retrieval runs. See chains/query_rewriter.py for details.
"""

from app.agents.state import AgentState
from app.retrieval.retriever import retrieve_context
from app.chains.query_rewriter import rewrite_query


def retrieve_node(state: AgentState) -> AgentState:
    search_query = rewrite_query(state["question"], state["session_id"])
    context, documents = retrieve_context(search_query)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "sheet": doc.metadata.get("sheet"),
        }
        for doc in documents
    ]

    return {**state, "context": context, "sources": sources}