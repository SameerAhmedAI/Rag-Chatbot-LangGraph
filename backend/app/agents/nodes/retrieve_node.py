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
from app.agents.nodes.base_node import AgentNode
from app.retrieval.retriever import Retriever
from app.chains.query_rewriter import QueryRewriter


class RetrieveNode(AgentNode):
    """Retrieves context for the RAG path using a history-resolved query."""

    def __init__(self, retriever: Retriever | None = None, rewriter: QueryRewriter | None = None):
        self._retriever = retriever or Retriever()
        self._rewriter = rewriter or QueryRewriter()

    def run(self, state: AgentState) -> AgentState:
        search_query = self._rewriter.rewrite(state["question"], state["session_id"])
        context, documents = self._retriever.retrieve(search_query)

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "sheet": doc.metadata.get("sheet"),
            }
            for doc in documents
        ]

        return {**state, "context": context, "sources": sources}


# Module-level instance so graph.py can wire this in directly.
retrieve_node = RetrieveNode()