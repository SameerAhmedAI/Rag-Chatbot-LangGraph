"""
Router node.
Decides whether the incoming question needs document retrieval (RAG path)
or can be answered as general conversation (e.g., "hi", "thanks", "who are you").
This demonstrates a genuine conditional edge in the graph, not a pass-through.

Bug fix (found during Task 3 demo testing): the router originally classified
using ONLY the raw question text, with no conversation history. Short,
context-dependent follow-ups like "why did that happen?" have no inherent
topic signal on their own, so the classifier would default to "general" —
skipping retrieval entirely and causing the generation step to answer from
no context, which produced confident but fabricated (hallucinated) answers.

Fix: the router prompt now receives recent conversation history, the same
way query_rewriter.py already does, so it can recognize that a short
follow-up is continuing a document-grounded conversation and route to "rag"
accordingly.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.agents.state import AgentState
from app.agents.nodes.base_node import AgentNode
from app.chains.memory import session_memory


class RouterNode(AgentNode):
    """Classifies the question as needing document retrieval ('rag') or not ('general')."""

    ROUTER_PROMPT = """Classify the user's LATEST message into exactly one category, \
using the conversation history for context when the message alone is ambiguous.

- "rag": the message asks a question that likely requires looking up \
information from documents (facts, data, specifics, "what does X say about Y"), \
OR it is a short follow-up (e.g. "why did that happen?", "what about the other one?") \
that continues a document-grounded conversation already visible in the history below.
- "general": the message is a greeting, small talk, thanks, or a question \
about the assistant itself that doesn't need document lookup, AND is not a \
follow-up to a document-grounded exchange in the history.

If the history shows the assistant previously answered using retrieved context, \
treat short pronoun-based follow-ups ("why", "how come", "what about that") as "rag" \
by default, since they almost always continue that same document-grounded thread.

Respond with ONLY the single word "rag" or "general" — nothing else.

Conversation history:
{history}

Latest message: {question}
"""

    # last 4 messages = last 2 turns, enough to tell whether a short follow-up
    # is continuing a document-grounded thread, without needing the full transcript
    RECENT_MESSAGE_WINDOW = 4

    def _format_history(self, session_id: str) -> str:
        history = session_memory.get_history(session_id)
        if not history:
            return "No prior conversation."

        recent = history[-self.RECENT_MESSAGE_WINDOW:]
        lines = []
        for msg in recent:
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def run(self, state: AgentState) -> AgentState:
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_template(self.ROUTER_PROMPT)
        chain = prompt | llm | StrOutputParser()

        history_text = self._format_history(state["session_id"])

        result = chain.invoke(
            {"question": state["question"], "history": history_text}
        ).strip().lower()
        route = "rag" if "rag" in result else "general"

        return {**state, "route": route}

    @staticmethod
    def route_decision(state: AgentState) -> str:
        """
        Conditional edge function: reads state['route'] and returns the
        name of the next node to visit.
        """
        return state.get("route", "rag")


# Module-level instances so graph.py can wire these in directly.
router_node = RouterNode()
route_decision = RouterNode.route_decision