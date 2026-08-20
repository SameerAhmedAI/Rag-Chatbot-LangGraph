"""
Router node.
Decides whether the incoming question needs document retrieval (RAG path)
or can be answered as general conversation (e.g., "hi", "thanks", "who are you").
This demonstrates a genuine conditional edge in the graph, not a pass-through.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.agents.state import AgentState
from app.agents.nodes.base_node import AgentNode


class RouterNode(AgentNode):
    """Classifies the question as needing document retrieval ('rag') or not ('general')."""

    ROUTER_PROMPT = """Classify the user's message into exactly one category:

- "rag": the message asks a question that likely requires looking up \
information from documents (facts, data, specifics, "what does X say about Y").
- "general": the message is a greeting, small talk, thanks, or a question \
about the assistant itself that doesn't need document lookup.

Respond with ONLY the single word "rag" or "general" — nothing else.

Message: {question}
"""

    def run(self, state: AgentState) -> AgentState:
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_template(self.ROUTER_PROMPT)
        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({"question": state["question"]}).strip().lower()
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