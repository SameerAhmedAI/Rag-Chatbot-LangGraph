"""
Generate node.
Produces a draft answer. Behaves differently depending on the route:
- "rag": grounded answer using retrieved context + chat history
- "general": lightweight conversational reply, no retrieval needed
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.agents.state import AgentState
from app.agents.nodes.base_node import AgentNode
from app.chains.memory import session_memory


class GenerateNode(AgentNode):
    """Generates a draft answer, grounded (RAG path) or conversational (general path)."""

    RAG_PROMPT = """You are a helpful AI assistant. Answer the question using ONLY \
the retrieved context below. If the answer isn't in the context, say so clearly.

Conversation history:
{history}

Retrieved context:
{context}

Question: {question}
"""

    GENERAL_PROMPT = """You are a helpful, friendly AI assistant for a document \
Q&A chatbot. Respond naturally and briefly to the user's message.

Conversation history:
{history}

Message: {question}
"""

    def _get_llm(self) -> ChatGroq:
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0.3,
        )

    def run(self, state: AgentState) -> AgentState:
        llm = self._get_llm()
        history = session_memory.format_history_for_prompt(state["session_id"])

        if state.get("route") == "rag":
            prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT)
            chain = prompt | llm | StrOutputParser()
            draft = chain.invoke(
                {
                    "question": state["question"],
                    "context": state.get("context", ""),
                    "history": history,
                }
            )
        else:
            prompt = ChatPromptTemplate.from_template(self.GENERAL_PROMPT)
            chain = prompt | llm | StrOutputParser()
            draft = chain.invoke({"question": state["question"], "history": history})

        return {**state, "draft_answer": draft}


# Module-level instance so graph.py can wire this in directly.
generate_node = GenerateNode()