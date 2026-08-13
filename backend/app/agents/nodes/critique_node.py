"""
Critique node.
Self-checks the draft answer against the retrieved context to catch
unsupported claims (basic hallucination guard). Only runs for the RAG path —
general chit-chat doesn't need fact-checking against documents.

This is what makes the graph "agentic" rather than a linear chain: the model
evaluates its own output and can trigger a refinement pass.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.agents.state import AgentState


CRITIQUE_PROMPT = """You are reviewing an AI-generated answer for accuracy.

Context that was available:
{context}

Question: {question}
Draft answer: {draft_answer}

Does the draft answer make any claims NOT supported by the context? \
Respond with ONLY "valid" if the answer is well-grounded in the context, \
or "refine" if it includes unsupported claims or hedges awkwardly and \
should be tightened.
"""

REFINE_PROMPT = """Rewrite the following answer so that it strictly reflects \
only what the context supports, is concise, and clearly states if information \
is missing. Do not add new information.

Context:
{context}

Original answer: {draft_answer}

Rewritten answer:
"""


def critique_node(state: AgentState) -> AgentState:
    # Only critique RAG answers — general chit-chat skips this check
    if state.get("route") != "rag":
        return {**state, "final_answer": state.get("draft_answer", ""), "needs_refinement": False}

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model, temperature=0)

    critique_chain = ChatPromptTemplate.from_template(CRITIQUE_PROMPT) | llm | StrOutputParser()
    verdict = critique_chain.invoke(
        {
            "context": state.get("context", ""),
            "question": state["question"],
            "draft_answer": state.get("draft_answer", ""),
        }
    ).strip().lower()

    needs_refinement = "refine" in verdict

    if not needs_refinement:
        return {**state, "final_answer": state.get("draft_answer", ""), "needs_refinement": False}

    refine_chain = ChatPromptTemplate.from_template(REFINE_PROMPT) | llm | StrOutputParser()
    refined = refine_chain.invoke(
        {
            "context": state.get("context", ""),
            "draft_answer": state.get("draft_answer", ""),
        }
    )

    return {**state, "final_answer": refined, "needs_refinement": True}