"""
Query rewriting for conversational retrieval.

Problem this solves: retrieval runs similarity search on the raw user question.
A follow-up like "why did that happen?" has weak/no semantic similarity to the
actual topic being discussed, because the pronoun ("that") is only resolvable
using conversation history — and history was never being passed into retrieval,
only into generation. This caused follow-up questions to retrieve irrelevant
or empty context even though the model "knew" what was being discussed.

Fix: before retrieval, if conversation history exists, ask the LLM to rewrite
the question into a standalone form that doesn't depend on prior context
(e.g., "why did that happen?" -> "why did the RNN perform worse than the LSTM?").
Retrieval then runs on the rewritten query. Generation still receives the
ORIGINAL question (not the rewritten one) so the answer responds to what the
user actually typed, not a machine paraphrase.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.chains.memory import get_history


REWRITE_PROMPT = """Given the conversation history and a new question, rewrite \
the new question into a standalone question that can be understood WITHOUT \
the history — resolve any pronouns or implicit references \
(e.g., "it", "that", "the second one", "why did it happen") into their \
actual subject.

If the new question is already standalone and doesn't depend on the history, \
return it unchanged.

Respond with ONLY the rewritten question — no preamble, no quotes, no explanation.

Conversation history:
{history}

New question: {question}

Standalone question:"""


def _format_history_for_rewrite(session_id: str) -> str:
    """
    Formats raw history messages for the rewrite prompt. Kept separate from
    chains/memory.py's format_history_for_prompt() since this only needs the
    last couple of turns — a full transcript isn't necessary just to resolve
    a pronoun, and keeping it short keeps the rewrite call fast.
    """
    history = get_history(session_id)
    if not history:
        return ""

    # last 4 messages = last 2 turns, enough to resolve most follow-ups
    recent = history[-4:]
    lines = []
    for msg in recent:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def rewrite_query(question: str, session_id: str) -> str:
    """
    Rewrites a question into a standalone form using recent conversation
    history. Returns the original question unchanged if there's no history
    yet (first turn in a session) — this adds zero extra latency/cost on
    the common single-turn case.
    """
    history_text = _format_history_for_rewrite(session_id)

    if not history_text:
        return question

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    rewritten = chain.invoke({"history": history_text, "question": question}).strip()

    # Safety net: if the rewrite comes back empty or absurdly long (model
    # misbehaving), fall back to the original question rather than breaking
    # retrieval entirely.
    if not rewritten or len(rewritten) > 500:
        return question

    return rewritten