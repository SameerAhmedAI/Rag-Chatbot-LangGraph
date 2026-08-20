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
from app.chains.memory import session_memory


class QueryRewriter:
    """Rewrites follow-up questions into standalone, retrieval-ready queries."""

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

    # last 4 messages = last 2 turns, enough to resolve most follow-ups without
    # needing a full transcript just to resolve a pronoun
    RECENT_MESSAGE_WINDOW = 4
    MAX_REWRITE_LENGTH = 500

    def __init__(self):
        self._chain = None  # built lazily, see _get_chain()

    def _get_chain(self):
        if self._chain is None:
            llm = ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.llm_model,
                temperature=0,
            )
            prompt = ChatPromptTemplate.from_template(self.REWRITE_PROMPT)
            self._chain = prompt | llm | StrOutputParser()
        return self._chain

    def _format_history_for_rewrite(self, session_id: str) -> str:
        history = session_memory.get_history(session_id)
        if not history:
            return ""

        recent = history[-self.RECENT_MESSAGE_WINDOW:]
        lines = []
        for msg in recent:
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def rewrite(self, question: str, session_id: str) -> str:
        """
        Rewrites a question into a standalone form using recent conversation
        history. Returns the original question unchanged if there's no history
        yet (first turn in a session) — this adds zero extra latency/cost on
        the common single-turn case.
        """
        history_text = self._format_history_for_rewrite(session_id)

        if not history_text:
            return question

        chain = self._get_chain()
        rewritten = chain.invoke({"history": history_text, "question": question}).strip()

        # Safety net: if the rewrite comes back empty or absurdly long (model
        # misbehaving), fall back to the original question rather than
        # breaking retrieval entirely.
        if not rewritten or len(rewritten) > self.MAX_REWRITE_LENGTH:
            return question

        return rewritten


# Module-level convenience wrapper so existing call sites
# (`rewrite_query(question, session_id)`) don't all need to change
# in the same commit.
_default_rewriter = QueryRewriter()


def rewrite_query(question: str, session_id: str) -> str:
    return _default_rewriter.rewrite(question, session_id)