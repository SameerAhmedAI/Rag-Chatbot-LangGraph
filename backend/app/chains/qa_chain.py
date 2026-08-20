"""
Conversational RAG QA chain.
This is the Intermediate-level deliverable: given a user question + session
history, retrieve relevant context, build a grounded prompt, and call the LLM.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.retrieval.retriever import Retriever
from app.chains.memory import session_memory
from app.chains.query_rewriter import QueryRewriter


class QAChain:
    """
    Full conversational RAG pipeline:
    1. Rewrite the question into a standalone form using recent history
       (resolves pronouns/follow-ups like "why did that happen?" before
       retrieval runs — see query_rewriter.py for why this is necessary)
    2. Retrieve relevant chunks using the REWRITTEN question
    3. Format chat history for the generation prompt
    4. Build and invoke the prompt chain using the ORIGINAL question
       (so the answer responds to what the user actually typed)
    5. Save the ORIGINAL turn to memory
    6. Return the answer + source metadata
    """

    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions strictly \
based on the provided context retrieved from the user's documents.

Rules:
- Only answer using information found in the context below. \
Do not use outside knowledge.
- If the answer is not present in the context, say so clearly \
instead of guessing.
- Cite which source(s) you used when relevant (e.g., "According to Source 1...").
- Keep answers concise and directly relevant to the question.
- Use the conversation history to resolve follow-up questions \
(e.g., "what about the second one?").

Conversation history:
{history}

Retrieved context:
{context}
"""

    def __init__(self, retriever: Retriever | None = None, rewriter: QueryRewriter | None = None):
        self._retriever = retriever or Retriever()
        self._rewriter = rewriter or QueryRewriter()
        self._chain = None  # built lazily, see _get_chain()

    def _get_llm(self) -> ChatGroq:
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0.2,
        )

    def _get_chain(self):
        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", "{question}"),
                ]
            )
            self._chain = prompt | self._get_llm() | StrOutputParser()
        return self._chain

    def answer(self, question: str, session_id: str = "default") -> dict:
        search_query = self._rewriter.rewrite(question, session_id)
        context, documents = self._retriever.retrieve(search_query)
        history = session_memory.format_history_for_prompt(session_id)

        chain = self._get_chain()
        answer_text = chain.invoke(
            {
                "question": question,
                "context": context,
                "history": history,
            }
        )

        session_memory.add_turn(session_id, question, answer_text)

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "sheet": doc.metadata.get("sheet"),
            }
            for doc in documents
        ]

        return {"answer": answer_text, "sources": sources, "rewritten_query": search_query}


# Module-level convenience wrapper so existing call sites
# (`answer_question(question, session_id)`) don't all need to change
# in the same commit.
_default_qa_chain = QAChain()


def answer_question(question: str, session_id: str = "default") -> dict:
    return _default_qa_chain.answer(question, session_id)