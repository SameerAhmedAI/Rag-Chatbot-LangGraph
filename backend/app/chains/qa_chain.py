"""
Conversational RAG QA chain.
This is the Intermediate-level deliverable: given a user question + session
history, retrieve relevant context, build a grounded prompt, and call the LLM.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.retrieval.retriever import retrieve_context
from app.chains.memory import format_history_for_prompt, add_turn
from app.chains.query_rewriter import rewrite_query


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


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0.2,
    )


def build_qa_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = get_llm()
    return prompt | llm | StrOutputParser()


def answer_question(question: str, session_id: str = "default") -> dict:
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
    search_query = rewrite_query(question, session_id)
    context, documents = retrieve_context(search_query)
    history = format_history_for_prompt(session_id)

    chain = build_qa_chain()
    answer = chain.invoke(
        {
            "question": question,
            "context": context,
            "history": history,
        }
    )

    add_turn(session_id, question, answer)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "sheet": doc.metadata.get("sheet"),
        }
        for doc in documents
    ]

    return {"answer": answer, "sources": sources, "rewritten_query": search_query}