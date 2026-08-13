"""
Retrieval pipeline.
Wraps the Chroma vector store as a LangChain retriever and formats
retrieved chunks into a single context string for prompting.
"""

from langchain_core.documents import Document

from app.vectorstore.chroma_store import get_vectorstore
from app.config import settings


def get_retriever(k: int | None = None):
    """
    Returns a LangChain retriever object backed by the Chroma vector store.
    """
    vectorstore = get_vectorstore()
    top_k = k or settings.top_k_results
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def format_context(documents: list[Document]) -> str:
    """
    Formats retrieved documents into a single context block for the prompt,
    with source attribution so the model can (and should) cite sources.
    """
    if not documents:
        return "No relevant context was found in the knowledge base."

    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        extra = ""
        if "page" in doc.metadata:
            extra = f", page {doc.metadata['page']}"
        elif "sheet" in doc.metadata:
            extra = f", sheet '{doc.metadata['sheet']}'"

        blocks.append(f"[Source {i}: {source}{extra}]\n{doc.page_content}")

    return "\n\n---\n\n".join(blocks)


def retrieve_context(query: str, k: int | None = None) -> tuple[str, list[Document]]:
    """
    Retrieve relevant chunks for a query and return both the formatted
    context string and the raw Document list (for citation metadata in the API response).
    """
    retriever = get_retriever(k=k)
    documents = retriever.invoke(query)
    context = format_context(documents)
    return context, documents