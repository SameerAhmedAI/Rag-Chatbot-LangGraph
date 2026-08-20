"""
Retrieval service.
Wraps the vector store repository, runs similarity search, and formats
retrieved chunks into a single context string for prompting.
"""

from langchain_core.documents import Document

from app.vectorstore.chroma_store import VectorStoreRepository
from app.config import settings


class Retriever:
    """
    Retrieval pipeline: given a query, get relevant chunks from the
    vector store and format them for prompt injection + citation.
    """

    def __init__(self, k: int | None = None):
        self.k = k or settings.top_k_results

    def get_langchain_retriever(self):
        """Returns a LangChain retriever object backed by the Chroma vector store."""
        return VectorStoreRepository.as_retriever(k=self.k)

    @staticmethod
    def format_context(documents: list[Document]) -> str:
        """
        Formats retrieved documents into a single context block for the
        prompt, with source attribution so the model can (and should)
        cite sources.
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

    def retrieve(self, query: str) -> tuple[str, list[Document]]:
        """
        Retrieve relevant chunks for a query and return both the formatted
        context string and the raw Document list (for citation metadata
        in the API response).
        """
        retriever = self.get_langchain_retriever()
        documents = retriever.invoke(query)
        context = self.format_context(documents)
        return context, documents


# Module-level convenience wrapper so existing call sites
# (`retrieve_context(query)`) don't all need to change in the same commit.
def retrieve_context(query: str, k: int | None = None) -> tuple[str, list[Document]]:
    return Retriever(k=k).retrieve(query)


def format_context(documents: list[Document]) -> str:
    return Retriever.format_context(documents)