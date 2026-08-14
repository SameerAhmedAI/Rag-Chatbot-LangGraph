RAG Chatbot with LangGraph

Intern Task 3 (Intermediate + Advanced Level) — a document-grounded conversational
AI chatbot built with Retrieval-Augmented Generation (RAG), LangChain, and
ChromaDB, extended with a LangGraph multi-agent workflow.

Author: Sameer Ahmed — AI Engineering Intern
Repository: https://github.com/SameerAhmedAI/rag-chatbot-langgraph

1. Project Overview

This project implements a chatbot that answers questions grounded strictly in
user-uploaded documents rather than the LLM's general training knowledge. It
was built in two tiers, matching the internship task's Intermediate and
Advanced requirements:

Intermediate Level: a conversational RAG pipeline — document ingestion,
embedding, vector indexing, retrieval, prompt engineering, and multi-turn
conversational QA — via LangChain and ChromaDB.
Advanced Level: the same retrieval backend, additionally routed through
a LangGraph multi-agent workflow with conditional routing and a
self-critique/refinement step.
2. Objectives
Support document ingestion across four formats: PDF, DOCX, TXT, and Excel.
Chunk and embed documents locally (no external embedding API cost).
Persist embeddings in a local vector database (ChromaDB).
Retrieve relevant context per query and ground LLM answers in that context.
Support multi-turn conversations, including follow-up questions that rely
on prior context (e.g., "why did that happen?").
Extend the pipeline with a LangGraph agent capable of routing between
document-grounded and general-conversation responses, and self-checking
its own answers against retrieved context before returning them.
3. System Architecture

Upload flows through the Loader Factory (routing by file extension to a
PDF/DOCX/TXT/Excel loader), then chunking, embedding, and ChromaDB indexing.
A user question flows through the query rewriter (resolving follow-ups using
session history), similarity search against ChromaDB, context formatting
with source citations, then either the LangChain QA chain (/chat,
Intermediate) or the LangGraph agent (/agent-chat, Advanced), both calling
the Groq LLM and updating session memory before returning the answer.

Full diagram source files (Mermaid format, view/export at
mermaid.live):

docs/architecture_diagram.mmd — end-to-end system architecture
docs/workflow_diagram.mmd — sequence diagram of upload and chat flows
docs/langgraph_state_diagram.mmd — LangGraph agent state machine
4. Flow Diagram

See docs/workflow_diagram.mmd. Summary: a document upload flows through the
ingestion, chunking, embedding, indexing pipeline once at upload time. A
chat query flows through history retrieval, query rewriting, similarity
search, context formatting, LLM generation, and memory update, on every turn.

5. State Diagram (LangGraph Agent)

See docs/langgraph_state_diagram.mmd. The advanced-level agent uses a
StateGraph with four nodes:

router — classifies the question as needing document retrieval
(rag) or general conversation (general) and branches via a
conditional edge.
retrieve — pulls top-k relevant chunks from ChromaDB, using a
rewritten (history-resolved) version of the question (RAG path only).
generate — produces a draft answer grounded in retrieved context, or
a plain conversational reply on the general path.
critique — self-checks the draft against retrieved context and
rewrites it if it finds unsupported claims, before returning the final
answer.
6. Methodology
Ingestion: a format-specific loader per file type (PDF via pypdf,
DOCX via python-docx, TXT via direct read, Excel via pandas), unified
behind a single loader_factory.py entry point so the rest of the
application is format-agnostic.
Chunking: RecursiveCharacterTextSplitter (1000 characters, 150
character overlap), splitting on paragraph, sentence, and word boundaries
in that priority order to preserve semantic coherence.
Embedding: sentence-transformers (all-MiniLM-L6-v2), run locally
on CPU, no external embedding API cost or rate limit.
Vector storage: ChromaDB, persisted to disk so the index survives
server restarts.
Retrieval: top-k similarity search (k=4 by default), with retrieved
chunks formatted alongside source filename, page, and sheet metadata for
citation in responses.
Query rewriting: before retrieval, if conversation history exists,
the raw question is rewritten into a standalone form that resolves
pronouns and implicit references (e.g., "why did that happen?" becomes a
fully-specified question). See Section 9 for why this was necessary.
Conversational memory: a per-session_id in-memory message history,
capped at the most recent 12 messages (6 turns), injected into the
generation prompt.
Agent workflow: LangGraph StateGraph with conditional routing and a
self-critique and refinement loop, described in Section 5.
7. Technology Stack
Layer	Technology
API	FastAPI
LLM	Groq (Llama 3.3 70B)
Orchestration	LangChain, LangGraph
Vector DB	ChromaDB
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Document parsing	pypdf, python-docx, openpyxl / pandas
Testing	pytest
8. Experimental Results

Manual end-to-end verification was performed using a real technical PDF
document (about 13 pages) as the test corpus, with the following results:

Grounded factual query: "What was the RNN's test accuracy and how
does it compare to the LSTM?" was correctly answered with the specific
figures from the document (0.2719 vs 0.5492, roughly 2x difference), with
correct source and page citation.
Follow-up query (pronoun resolution): "Why did that happen?" sent in
the same session immediately after the above initially failed (see
Section 9, Challenge 1), and was correctly resolved after the
query-rewriting fix was added, answering with the relevant explanation
(the LSTM's ability to retain long-term dependencies versus the RNN's
limitation).
Multi-format ingestion: verified working for .txt, .pdf, .docx,
and .xlsx uploads, each correctly chunked and indexed.
LangGraph routing: verified that document-specific questions route to
"rag" and general conversational input routes to "general", with the
critique node only running on the RAG path as designed.
9. Challenges and Solutions

Challenge 1: Follow-up questions failed to retrieve relevant context.
Initial testing showed that a follow-up question like "why did that
happen?", sent immediately after a grounded question in the same session,
failed to retrieve relevant context. Root cause: retrieval ran a similarity
search on the raw question text, before any conversation history was
considered, so a pronoun-only follow-up had weak semantic similarity to
the actual topic being discussed, even though the LLM's generation step had
access to history.

Fix: added a query-rewriting step (chains/query_rewriter.py) that runs
before retrieval whenever session history exists. It sends the recent
conversation history plus the new question to the LLM with an instruction
to produce a standalone, pronoun-resolved question, and retrieval runs on
that rewritten query instead of the raw one. Generation still receives the
user's original question, not the rewritten version, so the answer
responds to what the user actually typed. Verified fixed via repeated
end-to-end testing (Section 8).

Challenge 2: Environment file (.env) path confusion during setup.
Early setup attempts created .env at the project root instead of inside
backend/, where config.py (via pydantic-settings) actually resolves
it relative to the working directory uvicorn is run from. Resolved by
confirming backend/.env is the correct, required location and documenting
it explicitly in the Setup section below.

Challenge 3: Excel ingestion needed a hand-rolled loader. No
off-the-shelf LangChain Excel loader reliably handles arbitrary multi-sheet
layouts. Solved by hand-rolling the loader with pandas, iterating every
sheet and batching rows into readable chunks, rather than depending on a
generic loader.

10. Project Structure

rag-chatbot-langgraph/

backend/app/main.py, config.py
backend/app/ingestion/ (PDF/DOCX/TXT/Excel loaders + factory)
backend/app/embeddings/ (sentence-transformers wrapper)
backend/app/vectorstore/ (ChromaDB indexing + search)
backend/app/retrieval/ (retriever + context formatting)
backend/app/chains/ (LangChain QA chain, memory, query rewriter)
backend/app/agents/ (LangGraph state, nodes, graph)
backend/app/api/ (/upload, /chat, /agent-chat routes)
data/ (uploaded files + persisted vector db, gitignored)
docs/ (Mermaid diagrams + research manual)
tests/ (pytest tests)
requirements.txt
11. Setup

Run inside backend/:

cd backend
python -m venv venv
source venv/bin/activate (Windows: venv\Scripts\activate)
pip install -r requirements.txt

cp .env.example .env
Edit backend/.env (NOT the project root) and add your GROQ_API_KEY
(free at console.groq.com)

12. Running

cd backend
uvicorn app.main:app --reload --port 8000

API docs available at http://localhost:8000/docs

13. Endpoints
Method	Endpoint	Description
POST	/upload	Upload and index a document (PDF/DOCX/TXT/XLSX)
POST	/chat	Conversational RAG QA via LangChain (Intermediate)
POST	/agent-chat	Conversational RAG QA via LangGraph agent (Advanced)
GET	/health	Health check + indexed chunk count
14. Testing

pytest tests/ -v

15. Future Improvements
Persist chat memory to Redis/Postgres instead of in-process memory.
Add streaming responses (SSE) for both /chat and /agent-chat.
Add a re-ranking step after retrieval for higher precision on ambiguous
or ambiguously-related follow-up queries.
Extend LangGraph with a dedicated "clarify" node for ambiguous queries.
Add Graph RAG support for multi-hop reasoning over document relationships
(see docs/research_manual.md for a comparison against the current
Naive RAG approach used here).
16. References
LangChain docs: python.langchain.com
LangGraph docs: langchain-ai.github.io/langgraph
ChromaDB docs: docs.trychroma.com
Groq API docs: console.groq.com/docs
sentence-transformers docs: sbert.net
17. GitHub Repository

https://github.com/SameerAhmedAI/rag-chatbot-langgraph