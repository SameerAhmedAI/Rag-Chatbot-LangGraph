Engineering Knowledge Base Assistant — RAG Chatbot with LangGraph

Intern Task 3 (Intermediate + Advanced Level) — a document-grounded conversational
AI chatbot built with Retrieval-Augmented Generation (RAG), LangChain, and
ChromaDB, extended with a LangGraph multi-agent workflow.

Author: Sameer Ahmed — AI Engineering Intern
Repository: https://github.com/SameerAhmedAI/Rag-Chatbot-LangGraph

1. Project Overview

This project implements a chatbot for an internal engineering team's knowledge
base — project plans, incident postmortems, budget records, and API design
guidelines — answering questions grounded strictly in that document set rather
than the LLM's general training knowledge. It was built in two tiers, matching
the internship task's Intermediate and Advanced requirements:

Intermediate Level: a conversational RAG pipeline — document ingestion,
embedding, vector indexing, retrieval, prompt engineering, and multi-turn
conversational QA — via LangChain and ChromaDB.
Advanced Level: the same retrieval backend, additionally routed through
a LangGraph multi-agent workflow with conditional routing and a
self-critique/refinement step.

The system is architected around a small fictional engineering organization
("Falcon Horizon"), with documents that genuinely reference each other — an
incident postmortem cites the corpus-scale risk flagged in the project plan, a
budget sheet reflects tooling changes made in response to that incident, and
an API design guideline references the same incident's action items. This lets
the demonstration (Section 8) exercise real cross-document reasoning rather
than answering isolated questions against unrelated files.

A full walkthrough with real request/response evidence is available in
docs/demonstration.md.

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
Demonstrate the system against a coherent, focused case study rather than
arbitrary unrelated uploads, so cross-document reasoning is meaningfully
testable.

3. System Architecture

The backend is organized around object-oriented design patterns rather than
loose procedural functions:

Ingestion (Strategy pattern): a DocumentLoader abstract base class with
four concrete strategies (PDFLoader, DocxLoader, TxtLoader,
ExcelLoader), selected at runtime by LoaderFactory based on file
extension. Adding a new supported format means writing one new class and
registering it — no other code changes.
Vector storage (Repository pattern): VectorStoreRepository wraps all
ChromaDB persistence, chunking, and querying behind a single interface.
Retrieval and chains (Service classes): Retriever, QAChain,
QueryRewriter, and SessionMemory are each a class with a defined
interface, rather than free functions operating on a shared global dict.
SessionMemory in particular replaced a module-level mutable dict that
had no encapsulation — a real bug risk under concurrent requests, not
just a style issue.
Agent nodes (Strategy pattern): each LangGraph node (router, retrieve,
generate, critique) implements a shared AgentNode interface, making
each node an interchangeable, independently testable unit.

Upload flows through the Loader Factory (routing by file extension to a
PDF/DOCX/TXT/Excel loader), then chunking, embedding, and ChromaDB indexing.
A user question flows through the router (which now considers recent
conversation history — see Challenge 5 — to correctly classify short
follow-ups), then either the LangChain QA chain (/chat, Intermediate) or the
LangGraph agent (/agent-chat, Advanced). The RAG path additionally runs a
query rewriter (resolving follow-ups using session history) before similarity
search against ChromaDB, formats retrieved context with source citations,
calls the Groq LLM, and updates session memory before returning the answer.

Full diagram source files (Mermaid format, view/export at mermaid.live):

docs/architecture_file.mmd — end-to-end system architecture
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
conditional edge. Uses recent conversation history to correctly classify
short follow-ups that carry no topic signal on their own (see Challenge 5).
retrieve — pulls top-k relevant chunks from ChromaDB, using a
rewritten (history-resolved) version of the question (RAG path only).
generate — produces a draft answer grounded in retrieved context, or
a plain conversational reply on the general path.
critique — self-checks the draft against retrieved context and
rewrites it if it finds unsupported claims, before returning the final
answer.

6. Methodology
Ingestion: a format-specific loader strategy per file type (PDF via
pypdf, DOCX via python-docx, TXT via direct read, Excel via pandas),
selected at runtime by LoaderFactory so the rest of the application is
format-agnostic. See Section 3.
Chunking: RecursiveCharacterTextSplitter (1000 characters, 150
character overlap), splitting on paragraph, sentence, and word boundaries
in that priority order to preserve semantic coherence.
Embedding: sentence-transformers (all-MiniLM-L6-v2), run locally
on CPU, no external embedding API cost or rate limit.
Vector storage: ChromaDB via VectorStoreRepository, persisted to
disk so the index survives server restarts.
Retrieval: top-k similarity search (k=8, tuned up from an initial
default of 4 — see Challenge 4), with retrieved chunks formatted alongside
source filename, page, and sheet metadata for citation in responses.
Query rewriting: before retrieval, if conversation history exists,
the raw question is rewritten into a standalone form that resolves
pronouns and implicit references (e.g., "why did that happen?" becomes a
fully-specified question). See Challenge 1.
Agent routing: the LangGraph router additionally uses recent
conversation history (not just the raw question) to classify rag vs.
general, so short context-dependent follow-ups route correctly. See
Challenge 5.
Conversational memory: a per-session_id history managed by the
SessionMemory class, capped at the most recent 12 messages (6 turns),
injected into the generation prompt.
Agent workflow: LangGraph StateGraph with conditional routing and a
self-critique and refinement loop, described in Section 5.

7. Technology Stack
Layer	Technology
API	FastAPI
LLM	Groq (GPT-OSS-120B)
Orchestration	LangChain, LangGraph
Vector DB	ChromaDB
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Document parsing	pypdf, python-docx, openpyxl / pandas
Testing	pytest

Note: this project originally used Groq's llama-3.3-70b-versatile.
Groq decommissioned that model on August 16, 2026; the project was
migrated to openai/gpt-oss-120b (Groq's recommended replacement) on
August 21, 2026. See Section 10, Challenge 6.

8. Demonstration

A full walkthrough with real screenshots — server health check, live document
upload, single-document retrieval, cross-document reasoning, conversational
follow-up handling, and agent routing — is available in
docs/demonstration.md. It also documents the two refinements made during
testing (Challenges 4 and 5 below) with real before/after responses shown
side by side.

9. Experimental Results

Manual end-to-end verification was performed against the five-document
case-study corpus (Falcon Horizon project plan, DL-RAG-Toolkit technical
report, Q3 incident postmortem, engineering budget/headcount sheet, and
internal API design guidelines), via both /chat and /agent-chat.

Single-document retrieval: verified correct and grounded across all
five documents (e.g., "What deep learning models are covered in the
DL-RAG-Toolkit report?" correctly listed all five models with source
citation; "What is the rate limit for /chat and /agent-chat?" correctly
quoted the API guideline's rate-limiting section).
Cross-document reasoning: "What caused the Q3 latency incident, and
how does it relate to the corpus size challenges mentioned in the Falcon
Horizon plan?" was correctly answered by synthesizing the incident
postmortem's root cause (an HNSW ef_construction configuration change)
with a scalability risk flagged in a separate, unrelated project-planning
document — the model correctly identified that the incident exposed
exactly the risk the plan had warned about.
Follow-up query (pronoun resolution): "What was the compute budget
variance in Q2 2026?" followed immediately by "Why did that happen?" in
the same session was correctly resolved, returning the specific cause
(increased indexing volume) with correct source citation. See Challenge 1
for the underlying fix and Challenge 5 for a related agent-routing fix.
Multi-format ingestion: verified working for .txt, .pdf, .docx,
and .xlsx uploads, including a live upload captured in the demonstration.
LangGraph routing: verified that document-specific questions and
context-dependent follow-ups route to "rag", and general conversational
input (e.g., "Hello, how are you?") routes to "general" with retrieval
correctly skipped, with the critique node only running on the RAG path
as designed.

10. Challenges and Solutions

Challenge 1: Follow-up questions failed to retrieve relevant context.
Initial testing showed that a follow-up question like "why did that
happen?", sent immediately after a grounded question in the same session,
failed to retrieve relevant context. Root cause: retrieval ran a similarity
search on the raw question text, before any conversation history was
considered, so a pronoun-only follow-up had weak semantic similarity to
the actual topic being discussed, even though the LLM's generation step had
access to history.

Fix: added a query-rewriting step (chains/query_rewriter.py, now the
QueryRewriter class) that runs before retrieval whenever session
history exists. It sends the recent conversation history plus the new
question to the LLM with an instruction to produce a standalone,
pronoun-resolved question, and retrieval runs on that rewritten query
instead of the raw one. Generation still receives the user's original
question, not the rewritten version, so the answer responds to what the
user actually typed. Verified fixed via repeated end-to-end testing
(Section 9).

Challenge 2: Environment file (.env) path confusion during setup.
Early setup attempts created .env at the project root instead of inside
backend/, where config.py (via pydantic-settings) actually resolves
it relative to the working directory uvicorn is run from. Resolved by
confirming backend/.env is the correct, required location and documenting
it explicitly in the Setup section below.

Challenge 3: Excel ingestion needed a hand-rolled loader. No
off-the-shelf LangChain Excel loader reliably handles arbitrary multi-sheet
layouts. Solved by hand-rolling the loader with pandas (now the
ExcelLoader strategy class), iterating every sheet and batching rows
into readable chunks, rather than depending on a generic loader.

Challenge 4: Compound cross-document questions can under-retrieve one
sub-topic. Testing showed that single-concept questions (e.g., "what
does the API guideline say about rate limiting?") retrieve correctly
regardless of which document they target. However, a compound question
asking about two distinct topics across two different documents in one
query — "What tooling budget changes happened after the Q3 incident, and
what does the API design guideline document say about handling similar
issues in the future?" — initially returned the correct answer for the
budget half but reported no relevant content for the API guidelines half,
even though that document was confirmed to be fully indexed and
independently retrievable (verified by asking a direct single-document
question about the same file). Root cause: a single embedding vector must
represent both parts of a compound question simultaneously, and the two
sub-topics do not weight evenly in that vector, so the weaker-matching
half can fall outside the top-k retrieval window.

Fix (partial): increased top_k from 4 to 8, which improved but did not
fully resolve compound-question retrieval. This is a known limitation of
naive single-hop RAG architectures — a single retrieval pass cannot
reliably serve two independent information needs from one query vector.
See docs/research_manual.md for a fuller comparison against Graph RAG,
which handles multi-hop, multi-entity questions more reliably. A proper
fix (query decomposition) is listed under Future Improvements.

Challenge 5: LangGraph router misclassified context-dependent
follow-ups. The router originally classified each question using only
the raw question text, with no conversation history — unlike the query
rewriter (Challenge 1), which already had history access. A short,
context-dependent follow-up like "why did that happen?" carries no topic
signal on its own, so the router would default to "general" and skip
retrieval, even when the question was clearly continuing a prior
document-grounded exchange.

Fix: RouterNode now receives the last two conversation turns