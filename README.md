# RAG Chatbot – Tsinghua University Knowledge Base

This project is a Retrieval-Augmented Generation (RAG) system that crawls, processes, and enables question-answering over the Tsinghua School of Software website.

The system ingests website content, builds a searchable vector database, and uses an LLM to generate answers with citations from the original source pages.

---

## Project Objective

Build a production-style RAG chatbot that can:
- Crawl a full website corpus (~850 pages)
- Extract and structure article content
- Support semantic search over documents
- Generate grounded answers using retrieved context
- Provide citation-backed responses

---

## Deliverables

### Phase 1
- Website crawler for full-site data extraction
- Filtering and extraction of article pages (~650 articles)
- Structured JSON data format with:
  - URL
  - Title
  - Content
  - Date (if available)
  - Images (optional metadata)

### Phase 2
- Text chunking strategy for RAG ingestion
- Embedding generation pipeline
- Vector database setup (FAISS)
- Retrieval system implementation

### Phase 3
- LLM-based response generation
- Citation-based answer formatting
- FastAPI backend for chatbot API
- Authentication layer
- Simple web chat UI (Streamlit or similar)
- Deployment (Vercel / Railway / VPS)

---

## System Architecture (Planned)

Crawler → Ingestion → Chunking → Embeddings → Vector Store (FAISS) → Retrieval → LLM → Chat UI

---

## Tech Stack

- Python 3.11
- BeautifulSoup (web crawling)
- Requests (HTTP client)
- Sentence Transformers (embeddings)
- FAISS (vector similarity search)
- FastAPI (backend API)
- Streamlit (chat UI)
- dotenv (configuration management)

---

## How to Run (WIP)

> Instructions will be updated as system components are completed.

---

## Data Source

- Tsinghua School of Software website
- Full-site crawl respecting robots.txt and rate limiting (1–2 req/sec)

---

## Notes

- This project is being built incrementally as part of a technical assessment.
- The RAG pipeline is being implemented from scratch (no full framework wrappers used).
- All crawling respects ethical scraping practices.