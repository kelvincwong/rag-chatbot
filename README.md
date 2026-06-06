# RAG Chatbot – Tsinghua University Knowledge Base

This project is a Retrieval-Augmented Generation (RAG) system that crawls, processes, and enables question-answering over the Tsinghua School of Software website.

The system ingests website content, builds a searchable vector database, and uses an LLM to generate grounded answers with citations from the original source pages.

---

## Key Features

- Full-site crawler (~850 pages)
- Semantic search using FAISS
- SentenceTransformer embeddings
- Bilingual support (Chinese + English)
- LLM-based answer generation
- Source-cited responses
- Authentication-protected chatbot
- Web-based chat UI

---

## System Architecture

Crawler → Ingestion → Chunking → Embeddings → Vector Store (FAISS) → Retrieval → LLM → Answer + Sources → UI

---

## Tech Stack

- Python 3.11
- FastAPI (backend API)
- BeautifulSoup (web crawling)
- Requests (HTTP client)
- Sentence Transformers (embeddings)
- FAISS (vector search)
- HTML / JavaScript (frontend UI)

---

## How to Run

### Install dependencies

pip install -r requirements.txt

### Start the server

uvicorn app.main:app --reload

### Open the application

- Chat UI: http://127.0.0.1:8000/ui  
- API Docs: http://127.0.0.1:8000/docs  

---

## API Endpoints

### POST /chat

Send a user query to the RAG system.

#### Request

{
  "message": "What is the School of Software?"
}

#### Response

{
  "query": "What is the School of Software?",
  "answer": "The School of Software at Tsinghua University is ...",
  "sources": [
    {
      "title": "Official Article Title",
      "url": "https://www.thss.tsinghua.edu.cn/..."
    }
  ]
}

---

## Authentication

The chatbot is protected using simple HTTP Basic Authentication.

### Demo credentials

username: admin  
password: admin123  

---

## Data Source

- Tsinghua School of Software website: https://www.thss.tsinghua.edu.cn
- Full-site crawl with polite rate limiting (1–2 requests/second)
- Structured extraction of article pages (news, announcements, faculty info)
- Missing or unverified robots.txt handling

---

## System Notes

- Built from scratch without using full RAG frameworks
- All crawling respects ethical scraping practices
- LLM outputs are grounded using retrieved context from FAISS index
- Sources are returned separately and rendered in the UI

---

## Future Improvements

- Add reranking model for better retrieval accuracy
- Improve UI (React-based chat interface)
- Add streaming LLM responses
- Improve chunking strategy for long documents
- Add hybrid search (BM25 + vector search)
- Improve multilingual embedding performance