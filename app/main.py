from fastapi import FastAPI
from pydantic import BaseModel

from scripts.query import retrieve
from app.llm import generate_answer

app = FastAPI(title="RAG Chatbot API")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    query = req.message

    results = retrieve(query)

    answer = generate_answer(query, results)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {"title": r["title"], "url": r["url"]}
            for r in results
        ]
    }