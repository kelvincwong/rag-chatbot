from fastapi import FastAPI, Depends
from pydantic import BaseModel

from scripts.query import retrieve
from app.llm import generate_answer
from app.auth import authenticate
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RAG Chatbot API")
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest, user: str = Depends(authenticate)):
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

app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")
