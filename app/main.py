from fastapi import FastAPI
from pydantic import BaseModel

from scripts.query import retrieve

app = FastAPI(title="RAG Chatbot API")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    query = req.message

    results = retrieve(query)

    return {
        "query": query,
        "results": results
    }