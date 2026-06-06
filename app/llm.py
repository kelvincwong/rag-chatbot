from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

client = Groq(api_key=api_key)

def generate_answer(query, results):
    context = "\n\n".join([
        f"Title: {r['title']}\nURL: {r['url']}\nText: {r['text']}"
        for r in results[:5]
    ])

    prompt = f"""
You are a RAG assistant for Tsinghua School of Software.

Rules:
- Use ONLY provided context
- If not in context, say "not found in corpus"
- Do NOT hallucinate
- Always base answer on sources
- Do not copy text directly. Summarize and rephrase in natural language.
- Do not include citations, sources, or references in the answer.
- All sources will be provided separately by the system.

Constraints:
Do not copy text directly. Summarize and rephrase in natural language.

Context:
{context}

Question:
{query}

Return a clear answer with references.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Strict RAG assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content