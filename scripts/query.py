import json
import faiss
import numpy as np
#from sentence_transformers import SentenceTransformer

INDEX_FILE = "app/data/processed/faiss.index"
META_FILE = "app/data/processed/meta.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

model = None
index = None
metadata = None


def load_metadata():
    data = []
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def deduplicate(results):
    seen = set()
    cleaned = []

    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        cleaned.append(r)

    return cleaned

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
    return model


def get_index():
    global index
    if index is None:
        index = faiss.read_index(INDEX_FILE)
    return index


def get_metadata():
    global metadata
    if metadata is None:
        metadata = load_metadata()
    return metadata

# model = SentenceTransformer(MODEL_NAME)
# index = faiss.read_index(INDEX_FILE)
# metadata = load_metadata()

def retrieve(query: str, k: int = 5):
    model = get_model()
    index = get_index()
    metadata = get_metadata()

    query_vec = model.encode(query, normalize_embeddings=True)
    query_vec = np.array([query_vec])

    D, I = index.search(np.array(query_vec), k=k)

    results = []

    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        
        item = metadata[idx]

        results.append({
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "date": item.get("date", ""),
            "text": item.get("text", ""),
            "score": float(score)
        })

    results = [r for r in results if len(r["text"]) > 80]
    results = deduplicate(results)

    return results[:5]


#--------------------
#CLI code for testing
#--------------------
# def main():
#     model = SentenceTransformer(MODEL_NAME)

#     index = faiss.read_index(INDEX_FILE)
#     metadata = load_metadata()

#     print("RAG Query System Ready. Press Enter to exit.")
    
#     while True:
#         query = input("\nAsk: ")
#         if not query:
#             break

#         query_vec = model.encode([query], normalize_embeddings=True)

#         D, I = index.search(np.array(query_vec), k=5)

#         results = []

#         for idx, score in zip(I[0], D[0]):

#             if idx < 0 or idx >= len(metadata):
#                 continue

#             item = metadata[idx]

#             url = item.get("url", "")
#             title = item.get("title", "")

#             results.append({
#                 "url": url,
#                 "title": title,
#                 "date": item.get("date"),
#                 "text": item.get("text"),
#                 "score": float(score)
#             })

#         # ---------------------------
#         # Rerank pipeline
#         # ---------------------------
#         results = [r for r in results if len(r["text"]) > 80]

#         # intent = detect_intent(query)

#         # results = rerank_results(results, intent)
#         results = deduplicate(results)

#         results = results[:5]

#         # ---------------------------
#         # Output
#         # ---------------------------
#         print("\nTop results:\n")

#         for rank, r in enumerate(results):
#             print(f"\n[{rank+1}] {r['title']}")
#             print(f"URL: {r['url']}")
#             print(f"DATE: {r['date']}")
#             print(f"SCORE: {r['score']}")
#             # print(f"FINAL SCORE: {r['final_score']}")
#             print(f"TEXT: {r['text'][:200]}...")


# if __name__ == "__main__":
#     main()