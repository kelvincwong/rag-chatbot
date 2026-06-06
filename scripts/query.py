import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "app/data/processed/faiss.index"
META_FILE = "app/data/processed/meta.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_metadata():
    data = []
    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# def detect_intent(query):
#     if any(k in query for k in ["历史", "成立", "发展", "沿革", "简介", "概况"]):
#         return "intro"
#     if any(k in query for k in ["研究", "论文", "实验室"]):
#         return "research"
#     if any(k in query for k in ["新闻", "活动"]):
#         return "news"
#     return "general"

# def rule_score(doc, intent):
#     score = 0

#     url = doc.get("url", "")
#     entity = doc.get("entity", "")
#     title = doc.get("title", "")
#     text = doc.get("text", "")
#     section = doc.get("section_hint", "")

#     # --- intro boost ---
#     if intent == "intro":
#         if doc.get("entity") and "学院" in doc.get("entity"):
#             score += 1.5

#     if "xygk" in url:
#         score += 3

#     # --- news penalty ---
#     if "info" in url or "news" in url:
#         score -= 1

#     # --- intent-specific logic ---
#     if intent == "intro":
#         if "研究中心" in title or "联合" in title:
#             score -= 2

#         if any(k in title for k in ["简介", "概况", "历史", "沿革"]):
#             score += 2

#     if intent == "intro":
#         if any(k in section for k in ["历史", "简介", "概况", "沿革"]):
#             score += 2

#     return score

# def type_boost(doc):
#     url = doc.get("url", "")
#     title = doc.get("title", "")
#     section = doc.get("section_hint", "")

#     boost = 0

#     # STRONG boost for overview pages
#     if "xygk" in url or "学院简介" in title:
#         boost += 2.0

#     # MEDIUM boost for institutional pages
#     if "学院" in title and "软件" in title:
#         boost += 1.0

#     # PENALIZE research/lab pages
#     if "研究所" in title or "实验室" in title:
#         boost -= 1.5

#     if "info" in url:
#         boost -= 0.5

#     if "历史" in section or "沿革" in section:
#         boost += 1.5

#     return boost

# def rerank_results(results, intent):
#     for r in results:
#         boost = rule_score(r, intent)
#         type_b = type_boost(r)

#         entity = r.get("entity", "")

#         entity_boost = 0
#         if entity:
#             if entity in (r.get("text", "") + r.get("title", "")):
#                 entity_boost += 1.0
#             else:
#                 entity_boost += 0.3  # weak global boost

#         r["final_score"] = r["score"] + boost + type_b + entity_boost

#     results.sort(key=lambda x: x["final_score"], reverse=True)
#     return results

def deduplicate(results):
    seen = set()
    cleaned = []

    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        cleaned.append(r)

    return cleaned

def main():
    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(INDEX_FILE)
    metadata = load_metadata()

    print("RAG Query System Ready. Press Enter to exit.")
    
    while True:
        query = input("\nAsk: ")
        if not query:
            break

        query_vec = model.encode([query], normalize_embeddings=True)

        D, I = index.search(np.array(query_vec), k=5)

        results = []

        for idx, score in zip(I[0], D[0]):

            if idx < 0 or idx >= len(metadata):
                continue

            item = metadata[idx]

            url = item.get("url", "")
            title = item.get("title", "")

            results.append({
                "url": url,
                "title": title,
                "date": item.get("date"),
                "text": item.get("text"),
                "score": float(score)
            })

        # ---------------------------
        # Rerank pipeline
        # ---------------------------
        results = [r for r in results if len(r["text"]) > 80]

        # intent = detect_intent(query)

        # results = rerank_results(results, intent)
        results = deduplicate(results)

        results = results[:5]

        # ---------------------------
        # Output
        # ---------------------------
        print("\nTop results:\n")

        for rank, r in enumerate(results):
            print(f"\n[{rank+1}] {r['title']}")
            print(f"URL: {r['url']}")
            print(f"DATE: {r['date']}")
            print(f"SCORE: {r['score']}")
            # print(f"FINAL SCORE: {r['final_score']}")
            print(f"TEXT: {r['text'][:200]}...")


if __name__ == "__main__":
    main()