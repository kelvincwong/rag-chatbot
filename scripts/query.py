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

def detect_intent(query):
    if any(k in query for k in ["是什么", "介绍", "概况", "简介"]):
        return "intro"
    if any(k in query for k in ["研究", "论文", "实验室"]):
        return "research"
    if any(k in query for k in ["新闻", "活动"]):
        return "news"
    return "general"

def rule_score(doc, intent):
    score = 0

    url = doc.get("url", "")
    title = doc.get("title", "")
    text = doc.get("text", "")

    # --- intro boost ---
    intro_signals = ["简介", "概况", "学院", "介绍"]
    if any(k in title for k in intro_signals):
        score += 3

    if "xygk" in url:
        score += 3

    # --- news penalty ---
    if "info" in url or "news" in url:
        score -= 1

    # --- intent-specific logic ---
    if intent == "intro":
        if "研究中心" in title or "联合" in title:
            score -= 2

    return score

def type_boost(doc):
    url = doc["url"]
    title = doc["title"]

    boost = 0

    # STRONG boost for overview pages
    if "xygk" in url or "学院简介" in title:
        boost += 2.0

    # MEDIUM boost for institutional pages
    if "学院" in title and "软件" in title:
        boost += 1.0

    # PENALIZE research/lab pages
    if "研究所" in title or "实验室" in title:
        boost -= 1.5

    if "info" in url:
        boost -= 0.5

    return boost

def rerank_results(results, intent):
    for r in results:
        boost = rule_score(r, intent)
        type_b = type_boost(r)

        r["final_score"] = r["score"] + boost + type_b

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results

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

        results = [r for r in results if len(r["text"]) > 80]

        # ---------------------------
        # Rerank pipeline
        # ---------------------------
        intent = detect_intent(query)

        results = rerank_results(results, intent)
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
            print(f"FINAL SCORE: {r['final_score']}")
            print(f"TEXT: {r['text'][:200]}...")


if __name__ == "__main__":
    main()