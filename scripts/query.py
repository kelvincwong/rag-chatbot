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


def main():
    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(INDEX_FILE)
    metadata = load_metadata()

    while True:
        query = input("\nAsk: ")
        if not query:
            break

        query_vec = model.encode([query], normalize_embeddings=True)

        D, I = index.search(np.array(query_vec), k=5)

        print("\nTop results:\n")

        for rank, idx in enumerate(I[0]):
            item = metadata[idx]

            print(f"\n[{rank+1}] {item['title']}")
            print(f"URL: {item['url']}")
            print(f"DATE: {item['date']}")
            print(f"TEXT: {item['text'][:200]}...")


if __name__ == "__main__":
    main()