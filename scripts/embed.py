import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INPUT_FILE = "app/data/processed/chunks.jsonl"
INDEX_FILE = "app/data/processed/faiss.index"
META_FILE = "app/data/processed/meta.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------
# Load JSONL
# ---------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ---------------------------
# Save JSONL (metadata mapping)
# ---------------------------
def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# Main embedding pipeline
# ---------------------------
def main():
    print("Loading chunks...")
    chunks = load_jsonl(INPUT_FILE)

    print(f"Loaded {len(chunks)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["embedding_text"] for c in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)  # cosine similarity (because normalized)
    index.add(embeddings)

    print(f"Index size: {index.ntotal}")

    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_FILE)

    print("Saving metadata...")

    meta = []

    for i, c in enumerate(chunks):
        c["faiss_id"] = i
        meta.append(c)

    save_jsonl(META_FILE, meta)

    print("Done.")


if __name__ == "__main__":
    main()