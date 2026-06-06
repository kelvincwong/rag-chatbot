import json
import re

INPUT_FILE = "app/data/raw/raw_pages.jsonl"
OUTPUT_FILE = "app/data/processed/chunks.jsonl"

MAX_CHUNK_SIZE = 500
OVERLAP_SENTENCES = 2


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
# Sentence splitter (Chinese + basic punctuation)
# ---------------------------
def split_sentences(text):
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[。！？.!?])", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------
# Sentence-aware chunking
# ---------------------------
def chunk_sentences(sentences, max_size=MAX_CHUNK_SIZE):
    chunks = []
    current = []

    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # if adding sentence exceeds limit, flush chunk
        if current_len + sent_len > max_size and current:
            chunks.append("".join(current))
            
            # overlap: keep last N sentences
            current = current[-OVERLAP_SENTENCES:]
            current_len = sum(len(s) for s in current)

        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append("".join(current))

    return chunks


# ---------------------------
# Basic cleanup
# ---------------------------
def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ---------------------------
# Write JSONL
# ---------------------------
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    pages = load_jsonl(INPUT_FILE)

    all_chunks = []

    for page in pages:
        url = page.get("url")
        title = page.get("title", "")
        date = page.get("date", None)

        text = clean_text(page.get("content", ""))

        sentences = split_sentences(text)
        chunks = chunk_sentences(sentences)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "url": url,
                "title": title,
                "date": date,
                "chunk_id": idx,
                "text": chunk
            })

    write_jsonl(OUTPUT_FILE, all_chunks)

    print(f"Done.")
    print(f"Pages: {len(pages)} | Chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()