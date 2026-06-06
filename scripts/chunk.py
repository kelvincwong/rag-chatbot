import json
import re
import hashlib
import tiktoken
enc = tiktoken.get_encoding("o200k_base")

INPUT_FILE = "app/data/raw/raw_pages.jsonl"
OUTPUT_FILE = "app/data/processed/chunks.jsonl"

MAX_TOKENS = 800
OVERLAP_SENTENCES = 3


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
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------
# Sentence-aware chunking
# ---------------------------
def chunk_sentences(sentences, max_size=MAX_TOKENS):
    chunks = []
    current = []

    current_len = 0

    for sent in sentences:
        sent_len = token_len(sent)

        # heuristic: force boundary on year / section shifts
        if re.search(r"(\d{4}年|^\d{4})", sent):
            if current:
                chunks.append("".join(current))

                overlap = current[-OVERLAP_SENTENCES:]
                current = overlap.copy()
                current_len = sum(token_len(s) for s in current)
            continue

        # if adding sentence exceeds limit, flush chunk
        if current_len + sent_len > max_size and current:
            chunks.append("".join(current))

            overlap = current[-OVERLAP_SENTENCES:]
            current = overlap.copy()
            current_len = sum(token_len(s) for s in current)

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

def token_len(text):
    return len(enc.encode(text))

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
        doc_type = page.get("doc_type")
        entity = page.get("entity")
        is_entity_homepage = page.get("is_entity_homepage", False)
        doc_id = hashlib.md5(url.encode()).hexdigest()

        text = clean_text(page.get("content", ""))

        # paragraph split
        paragraphs = re.split(r"\n+", text)

        sentences = []
        for p in paragraphs:
            sentences.extend(split_sentences(p))

        chunks = chunk_sentences(sentences)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "url": url,
                "title": title,
                "date": date,
                "entity": entity,
                "doc_type": doc_type,
                "is_entity_homepage": is_entity_homepage,

                "section_hint": title,
                "source_url": url,

                "text": chunk,
                "embedding_text": f"""
                Entity: {entity}
                Title: {title}
                DocType: {doc_type}

                {chunk}
                """.strip()
            })

    write_jsonl(OUTPUT_FILE, all_chunks)

    print(f"Done.")
    print(f"Pages: {len(pages)} | Chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()