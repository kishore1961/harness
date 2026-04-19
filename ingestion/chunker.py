from loguru import logger
from typing import List


def chunk_text(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Sliding window chunker.
    Takes pages from pdf_loader, returns chunks with metadata.
    chunk_size and chunk_overlap are in words (not tokens).
    Close enough for our purposes without needing a tokenizer.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if len(chunk_words) > 50:  # skip tiny trailing chunks
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id:04d}",
                    "text": chunk_text,
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "word_count": len(chunk_words)
                })
                chunk_id += 1

            if end == len(words):
                break
            start += chunk_size - chunk_overlap

    logger.info(
        f"Created {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


if __name__ == "__main__":
    from ingestion.pdf_loader import load_pdf

    pages = load_pdf("data/budget_2024.pdf")

    for size in [300, 500, 800]:
        chunks = chunk_text(pages, chunk_size=size, chunk_overlap=int(size * 0.2))
        print(f"\nChunk size {size}: {len(chunks)} chunks")
        print(f"Sample chunk: {chunks[0]['text'][:200]}")