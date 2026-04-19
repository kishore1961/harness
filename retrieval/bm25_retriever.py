from ingestion.indexer import load_bm25_index
from loguru import logger


def bm25_retrieve(
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    Sparse BM25 retrieval.
    Returns top_k chunks with BM25 scores.
    """
    bm25, chunks = load_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top_k indices
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "text": chunks[idx]["text"],
            "metadata": {
                "page_number": chunks[idx]["page_number"],
                "source": chunks[idx]["source"],
                "chunk_size": chunks[idx]["chunk_size"],
                "word_count": chunks[idx]["word_count"]
            },
            "score": float(scores[idx]),
            "rank": rank + 1,
            "retrieval_method": "bm25"
        })

    logger.debug(f"BM25 retrieved {len(results)} chunks for: '{query[:50]}'")
    return results


if __name__ == "__main__":
    query = "fiscal deficit GDP target percentage"
    results = bm25_retrieve(query, top_k=5)
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} BM25 results:\n")
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Page: {r['metadata']['page_number']}")
        print(f"Text: {r['text'][:200]}")
        print("---")