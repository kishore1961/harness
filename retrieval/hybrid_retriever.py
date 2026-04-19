from retrieval.dense_retriever import dense_retrieve
from retrieval.bm25_retriever import bm25_retrieve
from loguru import logger


def hybrid_retrieve(
    query: str,
    top_k: int = 10,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> list[dict]:
    """
    Hybrid retrieval: combine dense + BM25 using Reciprocal Rank Fusion.
    RRF is better than weighted score addition because scores are not comparable.
    dense_weight and bm25_weight control RRF contribution.
    """
    dense_results = dense_retrieve(query, top_k=top_k * 2)
    bm25_results = bm25_retrieve(query, top_k=top_k * 2)

    # Reciprocal Rank Fusion
    k = 60  # RRF constant
    rrf_scores = {}

    for rank, result in enumerate(dense_results):
        text = result["text"]
        rrf_scores.setdefault(text, {"score": 0, "result": result})
        rrf_scores[text]["score"] += dense_weight * (1 / (k + rank + 1))

    for rank, result in enumerate(bm25_results):
        text = result["text"]
        rrf_scores.setdefault(text, {"score": 0, "result": result})
        rrf_scores[text]["score"] += bm25_weight * (1 / (k + rank + 1))

    # Sort by RRF score
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    results = []
    for rank, item in enumerate(sorted_results):
        result = item["result"].copy()
        result["score"] = item["score"]
        result["rank"] = rank + 1
        result["retrieval_method"] = "hybrid"
        results.append(result)

    logger.debug(f"Hybrid retrieved {len(results)} chunks for: '{query[:50]}'")
    return results


if __name__ == "__main__":
    query = "What is the fiscal deficit target for 2024-25?"
    results = hybrid_retrieve(query, top_k=5)
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} hybrid results:\n")
    for r in results:
        print(f"Rank {r['rank']} | RRF Score: {r['score']:.6f} | Page: {r['metadata']['page_number']}")
        print(f"Text: {r['text'][:200]}")
        print("---")