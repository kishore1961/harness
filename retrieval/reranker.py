from sentence_transformers import CrossEncoder
from loguru import logger

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker() -> CrossEncoder:
    """Singleton reranker."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker loaded")
    return _reranker


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Rerank chunks using cross-encoder.
    Cross-encoder reads query+chunk together — more accurate than bi-encoder.
    Takes top_k after reranking.
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
        chunk["original_rank"] = chunk["rank"]

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    for rank, chunk in enumerate(reranked):
        chunk["rank"] = rank + 1

    logger.debug(f"Reranked to top {len(reranked)} chunks")
    return reranked


if __name__ == "__main__":
    from retrieval.hybrid_retriever import hybrid_retrieve

    query = "What is the fiscal deficit target for 2024-25?"

    # First hybrid retrieve
    candidates = hybrid_retrieve(query, top_k=10)
    print(f"Before reranking — top 3:")
    for r in candidates[:3]:
        print(f"  Rank {r['rank']}: {r['text'][:150]}")

    # Then rerank
    reranked = rerank(query, candidates, top_k=5)
    print(f"\nAfter reranking — top 3:")
    for r in reranked[:3]:
        print(f"  Rank {r['rank']} (was {r['original_rank']}) | Score: {r['rerank_score']:.4f}")
        print(f"  {r['text'][:150]}")