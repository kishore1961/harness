from ingestion.indexer import get_chroma_collection
from ingestion.embedder import embed_query
from loguru import logger


def dense_retrieve(
    query: str,
    top_k: int = 10
) -> list[dict]:
    collection = get_chroma_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": 1 - dist,
            "rank": i + 1,
            "retrieval_method": "dense"
        })

    logger.debug(f"Dense retrieved {len(chunks)} chunks for: '{query[:50]}'")
    return chunks


if __name__ == "__main__":
    query = "What is the fiscal deficit target for 2026-27?"
    results = dense_retrieve(query, top_k=5)
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:\n")
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Page: {r['metadata']['page_number']}")
        print(f"Text: {r['text'][:200]}")
        print("---")
