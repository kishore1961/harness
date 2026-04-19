from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    """Singleton — load model once, reuse."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts.
    Returns numpy array of shape (len(texts), embedding_dim).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    model = get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    return embedding[0]


if __name__ == "__main__":
    test_texts = [
        "The fiscal deficit target for 2024-25 is 5.1% of GDP",
        "PM Awas Yojana urban allocation is increased",
        "New tax slabs introduced under new tax regime"
    ]
    embeddings = embed_texts(test_texts)
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")

    query_emb = embed_query("What is the fiscal deficit?")
    print(f"\nQuery embedding shape: {query_emb.shape}")