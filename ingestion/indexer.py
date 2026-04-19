import chromadb
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma_db")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
COLLECTION_NAME = "budget_2024"


def build_chroma_index(chunks: list[dict], embeddings) -> chromadb.Collection:
    """Store chunks and embeddings in ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # ChromaDB expects lists
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "page_number": c["page_number"],
            "source": c["source"],
            "chunk_size": c["chunk_size"],
            "word_count": c["word_count"]
        }
        for c in chunks
    ]

    # Insert in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        logger.info(f"Inserted batch {i//batch_size + 1}")

    logger.info(f"ChromaDB collection built: {collection.count()} chunks")
    return collection


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Build BM25 index from chunks."""
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # Save index + chunks for later use
    Path(BM25_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    logger.info(f"BM25 index built and saved: {BM25_INDEX_PATH}")
    return bm25


def load_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Load saved BM25 index."""
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def get_chroma_collection() -> chromadb.Collection:
    """Get existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    from ingestion.pdf_loader import load_pdf
    from ingestion.chunker import chunk_text
    from ingestion.embedder import embed_texts

    logger.info("Starting full ingestion pipeline...")

    # Load
    pages = load_pdf("data/budget_2024.pdf")

    # Chunk with default config (500 words, 100 overlap)
    chunks = chunk_text(pages, chunk_size=500, chunk_overlap=100)

    # Embed
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Index
    build_chroma_index(chunks, embeddings)
    build_bm25_index(chunks)

    logger.info("Ingestion complete!")
    print(f"\nSummary:")
    print(f"  Pages: {len(pages)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  ChromaDB: {CHROMA_DIR}")
    print(f"  BM25 index: {BM25_INDEX_PATH}")