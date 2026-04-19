import yaml
import time
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

_current_chunk_size = 500  # track what is currently indexed


def reindex_if_needed(chunk_size: int):
    """Re-run ingestion if chunk size changed."""
    global _current_chunk_size
    if chunk_size == _current_chunk_size:
        logger.info(f"Chunk size {chunk_size} already indexed. Skipping re-index.")
        return

    logger.info(f"Chunk size changed {_current_chunk_size} -> {chunk_size}. Re-indexing...")
    from ingestion.pdf_loader import load_pdf
    from ingestion.chunker import chunk_text
    from ingestion.embedder import embed_texts
    from ingestion.indexer import build_chroma_index, build_bm25_index

    pages = load_pdf("data/budget_2024.pdf")
    overlap = int(chunk_size * 0.2)
    chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=overlap)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    build_chroma_index(chunks, embeddings)
    build_bm25_index(chunks)

    _current_chunk_size = chunk_size
    logger.info(f"Re-indexed with chunk_size={chunk_size}, chunks={len(chunks)}")


def load_experiment_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(config: dict) -> dict:
    from evaluation.deepeval_runner import run_deepeval
    from evaluation.ragas_runner import run_ragas
    from experiments.mlflow_logger import log_experiment

    logger.info(f"Starting experiment: {config.get('experiment_name')}")
    logger.info(f"Config: {config}")

    # Re-index if chunk size changed
    reindex_if_needed(config.get("chunk_size", 500))

    start = time.time()

    logger.info("Running DeepEval...")
    deepeval_scores = run_deepeval(config=config)

    logger.info("Running Ragas...")
    ragas_scores = run_ragas(config=config)

    elapsed = time.time() - start

    run_id = log_experiment(
        config,
        deepeval_scores,
        ragas_scores,
        {"experiment_duration_seconds": round(elapsed, 1)}
    )

    results = {
        "config": config,
        "deepeval_scores": deepeval_scores,
        "ragas_scores": ragas_scores,
        "mlflow_run_id": run_id,
        "duration_seconds": round(elapsed, 1)
    }

    logger.info(f"Experiment complete: {config.get('experiment_name')}")
    logger.info(f"DeepEval: {deepeval_scores}")
    logger.info(f"Ragas: {ragas_scores}")
    logger.info(f"MLflow run: {run_id}")

    return results


if __name__ == "__main__":
    config = {
        "experiment_name": "exp_01_baseline",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    }
    results = run_experiment(config)
    print("\n=== EXPERIMENT RESULTS ===")
    print(f"Experiment: {results['config']['experiment_name']}")
    print(f"Duration: {results['duration_seconds']}s")
    print("\nDeepEval:")
    for k, v in results['deepeval_scores'].items():
        print(f"  {k}: {v}")
    print("\nRagas:")
    for k, v in results['ragas_scores'].items():
        print(f"  {k}: {v}")
    print(f"\nMLflow run: {results['mlflow_run_id']}")
