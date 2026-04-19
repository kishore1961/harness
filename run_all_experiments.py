"""
Master experiment runner.
Runs all 10 experiments sequentially.
Results logged to MLflow automatically.
Run: python run_all_experiments.py
Expected time: 30-60 minutes depending on rate limits.
"""
import json
import time
from pathlib import Path
from loguru import logger
from experiments.experiment_runner import run_experiment

EXPERIMENTS = [
    {
        "experiment_name": "exp_01_baseline",
        "description": "Baseline: hybrid, rerank, llama-3.1-8b, chunk=500",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_02_chunk_300",
        "description": "Chunk size 300 vs baseline 500",
        "chunk_size": 300,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_03_chunk_800",
        "description": "Chunk size 800 vs baseline 500",
        "chunk_size": 800,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_04_dense_only",
        "description": "Dense only retrieval",
        "chunk_size": 500,
        "retrieval_method": "dense",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_05_bm25_only",
        "description": "BM25 only retrieval",
        "chunk_size": 500,
        "retrieval_method": "bm25",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_06_no_rerank",
        "description": "Hybrid without reranking",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": False,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_07_llama_70b",
        "description": "Llama 3.3 70B generator",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.3-70b",
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_08_claude_haiku",
        "description": "Claude Haiku generator vs OSS",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "claude-haiku",
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_09_prompt_v1_1",
        "description": "Prompt v1.1 calibrated",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.1"
    },
    {
        "experiment_name": "exp_10_prompt_v1_2",
        "description": "Prompt v1.2 with citations",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.2"
    },
]

WAIT_BETWEEN_EXPERIMENTS = 30  # seconds — let rate limits recover


def main():
    all_results = []
    failed = []

    logger.info(f"Starting all {len(EXPERIMENTS)} experiments")
    logger.info(f"Waiting {WAIT_BETWEEN_EXPERIMENTS}s between experiments")

    for i, config in enumerate(EXPERIMENTS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i+1}/{len(EXPERIMENTS)}: {config['experiment_name']}")
        logger.info(f"{'='*60}")

        try:
            result = run_experiment(config)
            all_results.append(result)

            # Print summary
            print(f"\n✅ {config['experiment_name']} complete")
            print(f"   DeepEval faithfulness: {result['deepeval_scores'].get('faithfulness', 'N/A')}")
            print(f"   Ragas faithfulness:    {result['ragas_scores'].get('faithfulness', 'N/A')}")
            print(f"   MLflow run: {result['mlflow_run_id']}")

        except Exception as e:
            logger.error(f"Experiment {config['experiment_name']} FAILED: {e}")
            failed.append(config['experiment_name'])

        # Wait between experiments to avoid rate limits
        if i < len(EXPERIMENTS) - 1:
            logger.info(f"Waiting {WAIT_BETWEEN_EXPERIMENTS}s before next experiment...")
            time.sleep(WAIT_BETWEEN_EXPERIMENTS)

    # Save all results to JSON
    results_path = "experiments/all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print final summary
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Completed: {len(all_results)}/{len(EXPERIMENTS)}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Results saved to: {results_path}")
    print("\nSummary table:")
    print(f"{'Experiment':<25} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Recall':>6}")
    print("-" * 55)
    for r in all_results:
        name = r['config']['experiment_name']
        faith = r['deepeval_scores'].get('faithfulness', 0)
        relev = r['deepeval_scores'].get('answerrelevancy', 0)
        prec = r['deepeval_scores'].get('contextualprecision', 0)
        recall = r['deepeval_scores'].get('contextualrecall', 0)
        print(f"{name:<25} {faith:>6.3f} {relev:>6.3f} {prec:>6.3f} {recall:>6.3f}")


if __name__ == "__main__":
    main()
