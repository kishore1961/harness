import mlflow
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "rag-benchmark-harness"


def log_experiment(
    config: dict,
    deepeval_scores: dict,
    ragas_scores: dict,
    extra_metrics: dict = None
) -> str:
    """
    Log one experiment run to MLflow.
    Returns run_id.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=config.get("experiment_name", "unnamed")) as run:

        # Log config as params
        mlflow.log_params({
            "chunk_size": config.get("chunk_size"),
            "retrieval_method": config.get("retrieval_method"),
            "reranking": config.get("reranking"),
            "top_k": config.get("top_k"),
            "model_key": config.get("model_key"),
            "prompt_version": config.get("prompt_version"),
        })

        # Log DeepEval metrics with prefix
        for metric, score in deepeval_scores.items():
            mlflow.log_metric(f"deepeval_{metric}", score)

        # Log Ragas metrics with prefix
        for metric, score in ragas_scores.items():
            mlflow.log_metric(f"ragas_{metric}", score)

        # Log extra metrics (latency, cost etc)
        if extra_metrics:
            mlflow.log_metrics(extra_metrics)

        run_id = run.info.run_id
        logger.info(f"MLflow run logged: {run_id}")
        return run_id


if __name__ == "__main__":
    # Test with dummy scores
    config = {
        "experiment_name": "test_run",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    }
    deepeval_scores = {
        "faithfulness": 0.82,
        "answerrelevancy": 0.79,
        "contextualprecision": 0.75,
        "contextualrecall": 0.71
    }
    ragas_scores = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.77,
        "context_precision": 0.73,
        "context_recall": 0.69,
        "answer_correctness": 0.74
    }
    run_id = log_experiment(config, deepeval_scores, ragas_scores)
    print(f"Run ID: {run_id}")
    print(f"View at: {MLFLOW_TRACKING_URI}")
