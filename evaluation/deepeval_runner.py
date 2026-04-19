import json
import os
import re
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def clean_text(text: str) -> str:
    text = text.replace('₹', 'Rs.')
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_deepeval(
    eval_set_path: str = "evaluation/eval_set.json",
    config: dict = None,
    judge_provider: str = None,
    judge_model: str = None
) -> dict:
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )
    from deepeval.test_case import LLMTestCase
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
    from evaluation.judge_client import get_deepeval_judge
    from pipeline import run_rag_pipeline

    if config is None:
        config = {
            "chunk_size": 500,
            "retrieval_method": "hybrid",
            "reranking": True,
            "top_k": 5,
            "model_key": "llama-3.1-8b",
            "prompt_version": "1.0"
        }

    if judge_provider:
        os.environ["JUDGE_PROVIDER"] = judge_provider
    if judge_model:
        os.environ["JUDGE_MODEL"] = judge_model

    judge = get_deepeval_judge()

    with open(eval_set_path) as f:
        eval_set = json.load(f)

    logger.info(f"Running DeepEval on {len(eval_set)} questions...")

    test_cases = []
    for item in eval_set:
        logger.info(f"  Running: {item['question'][:60]}...")
        result = run_rag_pipeline(item["question"], config, trace=False)

        cleaned_answer = clean_text(result["answer"])
        cleaned_context = [clean_text(c["text"]) for c in result["retrieved_chunks"]]
        cleaned_expected = clean_text(item["expected_answer"])
        cleaned_question = clean_text(item["question"])

        test_cases.append(LLMTestCase(
            input=cleaned_question,
            actual_output=cleaned_answer,
            expected_output=cleaned_expected,
            retrieval_context=cleaned_context,
            context=cleaned_context
        ))

    faithfulness_metric = FaithfulnessMetric(model=judge, threshold=0.5)
    relevancy_metric = AnswerRelevancyMetric(model=judge, threshold=0.5)
    precision_metric = ContextualPrecisionMetric(model=judge, threshold=0.5)
    recall_metric = ContextualRecallMetric(model=judge, threshold=0.5)

    metrics = [faithfulness_metric, relevancy_metric, precision_metric, recall_metric]

    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False)
    )

    # Collect scores per metric using index position
    metric_keys = ["faithfulness", "answerrelevancy", "contextualprecision", "contextualrecall"]
    scores = {k: [] for k in metric_keys}

    for test_result in results.test_results:
        if test_result.metrics_data:
            for i, metric_data in enumerate(test_result.metrics_data):
                if i < len(metric_keys) and metric_data.score is not None:
                    scores[metric_keys[i]].append(metric_data.score)

    averaged = {}
    for key, vals in scores.items():
        if vals:
            averaged[key] = round(sum(vals) / len(vals), 4)
        else:
            averaged[key] = 0.0

    logger.info(f"DeepEval scores: {averaged}")
    return averaged


if __name__ == "__main__":
    config = {
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    }
    scores = run_deepeval(config=config)
    print("\nDeepEval Results:")
    for metric, score in scores.items():
        print(f"  {metric}: {score}")
