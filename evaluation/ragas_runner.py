# evaluation/ragas_runner.py
import logging
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("langsmith.client").setLevel(logging.CRITICAL)

import json
import os
import re
import math
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "none"
os.environ["OPENAI_API_KEY"] = "dummy-not-used"

RAGAS_JUDGE_PROVIDER = os.getenv("RAGAS_JUDGE_PROVIDER", "anthropic")
RAGAS_JUDGE_MODEL    = os.getenv("RAGAS_JUDGE_MODEL", "claude-haiku-4-5-20251001")


def clean_text(text: str) -> str:
    text = text.replace('₹', 'Rs.')
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _get_ragas_judge():
    from ragas.llms import LangchainLLMWrapper

    if RAGAS_JUDGE_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        logger.info(f"Ragas judge: Groq / {RAGAS_JUDGE_MODEL}")
        llm = ChatGroq(
            model=os.getenv("RAGAS_JUDGE_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
    elif RAGAS_JUDGE_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        logger.info(f"Ragas judge: Anthropic / {RAGAS_JUDGE_MODEL}")
        llm = ChatAnthropic(
            model=os.getenv("RAGAS_JUDGE_MODEL", "claude-haiku-4-5-20251001"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
            max_tokens=2048
        )
    else:
        raise ValueError(f"Unknown RAGAS_JUDGE_PROVIDER: '{RAGAS_JUDGE_PROVIDER}'")

    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )


def run_ragas(
    eval_set_path: str = "evaluation/eval_set.json",
    config: dict = None
) -> dict:
    # Use old-style ragas.metrics — these still support LangchainLLMWrapper
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            context_precision,
            context_recall,
            answer_correctness,
        )
    from datasets import Dataset
    from pipeline import run_rag_pipeline

    if config is None:
        config = {
            "chunk_size": 500,
            "retrieval_method": "hybrid",
            "reranking": True,
            "top_k": 5,
            "model_key": "claude-haiku",
            "prompt_version": "1.0"
        }

    with open(eval_set_path) as f:
        eval_set = json.load(f)

    logger.info(f"Running Ragas on {len(eval_set)} questions...")

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in eval_set:
        logger.info(f"  Running: {item['question'][:60]}...")
        result = run_rag_pipeline(item["question"], config, trace=False)
        questions.append(clean_text(item["question"]))
        answers.append(clean_text(result["answer"]))
        contexts.append([clean_text(c["text"]) for c in result["retrieved_chunks"]])
        ground_truths.append(clean_text(item["expected_answer"]))

    # Old ragas API uses these column names
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    judge_llm        = _get_ragas_judge()
    judge_embeddings = _get_ragas_embeddings()

    # Old-style: assign llm after import
    metrics_list = [faithfulness, context_precision, context_recall, answer_correctness]
    for m in metrics_list:
        m.llm = judge_llm
        if hasattr(m, 'embeddings'):
            m.embeddings = judge_embeddings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ragas_evaluate(
            dataset,
            metrics=metrics_list,
            raise_exceptions=False,
        )

    # Extract scores — old API returns dict-like result
    scores = {}
    metric_keys = {
        "faithfulness":      "faithfulness",
        "context_precision": "context_precision",
        "context_recall":    "context_recall",
        "answer_correctness": "answer_correctness",
    }

    try:
        df = result.to_pandas()
        logger.debug(f"Ragas result columns: {list(df.columns)}")
        for ragas_key, our_key in metric_keys.items():
            try:
                val = float(df[ragas_key].mean())
                scores[our_key] = 0.0 if math.isnan(val) else round(val, 4)
            except Exception as e:
                logger.warning(f"Could not extract {ragas_key}: {e}")
                scores[our_key] = 0.0
    except Exception as e:
        logger.error(f"Could not parse Ragas result: {e}")
        scores = {k: 0.0 for k in metric_keys.values()}

    logger.info(f"Ragas scores: {scores}")
    return scores


if __name__ == "__main__":
    config = {
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "claude-haiku",
        "prompt_version": "1.0"
    }
    scores = run_ragas(config=config)
    print("\nRagas Results:")
    for metric, score in scores.items():
        print(f"  {metric}: {score}")