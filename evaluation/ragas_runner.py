import json
import os
import re
import math
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Stop ragas from trying OpenAI
os.environ["OPENAI_API_KEY"] = "dummy-not-used"


def clean_text(text: str) -> str:
    text = text.replace('₹', 'Rs.')
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_ragas(
    eval_set_path: str = "evaluation/eval_set.json",
    config: dict = None
) -> dict:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        context_precision,
        context_recall,
        answer_correctness,
    )
    from ragas.metrics.critique import harmfulness
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from datasets import Dataset
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

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # LLM judge — Groq
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    judge_llm = LangchainLLMWrapper(groq_llm)

    # Embeddings — local, no API needed
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # Use only metrics that work without OpenAI embeddings
    metrics_list = [
        faithfulness,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    for m in metrics_list:
        m.llm = judge_llm
        if hasattr(m, 'embeddings'):
            m.embeddings = judge_embeddings

    result = ragas_evaluate(
        dataset,
        metrics=metrics_list,
        raise_exceptions=False
    )

    scores = {}
    for metric in ["faithfulness", "context_precision",
                   "context_recall", "answer_correctness"]:
        try:
            val = float(result[metric])
            scores[metric] = 0.0 if math.isnan(val) else round(val, 4)
        except Exception:
            scores[metric] = 0.0

    logger.info(f"Ragas scores: {scores}")
    return scores


if __name__ == "__main__":
    config = {
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    }
    scores = run_ragas(config=config)
    print("\nRagas Results:")
    for metric, score in scores.items():
        print(f"  {metric}: {score}")
