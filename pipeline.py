from dotenv import load_dotenv
from loguru import logger

from retrieval.dense_retriever import dense_retrieve
from retrieval.bm25_retriever import bm25_retrieve
from retrieval.hybrid_retriever import hybrid_retrieve
from retrieval.reranker import rerank
from generation.prompt_manager import build_prompt
from generation.llm_client import call_llm
from observability.langfuse_client import trace_rag_call
from observability.langsmith_client import trace_rag_call_langsmith

load_dotenv()


def run_rag_pipeline(query: str, config: dict, trace: bool = True) -> dict:
    retrieval_method = config.get("retrieval_method", "hybrid")
    top_k = config.get("top_k", 5)
    reranking = config.get("reranking", True)
    model_key = config.get("model_key", "llama-3.1-8b")
    prompt_version = config.get("prompt_version", "1.0")

    logger.info(f"Running RAG | retrieval={retrieval_method} | reranking={reranking} | model={model_key}")

    # retrieval
    if retrieval_method == "dense":
        candidates = dense_retrieve(query, top_k=top_k * 2 if reranking else top_k)
    elif retrieval_method == "bm25":
        candidates = bm25_retrieve(query, top_k=top_k * 2 if reranking else top_k)
    else:
        candidates = hybrid_retrieve(query, top_k=top_k * 2 if reranking else top_k)

    # reranking
    if reranking:
        final_chunks = rerank(query, candidates, top_k=top_k)
    else:
        final_chunks = candidates[:top_k]

    # prompt build
    system_prompt, user_prompt = build_prompt(query, final_chunks, version=prompt_version)

    # generation
    llm_result = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_key=model_key
    )

    # traces
    langfuse_trace_id = None
    langsmith_run_id = None
    if trace:
        try:
            langfuse_trace_id = trace_rag_call(query, final_chunks, llm_result, config)
        except Exception as e:
            logger.warning(f"LangFuse trace failed: {e}")

        try:
            langsmith_run_id = trace_rag_call_langsmith(query, final_chunks, llm_result, config)
        except Exception as e:
            logger.warning(f"LangSmith trace failed: {e}")

    return {
        "query": query,
        "answer": llm_result["answer"],
        "retrieved_chunks": final_chunks,
        "llm_result": llm_result,
        "config": config,
        "langfuse_trace_id": langfuse_trace_id,
        "langsmith_run_id": langsmith_run_id
    }


if __name__ == "__main__":
    config = {
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": "llama-3.1-8b",
        "prompt_version": "1.0"
    }

    questions = [
        "What is the fiscal deficit target for 2026-27?",
        "What is the estimated nominal GDP growth for 2026-27?",
        "What tax changes are mentioned for 2026-27?"
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        result = run_rag_pipeline(q, config, trace=True)
        print(f"A: {result['answer']}")
        print(f"Latency: {result['llm_result']['latency_ms']} ms")
        print(f"Tokens: {result['llm_result']['total_tokens']}")
        print(f"Trace (LangFuse): {result['langfuse_trace_id']}")
        print(f"Trace (LangSmith): {result['langsmith_run_id']}")
        print("\nTop retrieved chunk preview:")
        if result["retrieved_chunks"]:
            print(result["retrieved_chunks"][0]["text"][:300])
