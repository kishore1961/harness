import os
from dotenv import load_dotenv
from langfuse import Langfuse
from loguru import logger

load_dotenv()

_langfuse = None


def get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        logger.info("LangFuse client initialized")
    return _langfuse


def trace_rag_call(
    query: str,
    retrieved_chunks: list[dict],
    llm_result: dict,
    config: dict,
    trace_name: str = "rag_query"
) -> str:
    """
    Trace a full RAG call in LangFuse.
    Returns trace_id for linking in dashboard.
    """
    lf = get_langfuse()

    trace = lf.trace(
        name=trace_name,
        input=query,
        metadata={
            "model": config.get("model_key"),
            "retrieval_method": config.get("retrieval_method"),
            "chunk_size": config.get("chunk_size"),
            "top_k": config.get("top_k"),
            "prompt_version": config.get("prompt_version"),
            "reranking": config.get("reranking", False)
        }
    )

    # Span 1: Retrieval
    retrieval_span = trace.span(
        name="retrieval",
        input=query,
        output={
            "num_chunks": len(retrieved_chunks),
            "top_chunk_preview": retrieved_chunks[0]["text"][:200] if retrieved_chunks else ""
        },
        metadata={"retrieval_method": config.get("retrieval_method")}
    )
    retrieval_span.end()

    # Span 2: LLM Generation
    generation_span = trace.generation(
        name="llm_generation",
        model=llm_result.get("model_name"),
        input={"role": "user", "content": "[assembled prompt]"},
        output=llm_result.get("answer"),
        usage={
            "input": llm_result.get("input_tokens"),
            "output": llm_result.get("output_tokens"),
            "total": llm_result.get("total_tokens")
        },
        metadata={
            "latency_ms": llm_result.get("latency_ms"),
            "cost_usd": llm_result.get("cost_usd"),
            "provider": llm_result.get("provider")
        }
    )
    generation_span.end()

    trace.update(output=llm_result.get("answer"))
    lf.flush()

    logger.info(f"LangFuse trace created: {trace.id}")
    return trace.id


if __name__ == "__main__":
    trace_id = trace_rag_call(
        query="What is the fiscal deficit?",
        retrieved_chunks=[
            {"text": "The fiscal deficit is 5.1% of GDP", "metadata": {"page_number": 3}}
        ],
        llm_result={
            "answer": "The fiscal deficit target is 5.1% of GDP for 2024-25.",
            "input_tokens": 150,
            "output_tokens": 20,
            "total_tokens": 170,
            "latency_ms": 450,
            "cost_usd": 0.0,
            "provider": "groq",
            "model_name": "llama-3.1-8b-instant"
        },
        config={
            "model_key": "llama-3.1-8b",
            "retrieval_method": "hybrid",
            "chunk_size": 500,
            "top_k": 5,
            "prompt_version": "1.0",
            "reranking": True
        }
    )
    print(f"\nTrace ID: {trace_id}")
    print("Check your LangFuse dashboard at https://cloud.langfuse.com")