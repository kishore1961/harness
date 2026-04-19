import os
from dotenv import load_dotenv
from loguru import logger
from langsmith import traceable

load_dotenv()

# Essential Environment Variables for Auto-tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-benchmark-harness"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

@traceable(run_type="chain", name="rag_query")
def trace_rag_call_langsmith(
    query: str,
    retrieved_chunks: list[dict],
    llm_result: dict,
    config: dict
) -> str:
    """
    Uses the @traceable decorator. 
    This is the most robust way to ensure traces hit LangSmith.
    """
    # Metadata for the trace
    metadata = {
        "model": config.get("model_key"),
        "retrieval_method": config.get("retrieval_method"),
        "total_tokens": llm_result.get("total_tokens")
    }
    
    logger.info("LangSmith auto-trace initiated via decorator")
    return "auto-traced"

if __name__ == "__main__":
    # Test call
    res = trace_rag_call_langsmith(
        "What is the fiscal deficit?", 
        [], 
        {"answer": "4.3%", "total_tokens": 100}, 
        {"model_key": "llama-3.1-8b"}
    )
    print("Check https://smith.langchain.com - Look for 'rag_query' in your project.")