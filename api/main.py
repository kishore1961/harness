from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

app = FastAPI(
    title="RAG Benchmark Harness",
    description="Production-grade RAG system with full LLMOps observability",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str
    model_key: Optional[str] = "llama-3.1-8b"
    retrieval_method: Optional[str] = "hybrid"
    reranking: Optional[bool] = True
    top_k: Optional[int] = 5
    prompt_version: Optional[str] = "1.0"


class QueryResponse(BaseModel):
    question: str
    answer: str
    model_key: str
    retrieval_method: str
    latency_ms: float
    total_tokens: int
    cost_usd: float
    langfuse_trace_id: Optional[str]
    top_chunk_preview: str


class ExperimentRequest(BaseModel):
    experiment_name: str
    chunk_size: Optional[int] = 500
    retrieval_method: Optional[str] = "hybrid"
    reranking: Optional[bool] = True
    top_k: Optional[int] = 5
    model_key: Optional[str] = "llama-3.1-8b"
    prompt_version: Optional[str] = "1.0"


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-benchmark-harness"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        from pipeline import run_rag_pipeline

        config = {
            "chunk_size": 500,
            "retrieval_method": request.retrieval_method,
            "reranking": request.reranking,
            "top_k": request.top_k,
            "model_key": request.model_key,
            "prompt_version": request.prompt_version
        }

        result = run_rag_pipeline(request.question, config, trace=True)

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            model_key=request.model_key,
            retrieval_method=request.retrieval_method,
            latency_ms=result["llm_result"]["latency_ms"],
            total_tokens=result["llm_result"]["total_tokens"],
            cost_usd=result["llm_result"]["cost_usd"],
            langfuse_trace_id=result.get("langfuse_trace_id"),
            top_chunk_preview=result["retrieved_chunks"][0]["text"][:300]
            if result["retrieved_chunks"] else ""
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments")
def get_experiments():
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("rag-benchmark-harness")

        if not experiment:
            return {"experiments": []}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        results = []
        for run in runs:
            results.append({
                "run_id": run.info.run_id,
                "experiment_name": run.data.params.get("experiment_name", run.info.run_name),
                "model_key": run.data.params.get("model_key"),
                "retrieval_method": run.data.params.get("retrieval_method"),
                "chunk_size": run.data.params.get("chunk_size"),
                "reranking": run.data.params.get("reranking"),
                "prompt_version": run.data.params.get("prompt_version"),
                "deepeval_faithfulness": run.data.metrics.get("deepeval_faithfulness"),
                "deepeval_answerrelevancy": run.data.metrics.get("deepeval_answerrelevancy"),
                "deepeval_contextualprecision": run.data.metrics.get("deepeval_contextualprecision"),
                "deepeval_contextualrecall": run.data.metrics.get("deepeval_contextualrecall"),
                "ragas_faithfulness": run.data.metrics.get("ragas_faithfulness"),
                "ragas_context_recall": run.data.metrics.get("ragas_context_recall"),
                "ragas_answer_correctness": run.data.metrics.get("ragas_answer_correctness"),
                "duration_seconds": run.data.metrics.get("experiment_duration_seconds"),
            })

        return {"experiments": results, "total": len(results)}

    except Exception as e:
        logger.error(f"Failed to fetch experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def get_models():
    return {
        "groq": ["llama-3.1-8b", "llama-3.3-70b"],
        "anthropic": ["claude-haiku", "claude-sonnet"],
        "judge": "claude-haiku-4-5-20251001"
    }


@app.get("/prompts")
def get_prompts():
    import yaml
    from pathlib import Path
    prompts = {}
    for version in ["1.0", "1.1", "1.2"]:
        path = Path(f"prompts/v{version}.yaml")
        if path.exists():
            with open(path) as f:
                prompts[f"v{version}"] = yaml.safe_load(f)
    return prompts
