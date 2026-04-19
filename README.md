# RAG Benchmark Harness

> Systematic benchmarking of RAG configurations with full LLMOps observability.
> Built on India Union Budget 2026-27.

[![Nightly Eval](https://github.com/YOUR_USERNAME/rag-benchmark-harness/actions/workflows/nightly_eval.yml/badge.svg)](https://github.com/YOUR_USERNAME/rag-benchmark-harness/actions/workflows/nightly_eval.yml)

## What This Is

A reference implementation answering: **which RAG configuration gives the best quality-to-cost ratio?**

Benchmarks 5 configurations across chunking, retrieval method, reranking, and model choice.
Instrumented with LangFuse + LangSmith observability, MLflow experiment tracking,
DeepEval + Ragas evaluation, and GitHub Actions nightly regression detection.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/rag-benchmark-harness
cd rag-benchmark-harness
cp .env.example .env
# Fill in your API keys in .env
docker compose up