"""
Master experiment runner.
Runs all 10 experiments sequentially.
Results logged to MLflow automatically.

Run: python run_all_experiments.py

Expected time: 30-60 minutes depending on rate limits.

Which model runs is controlled by .env:
  LLM_MODEL_KEY = claude-haiku   (default)
  LLM_MODEL_KEY = llama-3.1-8b
  LLM_MODEL_KEY = llama-3.3-70b
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from experiments.experiment_runner import run_experiment

load_dotenv()

# ── Create timestamped run directory ────────────────────────────
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("logs") / RUN_TIMESTAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── Redirect ALL stdout/stderr to log file + terminal ──────────
class TeeWriter:
    """Write to both terminal and file simultaneously."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return False

# Open master log file
MASTER_LOG_PATH = RUN_DIR / "full_console.log"
_log_file = open(MASTER_LOG_PATH, "w")
sys.stdout = TeeWriter(sys.__stdout__, _log_file)
sys.stderr = TeeWriter(sys.__stderr__, _log_file)

# ── Configure loguru to also write to a structured log file ─────
logger.remove()  # remove default handler

# Console handler (INFO level, colored)
logger.add(
    sys.__stderr__,
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
)

# File handler (DEBUG level, full detail)
logger.add(
    RUN_DIR / "structured.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
           "{name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
)

# Read the active model from .env
LLM_MODEL_KEY = os.getenv("LLM_MODEL_KEY", "claude-haiku")

EXPERIMENTS = [
    {
        "experiment_name": "exp_01_baseline",
        "description": "Baseline: chunk=500, hybrid, rerank, prompt v1.0",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_02_chunk_300",
        "description": "Chunk size 300 vs baseline 500",
        "chunk_size": 300,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_03_chunk_800",
        "description": "Chunk size 800 vs baseline 500",
        "chunk_size": 800,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_04_dense_only",
        "description": "Dense-only retrieval",
        "chunk_size": 500,
        "retrieval_method": "dense",
        "reranking": True,
        "top_k": 5,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_05_bm25_only",
        "description": "BM25-only retrieval",
        "chunk_size": 500,
        "retrieval_method": "bm25",
        "reranking": True,
        "top_k": 5,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_06_no_rerank",
        "description": "Hybrid without reranking",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": False,
        "top_k": 5,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_07_top_k_3",
        "description": "Smaller context window: top_k=3",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 3,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_08_top_k_10",
        "description": "Larger context window: top_k=10",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 10,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.0"
    },
    {
        "experiment_name": "exp_09_prompt_v1_1",
        "description": "Prompt v1.1 — uncertainty calibration",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.1"
    },
    {
        "experiment_name": "exp_10_prompt_v1_2",
        "description": "Prompt v1.2 — with citations",
        "chunk_size": 500,
        "retrieval_method": "hybrid",
        "reranking": True,
        "top_k": 5,
        "model_key": LLM_MODEL_KEY,
        "prompt_version": "1.2"
    },
]

WAIT_BETWEEN_EXPERIMENTS = int(os.getenv("WAIT_BETWEEN_EXPERIMENTS", "5"))


def main():
    all_results = []
    failed = []

    # ── Save run config snapshot ─────────────────────────────────
    run_config = {
        "run_timestamp": RUN_TIMESTAMP,
        "llm_model_key": LLM_MODEL_KEY,
        "wait_between_experiments": WAIT_BETWEEN_EXPERIMENTS,
        "total_experiments": len(EXPERIMENTS),
        "experiments": EXPERIMENTS,
    }
    with open(RUN_DIR / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info(f"Run directory: {RUN_DIR}")
    logger.info(f"Master log: {MASTER_LOG_PATH}")
    logger.info(f"Active LLM: {LLM_MODEL_KEY}")
    logger.info(f"Starting all {len(EXPERIMENTS)} experiments")
    logger.info(f"Waiting {WAIT_BETWEEN_EXPERIMENTS}s between experiments")

    for i, config in enumerate(EXPERIMENTS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i+1}/{len(EXPERIMENTS)}: {config['experiment_name']}")
        logger.info(f"{'='*60}")

        try:
            result = run_experiment(config)
            all_results.append(result)

            print(f"\n✅ {config['experiment_name']} complete")
            print(f"   DeepEval faithfulness: {result['deepeval_scores'].get('faithfulness', 'N/A')}")
            print(f"   Ragas faithfulness:    {result['ragas_scores'].get('faithfulness', 'N/A')}")
            print(f"   MLflow run:            {result['mlflow_run_id']}")

            # Save per-experiment result
            exp_result_path = RUN_DIR / f"{config['experiment_name']}_result.json"
            with open(exp_result_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Experiment {config['experiment_name']} FAILED: {e}")
            failed.append(config['experiment_name'])

            # Save failure info
            fail_path = RUN_DIR / f"{config['experiment_name']}_FAILED.json"
            with open(fail_path, "w") as f:
                json.dump({
                    "experiment_name": config["experiment_name"],
                    "error": str(e),
                    "config": config,
                }, f, indent=2)

        if i < len(EXPERIMENTS) - 1:
            logger.info(f"Waiting {WAIT_BETWEEN_EXPERIMENTS}s before next experiment...")
            time.sleep(WAIT_BETWEEN_EXPERIMENTS)

    # ── Save all results ─────────────────────────────────────────
    results_path = RUN_DIR / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Also save to the original location for backward compatibility
    with open("experiments/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Summary table ────────────────────────────────────────────
    summary_lines = []
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("ALL EXPERIMENTS COMPLETE")
    summary_lines.append(f"{'='*60}")
    summary_lines.append(f"Run directory: {RUN_DIR}")
    summary_lines.append(f"Model used:    {LLM_MODEL_KEY}")
    summary_lines.append(f"Completed:     {len(all_results)}/{len(EXPERIMENTS)}")
    if failed:
        summary_lines.append(f"Failed:        {failed}")
    summary_lines.append(f"Results:       {results_path}")

    header = f"\n{'Experiment':<25} {'D-Faith':>7} {'D-Relev':>7} {'D-Prec':>7} {'D-Recall':>8} | {'R-Faith':>7} {'R-Prec':>7} {'R-Recall':>8} {'R-Corr':>7}"
    summary_lines.append(header)
    summary_lines.append("-" * 105)

    for r in all_results:
        name   = r['config']['experiment_name']
        d_f    = r['deepeval_scores'].get('faithfulness', 0)
        d_r    = r['deepeval_scores'].get('answerrelevancy', 0)
        d_p    = r['deepeval_scores'].get('contextualprecision', 0)
        d_rc   = r['deepeval_scores'].get('contextualrecall', 0)
        r_f    = r['ragas_scores'].get('faithfulness', 0)
        r_p    = r['ragas_scores'].get('context_precision', 0)
        r_rc   = r['ragas_scores'].get('context_recall', 0)
        r_c    = r['ragas_scores'].get('answer_correctness', 0)
        line = f"{name:<25} {d_f:>7.3f} {d_r:>7.3f} {d_p:>7.3f} {d_rc:>8.3f} | {r_f:>7.3f} {r_p:>7.3f} {r_rc:>8.3f} {r_c:>7.3f}"
        summary_lines.append(line)

    # Print and save summary
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(RUN_DIR / "summary.txt", "w") as f:
        f.write(summary_text)

    logger.info(f"All logs saved to: {RUN_DIR}/")
    logger.info("Files in run directory:")
    for p in sorted(RUN_DIR.iterdir()):
        logger.info(f"  {p.name}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure log file is closed properly
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_file.close()
        print(f"\n📁 All logs saved to: {RUN_DIR}/")