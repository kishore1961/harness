"""
Checks if latest eval scores have regressed vs baseline.
Exits with code 1 if regression detected — fails GitHub Actions.
"""
import json
import sys
from pathlib import Path
from loguru import logger

BASELINE = {
    "faithfulness": 0.75,
    "answerrelevancy": 0.90,
    "contextualprecision": 0.80,
    "contextualrecall": 0.75,
}

REGRESSION_THRESHOLD = 0.10  # 10% drop triggers failure


def check():
    results_path = Path("experiments/all_results.json")
    if not results_path.exists():
        logger.warning("No results file found. Skipping regression check.")
        sys.exit(0)

    with open(results_path) as f:
        results = json.load(f)

    if not results:
        logger.warning("Empty results. Skipping.")
        sys.exit(0)

    latest = results[-1]["deepeval_scores"]
    regressions = []

    for metric, baseline_val in BASELINE.items():
        current_val = latest.get(metric, 0)
        drop = baseline_val - current_val
        drop_pct = drop / baseline_val if baseline_val > 0 else 0

        if drop_pct > REGRESSION_THRESHOLD:
            regressions.append(
                f"{metric}: baseline={baseline_val:.2f}, "
                f"current={current_val:.2f}, "
                f"drop={drop_pct*100:.1f}%"
            )

    if regressions:
        logger.error("REGRESSION DETECTED:")
        for r in regressions:
            logger.error(f"  {r}")
        sys.exit(1)
    else:
        logger.info("No regression detected. All metrics within threshold.")
        sys.exit(0)


if __name__ == "__main__":
    check()
