# llm_tests/count_tokens.py
"""
Exact token counter for your benchmark.
Reads your actual log files and counts every token used.

Run: python llm_tests/count_tokens.py
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# ── Method 1: Parse your actual log files ────────────────────
def count_from_logs():
    print("\n" + "="*60)
    print("METHOD 1: Count from Actual Log Files")
    print("="*60)

    log_dir = Path("logs")
    if not log_dir.exists():
        print("❌ No logs/ directory found")
        print("   Run at least one experiment first")
        return None

    # Find all log runs
    run_dirs = sorted([d for d in log_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        print("❌ No run directories found in logs/")
        return None

    print(f"Found {len(run_dirs)} run(s) in logs/")

    all_stats = []

    for run_dir in run_dirs:
        log_file = run_dir / "full_console.log"
        if not log_file.exists():
            continue

        print(f"\n  Run: {run_dir.name}")

        content = log_file.read_text(errors="ignore")

        # Extract every "tokens=XXXX" from LLM call lines
        # Pattern: LLM call | model=... | latency=...ms | tokens=XXXX
        token_pattern = re.compile(
            r'LLM call \| model=(\S+) \| latency=(\d+)ms \| tokens=(\d+) \| cost=\$([0-9.]+)'
        )

        matches = token_pattern.findall(content)

        if not matches:
            print(f"    ⚠️  No LLM call lines found in log")
            continue

        total_tokens = 0
        total_cost   = 0.0
        total_calls  = 0
        model_stats  = {}

        for model, latency_ms, tokens, cost in matches:
            t = int(tokens)
            c = float(cost)
            l = int(latency_ms)

            total_tokens += t
            total_cost   += c
            total_calls  += 1

            if model not in model_stats:
                model_stats[model] = {
                    "calls": 0, "tokens": 0,
                    "cost": 0.0, "latencies": []
                }
            model_stats[model]["calls"]     += 1
            model_stats[model]["tokens"]    += t
            model_stats[model]["cost"]      += c
            model_stats[model]["latencies"].append(l)

        print(f"    Total LLM calls:  {total_calls}")
        print(f"    Total tokens:     {total_tokens:,}")
        print(f"    Total cost:       ${total_cost:.6f}")

        for model, stats in model_stats.items():
            avg_lat = sum(stats["latencies"]) // len(stats["latencies"])
            avg_tok = stats["tokens"] // stats["calls"]
            print(f"\n    Model: {model}")
            print(f"      Calls:       {stats['calls']}")
            print(f"      Tokens:      {stats['tokens']:,} total")
            print(f"      Avg tokens:  {avg_tok} per call")
            print(f"      Cost:        ${stats['cost']:.6f}")
            print(f"      Avg latency: {avg_lat}ms")

        all_stats.append({
            "run": run_dir.name,
            "calls": total_calls,
            "tokens": total_tokens,
            "cost": total_cost,
        })

    return all_stats


# ── Method 2: Count from all_results.json ────────────────────
def count_from_results():
    print("\n" + "="*60)
    print("METHOD 2: Count from experiments/all_results.json")
    print("="*60)

    results_path = Path("experiments/all_results.json")
    if not results_path.exists():
        print("❌ experiments/all_results.json not found")
        return None

    with open(results_path) as f:
        results = json.load(f)

    print(f"Found {len(results)} completed experiments\n")

    grand_total_tokens = 0
    grand_total_cost   = 0.0

    print(f"  {'Experiment':<25} {'DeepEval':>10} {'Ragas':>10} {'Total':>10}")
    print(f"  {'-'*58}")

    for r in results:
        name = r["config"]["experiment_name"]

        # These are the scores — not tokens
        # But we can estimate from duration and known rates
        duration = r.get("duration_seconds", 0)

        print(f"  {name:<25} {'N/A':>10} {'N/A':>10} {duration:>9.0f}s")

    print(f"\n  ℹ️  all_results.json stores scores, not raw token counts")
    print(f"     Use Method 1 (log parsing) for exact token counts")
    print(f"     Use Method 3 (live counter) going forward")

    return results


# ── Method 3: Live token counter to add to your pipeline ─────
def show_live_counter_code():
    print("\n" + "="*60)
    print("METHOD 3: Live Token Counter (Add to Your Pipeline)")
    print("="*60)

    print("""
  Add this to your experiment_runner.py to track tokens exactly:

  ┌─────────────────────────────────────────────────────────┐
  │  # In experiments/experiment_runner.py                  │
  │                                                         │
  │  TOKEN_TRACKER = {                                      │
  │      "rag_calls": 0,                                    │
  │      "rag_tokens": 0,                                   │
  │      "rag_cost": 0.0,                                   │
  │      "judge_calls": 0,                                  │
  │      "judge_tokens": 0,                                 │
  │      "judge_cost": 0.0,                                 │
  │  }                                                      │
  └─────────────────────────────────────────────────────────┘

  Your llm_client.py already logs:
    tokens=2053 | cost=$0.00059125

  The log parser (Method 1) already captures this exactly.
""")


# ── Method 4: Estimate future runs ───────────────────────────
def estimate_future_runs(log_stats):
    print("\n" + "="*60)
    print("METHOD 4: Accurate Future Estimates Based on Real Data")
    print("="*60)

    if not log_stats:
        print("  ❌ No log data available — run experiments first")
        return

    # Use the most recent complete run
    latest = log_stats[-1]
    tokens_per_run  = latest["tokens"]
    cost_per_run    = latest["cost"]
    calls_per_run   = latest["calls"]

    print(f"\n  Based on your most recent actual run:")
    print(f"  Run timestamp:    {latest['run']}")
    print(f"  Total LLM calls:  {calls_per_run}")
    print(f"  Total tokens:     {tokens_per_run:,}")
    print(f"  Total cost:       ${cost_per_run:.6f}")

    if calls_per_run > 0:
        avg_tok_per_call  = tokens_per_run // calls_per_run
        avg_cost_per_call = cost_per_run / calls_per_run
        print(f"\n  Per call average:")
        print(f"    Tokens: {avg_tok_per_call}")
        print(f"    Cost:   ${avg_cost_per_call:.8f}")

    # Project to different scales
    print(f"\n  Projections:")
    print(f"  {'Scale':<30} {'Tokens':>12} {'Cost (Claude)':>14} {'Cost (Groq 8B)':>15}")
    print(f"  {'-'*72}")

    scales = [
        ("1 experiment (current)",    1),
        ("10 experiments (full run)", 10),
        ("50 experiments",            50),
        ("100 experiments",          100),
        ("3 models × 10 exp",         30),
    ]

    for label, multiplier in scales:
        tok  = tokens_per_run * multiplier
        # Claude haiku pricing: $0.25/1M input, $1.25/1M output
        cost_claude = tok * 0.0000007  # blended rate
        # Groq llama-3.1-8b: $0.05/1M input, $0.08/1M output
        cost_groq   = tok * 0.000000065  # blended rate
        print(f"  {label:<30} {tok:>12,} {cost_claude:>13.4f}$ {cost_groq:>14.4f}$")

    # Provider limits check
    print(f"\n  Provider daily limits vs your usage:")
    print(f"  {'Provider':<20} {'Free tokens/day':>18} {'Your 10-exp':>12} {'Fits?':>8}")
    print(f"  {'-'*60}")

    ten_exp_tokens = tokens_per_run * 10

    providers = [
        ("Groq",         500_000),
        ("Gemini",       None),        # request-based, not token-based
        ("Together AI",  None),        # paid per token
        ("OpenRouter",   200_000),
        ("Cerebras",     1_000_000),
    ]

    for name, daily_limit in providers:
        if daily_limit is None:
            print(f"  {name:<20} {'N/A (req-based)':>18} {ten_exp_tokens:>12,} {'Check RPM':>8}")
        else:
            fits = "✅ Yes" if ten_exp_tokens <= daily_limit else "⚠️  No"
            print(f"  {name:<20} {daily_limit:>18,} {ten_exp_tokens:>12,} {fits:>8}")


# ── Method 5: Per-experiment breakdown ───────────────────────
def breakdown_per_experiment():
    print("\n" + "="*60)
    print("METHOD 5: Per-Experiment Token Breakdown")
    print("="*60)

    log_dir = Path("logs")
    if not log_dir.exists():
        return

    run_dirs = sorted([d for d in log_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        return

    # Use the latest run
    latest_run = run_dirs[-1]
    log_file   = latest_run / "full_console.log"

    if not log_file.exists():
        return

    print(f"  Analyzing: {latest_run.name}")
    content = log_file.read_text(errors="ignore")

    # Find experiment boundaries and token usage
    exp_pattern   = re.compile(r'Experiment \d+/\d+: (exp_\w+)')
    token_pattern = re.compile(r'tokens=(\d+) \| cost=\$([0-9.]+)')

    # Split by experiment
    sections = re.split(r'={60}', content)

    current_exp = "pre_experiment"
    exp_tokens  = {}
    exp_costs   = {}

    for section in sections:
        # Check if this section has an experiment name
        exp_match = exp_pattern.search(section)
        if exp_match:
            current_exp = exp_match.group(1)

        # Count tokens in this section
        tok_matches = token_pattern.findall(section)
        for tokens, cost in tok_matches:
            if current_exp not in exp_tokens:
                exp_tokens[current_exp] = 0
                exp_costs[current_exp]  = 0.0
            exp_tokens[current_exp] += int(tokens)
            exp_costs[current_exp]  += float(cost)

    if exp_tokens:
        print(f"\n  {'Experiment':<28} {'Tokens':>10} {'Cost':>12}")
        print(f"  {'-'*52}")

        grand_tok  = 0
        grand_cost = 0.0

        for exp, tokens in sorted(exp_tokens.items()):
            cost = exp_costs.get(exp, 0.0)
            grand_tok  += tokens
            grand_cost += cost
            print(f"  {exp:<28} {tokens:>10,} {cost:>11.6f}$")

        print(f"  {'-'*52}")
        print(f"  {'TOTAL':<28} {grand_tok:>10,} {grand_cost:>11.6f}$")
        print(f"\n  Average per experiment: {grand_tok//max(len(exp_tokens),1):,} tokens")
    else:
        print("  ⚠️  Could not parse per-experiment breakdown")
        print("      (Logs may use different format)")


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXACT TOKEN COUNTER FOR YOUR BENCHMARK")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Method 1: parse logs
    log_stats = count_from_logs()

    # Method 2: results json
    count_from_results()

    # Method 3: show live counter code
    show_live_counter_code()

    # Method 4: accurate estimates
    estimate_future_runs(log_stats)

    # Method 5: per-experiment breakdown
    breakdown_per_experiment()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
  My original estimate of 510,000 tokens was based on:
    • RAG tokens:   read from your actual logs ✅ accurate
    • Judge tokens: guessed ~800/call ⚠️  estimated

  This script gives you the EXACT number from real logs.
  Use it after each run to know precisely what you spent.

  For resume/documentation, use the real numbers from this
  script — not rounded estimates.
    """)


if __name__ == "__main__":
    main()