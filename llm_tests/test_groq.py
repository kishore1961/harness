# llm_tests/test_groq.py
"""
Groq API - Complete Test Suite
Tests: connectivity, all models, token counting, 
       rate limits, streaming, timing, cost estimation

Run: python llm_tests/test_groq.py
"""

import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Groq free tier models ─────────────────────────────────────
GROQ_MODELS = {
    "llama-3.1-8b-instant": {
        "description": "Fast, lightweight — good for RAG generation",
        "context_window": 128000,
        "cost_per_1M_input":  0.05,
        "cost_per_1M_output": 0.08,
    },
    "llama-3.3-70b-versatile": {
        "description": "Smarter — good for evaluation/judge",
        "context_window": 128000,
        "cost_per_1M_input":  0.59,
        "cost_per_1M_output": 0.79,
    },
    "mixtral-8x7b-32768": {
        "description": "Long context — good for large documents",
        "context_window": 32768,
        "cost_per_1M_input":  0.24,
        "cost_per_1M_output": 0.24,
    },
    "gemma2-9b-it": {
        "description": "Google Gemma — good alternative to Llama",
        "context_window": 8192,
        "cost_per_1M_input":  0.20,
        "cost_per_1M_output": 0.20,
    },
}


# ── Test 1: Basic connectivity ────────────────────────────────
def test_connectivity():
    print("\n" + "="*60)
    print("TEST 1: Basic Connectivity")
    print("="*60)

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env")
        print("   Add this to your .env file:")
        print("   GROQ_API_KEY=gsk_your_key_here")
        return False

    print(f"✅ API key found: {GROQ_API_KEY[:8]}...{GROQ_API_KEY[-4:]}")

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client created successfully")
        return client
    except ImportError:
        print("❌ groq package not installed")
        print("   Run: pip install groq")
        return False
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False


# ── Test 2: List available models ─────────────────────────────
def test_list_models(client):
    print("\n" + "="*60)
    print("TEST 2: Available Models on Your Account")
    print("="*60)

    try:
        models = client.models.list()
        print(f"✅ Total models available: {len(models.data)}")
        print("\nModels:")
        for m in sorted(models.data, key=lambda x: x.id):
            print(f"  • {m.id}")
        return True
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return False


# ── Test 3: Simple generation — all models ────────────────────
def test_all_models(client):
    print("\n" + "="*60)
    print("TEST 3: Simple Generation — All Models")
    print("="*60)

    prompt = "What is 2 + 2? Reply in one word."
    results = {}

    for model_id, info in GROQ_MODELS.items():
        print(f"\n  Model: {model_id}")
        print(f"  Info:  {info['description']}")

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            latency = round((time.time() - start) * 1000)

            answer      = response.choices[0].message.content.strip()
            input_tok   = response.usage.prompt_tokens
            output_tok  = response.usage.completion_tokens
            total_tok   = response.usage.total_tokens

            # Cost calculation
            cost = (
                (input_tok  / 1_000_000) * info["cost_per_1M_input"] +
                (output_tok / 1_000_000) * info["cost_per_1M_output"]
            )

            print(f"  ✅ Answer:   {answer}")
            print(f"     Latency:  {latency}ms")
            print(f"     Tokens:   {input_tok} in + {output_tok} out = {total_tok} total")
            print(f"     Cost:     ${cost:.8f}")

            results[model_id] = {
                "status":     "ok",
                "answer":     answer,
                "latency_ms": latency,
                "tokens":     total_tok,
                "cost_usd":   cost,
            }

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[model_id] = {"status": "failed", "error": str(e)}

    return results


# ── Test 4: RAG-style prompt (realistic workload) ─────────────
def test_rag_prompt(client):
    print("\n" + "="*60)
    print("TEST 4: RAG-Style Prompt (Realistic Workload)")
    print("="*60)

    context = """
    The fiscal deficit target for 2026-27 is 4.4 percent of GDP.
    The fiscal deficit estimate for 2025-26 is 4.8 percent of GDP.
    The government has committed to a medium-term fiscal consolidation path.
    Total expenditure for 2026-27 is estimated at Rs. 50,65,345 crore.
    Capital expenditure is projected at Rs. 11,21,490 crore for 2026-27.
    """

    question = "What is the fiscal deficit target for 2026-27?"

    prompt = f"""You are a helpful assistant. Answer the question using only the context provided.

Context:
{context}

Question: {question}

Answer concisely and accurately."""

    model = "llama-3.1-8b-instant"
    print(f"  Model: {model}")
    print(f"  Question: {question}")

    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        latency = round((time.time() - start) * 1000)

        answer    = response.choices[0].message.content.strip()
        input_tok = response.usage.prompt_tokens
        out_tok   = response.usage.completion_tokens

        info      = GROQ_MODELS[model]
        cost      = (
            (input_tok / 1_000_000) * info["cost_per_1M_input"] +
            (out_tok   / 1_000_000) * info["cost_per_1M_output"]
        )

        print(f"\n  ✅ Answer:   {answer}")
        print(f"     Latency:  {latency}ms")
        print(f"     Tokens:   {input_tok} in + {out_tok} out")
        print(f"     Cost:     ${cost:.8f}")
        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


# ── Test 5: Speed benchmark ───────────────────────────────────
def test_speed(client):
    print("\n" + "="*60)
    print("TEST 5: Speed Benchmark (5 consecutive calls)")
    print("="*60)

    model   = "llama-3.1-8b-instant"
    prompt  = "Name the capital of France."
    latencies = []

    print(f"  Model: {model}")
    print(f"  Running 5 calls...\n")

    for i in range(5):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            latency = round((time.time() - start) * 1000)
            latencies.append(latency)
            tokens = response.usage.total_tokens
            print(f"  Call {i+1}: {latency}ms | {tokens} tokens")
            time.sleep(1)   # be polite to the API
        except Exception as e:
            print(f"  Call {i+1}: ❌ {e}")

    if latencies:
        print(f"\n  Average latency: {sum(latencies)//len(latencies)}ms")
        print(f"  Fastest:         {min(latencies)}ms")
        print(f"  Slowest:         {max(latencies)}ms")

    return latencies


# ── Test 6: Token limit test ──────────────────────────────────
def test_token_limits(client):
    print("\n" + "="*60)
    print("TEST 6: Token Limit Awareness")
    print("="*60)

    # Simulate a large RAG context (5 chunks × ~500 tokens each)
    large_context = "The government allocated funds for infrastructure. " * 200
    prompt = f"""Context: {large_context}

Question: What did the government allocate funds for?
Answer briefly."""

    model = "llama-3.1-8b-instant"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )

        input_tok = response.usage.prompt_tokens
        out_tok   = response.usage.completion_tokens
        ctx_limit = GROQ_MODELS[model]["context_window"]

        print(f"  Model context window: {ctx_limit:,} tokens")
        print(f"  Your prompt used:     {input_tok:,} tokens")
        print(f"  Remaining:            {ctx_limit - input_tok:,} tokens")
        print(f"  Output tokens:        {out_tok}")
        print(f"  ✅ Large context handled successfully")
        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


# ── Test 7: Full benchmark cost estimate ─────────────────────
def test_cost_estimate():
    print("\n" + "="*60)
    print("TEST 7: Full Benchmark Cost Estimate")
    print("="*60)

    # Based on your actual logs
    per_experiment = {
        "rag_calls":            10,   # 5 deepeval + 5 ragas
        "avg_rag_tokens":       1900,
        "deepeval_judge_calls": 20,
        "avg_judge_tokens":     800,
        "ragas_judge_calls":    20,
        "avg_ragas_tokens":     800,
    }

    total_rag_tokens    = (per_experiment["rag_calls"] *
                           per_experiment["avg_rag_tokens"])
    total_judge_tokens  = (
        (per_experiment["deepeval_judge_calls"] +
         per_experiment["ragas_judge_calls"]) *
        per_experiment["avg_judge_tokens"]
    )
    total_per_exp       = total_rag_tokens + total_judge_tokens
    total_10_exp        = total_per_exp * 10

    print(f"\n  Per experiment:")
    print(f"    RAG tokens:    {total_rag_tokens:,}")
    print(f"    Judge tokens:  {total_judge_tokens:,}")
    print(f"    Total:         {total_per_exp:,}")
    print(f"\n  For 10 experiments:")
    print(f"    Total tokens:  {total_10_exp:,}")

    print(f"\n  Cost per provider (10 experiments):")
    print(f"  {'Model':<35} {'Input+Output':>15} {'Cost':>10}")
    print(f"  {'-'*62}")

    estimates = [
        ("Groq llama-3.1-8b-instant",    0.05,  0.08),
        ("Groq llama-3.3-70b-versatile",  0.59,  0.79),
        ("Groq mixtral-8x7b",             0.24,  0.24),
        ("OpenAI gpt-4o-mini",            0.15,  0.60),
        ("OpenAI gpt-4o",                 2.50, 10.00),
        ("Anthropic claude-haiku",         0.25,  1.25),
    ]

    # Assume 70% input, 30% output split
    for name, inp_rate, out_rate in estimates:
        inp_tok = int(total_10_exp * 0.70)
        out_tok = int(total_10_exp * 0.30)
        cost    = (inp_tok / 1_000_000 * inp_rate +
                   out_tok / 1_000_000 * out_rate)
        print(f"  {name:<35} {total_10_exp:>15,} {cost:>9.4f}$")

    print(f"\n  Groq free tier limit: 500,000 tokens/day")
    print(f"  Your 10-exp usage:    {total_10_exp:,} tokens")
    if total_10_exp <= 500_000:
        print(f"  ✅ Fits in ONE day of free Groq tier")
    else:
        days = total_10_exp / 500_000
        print(f"  ⚠️  Needs {days:.1f} days of Groq free tier")


# ── Test 8: JSON output (for judge use) ──────────────────────
def test_json_output(client):
    print("\n" + "="*60)
    print("TEST 8: JSON Output (Judge Use Case)")
    print("="*60)

    prompt = """Evaluate this RAG answer for faithfulness.

Context: The fiscal deficit target for 2026-27 is 4.4% of GDP.
Answer: The fiscal deficit target for 2026-27 is 4.4% of GDP.

Return ONLY valid JSON in this exact format:
{"score": 1.0, "reason": "The answer matches the context exactly"}"""

    model = "llama-3.3-70b-versatile"
    print(f"  Model: {model}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluation assistant. Always respond with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        print(f"  Raw response: {raw}")

        parsed = json.loads(raw)
        print(f"  ✅ Valid JSON: score={parsed.get('score')}")
        print(f"     Reason: {parsed.get('reason')}")
        return True

    except json.JSONDecodeError:
        print(f"  ⚠️  Invalid JSON returned — retry logic needed")
        return False
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


# ── Main runner ───────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GROQ API — COMPLETE TEST SUITE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Test 1: connectivity
    client = test_connectivity()
    if not client:
        print("\n❌ Cannot proceed — fix API key first")
        return

    # Test 2: list models
    test_list_models(client)

    # Test 3: all models
    model_results = test_all_models(client)

    # Test 4: RAG prompt
    test_rag_prompt(client)

    # Test 5: speed
    test_speed(client)

    # Test 6: token limits
    test_token_limits(client)

    # Test 7: cost estimate (no API call needed)
    test_cost_estimate()

    # Test 8: JSON output
    test_json_output(client)

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("GROQ TEST SUMMARY")
    print("="*60)

    working = [m for m, r in model_results.items() if r["status"] == "ok"]
    failed  = [m for m, r in model_results.items() if r["status"] != "ok"]

    print(f"✅ Working models: {len(working)}")
    for m in working:
        r = model_results[m]
        print(f"   • {m}: {r['latency_ms']}ms | {r['tokens']} tokens")

    if failed:
        print(f"❌ Failed models: {len(failed)}")
        for m in failed:
            print(f"   • {m}: {model_results[m].get('error')}")

    print(f"\n📋 Recommended for your harness:")
    print(f"   RAG generation: llama-3.1-8b-instant  (fastest, free)")
    print(f"   Judge LLM:      llama-3.3-70b-versatile (smartest, free)")
    print(f"\n📁 Next step: python llm_tests/test_gemini.py")


if __name__ == "__main__":
    main()