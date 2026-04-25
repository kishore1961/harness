# llm_tests/test_gemini.py
"""
Google Gemini API - Complete Test Suite
Tests: connectivity, all models, token counting,
       RAG prompt, speed, cost estimation, JSON output

Run: python llm_tests/test_gemini.py

Get API key: https://aistudio.google.com/apikey
  1. Go to https://aistudio.google.com
  2. Sign in with Google account
  3. Click "Get API Key" top right
  4. Click "Create API key in new project"
  5. Copy key — starts with AIza...
  6. Add to .env: GEMINI_API_KEY=AIza...
"""

import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Gemini free tier models ───────────────────────────────────
GEMINI_MODELS = {
    "gemini-2.0-flash": {
        "description": "Latest + fastest — best free model",
        "context_window": 1_048_576,   # 1M tokens!
        "cost_per_1M_input":  0.10,
        "cost_per_1M_output": 0.40,
        "free_limit_rpm":     15,
        "free_limit_day":     1500,
    },
    "gemini-1.5-flash": {
        "description": "Stable, reliable — good for production",
        "context_window": 1_048_576,
        "cost_per_1M_input":  0.075,
        "cost_per_1M_output": 0.30,
        "free_limit_rpm":     15,
        "free_limit_day":     1500,
    },
    "gemini-1.5-flash-8b": {
        "description": "Smallest + cheapest Gemini",
        "context_window": 1_048_576,
        "cost_per_1M_input":  0.0375,
        "cost_per_1M_output": 0.15,
        "free_limit_rpm":     15,
        "free_limit_day":     1500,
    },
}


# ── Test 1: Basic connectivity ────────────────────────────────
def test_connectivity():
    print("\n" + "="*60)
    print("TEST 1: Basic Connectivity")
    print("="*60)

    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env")
        print("\n   How to get it:")
        print("   1. Go to https://aistudio.google.com")
        print("   2. Sign in with Google")
        print("   3. Click 'Get API Key' top right")
        print("   4. Create API key in new project")
        print("   5. Add to .env: GEMINI_API_KEY=AIza...")
        return False

    print(f"✅ API key found: {GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-4:]}")

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini client configured successfully")
        return genai
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("   Run: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Failed to configure: {e}")
        return False


# ── Test 2: List available models ─────────────────────────────
def test_list_models(genai):
    print("\n" + "="*60)
    print("TEST 2: Available Models on Your Account")
    print("="*60)

    try:
        models = list(genai.list_models())
        # Filter to generative models only
        gen_models = [
            m for m in models
            if "generateContent" in m.supported_generation_methods
        ]
        print(f"✅ Total generative models: {len(gen_models)}")
        print("\nModels:")
        for m in sorted(gen_models, key=lambda x: x.name):
            print(f"  • {m.name.replace('models/', '')}")
        return True
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return False


# ── Test 3: Simple generation — all models ────────────────────
def test_all_models(genai):
    print("\n" + "="*60)
    print("TEST 3: Simple Generation — All Models")
    print("="*60)

    prompt = "What is 2 + 2? Reply in one word only."
    results = {}

    for model_id, info in GEMINI_MODELS.items():
        print(f"\n  Model: {model_id}")
        print(f"  Info:  {info['description']}")

        try:
            model = genai.GenerativeModel(model_id)

            start    = time.time()
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=10,
                )
            )
            latency = round((time.time() - start) * 1000)

            answer    = response.text.strip()
            input_tok = response.usage_metadata.prompt_token_count
            out_tok   = response.usage_metadata.candidates_token_count
            total_tok = response.usage_metadata.total_token_count

            cost = (
                (input_tok / 1_000_000) * info["cost_per_1M_input"] +
                (out_tok   / 1_000_000) * info["cost_per_1M_output"]
            )

            print(f"  ✅ Answer:   {answer}")
            print(f"     Latency:  {latency}ms")
            print(f"     Tokens:   {input_tok} in + {out_tok} out = {total_tok} total")
            print(f"     Cost:     ${cost:.8f}")
            print(f"     Free/day: {info['free_limit_day']} requests")

            results[model_id] = {
                "status":     "ok",
                "answer":     answer,
                "latency_ms": latency,
                "tokens":     total_tok,
                "cost_usd":   cost,
            }

            time.sleep(2)   # respect rate limits

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[model_id] = {"status": "failed", "error": str(e)}

    return results


# ── Test 4: RAG-style prompt ──────────────────────────────────
def test_rag_prompt(genai):
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

    prompt = f"""You are a helpful assistant. Answer using only the context.

Context:
{context}

Question: {question}

Answer concisely and accurately."""

    model_id = "gemini-2.0-flash"
    print(f"  Model:    {model_id}")
    print(f"  Question: {question}")

    try:
        model    = genai.GenerativeModel(model_id)
        start    = time.time()
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=200,
            )
        )
        latency = round((time.time() - start) * 1000)

        answer    = response.text.strip()
        input_tok = response.usage_metadata.prompt_token_count
        out_tok   = response.usage_metadata.candidates_token_count
        info      = GEMINI_MODELS[model_id]
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
def test_speed(genai):
    print("\n" + "="*60)
    print("TEST 5: Speed Benchmark (5 calls)")
    print("="*60)

    model_id  = "gemini-2.0-flash"
    model     = genai.GenerativeModel(model_id)
    prompt    = "Name the capital of France. One word."
    latencies = []

    print(f"  Model: {model_id}")
    print(f"  Running 5 calls (with 4s gap for rate limit)...\n")

    for i in range(5):
        try:
            start    = time.time()
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=5,
                )
            )
            latency = round((time.time() - start) * 1000)
            latencies.append(latency)
            tokens = response.usage_metadata.total_token_count
            print(f"  Call {i+1}: {latency}ms | {tokens} tokens")
            time.sleep(4)   # 15 RPM = 1 request per 4 seconds
        except Exception as e:
            print(f"  Call {i+1}: ❌ {e}")
            time.sleep(5)

    if latencies:
        print(f"\n  Average latency: {sum(latencies)//len(latencies)}ms")
        print(f"  Fastest:         {min(latencies)}ms")
        print(f"  Slowest:         {max(latencies)}ms")
        print(f"\n  ⚠️  Free tier limit: 15 requests/minute")
        print(f"     That means 1 request every 4 seconds")
        print(f"     Your benchmark needs ~60 calls per experiment")
        print(f"     Time needed: ~4 minutes per experiment (free tier)")

    return latencies


# ── Test 6: 1M context window demo ───────────────────────────
def test_large_context(genai):
    print("\n" + "="*60)
    print("TEST 6: Large Context Window (Gemini's Superpower)")
    print("="*60)

    # Simulate a much larger document
    large_context = "The government allocated funds for infrastructure development. " * 500
    prompt = f"""Context: {large_context}

Question: What did the government allocate funds for?
Answer in one sentence."""

    model_id = "gemini-2.0-flash"
    model    = genai.GenerativeModel(model_id)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=50,
            )
        )

        input_tok = response.usage_metadata.prompt_token_count
        ctx_limit = GEMINI_MODELS[model_id]["context_window"]

        print(f"  Model context window: {ctx_limit:,} tokens (1 MILLION!)")
        print(f"  Your prompt used:     {input_tok:,} tokens")
        print(f"  Remaining:            {ctx_limit - input_tok:,} tokens")
        print(f"  ✅ Answer: {response.text.strip()[:100]}")
        print(f"\n  💡 Gemini can process entire books in one call")
        print(f"     vs Claude/GPT which have 128K-200K limits")
        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


# ── Test 7: JSON output ───────────────────────────────────────
def test_json_output(genai):
    print("\n" + "="*60)
    print("TEST 7: JSON Output (Judge Use Case)")
    print("="*60)

    prompt = """Evaluate this RAG answer for faithfulness.

Context: The fiscal deficit target for 2026-27 is 4.4% of GDP.
Answer: The fiscal deficit target for 2026-27 is 4.4% of GDP.

Return ONLY valid JSON:
{"score": 1.0, "reason": "explanation here"}"""

    model_id = "gemini-2.0-flash"
    model    = genai.GenerativeModel(
        model_id,
        system_instruction="You are an evaluation assistant. Always respond with valid JSON only. No markdown."
    )
    print(f"  Model: {model_id}")

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=200,
            )
        )

        raw = response.text.strip()
        # Clean markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        print(f"  Raw response: {raw}")

        parsed = json.loads(raw)
        print(f"  ✅ Valid JSON: score={parsed.get('score')}")
        print(f"     Reason: {parsed.get('reason')}")
        return True

    except json.JSONDecodeError:
        print(f"  ⚠️  Invalid JSON — needs cleaning")
        return False
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


# ── Test 8: Cost + rate limit summary ────────────────────────
def test_cost_and_limits():
    print("\n" + "="*60)
    print("TEST 8: Cost + Rate Limit Summary for Your Benchmark")
    print("="*60)

    total_tokens = 510_000
    total_calls  = 10 * 60   # 10 experiments × ~60 calls each

    print(f"\n  Your benchmark stats:")
    print(f"    Total tokens:  {total_tokens:,}")
    print(f"    Total API calls: {total_calls}")

    print(f"\n  {'Model':<25} {'Cost':>8} {'Calls/day limit':>16} {'Fits?':>8}")
    print(f"  {'-'*60}")

    for model_id, info in GEMINI_MODELS.items():
        inp = int(total_tokens * 0.70)
        out = int(total_tokens * 0.30)
        cost = (
            (inp / 1_000_000) * info["cost_per_1M_input"] +
            (out / 1_000_000) * info["cost_per_1M_output"]
        )
        fits = "✅ Yes" if total_calls <= info["free_limit_day"] else "⚠️  No"
        print(f"  {model_id:<25} ${cost:>6.4f} {info['free_limit_day']:>16,} {fits:>8}")

    print(f"\n  ⚠️  Rate limit: 15 requests/minute on free tier")
    print(f"     Add time.sleep(4) between calls")
    print(f"     Or upgrade to paid for 1000+ RPM")


# ── Main runner ───────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GOOGLE GEMINI API — COMPLETE TEST SUITE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Test 1: connectivity
    genai = test_connectivity()
    if not genai:
        print("\n❌ Cannot proceed — fix API key first")
        return

    # Test 2: list models
    test_list_models(genai)

    # Test 3: all models
    model_results = test_all_models(genai)

    # Test 4: RAG prompt
    test_rag_prompt(genai)

    # Test 5: speed
    test_speed(genai)

    # Test 6: large context
    test_large_context(genai)

    # Test 7: JSON output
    test_json_output(genai)

    # Test 8: cost summary (no API call)
    test_cost_and_limits()

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("GEMINI TEST SUMMARY")
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
            print(f"   • {m}: {model_results[m].get('error', '')[:80]}")

    print(f"\n📋 Recommended for your harness:")
    print(f"   RAG generation: gemini-2.0-flash (fastest, 1M context, free)")
    print(f"   Judge LLM:      gemini-2.0-flash (consistent JSON)")
    print(f"\n⚠️  Key difference from Groq:")
    print(f"   Groq:   ~200ms latency, 500K tokens/day limit")
    print(f"   Gemini: ~500ms latency, 1500 requests/day limit")
    print(f"   Gemini wins on: context window (1M vs 128K)")
    print(f"   Groq wins on:   raw speed")
    print(f"\n📁 Next step: python llm_tests/test_openai.py")


if __name__ == "__main__":
    main()