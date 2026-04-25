# llm_tests/test_gemini_v2.py
"""
Google Gemini API - Complete Test Suite (New SDK)
Uses: google-genai (not deprecated google.generativeai)

Run: python llm_tests/test_gemini_v2.py

Get API key: https://aistudio.google.com/apikey
  1. Go to https://aistudio.google.com
  2. Sign in with Google account  
  3. Click "Get API Key" top right
  4. "Create API key in new project"
  5. Add to .env: GEMINI_API_KEY=AIza...
"""

import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODELS = {
    "gemini-2.0-flash": {
        "description": "Latest + fastest — best free model",
        "context_window": 1_048_576,
        "cost_per_1M_input":  0.10,
        "cost_per_1M_output": 0.40,
        "free_rpm":  15,
        "free_rpd":  1500,
    },
    "gemini-2.5-flash": {
        "description": "Newest — thinking model, very capable",
        "context_window": 1_048_576,
        "cost_per_1M_input":  0.15,
        "cost_per_1M_output": 0.60,
        "free_rpm":  10,
        "free_rpd":  500,
    },
    "gemini-2.0-flash-lite": {
        "description": "Smallest + cheapest Gemini",
        "context_window": 1_048_576,
        "cost_per_1M_input":  0.075,
        "cost_per_1M_output": 0.30,
        "free_rpm":  30,
        "free_rpd":  1500,
    },
}


# ── Test 1: Connectivity ──────────────────────────────────────
def test_connectivity():
    print("\n" + "="*60)
    print("TEST 1: Basic Connectivity (New google-genai SDK)")
    print("="*60)

    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env")
        print("\n   Steps:")
        print("   1. Go to https://aistudio.google.com")
        print("   2. Sign in with Google")
        print("   3. Get API Key → Create API key")
        print("   4. Add to .env: GEMINI_API_KEY=AIza...")
        return None

    print(f"✅ API key found: {GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-4:]}")

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ New google-genai client created")
        print("   SDK: google-genai (not deprecated)")
        return client
    except ImportError:
        print("❌ google-genai not installed")
        print("   Run: pip install google-genai")
        return None
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


# ── Test 2: List models ───────────────────────────────────────
def test_list_models(client):
    print("\n" + "="*60)
    print("TEST 2: Available Models")
    print("="*60)

    try:
        models = list(client.models.list())
        gen_models = [
            m for m in models
            if "generateContent" in (m.supported_actions or [])
            or hasattr(m, 'name')
        ]
        print(f"✅ Models found: {len(models)}")
        print("\nTop models:")
        for m in models[:15]:
            name = getattr(m, 'name', str(m))
            name = name.replace("models/", "")
            print(f"  • {name}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


# ── Test 3: Simple generation ─────────────────────────────────
def test_all_models(client):
    print("\n" + "="*60)
    print("TEST 3: Simple Generation — All Models")
    print("="*60)

    from google.genai import types

    prompt  = "What is 2 + 2? Reply in one word only."
    results = {}

    for model_id, info in GEMINI_MODELS.items():
        print(f"\n  Model: {model_id}")
        print(f"  Info:  {info['description']}")

        try:
            start    = time.time()
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
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

            print(f"  ✅ Answer:     {answer}")
            print(f"     Latency:   {latency}ms")
            print(f"     Tokens:    {input_tok} in + {out_tok} out = {total_tok}")
            print(f"     Cost:      ${cost:.8f}")
            print(f"     Free RPD:  {info['free_rpd']} requests/day")

            results[model_id] = {
                "status":     "ok",
                "answer":     answer,
                "latency_ms": latency,
                "tokens":     total_tok,
                "cost_usd":   cost,
            }

            time.sleep(4)   # respect 15 RPM limit

        except Exception as e:
            err = str(e)[:120]
            print(f"  ❌ Failed: {err}")

            # Diagnose the error
            if "429" in err and "limit: 0" in err:
                print(f"  ℹ️  Quota limit is 0 — possible causes:")
                print(f"     1. Gemini API not enabled in your project")
                print(f"     2. Daily free quota exhausted")
                print(f"     3. Region restriction")
                print(f"     Fix: https://console.cloud.google.com/apis/library")
                print(f"          Search 'Generative Language API' → Enable")
            elif "429" in err:
                print(f"  ℹ️  Rate limited — wait 60 seconds and retry")
            elif "404" in err:
                print(f"  ℹ️  Model not found in this API version")

            results[model_id] = {"status": "failed", "error": err}

    return results


# ── Test 4: RAG prompt ────────────────────────────────────────
def test_rag_prompt(client):
    print("\n" + "="*60)
    print("TEST 4: RAG-Style Prompt")
    print("="*60)

    from google.genai import types

    context = """
    The fiscal deficit target for 2026-27 is 4.4 percent of GDP.
    The fiscal deficit estimate for 2025-26 is 4.8 percent of GDP.
    Total expenditure for 2026-27 is estimated at Rs. 50,65,345 crore.
    Capital expenditure is projected at Rs. 11,21,490 crore for 2026-27.
    """
    question = "What is the fiscal deficit target for 2026-27?"
    prompt   = f"""Answer using only the context provided.

Context: {context}
Question: {question}
Answer:"""

    model_id = "gemini-2.0-flash"
    print(f"  Model:    {model_id}")
    print(f"  Question: {question}")

    try:
        start    = time.time()
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=200,
            )
        )
        latency   = round((time.time() - start) * 1000)
        answer    = response.text.strip()
        input_tok = response.usage_metadata.prompt_token_count
        out_tok   = response.usage_metadata.candidates_token_count
        info      = GEMINI_MODELS[model_id]
        cost      = (
            (input_tok / 1_000_000) * info["cost_per_1M_input"] +
            (out_tok   / 1_000_000) * info["cost_per_1M_output"]
        )

        print(f"\n  ✅ Answer:  {answer}")
        print(f"     Latency: {latency}ms")
        print(f"     Tokens:  {input_tok} in + {out_tok} out")
        print(f"     Cost:    ${cost:.8f}")
        return True

    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:120]}")
        return False


# ── Test 5: JSON output ───────────────────────────────────────
def test_json_output(client):
    print("\n" + "="*60)
    print("TEST 5: JSON Output (Judge Use Case)")
    print("="*60)

    from google.genai import types

    prompt = """Evaluate faithfulness of this RAG answer.

Context: The fiscal deficit target for 2026-27 is 4.4% of GDP.
Answer:  The fiscal deficit target for 2026-27 is 4.4% of GDP.

Return ONLY valid JSON, no markdown:
{"score": 1.0, "reason": "explanation"}"""

    model_id = "gemini-2.0-flash"
    print(f"  Model: {model_id}")

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=200,
                system_instruction=(
                    "You are an evaluation assistant. "
                    "Always respond with valid JSON only. No markdown."
                )
            )
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        print(f"  Raw: {raw}")

        parsed = json.loads(raw)
        print(f"  ✅ Valid JSON: score={parsed.get('score')}")
        print(f"     Reason: {parsed.get('reason')}")
        return True

    except json.JSONDecodeError:
        print(f"  ⚠️  Invalid JSON returned")
        return False
    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:120]}")
        return False


# ── Test 6: Diagnose quota issue ──────────────────────────────
def diagnose_quota():
    print("\n" + "="*60)
    print("TEST 6: Quota Diagnosis")
    print("="*60)

    print("""
  If you saw 'limit: 0' errors, here is what to check:

  Step 1 — Enable the API:
    https://console.cloud.google.com/apis/library
    Search: "Generative Language API"
    Click: ENABLE

  Step 2 — Check your project:
    https://aistudio.google.com
    Make sure you are using the same Google account
    that has the API key

  Step 3 — Check rate limits:
    https://ai.dev/rate-limit
    Free tier: 15 req/min, 1500 req/day for gemini-2.0-flash

  Step 4 — Try gemini-2.0-flash-lite instead:
    It has 30 req/min limit (more generous)

  Step 5 — If all else fails:
    Add billing to Google Cloud ($0 charge unless you exceed free)
    This unlocks higher quotas immediately
    Go to: https://console.cloud.google.com/billing
    """)


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GOOGLE GEMINI API — TEST SUITE v2 (New SDK)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    client = test_connectivity()
    if not client:
        print("\n❌ Cannot proceed — fix API key first")
        return

    test_list_models(client)
    model_results = test_all_models(client)
    test_rag_prompt(client)
    test_json_output(client)
    diagnose_quota()

    # Summary
    print("\n" + "="*60)
    print("GEMINI TEST SUMMARY")
    print("="*60)

    working = [m for m, r in model_results.items() if r["status"] == "ok"]
    failed  = [m for m, r in model_results.items() if r["status"] != "ok"]

    if working:
        print(f"✅ Working models: {len(working)}")
        for m in working:
            r = model_results[m]
            print(f"   • {m}: {r['latency_ms']}ms | {r['tokens']} tokens")
    else:
        print("❌ No models working")
        print("\n   Most likely cause: Gemini API not enabled")
        print("   Fix: https://console.cloud.google.com/apis/library")
        print("        Search 'Generative Language API' → Enable")

    if failed:
        print(f"\n❌ Failed models: {len(failed)}")
        for m in failed:
            print(f"   • {m}")

    print(f"\n📋 Gemini vs Groq comparison:")
    print(f"   Groq:   ✅ Works immediately, 200ms, 500K tokens/day")
    print(f"   Gemini: ⚠️  Needs API enabled, 1M context window")
    print(f"\n📁 Next step: python llm_tests/test_openai.py")


if __name__ == "__main__":
    main()