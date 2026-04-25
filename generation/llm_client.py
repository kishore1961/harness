import os
import time
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Model registries ──────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.1-8b":  "llama-3.1-8b-instant",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
}

ANTHROPIC_MODELS = {
    "claude-haiku":  "claude-haiku-4-5-20251001",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
}

ALL_MODELS = {**GROQ_MODELS, **ANTHROPIC_MODELS}

# ── Read default model from .env ──────────────────────────────────
# Set LLM_MODEL_KEY in .env to control which model runs.
# e.g. LLM_MODEL_KEY=claude-haiku  or  LLM_MODEL_KEY=llama-3.1-8b
DEFAULT_MODEL_KEY = os.getenv("LLM_MODEL_KEY", "claude-haiku")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model_key: str = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> dict:
    # Fall back to .env default if caller doesn't specify
    model_key = model_key or DEFAULT_MODEL_KEY

    if model_key not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(ALL_MODELS.keys())}")

    start_time = time.time()

    if model_key in GROQ_MODELS:
        result = _call_groq(system_prompt, user_prompt, GROQ_MODELS[model_key], temperature, max_tokens)
    else:
        result = _call_anthropic(system_prompt, user_prompt, ANTHROPIC_MODELS[model_key], temperature, max_tokens)

    latency_ms = (time.time() - start_time) * 1000
    result["latency_ms"] = round(latency_ms, 2)
    result["model_key"] = model_key

    logger.debug(
        f"LLM call | model={model_key} | latency={latency_ms:.0f}ms | "
        f"tokens={result.get('total_tokens', 'N/A')} | cost=${result.get('cost_usd', 0)}"
    )
    return result


def _call_groq(system_prompt, user_prompt, model_name, temperature, max_tokens) -> dict:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    max_retries = 5
    wait = 15

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {
                "answer":        response.choices[0].message.content,
                "input_tokens":  response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens":  response.usage.total_tokens,
                "provider":      "groq",
                "model_name":    model_name,
                "cost_usd":      0.0   # Groq free tier — no billing
            }
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                logger.warning(f"Groq rate limit. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                raise e

    raise RuntimeError("Max retries exceeded for Groq")


def _call_anthropic(system_prompt, user_prompt, model_name, temperature, max_tokens) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    # Haiku pricing: $0.25/M input, $1.25/M output
    cost = (input_tokens * 0.00000025) + (output_tokens * 0.00000125)

    return {
        "answer":        response.content[0].text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "provider":      "anthropic",
        "model_name":    model_name,
        "cost_usd":      round(cost, 8)
    }


if __name__ == "__main__":
    print(f"Active model from .env: {DEFAULT_MODEL_KEY}\n")
    test_system = "You are a helpful assistant."
    test_user   = "What is 2+2? Answer in one sentence."

    result = call_llm(test_system, test_user)
    print(f"Answer:  {result['answer']}")
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Tokens:  {result['total_tokens']}")
    print(f"Cost:    ${result['cost_usd']}")