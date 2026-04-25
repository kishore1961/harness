# evaluation/judge_client.py

import os
import re
import time
import json
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "anthropic")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")

JSON_SYSTEM_PROMPT = """You are an evaluation assistant.
You MUST respond with valid JSON only.
Do not include any text before or after the JSON.
Do not use backslashes inside string values.
Do not use newlines inside string values.
Use only simple ASCII characters in your responses."""


def get_deepeval_judge():
    from deepeval.models.base_model import DeepEvalBaseLLM

    def _clean(text: str) -> str:
        """Strip markdown fences and normalize JSON."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r'\\(?!["\/bfnrt]|u[0-9a-fA-F]{4})', r'', text)
        return text.strip()

    def _parse_with_schema(raw: str, schema):
        """
        If DeepEval passes a Pydantic schema, parse JSON and
        return a schema instance. Otherwise return raw string.
        """
        if schema is None:
            return raw
        try:
            data = json.loads(raw)
            return schema(**data)
        except Exception:
            # Return raw string and let DeepEval handle it
            return raw

    # ── Groq Judge ────────────────────────────────────────────────
    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self, model_name: str):
            self.model_name = model_name
            from groq import Groq
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        def load_model(self):
            return self.client

        def generate(self, prompt: str, schema=None) -> str:
            max_retries = 5
            wait = 10
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": JSON_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=2048
                    )
                    raw = _clean(response.choices[0].message.content)
                    return _parse_with_schema(raw, schema)
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        logger.warning(f"Groq rate limit. Waiting {wait}s... ({attempt+1}/{max_retries})")
                        time.sleep(wait)
                        wait *= 2
                    else:
                        raise e
            raise RuntimeError("Max retries exceeded for Groq judge")

        async def a_generate(self, prompt: str, schema=None) -> str:
            return self.generate(prompt, schema=schema)

        def get_model_name(self) -> str:
            return self.model_name

    # ── Anthropic Judge ───────────────────────────────────────────
    class AnthropicJudge(DeepEvalBaseLLM):
        def __init__(self, model_name: str):
            self.model_name = model_name
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        def load_model(self):
            return self.client

        def generate(self, prompt: str, schema=None):
            max_retries = 4
            wait = 8

            for attempt in range(max_retries):
                try:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=2048,
                        temperature=0,        # deterministic → fewer JSON errors
                        system=JSON_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    raw = _clean(response.content[0].text)

                    # Validate it is parseable JSON before returning
                    json.loads(raw)

                    return _parse_with_schema(raw, schema)

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"AnthropicJudge got invalid JSON "
                        f"(attempt {attempt+1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {wait}s...")
                        time.sleep(wait)
                        wait *= 2
                    else:
                        logger.error("All retries exhausted — returning empty JSON")
                        # Return a safe fallback so DeepEval scores 0
                        # instead of crashing the entire experiment
                        fallback = '{"reason": "parse_error", "score": 0}'
                        return _parse_with_schema(fallback, schema)

                except Exception as e:
                    if "overloaded" in str(e).lower() or "529" in str(e):
                        logger.warning(
                            f"Anthropic overloaded, waiting {wait}s "
                            f"(attempt {attempt+1}/{max_retries})"
                        )
                        time.sleep(wait)
                        wait *= 2
                    else:
                        raise

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema=schema)

        def get_model_name(self) -> str:
            return self.model_name

    # ── Route by JUDGE_PROVIDER ───────────────────────────────────
    if JUDGE_PROVIDER == "groq":
        model = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
        logger.info(f"Judge: Groq / {model}")
        return GroqJudge(model_name=model)
    elif JUDGE_PROVIDER == "anthropic":
        model = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")
        logger.info(f"Judge: Anthropic / {model}")
        return AnthropicJudge(model_name=model)
    else:
        raise ValueError(
            f"Unknown JUDGE_PROVIDER: '{JUDGE_PROVIDER}'. Use 'groq' or 'anthropic'"
        )


if __name__ == "__main__":
    print(f"JUDGE_PROVIDER={JUDGE_PROVIDER}  JUDGE_MODEL={JUDGE_MODEL}\n")
    judge = get_deepeval_judge()
    result = judge.generate('Return this exact JSON: {"truths": ["The sky is blue"]}')
    print(f"Judge response: {result}")
    print(f"Type: {type(result)}")