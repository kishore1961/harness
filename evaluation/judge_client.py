import os
import time
import re
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")

JSON_SYSTEM_PROMPT = """You are an evaluation assistant.
You MUST respond with valid JSON only.
Do not include any text before or after the JSON.
Do not use backslashes inside string values.
Do not use newlines inside string values.
Use only simple ASCII characters in your responses."""


def get_deepeval_judge():
    from deepeval.models.base_model import DeepEvalBaseLLM
    from groq import Groq
    import anthropic

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        def load_model(self):
            return self.client

        def _clean(self, text: str) -> str:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', text)
            return text.strip()

        def generate(self, prompt: str, schema=None) -> str:
            max_retries = 5
            wait = 10  # start with 10 seconds

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": JSON_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=2048
                    )
                    return self._clean(response.choices[0].message.content)

                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        logger.warning(f"Rate limit hit. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                        wait *= 2  # exponential backoff
                    else:
                        raise e

            raise RuntimeError("Max retries exceeded for Groq judge")

        async def a_generate(self, prompt: str, schema=None) -> str:
            return self.generate(prompt, schema=schema)

        def get_model_name(self) -> str:
            return self.model_name

    class AnthropicJudge(DeepEvalBaseLLM):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        def load_model(self):
            return self.client

        def _clean(self, text: str) -> str:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', text)
            return text.strip()

        def generate(self, prompt: str, schema=None) -> str:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=JSON_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._clean(response.content[0].text)

        async def a_generate(self, prompt: str, schema=None) -> str:
            return self.generate(prompt, schema=schema)

        def get_model_name(self) -> str:
            return self.model_name

    if JUDGE_PROVIDER == "groq":
        logger.info(f"Judge: Groq / {JUDGE_MODEL}")
        return GroqJudge(model_name=JUDGE_MODEL)
    elif JUDGE_PROVIDER == "anthropic":
        model = os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-3-haiku-20240307")
        logger.info(f"Judge: Anthropic / {model}")
        return AnthropicJudge(model_name=model)
    else:
        raise ValueError(f"Unknown judge provider: {JUDGE_PROVIDER}")


if __name__ == "__main__":
    judge = get_deepeval_judge()
    result = judge.generate('Return this exact JSON: {"truths": ["The sky is blue"]}')
    print(f"Judge response: {result}")
    print(f"Type: {type(result)}")
