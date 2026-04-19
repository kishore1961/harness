import yaml
from pathlib import Path
from loguru import logger


def load_prompt(version: str = "1.0") -> dict:
    """Load prompt template by version."""
    path = Path(f"prompts/v{version}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Prompt version {version} not found at {path}")
    with open(path) as f:
        prompt = yaml.safe_load(f)
    logger.debug(f"Loaded prompt version {version}")
    return prompt


def build_prompt(
    question: str,
    context_chunks: list[dict],
    version: str = "1.0"
) -> tuple[str, str]:
    """
    Assemble system + user prompt from template.
    Returns (system_prompt, user_prompt).
    """
    prompt_config = load_prompt(version)
    context = "\n\n---\n\n".join([
        f"[Page {c['metadata']['page_number']}]\n{c['text']}"
        for c in context_chunks
    ])
    user_prompt = prompt_config["user_template"].format(
        context=context,
        question=question
    )
    return prompt_config["system_prompt"], user_prompt


if __name__ == "__main__":
    # Test with dummy chunks
    dummy_chunks = [
        {"text": "The fiscal deficit is 5.1% of GDP", "metadata": {"page_number": 3}},
        {"text": "PM Awas Yojana gets increased allocation", "metadata": {"page_number": 7}}
    ]
    system, user = build_prompt(
        "What is the fiscal deficit?",
        dummy_chunks,
        version="1.0"
    )
    print("SYSTEM PROMPT:")
    print(system)
    print("\nUSER PROMPT:")
    print(user)