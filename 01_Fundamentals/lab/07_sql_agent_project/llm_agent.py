"""Smoke test for the OpenAI chat model used by the SQL agent project."""

from pathlib import Path
import os
import warnings

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_environment() -> None:
    """Load environment variables from local and repository .env files.

    Precedence is:
    1. Current working directory .env
    2. This SQL project directory .env
    3. Repository root .env

    Later files fill only missing values.
    """
    env_paths = [
        Path.cwd() / ".env",
        PROJECT_DIR / ".env",
        PROJECT_ROOT / ".env",
    ]
    seen = set()

    for env_path in env_paths:
        resolved = env_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            load_dotenv(resolved, override=False)


def create_llm() -> ChatOpenAI:
    """Create the LangChain OpenAI chat model."""
    load_environment()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Add it to .env in the current directory, "
            "the SQL project folder, or the repository root."
        )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "512")),
    )


if __name__ == "__main__":
    llm = create_llm()
    response = llm.invoke("What is the capital of Ontario? Answer in one short sentence.")
    print(response.content)
