"""Natural-language SQL agent for the Chinook MySQL database.

Environment variables are loaded from .env in the current directory, this
project directory, or the repository root.
Required:
    OPENAI_API_KEY

Optional:
    OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
    MYSQL_USERNAME, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE
"""

from pathlib import Path
from urllib.parse import quote_plus
import argparse
import os
import warnings

from dotenv import load_dotenv
from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase

from llm_agent import create_llm


warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT = "How many Album are there in the database?"


def load_environment() -> None:
    """Load local and repository environment variables.

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


def mysql_uri_from_env() -> str:
    """Build a SQLAlchemy MySQL URI from environment variables."""
    load_environment()

    username = os.getenv("MYSQL_USERNAME", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    database = os.getenv("MYSQL_DATABASE", "Chinook")

    escaped_username = quote_plus(username)
    escaped_password = quote_plus(password)

    return (
        f"mysql+mysqlconnector://{escaped_username}:{escaped_password}"
        f"@{host}:{port}/{database}"
    )


def create_database() -> SQLDatabase:
    """Create the LangChain SQLDatabase wrapper."""
    return SQLDatabase.from_uri(mysql_uri_from_env())


def create_agent(verbose: bool = True):
    """Create a LangChain SQL agent connected to Chinook."""
    llm = create_llm()
    db = create_database()

    return create_sql_agent(
        llm=llm,
        db=db,
        verbose=verbose,
        handle_parsing_errors=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask natural-language questions of Chinook.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Natural-language question to send to the SQL agent.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose LangChain agent tracing.",
    )
    args = parser.parse_args()

    agent_executor = create_agent(verbose=not args.quiet)
    result = agent_executor.invoke({"input": args.prompt})

    print("\nNatural Language Query:")
    print(args.prompt)
    print("\nResponse:")
    print(result["output"])


if __name__ == "__main__":
    main()
