"""
Minimal LangFuse Utilities with Self-Hosted Support

Simple helpers for production observability.
"""

import os
from pathlib import Path
from langfuse import Langfuse

# Load .env file if exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env loading")

# Log Langfuse configuration at module load
print("\n" + "="*70)
print("🔧 LANGFUSE CONFIGURATION")
print("="*70)
print(f"📍 LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST', 'NOT SET')}")
print(f"🔑 LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY', 'NOT SET')[:20]}...")
print(f"🔐 LANGFUSE_SECRET_KEY: {os.getenv('LANGFUSE_SECRET_KEY', 'NOT SET')[:20]}...")
print("="*70 + "\n")

# Initialize client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)


def flush_traces():
    """Ensure all traces are sent to LangFuse."""
    langfuse_client.flush()
    print("✅ Traces flushed to LangFuse")


if __name__ == "__main__":
    print("LangFuse instrumentation utilities loaded")
    flush_traces()
