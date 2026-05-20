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

if not os.getenv("LANGFUSE_BASE_URL") and os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]

# Log Langfuse configuration at module load
print("\n" + "="*70)
print("🔧 LANGFUSE CONFIGURATION")
print("="*70)
print(f"📍 LANGFUSE_BASE_URL: {os.getenv('LANGFUSE_BASE_URL', 'NOT SET')}")
print(f"🔑 LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY', 'NOT SET')[:20]}...")
print(f"🔐 LANGFUSE_SECRET_KEY: {os.getenv('LANGFUSE_SECRET_KEY', 'NOT SET')[:20]}...")
print("="*70 + "\n")

# Initialize client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    base_url=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
    environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
)


def flush_traces():
    """Ensure all traces are sent to LangFuse."""
    langfuse_client.flush()
    print("✅ Traces flushed to LangFuse")


def shutdown_tracing():
    """Flush pending traces and shut down the Langfuse exporter cleanly."""
    langfuse_client.shutdown()
    print("✅ LangFuse tracing shut down cleanly")


if __name__ == "__main__":
    print("LangFuse instrumentation utilities loaded")
    flush_traces()
