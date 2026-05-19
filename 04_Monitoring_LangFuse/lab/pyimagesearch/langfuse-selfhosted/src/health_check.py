"""
vLLM Health Check

Verifies that vLLM server is running and accessible.
Run this before other examples to ensure everything is working.
"""

import sys
import httpx
from llm_utils import get_llm_client
from config import get_llm_config


def check_vllm_health(base_url: str = None, timeout: int = 5) -> bool:
    """
    Check if vLLM server is healthy.
    
    Args:
        base_url: vLLM server base URL (defaults to config.yaml)
        timeout: Request timeout in seconds
        
    Returns:
        True if server is healthy, False otherwise
    """
    # Load base_url from config if not provided
    if base_url is None:
        llm_config = get_llm_config()
        base_url = llm_config.get("base_url", "http://localhost:8000/v1")
        base_url = base_url.rstrip("/v1")
    
    health_url = f"{base_url}/health"
    models_url = f"{base_url}/v1/models"
    
    print(f"🔍 Checking vLLM server at {base_url}...")
    
    try:
        # Check health endpoint
        with httpx.Client(timeout=timeout) as client:
            response = client.get(health_url)
            if response.status_code == 200:
                print(f"  ✅ Health check passed")
            else:
                print(f"  ❌ Health check failed (status: {response.status_code})")
                return False
        
        # Check models endpoint
        with httpx.Client(timeout=timeout) as client:
            response = client.get(models_url)
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    print(f"  ✅ Models available: {[m['id'] for m in models]}")
                else:
                    print(f"  ⚠️  No models loaded yet (still initializing?)")
                    return False
            else:
                print(f"  ❌ Models endpoint failed (status: {response.status_code})")
                return False
        
        return True
        
    except httpx.ConnectError:
        print(f"  ❌ Connection failed - is vLLM running?")
        print(f"     Start with: docker-compose up -d")
        return False
    except httpx.TimeoutException:
        print(f"  ❌ Request timed out")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def test_llm_generation() -> bool:
    """Test simple LLM generation."""
    print("\n🔍 Testing LLM generation...")
    
    try:
        client = get_llm_client(timeout=30)
        response = client.chat.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Say 'OK' if you're working."}],
            max_tokens=10
        )
        
        answer = response.choices[0].message.content
        print(f"  ✅ Generation successful: {answer[:50]}...")
        return True
        
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        return False


def main():
    """Run all health checks."""
    print("\n" + "="*60)
    print("vLLM Health Check")
    print("="*60 + "\n")
    
    # Check vLLM health
    health_ok = check_vllm_health()
    
    if not health_ok:
        print("\n❌ vLLM is not ready. Please:")
        print("   1. Start vLLM: docker-compose up -d")
        print("   2. Wait 2-3 minutes for model download")
        print("   3. Run this script again")
        sys.exit(1)
    
    # Test generation
    generation_ok = test_llm_generation()
    
    if not generation_ok:
        print("\n⚠️  vLLM is running but generation failed.")
        print("   Check docker logs: docker-compose logs vllm")
        sys.exit(1)
    
    # All good!
    print("\n" + "="*60)
    print("✅ All checks passed! vLLM is ready to use.")
    print("="*60 + "\n")
    print("You can now run the other examples:")
    print("  python src/basic_llm_app.py")
    print("  python src/tracing_manual.py")
    print("  python src/tracing_decorator.py")
    print("  python src/evaluation_metrics.py")
    print()


if __name__ == "__main__":
    main()
