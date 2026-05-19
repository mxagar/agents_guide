"""
Manual Tracing with Low-Level Langfuse API

Shows explicit trace creation and management using Langfuse SDK directly.
This gives you full control but requires more code compared to decorators.
"""

from langfuse import Langfuse
from llm_utils import get_llm_client
from config import get_llm_config
import time

# Initialize Langfuse client
langfuse = Langfuse()

# Initialize vLLM client
client, model = get_llm_client(load_model_from_config=True)

# Get configuration
llm_config = get_llm_config()
temperature = llm_config.get("temperature", 0.7)
max_tokens = llm_config.get("max_tokens", 300)


def generate_with_manual_tracing(question: str) -> str:
    """
    Generate answer WITH manual trace creation.
    
    This gives you full control over every trace property:
    - Custom trace names and IDs
    - Granular span creation
    - Manual token counting
    - Custom metadata
    """
    
    print("🟡 Calling LLM with manual tracing...")
    
    # 1. Create trace manually
    trace = langfuse.trace(
        name="manual_llm_call",
        metadata={"method": "manual", "question": question}
    )
    
    # 2. Create span for LLM generation
    start_time = time.time()
    
    generation = trace.generation(
        name="llm_generation",
        model=model,
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        metadata={
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    )
    
    # 3. Make the actual LLM call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    latency_ms = (time.time() - start_time) * 1000
    answer = response.choices[0].message.content
    
    # 4. Update generation with results
    generation.update(
        output=answer,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        },
        metadata={
            "latency_ms": round(latency_ms, 2)
        }
    )
    
    print(f"   Tokens used: {response.usage.total_tokens}")
    print(f"   Latency: {latency_ms:.2f}ms")
    print(f"   ✅ Manually logged to Langfuse")
    print(f"   🔍 Trace ID: {trace.id}\n")
    
    return answer


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Manual Tracing Demo")
    print("="*70 + "\n")
    
    question = "What is deep learning?"
    print(f"Question: {question}\n")
    print("-" * 70 + "\n")
    
    # Generate with manual tracing
    answer = generate_with_manual_tracing(question)
    print(f"Answer: {answer}\n")
    
    print("=" * 70)
    print("\n📊 Manual Tracing vs Decorators:")
    print("   Manual (this file):")
    print("   • Full control over trace structure")
    print("   • More verbose code")
    print("   • Good for complex custom logging")
    print()
    print("   Decorators (recommended):")
    print("   • Clean @observe annotation")
    print("   • Less boilerplate")
    print("   • Automatic nesting")
    print("   • See: src/tracing_decorator.py")
    print("\n🔍 Check your dashboard: https://cloud.langfuse.com")
    print("=" * 70 + "\n")
    
    # Flush traces
    langfuse.flush()
