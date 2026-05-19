"""
Decorator-Based Tracing (Recommended Approach)

Uses @observe decorator for automatic tracing.
This is the cleanest and recommended way to add observability with Langfuse 2.x.
"""

import os
from langfuse.decorators import observe, langfuse_context
from llm_utils import get_llm_client
from config import get_llm_config

# Log Langfuse configuration at startup
print("\n" + "="*70)
print("🔧 LANGFUSE CONFIGURATION")
print("="*70)
print(f"📍 LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST', 'NOT SET')}")
print(f"🔑 LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY', 'NOT SET')[:20]}...")
print(f"🔐 LANGFUSE_SECRET_KEY: {os.getenv('LANGFUSE_SECRET_KEY', 'NOT SET')[:20]}...")
print("="*70 + "\n")

# Initialize vLLM client with model from config
client, model = get_llm_client(load_model_from_config=True)


@observe(name="generate_answer")
def generate_answer(question: str) -> str:
    """
    Generate answer with automatic tracing via @observe decorator.
    """
    import time
    
    # Load config
    llm_config = get_llm_config()
    temperature = llm_config.get("temperature", 0.7)
    max_tokens = llm_config.get("max_tokens", 300)
    
    langfuse_context.update_current_observation(
        input={"question": question, "model": model}
    )
    
    try:
        # Track latency
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        answer = response.choices[0].message.content
        
        # Update observation with usage metrics and latency
        langfuse_context.update_current_observation(
            output={"answer": answer},
            usage={
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            },
            metadata={"latency_ms": round(latency_ms, 2)}
        )
        
        print(f"📊 Latency: {latency_ms:.2f}ms")
        print(f"📊 Tokens: {response.usage.prompt_tokens} → {response.usage.completion_tokens} (total: {response.usage.total_tokens})")
        
        return answer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tip: Make sure vLLM is running (docker-compose up -d)")
        raise


@observe(name="llm_pipeline")
def run_pipeline(question: str):
    """
    Pipeline with nested @observe functions.
    The decorator automatically creates parent-child trace relationships.
    """
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}\n")
    
    langfuse_context.update_current_trace(
        name="decorator_pipeline",
        metadata={"method": "decorator"}
    )
    
    print("Generating answer with tracing...")
    answer = generate_answer(question)
    
    print(f"\n✅ Answer: {answer}\n")
    
    langfuse_context.update_current_observation(
        output={"answer": answer}
    )
    
    trace_id = langfuse_context.get_current_trace_id()
    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    print(f"🔍 View trace: {langfuse_host}/trace/{trace_id}")
    print(f"{'='*50}\n")
    
    return answer


if __name__ == "__main__":
    question = "Explain neural networks briefly"
    run_pipeline(question)
    
    # Flush traces before exit (critical for short-lived scripts)
    print("⏳ Flushing traces to Langfuse...")
    langfuse_context.flush()
    print("✅ Traces sent!\n")
