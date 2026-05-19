"""
Evaluation with Decorators + Custom Scoring

Shows how to combine @observe decorators with custom quality metrics.
"""

from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse
from llm_utils import get_llm_client
from config import get_llm_config, get_evaluation_config
import time

# Initialize Langfuse for manual scoring
langfuse = Langfuse()

# Initialize vLLM client
client, model = get_llm_client(load_model_from_config=True)


@observe(name="generate_and_score")
def generate_and_score(question: str) -> tuple[str, float]:
    """
    Generate answer with @observe decorator + custom quality scoring.
    
    The decorator handles tracing, we add custom metrics.
    """
    
    # Load configs
    llm_config = get_llm_config()
    eval_config = get_evaluation_config()
    
    temperature = llm_config.get("temperature", 0.7)
    max_tokens = llm_config.get("max_tokens", 300)
    
    min_length = eval_config.get("min_length", 20)
    good_length_threshold = eval_config.get("good_length_threshold", 100)
    max_latency_ms = eval_config.get("max_latency_ms", 5000)
    
    try:
        # Update trace with input
        langfuse_context.update_current_observation(
            input={"question": question, "model": model}
        )
        
        # Track latency
        start_time = time.time()
        
        # Make LLM call
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
        answer_length = len(answer)
        
        # Calculate quality score
        if answer_length < min_length:
            quality_score = 0.3
        elif answer_length >= good_length_threshold:
            quality_score = 1.0
        else:
            quality_score = 0.3 + (0.7 * (answer_length - min_length) / (good_length_threshold - min_length))
        
        # Update observation with results and custom metrics
        langfuse_context.update_current_observation(
            output={"answer": answer, "quality_score": quality_score},
            usage={
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            },
            metadata={
                "latency_ms": round(latency_ms, 2),
                "answer_length": answer_length
            }
        )
        
        # Score the trace
        langfuse_context.score_current_observation(
            name="quality",
            value=quality_score,
            comment=f"Based on answer length ({answer_length} chars)"
        )
        
        print(f"📊 Quality Score: {quality_score:.2f} (answer length: {answer_length} chars)")
        print(f"📊 Latency: {latency_ms:.2f}ms")
        print(f"📊 Tokens: {response.usage.prompt_tokens} → {response.usage.completion_tokens}")
        
        if latency_ms > max_latency_ms:
            print(f"⚠️  Latency warning: {round(latency_ms, 2)}ms > {max_latency_ms}ms")
        
        return answer, quality_score
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tip: Make sure vLLM is running (docker-compose up -d)")
        raise


@observe(name="evaluation_pipeline")
def run_evaluation(question: str):
    """Wrapper to create a trace context for the evaluation."""
    from datetime import datetime
    
    # Add timestamp to make each run unique
    langfuse_context.update_current_trace(
        metadata={"run_time": datetime.now().isoformat()}
    )
    
    answer, score = generate_and_score(question)
    
    print(f"\n✅ Answer: {answer}\n")
    print(f"📊 Quality Score: {score:.2f}\n")
    
    trace_id = langfuse_context.get_current_trace_id()
    if trace_id:
        print(f"🔍 View trace with scores: https://cloud.langfuse.com/trace/{trace_id}")
        print(f"📋 Trace ID: {trace_id}")
    print("="*50 + "\n")
    
    return answer, score


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Evaluation with Custom Scoring")
    print("="*50 + "\n")
    
    question = "What are neural networks?"
    print(f"Question: {question}\n")
    
    run_evaluation(question)
    
    # Flush traces before exit (critical for short-lived scripts)
    print("⏳ Flushing traces to Langfuse...")
    langfuse.flush()
    print("✅ Traces sent!\n")
    
    print("✅ Check your dashboard: https://cloud.langfuse.com")
    print("   (Traces may take 10-30 seconds to appear)\n")
