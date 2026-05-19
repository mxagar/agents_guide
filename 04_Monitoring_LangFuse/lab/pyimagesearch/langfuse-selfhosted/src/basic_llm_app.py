"""
Basic LLM Application (No Tracing Baseline)

Simple pipeline using local vLLM server.
This version has NO tracing - compare with tracing_decorator.py
"""

from llm_utils import get_llm_client
from config import get_llm_config

# Initialize vLLM client with model from config
client, model = get_llm_client(load_model_from_config=True)


def generate_answer(question: str) -> str:
    """Generate answer using vLLM - NO tracing."""
    # Load config
    llm_config = get_llm_config()
    temperature = llm_config.get("temperature", 0.7)
    max_tokens = llm_config.get("max_tokens", 300)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Make sure vLLM is running (docker-compose up -d)")
        raise


def run_simple_pipeline(question: str):
    """Simple pipeline without tracing - baseline example."""
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}\n")
    
    print("Generating answer (no tracing)...")
    answer = generate_answer(question)
    
    print(f"✅ Answer:\n{answer}\n")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    question = "What is machine learning?"
    run_simple_pipeline(question)
