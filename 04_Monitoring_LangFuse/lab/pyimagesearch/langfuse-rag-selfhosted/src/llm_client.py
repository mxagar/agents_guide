"""
Simple LLM Client with Tracing

vLLM-first client with basic retry logic and config-driven setup.
"""

import time
import os
from pathlib import Path
from typing import List, Dict
from langfuse import observe, get_client

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from llm_utils import get_llm_client
from config import get_llm_config


class TracedLLMClient:
    """LLM client with LangFuse tracing and retry logic."""
    
    def __init__(self, model: str = None, max_retries: int = 2, timeout: int = 60):
        """
        Initialize traced LLM client.
        
        Args:
            model: Model name (if None, loads from config)
            max_retries: Number of retry attempts
            timeout: Request timeout in seconds
        """
        self.client = get_llm_client(timeout=timeout)
        
        # Load model from config if not specified
        if model is None:
            llm_config = get_llm_config()
            model = llm_config.get("model", "meta-llama/Llama-2-7b-chat-hf")
        
        self.model = model
        self.max_retries = max_retries
    
    @observe(name="llm_completion", capture_input=False)
    def complete(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Generate completion with tracing and retry logic.
        
        Args:
            messages: Chat messages in OpenAI format
            **kwargs: Additional completion parameters
            
        Returns:
            Dict with 'content' and metadata
        """
        # Load default parameters from config
        llm_config = get_llm_config()
        temperature = kwargs.get("temperature", llm_config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", llm_config.get("max_tokens", 300))
        
        get_client().update_current_span(
            input={"messages": messages},
            metadata={
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "feature": "rag-generation",
            },
        )
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    name="rag-answer-generation",
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata={
                        "attempt": attempt + 1,
                        "feature": "rag-generation",
                    },
                )
                end_time = time.time()
                
                content = response.choices[0].message.content
                usage = response.usage.model_dump() if response.usage else {}
                
                get_client().update_current_span(
                    output={"content": content},
                    metadata={
                        "attempt": attempt + 1,
                        "model": self.model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "latency_ms": round((end_time - start_time) * 1000, 2)
                    }
                )
                
                return {
                    "content": content,
                    "usage": usage,
                    "success": True
                }
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
        
        # All retries failed
        error_msg = f"LLM call failed after {self.max_retries} attempts: {last_error}"
        get_client().update_current_span(
            level="ERROR",
            status_message=error_msg,
            output={"error": error_msg}
        )
        return {"content": None, "error": error_msg, "success": False}


if __name__ == "__main__":
    # Example usage
    client = TracedLLMClient()
    
    result = client.complete(
        messages=[
            {"role": "user", "content": "What is RAG in AI?"}
        ]
    )
    
    print(f"Response: {result['content']}")
    print(f"Tokens: {result.get('usage', {}).get('total_tokens', 'unknown')}")
    
    # View trace
    langfuse = get_client()
    trace_id = langfuse.get_current_trace_id()
    langfuse_base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    print(f"🔍 View trace: {langfuse_base_url}/trace/{trace_id}")
    langfuse.flush()
