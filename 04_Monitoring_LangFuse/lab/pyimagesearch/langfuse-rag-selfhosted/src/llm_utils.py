""" 
Reusable LLM client utilities.
Provides consistent OpenAI-compatible client configuration across all examples.
"""

import os
from typing import Union, Tuple
from langfuse.openai import OpenAI


def get_llm_client(timeout: int = 60, load_model_from_config: bool = False) -> Union[OpenAI, Tuple[OpenAI, str]]:
    """
    Get configured OpenAI-compatible client for vLLM with Langfuse tracing.
    
    Args:
        timeout: Request timeout in seconds (default: 60)
        load_model_from_config: If True, also returns model name from config
        
    Returns:
        OpenAI client instance, or tuple of (client, model_name) if load_model_from_config=True
        
    Note:
        Uses OPENAI_BASE_URL and OPENAI_API_KEY from environment.
        The Langfuse OpenAI wrapper preserves the OpenAI SDK API while tracing
        generations, model names, token usage, latency, and API errors.
        Defaults to vLLM running at http://localhost:8000/v1
    """
    # Check environment variables and warn if missing
    if os.getenv("OPENAI_BASE_URL") is None:
        print("⚠️  OPENAI_BASE_URL not found in environment. Using default http://localhost:8000/v1")
    
    if os.getenv("OPENAI_API_KEY") is None:
        print("⚠️  OPENAI_API_KEY not set. Using dummy key.")
    
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        timeout=timeout,
    )
    
    if load_model_from_config:
        from config import get_llm_config
        llm_config = get_llm_config()
        model = llm_config.get("model", "meta-llama/Llama-2-7b-chat-hf")
        return client, model
    
    return client
