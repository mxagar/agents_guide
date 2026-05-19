"""
Minimal 3-Step Agent Workflow

search → analyze → answer
"""

import os
from pathlib import Path
from langfuse.decorators import observe, langfuse_context

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from llm_client import TracedLLMClient
from retriever import TracedRetriever


@observe(name="agent_search")
def agent_search(query: str, retriever: TracedRetriever):
    """Step 1: Search for information."""
    docs = retriever.retrieve(query=query, top_k=3)
    return docs


@observe(name="agent_analyze")
def agent_analyze(query: str, docs: list, llm_client: TracedLLMClient):
    """Step 2: Analyze query intent."""
    doc_preview = "\n".join([d['content'][:80] for d in docs])
    
    messages = [
        {"role": "system", "content": "Analyze the user's intent in one sentence."},
        {"role": "user", "content": f"Query: {query}\n\nDocs:\n{doc_preview}\n\nIntent:"}
    ]
    
    result = llm_client.complete(messages)
    
    if not result.get("success", False):
        return "Unknown intent (LLM call failed)"
    
    return result.get("content", "Unknown intent")


@observe(name="agent_answer")
def agent_answer(query: str, docs: list, intent: str, llm_client: TracedLLMClient):
    """Step 3: Generate final answer."""
    context = "\n\n".join([f"[{i+1}] {d['content']}" for i, d in enumerate(docs)])
    
    messages = [
        {"role": "system", "content": f"User intent: {intent}\nAnswer based on context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    
    result = llm_client.complete(messages)
    
    if not result.get("success", False):
        return "No answer generated (LLM call failed)"
    
    return result.get("content", "No answer generated")


@observe(name="agent_workflow")
def run_agent_workflow(query: str, retriever: TracedRetriever, llm_client: TracedLLMClient):
    """
    3-step agent workflow with tracing.
    """
    print(f"\n{'='*50}")
    print(f"Agent Workflow")
    print(f"Query: {query}")
    print(f"{'='*50}\n")
    
    # Step 1: Search
    print("Step 1: Searching...")
    docs = agent_search(query, retriever)
    print(f"✅ Found {len(docs)} documents\n")
    
    # Step 2: Analyze
    print("Step 2: Analyzing intent...")
    intent = agent_analyze(query, docs, llm_client)
    print(f"✅ Intent: {intent}\n")
    
    # Step 3: Answer
    print("Step 3: Generating answer...")
    answer = agent_answer(query, docs, intent, llm_client)
    print(f"✅ Answer generated\n")
    
    langfuse_context.update_current_observation(
        output={"answer": answer, "intent": intent}
    )
    
    # Get trace URL with correct host
    trace_id = langfuse_context.get_current_trace_id()
    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    
    print(f"{'='*50}")
    print(f"Workflow Complete")
    print(f"🔍 View trace: {langfuse_host}/trace/{trace_id}")
    print(f"{'='*50}\n")
    
    return {"answer": answer, "intent": intent, "success": True}


if __name__ == "__main__":
    retriever = TracedRetriever()
    llm_client = TracedLLMClient()
    
    # Load docs
    data_path = Path(__file__).parent.parent / "data" / "sample_docs.txt"
    with open(data_path, "r") as f:
        docs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    
    retriever.index_documents(docs)
    
    # Run agent
    query = "How does machine learning work?"
    result = run_agent_workflow(query, retriever, llm_client)
    
    print(f"Final Answer:\n{result['answer']}\n")
