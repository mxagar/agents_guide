"""
Simple RAG Pipeline with Full Tracing

retrieve → build_prompt → generate → evaluate
"""

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

from llm_client import TracedLLMClient
from retriever import TracedRetriever
from evaluation import evaluate_rag_output


@observe(name="rag_pipeline", capture_input=False)
def run_rag_pipeline(
    question: str,
    retriever: TracedRetriever,
    llm_client: TracedLLMClient,
    top_k: int = 3
) -> Dict:
    """
    Full RAG pipeline with tracing.
    """
    print(f"\n{'='*50}")
    print(f"RAG Pipeline")
    print(f"Question: {question}")
    print(f"{'='*50}\n")
    
    langfuse = get_client()
    langfuse.update_current_trace(
        name="rag_pipeline",
        input={"question": question},
        tags=["rag", "self-hosted", "pyimagesearch"],
        metadata={"top_k": top_k, "feature": "rag"},
    )
    langfuse.update_current_span(
        input={"question": question, "top_k": top_k}
    )
    
    # Step 1: Retrieve
    print("Step 1: Retrieving documents...")
    docs = retriever.retrieve(query=question, top_k=top_k)
    
    if not docs:
        print("❌ No documents found")
        return {"answer": "No relevant information found.", "success": False}
    
    print(f"✅ Retrieved {len(docs)} documents\n")
    
    # Step 2: Build prompt
    print("Step 2: Building prompt...")
    context = "\n\n".join([f"[{i+1}] {d['content']}" for i, d in enumerate(docs)])
    messages = [
        {"role": "system", "content": "Answer based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
    ]
    
    # Step 3: Generate
    print("Step 3: Generating answer...")
    result = llm_client.complete(messages)
    
    if not result["success"]:
        print(f"❌ Generation failed: {result.get('error')}")
        return {"answer": None, "error": result.get("error"), "success": False}
    
    answer = result["content"]
    print(f"✅ Answer generated\n")
    
    # Step 4: Comprehensive evaluation
    print("Step 4: Evaluating quality...")
    evaluation_results = evaluate_rag_output(question, docs, answer)
    
    langfuse.update_current_trace(
        output={
            "answer": answer,
            "sources_count": len(docs),
            "evaluation": evaluation_results
        }
    )
    langfuse.update_current_span(
        output={
            "answer": answer,
            "sources_count": len(docs),
            "evaluation": evaluation_results
        }
    )
    
    print(f"✅ Evaluation complete")
    print(f"  Relevancy: {evaluation_results['relevancy_score']:.2f}")
    print(f"  Hallucination Risk: {evaluation_results['hallucination_risk']:.2f}")
    print(f"  Overall Quality: {evaluation_results['overall_quality']:.2f}")
    print(f"  Passed: {'✅' if evaluation_results['passed'] else '❌'}\n")
    
    # Get trace URL with correct host
    trace_id = langfuse.get_current_trace_id()
    langfuse_base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    
    print(f"{'='*50}")
    print(f"✅ Pipeline Complete")
    print(f"🔍 View trace: {langfuse_base_url}/trace/{trace_id}")
    print(f"{'='*50}\n")
    
    return {
        "answer": answer,
        "sources": docs,
        "evaluation": evaluation_results,
        "success": True
    }


if __name__ == "__main__":
    # Initialize
    retriever = TracedRetriever()
    llm_client = TracedLLMClient()
    
    # Load documents from file
    data_path = Path(__file__).parent.parent / "data" / "sample_docs.txt"
    with open(data_path, "r") as f:
        docs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    
    retriever.index_documents(docs)
    
    # Run RAG
    question = "What is machine learning?"
    result = run_rag_pipeline(question, retriever, llm_client)
    
    if result["success"]:
        print(f"Answer:\n{result['answer']}\n")
    
    get_client().flush()
