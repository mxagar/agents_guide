"""
RAG Evaluation with Relevancy and Hallucination Scoring

Provides automatic quality metrics for RAG outputs.
"""

from typing import List, Dict
from langfuse.decorators import observe, langfuse_context
from config import get_evaluation_config


@observe(name="evaluate_relevancy")
def evaluate_relevancy(query: str, retrieved_docs: List[Dict], answer: str) -> float:
    """
    Evaluate relevancy of answer to query and retrieved documents.
    
    Simple heuristic: Check if answer contains keywords from query and documents.
    
    Args:
        query: Original user query
        retrieved_docs: List of retrieved documents
        answer: Generated answer
        
    Returns:
        Relevancy score between 0 and 1
    """
    langfuse_context.update_current_observation(
        input={
            "query": query,
            "doc_count": len(retrieved_docs),
            "answer_length": len(answer)
        }
    )
    
    # Extract keywords from query (simple word splitting)
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    # Count keyword matches
    keyword_matches = len(query_words & answer_words)
    keyword_coverage = keyword_matches / max(len(query_words), 1)
    
    # Check if answer references retrieved docs
    doc_overlap = 0
    for doc in retrieved_docs:
        doc_words = set(doc.get("content", "").lower().split())
        overlap = len(doc_words & answer_words)
        doc_overlap += overlap
    
    doc_relevance = min(doc_overlap / max(len(answer_words), 1), 1.0)
    
    # Combine scores (weighted average)
    relevancy_score = 0.6 * keyword_coverage + 0.4 * doc_relevance
    
    langfuse_context.update_current_observation(
        output={
            "relevancy_score": relevancy_score,
            "keyword_coverage": keyword_coverage,
            "doc_relevance": doc_relevance
        }
    )
    
    return relevancy_score


@observe(name="evaluate_hallucination")
def evaluate_hallucination_risk(retrieved_docs: List[Dict], answer: str) -> float:
    """
    Estimate hallucination risk.
    
    Simple heuristic: Answers should be grounded in retrieved documents.
    High risk if answer contains many words NOT in retrieved docs.
    
    Args:
        retrieved_docs: List of retrieved documents
        answer: Generated answer
        
    Returns:
        Hallucination risk between 0 (low risk) and 1 (high risk)
    """
    langfuse_context.update_current_observation(
        input={
            "doc_count": len(retrieved_docs),
            "answer_length": len(answer)
        }
    )
    
    if not retrieved_docs:
        # No context = high hallucination risk
        langfuse_context.update_current_observation(
            output={"hallucination_risk": 1.0, "reason": "No retrieved documents"}
        )
        return 1.0
    
    # Combine all retrieved document text
    all_doc_text = " ".join([doc.get("content", "") for doc in retrieved_docs])
    doc_words = set(all_doc_text.lower().split())
    
    # Extract answer words (excluding common stop words)
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                  "of", "with", "is", "are", "was", "were", "be", "been", "being"}
    answer_words = set(answer.lower().split()) - stop_words
    
    if not answer_words:
        langfuse_context.update_current_observation(
            output={"hallucination_risk": 0.5, "reason": "Empty answer after stop word removal"}
        )
        return 0.5
    
    # Calculate grounding ratio
    grounded_words = answer_words & doc_words
    grounding_ratio = len(grounded_words) / len(answer_words)
    
    # Hallucination risk is inverse of grounding
    hallucination_risk = 1.0 - grounding_ratio
    
    langfuse_context.update_current_observation(
        output={
            "hallucination_risk": hallucination_risk,
            "grounding_ratio": grounding_ratio,
            "grounded_words": len(grounded_words),
            "total_words": len(answer_words)
        }
    )
    
    return hallucination_risk


@observe(name="evaluate_rag_output")
def evaluate_rag_output(query: str, retrieved_docs: List[Dict], answer: str) -> Dict:
    """
    Comprehensive RAG output evaluation.
    
    Args:
        query: Original user query
        retrieved_docs: List of retrieved documents
        answer: Generated answer
        
    Returns:
        Dict with relevancy_score, hallucination_risk, and overall_quality
    """
    langfuse_context.update_current_observation(
        input={
            "query": query,
            "doc_count": len(retrieved_docs),
            "answer_length": len(answer)
        }
    )
    
    # Evaluate relevancy
    relevancy_score = evaluate_relevancy(query, retrieved_docs, answer)
    
    # Evaluate hallucination risk
    hallucination_risk = evaluate_hallucination_risk(retrieved_docs, answer)
    
    # Calculate overall quality (high relevancy + low hallucination = high quality)
    overall_quality = (relevancy_score + (1.0 - hallucination_risk)) / 2.0
    
    # Load quality threshold from config
    eval_config = get_evaluation_config()
    min_quality = eval_config.get("min_quality_score", 0.6)
    
    results = {
        "relevancy_score": relevancy_score,
        "hallucination_risk": hallucination_risk,
        "overall_quality": overall_quality,
        "passed": overall_quality >= min_quality
    }
    
    # Log scores to LangFuse
    langfuse_context.score_current_observation(
        name="relevancy",
        value=relevancy_score,
        comment=f"Keyword and document relevance"
    )
    
    langfuse_context.score_current_observation(
        name="hallucination_risk",
        value=hallucination_risk,
        comment=f"Risk of ungrounded claims"
    )
    
    langfuse_context.score_current_observation(
        name="overall_quality",
        value=overall_quality,
        comment=f"Combined quality score (threshold: {min_quality})"
    )
    
    langfuse_context.update_current_observation(
        output=results
    )
    
    return results


if __name__ == "__main__":
    # Example evaluation
    query = "What is machine learning?"
    
    retrieved_docs = [
        {"content": "Machine learning is a subset of AI that enables systems to learn from data.", "score": 0.95},
        {"content": "Machine learning algorithms improve through experience and data.", "score": 0.87}
    ]
    
    answer = "Machine learning is a subset of artificial intelligence that allows systems to automatically learn and improve from experience without being explicitly programmed."
    
    print("\n" + "="*60)
    print("RAG Evaluation Example")
    print("="*60 + "\n")
    
    print(f"Query: {query}\n")
    print(f"Answer: {answer}\n")
    
    results = evaluate_rag_output(query, retrieved_docs, answer)
    
    print(f"Relevancy Score: {results['relevancy_score']:.2f}")
    print(f"Hallucination Risk: {results['hallucination_risk']:.2f}")
    print(f"Overall Quality: {results['overall_quality']:.2f}")
    print(f"Passed: {'✅' if results['passed'] else '❌'}\n")
    
    print("✅ Scores logged to LangFuse dashboard\n")
