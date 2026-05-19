"""
Document Retriever with Local Embeddings

Uses SentenceTransformers (no API costs) + FAISS for vector search.
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from langfuse.decorators import observe, langfuse_context
from config import get_embeddings_config, get_rag_config

try:
    import faiss
except ImportError:
    print("❌ Install FAISS: pip install faiss-cpu")
    raise


class TracedRetriever:
    """Document retriever with local embeddings and tracing."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize retriever with local embedding model.
        
        Args:
            model_name: Embedding model name (if None, loads from config)
        """
        # Load config
        if model_name is None:
            embeddings_config = get_embeddings_config()
            model_name = embeddings_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            self.dimension = embeddings_config.get("dimension", 384)
        else:
            self.dimension = 384  # Default for MiniLM
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
    
    @observe(name="embed_text")
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using local model."""
        langfuse_context.update_current_observation(
            input={"text_preview": text[:100]}
        )
        
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        
        langfuse_context.update_current_observation(
            output={"embedding_dim": len(embedding)}
        )
        
        return embedding.astype(np.float32)
    
    @observe(name="index_documents")
    def index_documents(self, documents: List[str]):
        """Index documents for retrieval."""
        if not documents:
            print("⚠️  No documents to index")
            return
        
        langfuse_context.update_current_observation(
            input={"document_count": len(documents)}
        )
        
        # Store documents
        self.documents = documents
        
        # Embed all documents
        print(f"Embedding {len(documents)} documents...")
        embeddings = self.model.encode(documents, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        langfuse_context.update_current_observation(
            output={"indexed_count": len(documents)}
        )
        
        print(f"✅ Indexed {len(documents)} documents")
    
    @observe(name="retrieve_documents")
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve most relevant documents.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (if None, loads from config)
            
        Returns:
            List of dicts with 'content', 'score', and 'rank'
        """
        # Load top_k from config if not specified
        if top_k is None:
            rag_config = get_rag_config()
            top_k = rag_config.get("top_k", 3)
        
        # Check if index is empty
        if self.index.ntotal == 0:
            print("⚠️  Index is empty, no documents to retrieve")
            langfuse_context.update_current_observation(
                output={"results": [], "error": "Empty index"}
            )
            return []
        
        langfuse_context.update_current_observation(
            input={"query": query, "top_k": top_k}
        )
        
        # Embed query
        query_embedding = self.embed(query).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):  # Safety check
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                relevance_score = 1.0 / (1.0 + float(distance))
                
                results.append({
                    "content": self.documents[idx],
                    "score": relevance_score,
                    "rank": rank + 1,
                    "distance": float(distance)
                })
        
        langfuse_context.update_current_observation(
            output={
                "result_count": len(results),
                "scores": [r["score"] for r in results],
                "results": [
                    {
                        "rank": r["rank"],
                        "score": r["score"],
                        "content_preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"]
                    }
                    for r in results
                ]
            }
        )
        
        return results


if __name__ == "__main__":
    # Example usage
    retriever = TracedRetriever()
    
    # Sample documents
    docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text."
    ]
    
    retriever.index_documents(docs)
    
    # Retrieve
    results = retriever.retrieve("What is deep learning?", top_k=2)
    
    print("\nResults:")
    for r in results:
        print(f"{r['rank']}. Score: {r['score']:.3f}")
        print(f"   {r['content']}\n")
