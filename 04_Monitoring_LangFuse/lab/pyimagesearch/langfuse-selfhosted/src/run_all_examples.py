"""
Run All Examples - Complete Lesson 3 Walkthrough

Executes all 4 example scripts in sequence to demonstrate
the full progression from baseline to production-grade observability.
"""

from pathlib import Path
import sys
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_example(module_name: str, description: str):
    """Import and run an example module."""
    print_section(description)
    
    try:
        # Import the module
        module = __import__(module_name)
        
        # Run the example
        if hasattr(module, 'main'):
            module.main()
        elif hasattr(module, 'run_pipeline'):
            question = "What is machine learning?"
            module.run_pipeline(question)
        elif hasattr(module, 'run_simple_pipeline'):
            question = "What is machine learning?"
            module.run_simple_pipeline(question)
        elif hasattr(module, 'generate_and_score'):
            question = "What are neural networks?"
            print(f"Question: {question}\n")
            answer, score = module.generate_and_score(question)
            print(f"Answer: {answer}\n")
            print(f"Quality Score: {score:.2f}\n")
            print("✅ Score logged to LangFuse dashboard\n")
        
        print("✅ Example completed successfully")
        time.sleep(2)  # Brief pause between examples
        
    except Exception as e:
        print(f"❌ Error running {module_name}: {e}")
        print("Continuing with remaining examples...\n")
        time.sleep(1)


def main():
    """Run all examples in sequence."""
    print("\n" + "="*70)
    print("  LESSON 3: LLM OBSERVABILITY WALKTHROUGH")
    print("  Running all examples in progression order")
    print("="*70)
    
    examples = [
        ("basic_llm_app", "Example 1: Baseline (No Tracing)"),
        ("tracing_manual", "Example 2: Manual Tracing with LangFuse SDK"),
        ("tracing_decorator", "Example 3: Decorator Tracing (Recommended)"),
        ("evaluation_metrics", "Example 4: With Latency Tracking + Quality Scoring"),
    ]
    
    for module_name, description in examples:
        run_example(module_name, description)
    
    # Final summary
    print_section("✅ ALL EXAMPLES COMPLETED")
    print("What you've learned:")
    print("  1. ✅ Baseline LLM calls without observability")
    print("  2. ✅ Manual tracing with LangFuse SDK")
    print("  3. ✅ Decorator-based tracing (production pattern)")
    print("  4. ✅ Automatic quality scoring and latency monitoring")
    print("\nNext Steps:")
    print("  - View your traces at: https://cloud.langfuse.com")
    print("  - Explore the dashboard to see latency, tokens, and scores")
    print("  - Try modifying configs/config.yaml to experiment")
    print("  - Ready for Lesson 4: Production RAG with Observability")
    print()


if __name__ == "__main__":
    main()
