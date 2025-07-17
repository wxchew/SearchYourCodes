#!/usr/bin/env python3

from search import compare_models, refine_query_code_specific

def demo_comparison():
    """
    Demo the comparison between no LLM refinement vs LLM refinement
    """
    query = "what happen when ase walker reaches the plus end of microtubule"
    
    print("🔍 LLM Query Refinement Demo")
    print("=" * 80)
    print(f"Testing query: '{query}'")
    
    # Show what the refinement produces
    refined = refine_query_code_specific(query)
    print(f"\nRefinement result:")
    print(f"  Original: '{query}'")
    print(f"  Refined:  '{refined}'")
    
    if refined == query:
        print("  Status: No changes (original preserved)")
    else:
        print("  Status: Enhanced with programming terms")
    
    print("\n" + "="*80)
    print("🔀 Comparing: No Refinement vs LLM Refinement")
    print("="*80)
    
    # Test without refinement
    print("\n1️⃣ Without LLM Refinement:")
    compare_models(query, k=3, enable_refinement=False)
    
    # Test with refinement
    print("\n2️⃣ With LLM Refinement:")
    compare_models(query, k=3, enable_refinement=True)

if __name__ == "__main__":
    demo_comparison()
