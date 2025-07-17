#!/usr/bin/env python3
"""
Enhanced Search Demo
Demonstrates the LLM-powered query refinement capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search import *

def demo_query_refinement():
    """Demo the query refinement capabilities"""
    
    print("🎯 LLM Query Refinement Demo")
    print("=" * 60)
    
    test_queries = [
        "memory allocation",
        "sort algorithm",
        "file read",
        "walker behavior",
        "thread safety",
        "error handling"
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        print("-" * 40)
        
        # Test intent-based refinement
        original, intent_refined, method = refine_query_with_llm(query, "intent")
        print(f"Intent-based: '{intent_refined}'")
        
        # Test code-specific refinement
        original, code_refined, method = refine_query_with_llm(query, "code")
        print(f"Code-specific: '{code_refined}'")
        
        # Test combined approach
        original, combined_refined, method = refine_query_with_llm(query, "both")
        print(f"Combined ({method}): '{combined_refined}'")

def demo_search_comparison():
    """Demo search with and without refinement"""
    
    print("\n\n🔍 Search Comparison Demo")
    print("=" * 60)
    
    test_query = "walker hand behavior at filament edge"
    
    print(f"Testing query: '{test_query}'")
    print("\n1. Without LLM refinement:")
    results_no_llm = search_code(test_query, model_type='unixcoder', k=3, enable_refinement=False)
    
    print(f"\nTop 3 results (no refinement):")
    for i, result in enumerate(results_no_llm[:3], 1):
        meta = result['metadata']
        score = result['similarity_score']
        print(f"  {i}. {meta.get('function_name', 'Unknown')} ({score:.3f})")
    
    print("\n2. With LLM refinement:")
    results_with_llm = search_code(test_query, model_type='unixcoder', k=3, enable_refinement=True)
    
    print(f"\nTop 3 results (with refinement):")
    for i, result in enumerate(results_with_llm[:3], 1):
        meta = result['metadata']
        score = result['similarity_score']
        print(f"  {i}. {meta.get('function_name', 'Unknown')} ({score:.3f})")
    
    # Show improvement
    if results_with_llm and results_no_llm:
        improved_score = results_with_llm[0]['similarity_score']
        original_score = results_no_llm[0]['similarity_score']
        if improved_score > original_score:
            improvement = ((improved_score - original_score) / original_score) * 100
            print(f"\n✅ LLM refinement improved top result by {improvement:.1f}%")
        else:
            print(f"\n📊 LLM refinement: Top scores comparable")

def main():
    """Main demo function"""
    
    print("🚀 Enhanced Code Search with LLM Query Refinement")
    print("=" * 80)
    print("Features:")
    print("  - 🤖 Ollama integration with Llama 3.2 and DeepSeek Coder")
    print("  - 🎯 Intent-based query refinement")
    print("  - 🔧 Code-specific query enhancement")
    print("  - 📊 Transparent refinement process")
    print("  - 🔄 Fallback to original query on errors")
    print()
    
    try:
        # Check if collections are available
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        if "unixcoder_snippets" not in collection_names:
            print("❌ UniXcoder collection not found. Please run ingester.py first.")
            return
        
        if QUERY_REFINEMENT_CONFIG["enabled"]:
            demo_query_refinement()
            demo_search_comparison()
        else:
            print("❌ LLM refinement is disabled in configuration")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Make sure Ollama is running and models are available")

if __name__ == "__main__":
    main()
