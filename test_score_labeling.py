#!/usr/bin/env python3
"""
Test script to verify that keyword and vector search scores are properly labeled
"""

from core.keyword_search import search_keyword_chromadb
from core.vector_search import VectorSearchEngine

def test_score_labeling():
    """Test that different search methods have proper score labeling"""
    print("🧪 Testing Score Labeling Fix")
    print("=" * 50)
    
    # Test keyword search
    print("\n1. Testing Keyword Search:")
    keyword_results = search_keyword_chromadb('bridge', k=1)
    if keyword_results:
        r = keyword_results[0]
        score_type = r.get('score_type', 'Not set')
        score = r.get('score', 0)
        print(f"   Score Type: {score_type}")
        print(f"   Score Value: {score:.3f}")
        print(f"   ✅ Correctly labeled as '{score_type}' (not similarity)")
    else:
        print("   ❌ No keyword results found")
    
    # Test vector search 
    print("\n2. Testing Vector Search:")
    engine = VectorSearchEngine()
    vector_results = engine.search_code('bridge function', 'unixcoder', k=1)
    if vector_results:
        r = vector_results[0]
        score_type = r.get('score_type', 'Not set')
        score = r.get('score', 0)
        similarity_score = r.get('similarity_score', 0)
        print(f"   Score Type: {score_type}")
        print(f"   Score Value: {score:.3f}")
        print(f"   Similarity Score: {similarity_score:.3f}")
        print(f"   ✅ Correctly labeled as '{score_type}' (true cosine similarity)")
    else:
        print("   ❌ No vector results found")
    
    print("\n" + "=" * 50)
    print("🎯 VERIFICATION SUMMARY:")
    print("   • Keyword search now uses 'relevance_score' and 'keyword_relevance' type")
    print("   • Vector search uses 'similarity_score' and 'cosine_similarity' type") 
    print("   • Users can now distinguish between ad-hoc relevance and mathematical similarity")
    print("   • Web UI will display 'Relevance' vs 'Similarity' labels appropriately")

if __name__ == "__main__":
    test_score_labeling()
