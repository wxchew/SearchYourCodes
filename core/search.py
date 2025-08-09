"""
Comprehensive Code Search System

This module provides a unified interface for multi-method code search using:
1. Keyword Search: Exact text matching with fuzzy search capabilities
2. UniXcoder: Code structure and programming pattern search
3. SBERT: Semantic understanding and conceptual search

The system integrates ChromaDB for vector storage and provides an interactive
comparison interface for evaluating different search approaches.

This module now acts as a backward-compatible facade over the new modular
search architecture.
"""

import os
import torch
import chromadb
from typing import List, Dict, Tuple

# Handle imports for both module and direct execution
try:
    # Try relative imports first (when run as module)
    from core.keyword_search import search_keyword_chromadb
    from core.vector_search import VectorSearchEngine
    from core.search_orchestrator import (
        SearchOrchestrator, 
        SearchResultFormatter,
        compare_models,
        search_all_methods,
        get_model_display_name,
        print_results
    )
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from core.keyword_search import search_keyword_chromadb
    from core.vector_search import VectorSearchEngine
    from core.search_orchestrator import (
        SearchOrchestrator, 
        SearchResultFormatter,
        compare_models,
        search_all_methods,
        get_model_display_name,
        print_results
    )

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize ChromaDB client for legacy compatibility
try:
    from core.config import get_chroma_db_path
    chroma_db_path = str(get_chroma_db_path())
except ImportError:
    # Fallback for standalone execution
    import os
    from pathlib import Path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    chroma_db_path = str(project_root / "data" / "chroma_db")

client = chromadb.PersistentClient(path=chroma_db_path)

# Initialize search components
print(f"Using device: {DEVICE}")
_search_orchestrator = SearchOrchestrator(verbose=False)
_result_formatter = SearchResultFormatter()

print("Search system initialized successfully!")


# Legacy compatibility functions - delegate to new modules
def preprocess_code_for_unixcoder(code_text: str) -> str:
    """Legacy function - delegates to VectorSearchEngine."""
    engine = VectorSearchEngine()
    return engine.preprocess_code_for_unixcoder(code_text)


def get_hf_embedding(query_text: str, model, tokenizer):
    """Legacy function - delegates to VectorSearchEngine."""
    engine = VectorSearchEngine()
    return engine.get_hf_embedding(query_text, 'unixcoder')


def get_sbert_embedding(query_text: str, model):
    """Legacy function - delegates to VectorSearchEngine."""
    engine = VectorSearchEngine()
    return engine.get_sbert_embedding(query_text)


def get_query_embedding(query_text: str, model_type: str = 'unixcoder'):
    """Legacy function - delegates to VectorSearchEngine."""
    engine = VectorSearchEngine()
    return engine.get_query_embedding(query_text, model_type)


def search_code(query_text: str, model_type: str = 'unixcoder', k: int = 5) -> List[Dict]:
    """Legacy function - delegates to VectorSearchEngine."""
    engine = VectorSearchEngine()
    return engine.search_code(query_text, model_type, k)


def search_keyword(query_text: str, k: int = 5) -> List[Dict]:
    """Legacy function - delegates to ChromaDB keyword search."""
    return search_keyword_chromadb(query_text, k)


def _run_comprehensive_tests():
    """Run comprehensive tests of all search components."""
    print("\n🧪 Running Comprehensive System Tests")
    print("=" * 60)
    
    test_query = "bridge"
    
    # Test 1: Direct API calls
    print("\n1. Testing Direct API Calls:")
    print("-" * 30)
    
    # Keyword search
    try:
        keyword_results = search_keyword_chromadb(test_query, k=2)
        print(f"✅ Keyword search: {len(keyword_results)} results")
        if keyword_results:
            r = keyword_results[0]
            print(f"   Score type: {r.get('score_type', 'Not set')}")
            print(f"   Score: {r.get('score', 0):.3f}")
    except Exception as e:
        print(f"❌ Keyword search failed: {e}")
    
    # Vector search
    try:
        engine = VectorSearchEngine()
        vector_results = engine.search_code(test_query, 'unixcoder', k=2)
        print(f"✅ Vector search: {len(vector_results)} results")
        if vector_results:
            r = vector_results[0]
            print(f"   Score type: {r.get('score_type', 'Not set')}")
            print(f"   Score: {r.get('score', 0):.3f}")
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
    
    # Test 2: Search orchestrator
    print("\n2. Testing Search Orchestrator:")
    print("-" * 30)
    
    try:
        orchestrator = SearchOrchestrator()
        keyword_res, unixcoder_res, minilm_res = orchestrator.search_all_methods(test_query, k=2)
        print(f"✅ Orchestrator: {len(keyword_res)} keyword, {len(unixcoder_res)} unixcoder, {len(minilm_res)} minilm")
    except Exception as e:
        print(f"❌ Orchestrator failed: {e}")
    
    # Test 3: Legacy compatibility
    print("\n3. Testing Legacy Compatibility:")
    print("-" * 30)
    
    try:
        legacy_results = search_keyword(test_query, k=2)
        print(f"✅ Legacy search_keyword: {len(legacy_results)} results")
        
        legacy_code_results = search_code(test_query, 'unixcoder', k=2)
        print(f"✅ Legacy search_code: {len(legacy_code_results)} results")
    except Exception as e:
        print(f"❌ Legacy compatibility failed: {e}")
    
    print(f"\n✅ System tests completed!")


def _test_webapp_functionality():
    """Test web application search functionality."""
    print("\n🌐 Testing Web Application Functionality")
    print("=" * 60)
    
    try:
        # Import web app components
        import sys
        import os
        
        # Add project root to path for web app imports
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from app.main import app
        # Note: views import is optional since we're testing main functionality
        
        print("✅ Web app imports successful")
        
        # Test search endpoint simulation
        test_query = "trapper"
        
        print(f"\n1. Testing search formatting for query: '{test_query}'")
        print("-" * 40)
        
        # Create a test client context
        with app.test_client() as client:
            # Test keyword search
            try:
                response = client.post('/search', 
                    json={'query': test_query, 'search_type': 'keyword'},
                    content_type='application/json'
                )
                print(f"✅ Keyword search endpoint: HTTP {response.status_code}")
                
                if response.status_code == 200:
                    # Check if response contains expected elements
                    data = response.get_data(as_text=True)
                    if 'results' in data.lower():
                        print("   ✅ Response contains results")
                    else:
                        print("   ⚠️  Response may not contain formatted results")
                        
            except Exception as e:
                print(f"❌ Keyword search endpoint failed: {e}")
            
            # Test vector search endpoints
            for search_type in ['code_structure', 'semantic']:
                try:
                    response = client.post('/search', 
                        json={'query': test_query, 'search_type': search_type},
                        content_type='application/json'
                    )
                    print(f"✅ {search_type} search endpoint: HTTP {response.status_code}")
                except Exception as e:
                    print(f"❌ {search_type} search endpoint failed: {e}")
        
        # Test result formatting functions directly
        print(f"\n2. Testing result formatting functions:")
        print("-" * 40)
        
        try:
            # Import formatting functions
            from app.main import _format_keyword_results, _format_semantic_results
            
            # Get some test results
            keyword_results = search_keyword_chromadb(test_query, k=2)
            
            if keyword_results:
                formatted = _format_keyword_results(keyword_results)
                print(f"✅ Keyword formatting: {len(formatted)} formatted results")
                
                if formatted:
                    r = formatted[0]
                    print(f"   Function: {r.get('function', 'Not set')}")
                    print(f"   File: {r.get('file', 'Not set')}")
                    print(f"   Score type: {r.get('score_type', 'Not set')}")
                    print(f"   Has score: {r.get('has_score', False)}")
            else:
                print("⚠️  No keyword results to format")
                
        except Exception as e:
            print(f"❌ Result formatting test failed: {e}")
        
        # Test score labeling
        print(f"\n3. Testing score type labeling:")
        print("-" * 40)
        
        try:
            keyword_results = search_keyword_chromadb(test_query, k=1)
            if keyword_results:
                r = keyword_results[0]
                score_type = r.get('score_type', 'Not set')
                print(f"✅ Keyword score type: '{score_type}'")
                if score_type == 'keyword_relevance':
                    print("   ✅ Correctly labeled as relevance (not similarity)")
                else:
                    print("   ⚠️  Score type may not be correctly set")
            
            engine = VectorSearchEngine()
            vector_results = engine.search_code(test_query, 'unixcoder', k=1)
            if vector_results:
                r = vector_results[0]
                score_type = r.get('score_type', 'Not set')
                print(f"✅ Vector score type: '{score_type}'")
                if score_type == 'cosine_similarity':
                    print("   ✅ Correctly labeled as cosine similarity")
                else:
                    print("   ⚠️  Score type may not be correctly set")
                    
        except Exception as e:
            print(f"❌ Score labeling test failed: {e}")
            
    except ImportError as e:
        print(f"❌ Could not import web app components: {e}")
        print("   Make sure the Flask app is properly set up")
    except Exception as e:
        print(f"❌ Web app testing failed: {e}")
    
    print(f"\n✅ Web app tests completed!")


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM Code Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python core/search.py                    # Interactive search
    python core/search.py --test            # Run comprehensive tests
    python core/search.py --test-webapp     # Test web app functionality
    python core/search.py --query "bridge"  # Single query search
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run comprehensive system tests'
    )
    
    parser.add_argument(
        '--test-webapp',
        action='store_true', 
        help='Test web application functionality'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Run a single search query'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Verify collections exist first
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        required_collections = ["unixcoder_snippets", "sbert_snippets"]
        missing_collections = [col for col in required_collections if col not in collection_names]
        
        if missing_collections:
            print(f"Missing collections: {missing_collections}")
            print("Please run the ingestion script first to create embeddings.")
            return 1
        
        if args.verbose:
            print(f"Available collections: {collection_names}")
            for col_name in required_collections:
                try:
                    collection = client.get_collection(col_name)
                    print(f"Collection '{col_name}' loaded with {collection.count()} chunks")
                except Exception as e:
                    print(f"Error accessing collection '{col_name}': {e}")
        
        # Handle different modes
        if args.test:
            print("🧪 Running Comprehensive System Tests")
            _run_comprehensive_tests()
            return 0
            
        elif args.test_webapp:
            print("🌐 Testing Web Application Functionality")
            _test_webapp_functionality()
            return 0
            
        elif args.query:
            print(f"🔍 Searching for: '{args.query}'")
            compare_models(args.query, k=5)
            return 0
            
        else:
            # Interactive mode
            _run_interactive_search()
            return 0
            
    except Exception as e:
        print(f"A critical error occurred: {e}")
        return 1


def _run_interactive_search():
    """Run the interactive search interface."""
    print("\n\nInteractive Code Search System")
    print("=" * 50)
    print("Search Methods:")
    print("  - KEYWORD: Literal text and keyword matching")
    print("  - UNIXCODER: Programming patterns and syntax")
    print("  - MINILM: Conceptual understanding and meaning")
    print("\nCommands:")
    print("  - Type your search query")
    print("  - 'test' - Run comprehensive system tests")
    print("  - 'webapp' - Test web app functionality")
    print("  - 'exit' or 'quit' - End session")
    
    while True:
        print("\n" + "="*80)
        query = input("Enter search query: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting search. Goodbye!")
            break
        
        if not query.strip():
            print("Query cannot be empty.")
            continue
        
        if query.lower() == 'test':
            _run_comprehensive_tests()
            continue
        
        if query.lower() == 'webapp':
            _test_webapp_functionality()
            continue

        compare_models(query, k=5)


if __name__ == "__main__":
    exit(main())
