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

# Import the new modular search components
from core.keyword_search import KeywordSearchEngine, search_keyword
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
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_db_path = os.path.join(project_root, "data", "chroma_db")

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


def main():
    """Main interactive search interface."""
    try:
        # Verify collections exist
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        print(f"Available collections: {collection_names}")
        
        required_collections = ["unixcoder_snippets", "sbert_snippets"]
        missing_collections = [col for col in required_collections if col not in collection_names]
        
        if missing_collections:
            print(f"Missing collections: {missing_collections}")
            print("Please run the ingestion script first to create embeddings.")
            return
        
        for col_name in required_collections:
            try:
                collection = client.get_collection(col_name)
                print(f"Collection '{col_name}' loaded with {collection.count()} chunks")
            except Exception as e:
                print(f"Error accessing collection '{col_name}': {e}")
                
        print("\n\nInteractive Code Search System")
        print("=" * 50)
        print("Search Methods:")
        print("  - KEYWORD: Literal text and keyword matching")
        print("  - UNIXCODER: Programming patterns and syntax")
        print("  - MINILM: Conceptual understanding and meaning")
        print("\nCommands:")
        print("  - Type your search query")
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

            compare_models(query, k=5)

    except Exception as e:
        print(f"A critical error occurred: {e}")


if __name__ == "__main__":
    main()
