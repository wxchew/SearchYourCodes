"""
Search Orchestrator Module

This module provides a unified interface for coordinating different search methods
including keyword search, vector search, and model comparison functionality.

Features:
- Unified search interface
- Multi-method search comparison
- Result formatting and display
- Search orchestration and coordination
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path

from core.keyword_search import search_keyword_chromadb
from core.vector_search import VectorSearchEngine


class SearchOrchestrator:
    """
    Main search orchestrator that coordinates different search methods.
    
    This class provides a unified interface for performing searches across
    multiple methods and comparing their results.
    """
    
    def __init__(self, 
                 vector_engine: Optional[VectorSearchEngine] = None,
                 verbose: bool = False):
        """
        Initialize the search orchestrator.
        
        Args:
            vector_engine: Vector search engine instance
            verbose: Enable verbose logging
        """
        self.vector_engine = vector_engine or VectorSearchEngine()
        self.verbose = verbose
    
    def search_all_methods(self, query: str, k: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Search using all available methods and return results.
        
        Args:
            query: Search query string
            k: Number of results per method
            
        Returns:
            Tuple of (keyword_results, unixcoder_results, minilm_results)
        """
        if self.verbose:
            print(f"Performing multi-method search for: '{query}'")
        
        # Keyword search using ChromaDB
        try:
            keyword_results = search_keyword_chromadb(query, k)
        except Exception as e:
            print(f"Keyword search failed: {e}")
            keyword_results = []
        
        # UniXcoder vector search
        try:
            unixcoder_results = self.vector_engine.search_code(query, 'unixcoder', k)
        except Exception as e:
            print(f"UniXcoder search failed: {e}")
            unixcoder_results = []
        
        # MiniLM vector search
        try:
            minilm_results = self.vector_engine.search_code(query, 'minilm', k)
        except Exception as e:
            print(f"MiniLM search failed: {e}")
            minilm_results = []
        
        return keyword_results, unixcoder_results, minilm_results
    
    def compare_models(self, query: str, k: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Compare different embedding models for the same query.
        
        Args:
            query: Search query string
            k: Number of results per model
            
        Returns:
            Tuple of (keyword_results, unixcoder_results, minilm_results)
        """
        return self.search_all_methods(query, k)
    
    def search_unified(self, query: str, methods: List[str] = None, k: int = 5) -> Dict[str, List[Dict]]:
        """
        Perform unified search across specified methods.
        
        Args:
            query: Search query string
            methods: List of methods to use ('keyword', 'unixcoder', 'minilm')
            k: Number of results per method
            
        Returns:
            Dictionary mapping method names to their results
        """
        if methods is None:
            methods = ['keyword', 'unixcoder', 'minilm']
        
        results = {}
        
        if 'keyword' in methods:
            try:
                results['keyword'] = search_keyword_chromadb(query, k)
            except Exception as e:
                print(f"Keyword search failed: {e}")
                results['keyword'] = []
        
        if 'unixcoder' in methods:
            try:
                results['unixcoder'] = self.vector_engine.search_code(query, 'unixcoder', k)
            except Exception as e:
                print(f"UniXcoder search failed: {e}")
                results['unixcoder'] = []
        
        if 'minilm' in methods:
            try:
                results['minilm'] = self.vector_engine.search_code(query, 'minilm', k)
            except Exception as e:
                print(f"MiniLM search failed: {e}")
                results['minilm'] = []
        
        return results


class SearchResultFormatter:
    """
    Utility class for formatting and displaying search results.
    """
    
    @staticmethod
    def get_model_display_name(model_type: str) -> str:
        """
        Get display name for model type.
        
        Args:
            model_type: Internal model type identifier
            
        Returns:
            Human-readable model name
        """
        display_names = {
            'keyword': 'Keyword Search',
            'unixcoder': 'UniXcoder (Code Structure)',
            'minilm': 'MiniLM (Semantic)'
        }
        return display_names.get(model_type, model_type.title())
    
    @staticmethod
    def print_results(results: List[Dict], query: str, model_type: str) -> None:
        """
        Print formatted search results.
        
        Args:
            results: List of search result dictionaries
            query: Original search query
            model_type: Type of model used for search
        """
        model_name = SearchResultFormatter.get_model_display_name(model_type)
        print(f"\n{model_name} Results for '{query}':")
        print("=" * 80)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.get('score', 0):.3f}")
            
            # Handle different result formats
            if 'file_path' in result:
                file_path = result['file_path']
                if 'line_number' in result:
                    print(f"   File: {file_path}:{result['line_number']}")
                else:
                    print(f"   File: {file_path}")
            
            # Display function/class context if available
            if result.get('function_name'):
                print(f"   Function: {result['function_name']}")
            if result.get('class_name'):
                print(f"   Class: {result['class_name']}")
            
            # Display content
            content = result.get('content', result.get('matched_text', ''))
            if content:
                # Truncate very long content
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"   Content: {content}")
            
            # Display context for keyword results
            if 'context' in result and model_type == 'keyword':
                context_lines = result['context'].split('\n')
                if len(context_lines) > 3:
                    # Show first and last few lines of context
                    print("   Context:")
                    for line in context_lines[:2]:
                        print(f"     {line}")
                    if len(context_lines) > 4:
                        print("     ...")
                    for line in context_lines[-2:]:
                        print(f"     {line}")
                else:
                    print(f"   Context:\n     " + "\n     ".join(context_lines))
            
            # Display matched keywords for keyword search
            if 'matched_keywords' in result:
                print(f"   Matched Keywords: {', '.join(result['matched_keywords'])}")
    
    @staticmethod
    def print_comparison_results(keyword_results: List[Dict], 
                               unixcoder_results: List[Dict], 
                               minilm_results: List[Dict], 
                               query: str) -> None:
        """
        Print comparison results from multiple search methods.
        
        Args:
            keyword_results: Keyword search results
            unixcoder_results: UniXcoder search results
            minilm_results: MiniLM search results
            query: Original search query
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE SEARCH RESULTS FOR: '{query}'")
        print(f"{'='*80}")
        
        # Print results for each method
        SearchResultFormatter.print_results(keyword_results, query, 'keyword')
        SearchResultFormatter.print_results(unixcoder_results, query, 'unixcoder')
        SearchResultFormatter.print_results(minilm_results, query, 'minilm')
        
        # Print summary
        print(f"\n{'='*80}")
        print("SEARCH SUMMARY:")
        print(f"Keyword Search: {len(keyword_results)} results")
        print(f"UniXcoder Search: {len(unixcoder_results)} results")
        print(f"MiniLM Search: {len(minilm_results)} results")
        print(f"{'='*80}")


# Global orchestrator instance for efficient reuse
_global_orchestrator = None

def _get_global_orchestrator() -> SearchOrchestrator:
    """Get or create global orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = SearchOrchestrator()
    return _global_orchestrator

# Convenience functions for backward compatibility
def compare_models(query: str, k: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Legacy function for model comparison - uses persistent orchestrator."""
    orchestrator = _get_global_orchestrator()
    return orchestrator.compare_models(query, k)


def search_all_methods(query: str, k: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Legacy function for all-methods search - uses persistent orchestrator."""
    orchestrator = _get_global_orchestrator()
    return orchestrator.search_all_methods(query, k)


def get_model_display_name(model_type: str) -> str:
    """Legacy function for model display names."""
    return SearchResultFormatter.get_model_display_name(model_type)


def print_results(results: List[Dict], query: str, model_type: str) -> None:
    """Legacy function for printing results."""
    SearchResultFormatter.print_results(results, query, model_type)
