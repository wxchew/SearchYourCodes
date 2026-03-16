"""
Unified Code Search Interface

Provides multi-method code search with result fusion:
1. Keyword: Exact text matching
2. UniXcoder: Code structure and programming patterns
3. SBERT: Semantic understanding (all-MiniLM-L6-v2)
4. Fused: Reciprocal Rank Fusion across all methods
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from core.device import get_device
from core.keyword_search import search_keyword_chromadb
from core.vector_search import VectorSearchEngine

print(f"Using device: {get_device()}")


def compare_models(query: str, k: int = 5,
                   filter_metadata: Optional[Dict] = None
                   ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Run all three search methods and return their results.

    Args:
        query: Search query text
        k: Number of results per method
        filter_metadata: Optional ChromaDB where clause for filtering

    Returns:
        Tuple of (keyword_results, unixcoder_results, sbert_results)
    """
    # Keyword search
    try:
        keyword_results = search_keyword_chromadb(query, k)
    except Exception as e:
        print(f"Keyword search failed: {e}")
        keyword_results = []

    # Vector search engine (shared instance)
    engine = VectorSearchEngine()

    # UniXcoder vector search
    try:
        unixcoder_results = engine.search_code(query, 'unixcoder', k,
                                               filter_metadata=filter_metadata)
    except Exception as e:
        print(f"UniXcoder search failed: {e}")
        unixcoder_results = []

    # SBERT vector search
    try:
        sbert_results = engine.search_code(query, 'sbert', k,
                                           filter_metadata=filter_metadata)
    except Exception as e:
        print(f"SBERT search failed: {e}")
        sbert_results = []

    return keyword_results, unixcoder_results, sbert_results


def fuse_results(keyword_results: List[Dict],
                 unixcoder_results: List[Dict],
                 sbert_results: List[Dict],
                 k_constant: int = 60,
                 max_results: int = 10) -> List[Dict]:
    """
    Combine results from all search methods using Reciprocal Rank Fusion (RRF).

    RRF score for a document d = sum(1 / (k + rank_i(d))) across all methods
    that return d.

    Args:
        keyword_results: Results from keyword search
        unixcoder_results: Results from UniXcoder search
        sbert_results: Results from SBERT search
        k_constant: RRF constant (default 60, standard value)
        max_results: Maximum fused results to return
    """
    # Track scores and best result data per document
    rrf_scores: Dict[str, float] = defaultdict(float)
    best_result: Dict[str, Dict] = {}
    source_methods: Dict[str, List[str]] = defaultdict(list)

    def _doc_key(result: Dict) -> str:
        """Generate a unique key for a result based on file + line range."""
        metadata = result.get('metadata', {})
        file_path = metadata.get('file_path', result.get('file_path', 'unknown'))
        start = metadata.get('start_line', result.get('start_line', 0))
        end = metadata.get('end_line', result.get('end_line', 0))
        return f"{file_path}:{start}-{end}"

    method_lists = [
        ('keyword', keyword_results),
        ('unixcoder', unixcoder_results),
        ('sbert', sbert_results),
    ]

    for method_name, results in method_lists:
        for rank, result in enumerate(results):
            key = _doc_key(result)
            rrf_scores[key] += 1.0 / (k_constant + rank + 1)
            source_methods[key].append(method_name)

            # Keep the result with the highest original score
            if key not in best_result or result.get('score', 0) > best_result[key].get('score', 0):
                best_result[key] = result

    # Sort by RRF score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    fused = []
    for key in sorted_keys[:max_results]:
        result = dict(best_result[key])
        result['rrf_score'] = round(rrf_scores[key], 4)
        result['source_methods'] = source_methods[key]
        result['score_type'] = 'rrf_fusion'
        fused.append(result)

    return fused


# Display helpers for CLI usage
def get_model_display_name(model_type: str) -> str:
    """Get display name for model type."""
    display_names = {
        'keyword': 'Keyword Search',
        'unixcoder': 'UniXcoder (Code Structure)',
        'sbert': 'SBERT (Semantic)',
        'minilm': 'SBERT (Semantic)'
    }
    return display_names.get(model_type, model_type.title())


def print_results(results: List[Dict], query: str, model_type: str) -> None:
    """Print formatted search results to terminal."""
    model_name = get_model_display_name(model_type)
    print(f"\n{model_name} Results for '{query}':")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.get('score', 0):.3f}")
        if 'file_path' in result:
            file_path = result['file_path']
            if 'line_number' in result:
                print(f"   File: {file_path}:{result['line_number']}")
            else:
                print(f"   File: {file_path}")
        if result.get('function_name'):
            print(f"   Function: {result['function_name']}")
        if result.get('class_name'):
            print(f"   Class: {result['class_name']}")
        content = result.get('content', result.get('matched_text', ''))
        if content:
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   Content: {content}")


print("Search system initialized successfully!")
