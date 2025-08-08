"""
ChromaDB-based Keyword Search Module

This module implements keyword search by querying the existing ChromaDB collections
instead of re-parsing files. This ensures consistency with vector search and
leverages the high-quality Tree-sitter parsing already performed.

Features:
- Operates on the same data as vector search
- Consistent parsing methodology  
- Fast text search within ChromaDB documents
- Rich metadata from AST parsing
- No redundant file parsing

Functions:
- search_keyword_chromadb: Main keyword search function (backward compatible)
- ChromaDBKeywordSearch: Advanced keyword search class with additional methods
"""

from typing import List, Dict, Optional, Set
import re
import chromadb
from pathlib import Path


class ChromaDBKeywordSearch:
    """
    Keyword search engine that operates on ChromaDB collections.
    
    This engine searches within the documents and metadata already stored
    in ChromaDB, ensuring consistency with vector search results.
    """
    
    def __init__(self, chroma_db_path: Optional[str] = None, verbose: bool = False):
        """
        Initialize ChromaDB keyword search.
        
        Args:
            chroma_db_path: Path to ChromaDB storage
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        # Initialize ChromaDB client
        if chroma_db_path is None:
            try:
                from core.config import get_chroma_db_path
                self.chroma_db_path = str(get_chroma_db_path())
            except ImportError:
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.chroma_db_path = os.path.join(project_root, "data", "chroma_db")
        else:
            self.chroma_db_path = chroma_db_path
        
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        
        # Cache collections
        self._collections = {}
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize and cache available collections."""
        try:
            collections = self.client.list_collections()
            for collection in collections:
                self._collections[collection.name] = self.client.get_collection(collection.name)
            
            if self.verbose:
                print(f"Initialized {len(self._collections)} collections: {list(self._collections.keys())}")
        except Exception as e:
            print(f"Error initializing collections: {e}")
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from search query.
        
        Args:
            query: Search query string
            
        Returns:
            List of extracted keywords
        """
        # Split on whitespace and remove empty strings
        keywords = [word.strip() for word in query.split() if word.strip()]
        return keywords
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate relevance score based on keyword matches.
        
        Args:
            text: Text to search
            keywords: Keywords to match
            
        Returns:
            Relevance score between 0 and 1
        """
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        total_keyword_length = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            total_keyword_length += len(keyword_lower)
            
            # Count exact matches
            if keyword_lower in text_lower:
                matches += 1
                # Bonus for whole word matches
                if re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                    matches += 0.5
        
        # Score based on proportion of keywords matched
        keyword_score = matches / len(keywords)
        
        # Bonus for keyword density
        if total_keyword_length > 0:
            density_bonus = min(0.2, (matches * 10) / len(text))
            keyword_score += density_bonus
        
        return min(1.0, keyword_score)
    
    def search_collection(self, 
                         collection_name: str, 
                         query: str, 
                         k: int = 10,
                         filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search within a specific ChromaDB collection.
        
        Args:
            collection_name: Name of the collection to search
            query: Search query string
            k: Maximum number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        if collection_name not in self._collections:
            if self.verbose:
                print(f"Collection '{collection_name}' not found")
            return []
        
        collection = self._collections[collection_name]
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        try:
            # Get all documents (we'll score them locally)
            # ChromaDB doesn't have built-in text search, so we retrieve all and filter
            all_results = collection.get(
                include=['documents', 'metadatas'],
                where=filter_metadata
            )
            
            scored_results = []
            
            for i, (document, metadata) in enumerate(zip(
                all_results['documents'], 
                all_results['metadatas']
            )):
                # Generate a doc_id for compatibility
                doc_id = f"chunk_{i}"
                # Create searchable text from document and metadata
                searchable_text = document
                
                # Add function/class names to searchable text for better matching
                if metadata.get('function_name'):
                    searchable_text += f" {metadata['function_name']}"
                if metadata.get('class_name'):
                    searchable_text += f" {metadata['class_name']}"
                if metadata.get('namespace'):
                    searchable_text += f" {metadata['namespace']}"
                
                # Calculate keyword relevance score
                score = self._calculate_keyword_score(searchable_text, keywords)
                
                if score > 0:  # Only include results with matches
                    result = {
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'score': score,
                        'relevance_score': score,  # Keyword relevance, not similarity
                        'score_type': 'relevance',  # Distinguish from cosine similarity
                        'query': query,
                        'collection': collection_name
                    }
                    scored_results.append(result)
            
            # Sort by score and limit results
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[:k]
            
        except Exception as e:
            if self.verbose:
                print(f"Error searching collection '{collection_name}': {e}")
            return []
    
    def search_all_collections(self, 
                              query: str, 
                              k: int = 10,
                              prefer_collection: Optional[str] = None) -> List[Dict]:
        """
        Search across all available collections.
        
        Args:
            query: Search query string
            k: Maximum number of results
            prefer_collection: Collection to prioritize in results
            
        Returns:
            Unified list of search results
        """
        all_results = []
        
        # Search each collection
        for collection_name in self._collections.keys():
            collection_results = self.search_collection(collection_name, query, k)
            all_results.extend(collection_results)
        
        # If preferring a specific collection, boost its scores
        if prefer_collection and prefer_collection in self._collections:
            for result in all_results:
                if result['collection'] == prefer_collection:
                    result['score'] *= 1.2  # 20% boost for preferred collection
                    result['relevance_score'] = result['score']
        
        # Sort by score and limit results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:k]
    
    def search_by_function(self, function_name: str, k: int = 5) -> List[Dict]:
        """
        Search for functions by name.
        
        Args:
            function_name: Function name to search for
            k: Maximum number of results
            
        Returns:
            List of function matches
        """
        results = []
        
        for collection_name in self._collections.keys():
            # Use metadata filtering for exact function name matches
            try:
                collection = self._collections[collection_name]
                exact_matches = collection.get(
                    where={"function_name": function_name},
                    include=['documents', 'metadatas']
                )
                
                for i, (document, metadata) in enumerate(zip(
                    exact_matches['documents'],
                    exact_matches['metadatas']
                )):
                    doc_id = f"func_{function_name}_{i}"
                    result = {
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'score': 1.0,  # Exact match
                        'relevance_score': 1.0,
                        'score_type': 'exact_match',
                        'query': function_name,
                        'collection': collection_name
                    }
                    results.append(result)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error in function search for collection '{collection_name}': {e}")
        
        return results[:k]
    
    def search_by_class(self, class_name: str, k: int = 5) -> List[Dict]:
        """
        Search for classes by name.
        
        Args:
            class_name: Class name to search for
            k: Maximum number of results
            
        Returns:
            List of class matches
        """
        results = []
        
        for collection_name in self._collections.keys():
            try:
                collection = self._collections[collection_name]
                exact_matches = collection.get(
                    where={"class_name": class_name},
                    include=['documents', 'metadatas']
                )
                
                for i, (document, metadata) in enumerate(zip(
                    exact_matches['documents'],
                    exact_matches['metadatas']
                )):
                    doc_id = f"class_{class_name}_{i}"
                    result = {
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'score': 1.0,  # Exact match
                        'relevance_score': 1.0,
                        'score_type': 'exact_match',
                        'query': class_name,
                        'collection': collection_name
                    }
                    results.append(result)
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error in class search for collection '{collection_name}': {e}")
        
        return results[:k]


def search_keyword_chromadb(query_text: str, k: int = 5) -> List[Dict]:
    """
    Convenience function for ChromaDB-based keyword search.
    
    Args:
        query_text: Search query string
        k: Maximum number of results to return
        
    Returns:
        List of search results from ChromaDB
    """
    try:
        engine = ChromaDBKeywordSearch(verbose=False)
        results = engine.search_all_collections(query_text, k)
        
        # Convert to expected format for compatibility
        formatted_results = []
        for result in results:
            metadata = result['metadata']
            formatted_result = {
                'id': result['id'],
                'content': result['content'],
                'metadata': {
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'start_line': metadata.get('start_line', 0),
                    'end_line': metadata.get('end_line', 0),
                    'function_name': metadata.get('function_name', ''),
                    'class_name': metadata.get('class_name', ''),
                    'namespace': metadata.get('namespace', ''),
                },
                'relevance_score': result['score'],  # Use relevance_score for keyword search
                'score': result['score'],
                'score_type': 'keyword_relevance',  # Clear labeling
                'model_type': 'chromadb_keyword',
                'query': query_text
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error in ChromaDB keyword search: {e}")
        return []


# Backward compatibility alias
search_keyword = search_keyword_chromadb
