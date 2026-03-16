"""
ChromaDB-based Keyword Search Module

Implements keyword search by querying existing ChromaDB collections
using server-side filtering where possible, with local scoring for ranking.
"""

from typing import List, Dict, Optional
import re

from core.db import get_chroma_client


class ChromaDBKeywordSearch:
    """
    Keyword search engine that operates on ChromaDB collections.

    Uses ChromaDB's where_document filtering for server-side pre-filtering,
    then scores matching documents locally for ranking.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.client = get_chroma_client()
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
        """Extract keywords from search query."""
        return [word.strip() for word in query.split() if word.strip()]

    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches (0 to 1)."""
        if not keywords:
            return 0.0

        text_lower = text.lower()
        matches = 0
        total_keyword_length = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()
            total_keyword_length += len(keyword_lower)
            if keyword_lower in text_lower:
                matches += 1
                if re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                    matches += 0.5

        keyword_score = matches / len(keywords)
        if total_keyword_length > 0:
            density_bonus = min(0.2, (matches * 10) / max(len(text), 1))
            keyword_score += density_bonus

        return min(1.0, keyword_score)

    def search_collection(self,
                         collection_name: str,
                         query: str,
                         k: int = 10,
                         filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search within a specific ChromaDB collection using server-side
        filtering per keyword, then score and rank locally.
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
            # Server-side pre-filter: fetch documents containing at least one keyword
            candidate_ids = set()
            candidates_by_id: Dict[str, Dict] = {}

            for keyword in keywords:
                where_doc = {"$contains": keyword}
                try:
                    matches = collection.get(
                        include=['documents', 'metadatas'],
                        where=filter_metadata,
                        where_document=where_doc,
                    )
                except Exception:
                    # Fallback: some ChromaDB versions may not support where_document + where together
                    matches = collection.get(
                        include=['documents', 'metadatas'],
                        where_document=where_doc,
                    )

                if not matches or not matches['documents']:
                    continue

                ids = matches.get('ids', [])
                for i, (doc_id, document, metadata) in enumerate(zip(
                    ids, matches['documents'], matches['metadatas']
                )):
                    if doc_id not in candidates_by_id:
                        candidates_by_id[doc_id] = {
                            'document': document,
                            'metadata': metadata,
                        }
                        candidate_ids.add(doc_id)

            # Score candidates locally
            scored_results = []
            for doc_id, data in candidates_by_id.items():
                document = data['document']
                metadata = data['metadata']

                searchable_text = document
                if metadata.get('function_name'):
                    searchable_text += f" {metadata['function_name']}"
                if metadata.get('class_name'):
                    searchable_text += f" {metadata['class_name']}"
                if metadata.get('namespace'):
                    searchable_text += f" {metadata['namespace']}"

                score = self._calculate_keyword_score(searchable_text, keywords)

                if score > 0:
                    scored_results.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'score': score,
                        'relevance_score': score,
                        'score_type': 'relevance',
                        'query': query,
                        'collection': collection_name
                    })

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
        """Search across all available collections."""
        all_results = []

        for collection_name in self._collections.keys():
            collection_results = self.search_collection(collection_name, query, k)
            all_results.extend(collection_results)

        if prefer_collection and prefer_collection in self._collections:
            for result in all_results:
                if result['collection'] == prefer_collection:
                    result['score'] *= 1.2
                    result['relevance_score'] = result['score']

        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:k]

    def search_by_function(self, function_name: str, k: int = 5) -> List[Dict]:
        """Search for functions by name using metadata filtering."""
        results = []
        for collection_name, collection in self._collections.items():
            try:
                exact_matches = collection.get(
                    where={"function_name": function_name},
                    include=['documents', 'metadatas']
                )
                ids = exact_matches.get('ids', [])
                for i, (doc_id, document, metadata) in enumerate(zip(
                    ids, exact_matches['documents'], exact_matches['metadatas']
                )):
                    results.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'score': 1.0,
                        'relevance_score': 1.0,
                        'score_type': 'exact_match',
                        'query': function_name,
                        'collection': collection_name
                    })
            except Exception as e:
                if self.verbose:
                    print(f"Error in function search for collection '{collection_name}': {e}")
        return results[:k]

    def search_by_class(self, class_name: str, k: int = 5) -> List[Dict]:
        """Search for classes by name using metadata filtering."""
        results = []
        for collection_name, collection in self._collections.items():
            try:
                exact_matches = collection.get(
                    where={"class_name": class_name},
                    include=['documents', 'metadatas']
                )
                ids = exact_matches.get('ids', [])
                for i, (doc_id, document, metadata) in enumerate(zip(
                    ids, exact_matches['documents'], exact_matches['metadatas']
                )):
                    results.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'score': 1.0,
                        'relevance_score': 1.0,
                        'score_type': 'exact_match',
                        'query': class_name,
                        'collection': collection_name
                    })
            except Exception as e:
                if self.verbose:
                    print(f"Error in class search for collection '{collection_name}': {e}")
        return results[:k]


def search_keyword_chromadb(query_text: str, k: int = 5) -> List[Dict]:
    """Convenience function for ChromaDB-based keyword search."""
    try:
        engine = ChromaDBKeywordSearch(verbose=False)
        results = engine.search_all_collections(query_text, k)

        formatted_results = []
        for result in results:
            metadata = result['metadata']
            formatted_results.append({
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
                'relevance_score': result['score'],
                'score': result['score'],
                'score_type': 'keyword_relevance',
                'model_type': 'chromadb_keyword',
                'query': query_text
            })

        return formatted_results

    except Exception as e:
        print(f"Error in ChromaDB keyword search: {e}")
        return []


# Backward compatibility alias
search_keyword = search_keyword_chromadb
