"""
Vector Search Module

Provides vector-based search using UniXcoder and SBERT embedding models
with ChromaDB storage.
"""

import re
from typing import List, Dict, Optional

import numpy as np

from core.db import get_chroma_client
from core.device import get_device

# Import OOP embedders
try:
    from core.embedders_oop import EmbedderFactory
except ImportError:
    print("Warning: OOP embedders not available, falling back to legacy approach")
    EmbedderFactory = None


class VectorSearchEngine:
    """
    Vector-based search engine using ChromaDB and various embedding models.
    """

    def __init__(self):
        self.device = get_device()
        self.client = get_chroma_client()

        # OOP Embedders (lazy loaded)
        self._unixcoder_embedder = None
        self._sbert_embedder = None

        self.model_configs = {
            'unixcoder': {
                'type': 'huggingface_automodel',
                'name': 'microsoft/unixcoder-base',
                'device': self.device,
                'pooling_method': 'mean'
            },
            'sbert': {
                'type': 'sentence_transformer',
                'name': 'all-MiniLM-L6-v2',
                'device': self.device
            }
        }

    @property
    def unixcoder_embedder(self):
        """Lazy load UniXcoder embedder."""
        if self._unixcoder_embedder is None and EmbedderFactory:
            self._unixcoder_embedder = EmbedderFactory.from_config(self.model_configs['unixcoder'])
        return self._unixcoder_embedder

    @property
    def sbert_embedder(self):
        """Lazy load SBERT embedder."""
        if self._sbert_embedder is None and EmbedderFactory:
            self._sbert_embedder = EmbedderFactory.from_config(self.model_configs['sbert'])
        return self._sbert_embedder

    def preprocess_code_for_unixcoder(self, code_text: str) -> str:
        """Preprocess code text for optimal UniXcoder embeddings."""
        lines = [line.rstrip() for line in code_text.split('\n')]
        lines = [line for line in lines if line.strip()]

        noise_patterns = [
            r'^\s*//.*$',
            r'/\*.*?\*/',
            r'^\s*#include.*$',
            r'^\s*using\s+namespace.*$',
        ]

        cleaned_lines = []
        for line in lines:
            cleaned_line = line
            for pattern in noise_patterns:
                cleaned_line = re.sub(pattern, '', cleaned_line, flags=re.MULTILINE)
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)

        important_lines = []
        for line in cleaned_lines:
            if any(keyword in line.lower() for keyword in ['function', 'class', 'def', 'void', 'int', 'double', 'float', 'bool']):
                important_lines.append(line)
            elif len(line.strip()) > 10 and not line.strip().startswith('//'):
                important_lines.append(line)

        result = '\n'.join(important_lines)
        if len(result) > 2000:
            result = result[:2000]
        return result

    def get_hf_embedding(self, query_text: str, model_type: str = 'unixcoder') -> np.ndarray:
        """Generate embedding using HuggingFace AutoModel via OOP embedders."""
        if model_type != 'unixcoder':
            raise ValueError(f"Unsupported HuggingFace model type: {model_type}")

        embedder = self.unixcoder_embedder
        if embedder is None:
            raise RuntimeError("UniXcoder embedder not available")

        processed_text = self.preprocess_code_for_unixcoder(query_text)
        embedding = embedder.embed([processed_text], show_progress=False)
        return embedding[0]

    def get_sbert_embedding(self, query_text: str) -> np.ndarray:
        """Generate embedding using Sentence-BERT model via OOP embedders."""
        embedder = self.sbert_embedder
        if embedder is None:
            raise RuntimeError("SBERT embedder not available")

        embedding = embedder.embed([query_text], show_progress=False)
        return embedding[0]

    def get_query_embedding(self, query_text: str, model_type: str = 'unixcoder') -> np.ndarray:
        """Generate query embedding based on model type."""
        if model_type == 'unixcoder':
            return self.get_hf_embedding(query_text, 'unixcoder')
        elif model_type in ['minilm', 'sbert']:
            return self.get_sbert_embedding(query_text)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def search_code(self, query_text: str, model_type: str = 'unixcoder',
                    k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for code using vector similarity.

        Args:
            query_text: Search query text
            model_type: Embedding model to use ('unixcoder' or 'sbert')
            k: Number of results to return
            filter_metadata: Optional ChromaDB where clause for metadata filtering
        """
        try:
            query_embedding = self.get_query_embedding(query_text, model_type)

            if model_type == 'unixcoder':
                collection_name = "unixcoder_snippets"
            elif model_type in ['minilm', 'sbert']:
                collection_name = "sbert_snippets"
            else:
                collection_name = f"code_chunks_{model_type}"

            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception:
                print(f"Collection '{collection_name}' not found. Please run the ingestion script first.")
                return []

            query_kwargs = {
                'query_embeddings': [query_embedding.tolist()],
                'n_results': k,
                'include': ['documents', 'metadatas', 'distances'],
            }
            if filter_metadata:
                query_kwargs['where'] = filter_metadata

            results = collection.query(**query_kwargs)

            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = max(0, min(1, 1 - distance / 2))

                formatted_results.append({
                    'content': doc,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                    'start_line': metadata.get('start_line', 0),
                    'end_line': metadata.get('end_line', 0),
                    'function_name': metadata.get('function_name', ''),
                    'class_name': metadata.get('class_name', ''),
                    'score': float(similarity),
                    'similarity_score': float(similarity),
                    'score_type': 'cosine_similarity',
                    'distance': float(distance)
                })

            return formatted_results

        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
