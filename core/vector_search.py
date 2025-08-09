"""
Vector Search Module

This module provides vector-based search functionality using various embedding models
including HuggingFace AutoModels and Sentence Transformers with ChromaDB storage.

Features:
- Multiple embedding model support (UniXcoder, SBERT/all-MiniLM-L6-v2, etc.)
- ChromaDB integration for vector storage
- Configurable similarity search
- Code preprocessing for optimal embeddings
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
import chromadb
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

"""
Vector-based code search using ChromaDB and various embedding models.

This module provides semantic search capabilities with different embedding
approaches for code understanding.
"""

import os
import numpy as np
import chromadb
from typing import List, Dict, Optional, Any
import re

# Automatically detect device
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import OOP embedders
try:
    from core.embedders_oop import EmbedderFactory
except ImportError:
    print("Warning: OOP embedders not available, falling back to legacy approach")
    EmbedderFactory = None


class VectorSearchEngine:
    """
    Vector-based search engine using ChromaDB and various embedding models.
    
    This engine provides semantic search capabilities using pre-computed embeddings
    stored in ChromaDB for efficient similarity search.
    """
    
    def __init__(self, chroma_db_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the vector search engine.
        
        Args:
            chroma_db_path: Path to ChromaDB storage
            device: Device to use for model inference
        """
        self.device = device if device != "auto" else DEVICE
        
        # Initialize ChromaDB
        if chroma_db_path is None:
            try:
                from core.config import get_chroma_db_path
                self.chroma_db_path = str(get_chroma_db_path())
            except ImportError:
                # Fallback for standalone execution
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.chroma_db_path = os.path.join(project_root, "data", "chroma_db")
        else:
            self.chroma_db_path = chroma_db_path
        
        self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        
        # OOP Embedders (lazy loaded)
        self._unixcoder_embedder = None
        self._sbert_embedder = None
        
        # Model configurations for OOP embedders
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
        self._unixcoder_model = None
        self._unixcoder_tokenizer = None
        self._minilm_model = None
    
    @property
    def unixcoder_embedder(self):
        """Lazy load UniXcoder embedder using OOP architecture."""
        if self._unixcoder_embedder is None:
            if EmbedderFactory:
                self._unixcoder_embedder = EmbedderFactory.from_config(self.model_configs['unixcoder'])
            else:
                # Fallback to legacy approach if OOP not available
                from transformers import AutoTokenizer, AutoModel
                model_name = 'microsoft/unixcoder-base'
                self._unixcoder_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._unixcoder_model = AutoModel.from_pretrained(model_name).to(self.device)
                self._unixcoder_model.eval()
                return self._unixcoder_model
        return self._unixcoder_embedder

    @property
    def sbert_embedder(self):
        """Lazy load SBERT embedder using OOP architecture."""
        if self._sbert_embedder is None:
            if EmbedderFactory:
                self._sbert_embedder = EmbedderFactory.from_config(self.model_configs['sbert'])
            else:
                # Fallback to legacy approach if OOP not available
                from sentence_transformers import SentenceTransformer
                self._minilm_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                return self._minilm_model
        return self._sbert_embedder

    def preprocess_code_for_unixcoder(self, code_text: str) -> str:
        """
        Preprocess code text for optimal UniXcoder embeddings.
        
        Args:
            code_text: Raw code text
            
        Returns:
            Preprocessed code text
        """
        # 1. Remove excessive whitespace and empty lines
        lines = [line.rstrip() for line in code_text.split('\n')]
        lines = [line for line in lines if line.strip()]
        
        # 2. Remove common noise patterns
        noise_patterns = [
            r'^\s*//.*$',  # Single-line comments
            r'/\*.*?\*/',  # Multi-line comments (simple)
            r'^\s*#include.*$',  # Include statements
            r'^\s*using\s+namespace.*$',  # Using namespace
        ]
        
        cleaned_lines = []
        for line in lines:
            cleaned_line = line
            for pattern in noise_patterns:
                cleaned_line = re.sub(pattern, '', cleaned_line, flags=re.MULTILINE)
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)
        
        # 3. Focus on function bodies and class definitions
        # This is a simplified approach - a more sophisticated parser could be used
        important_lines = []
        for line in cleaned_lines:
            # Keep lines that contain function definitions, class definitions, or substantial code
            if any(keyword in line.lower() for keyword in ['function', 'class', 'def', 'void', 'int', 'double', 'float', 'bool']):
                important_lines.append(line)
            elif len(line.strip()) > 10 and not line.strip().startswith('//'):
                important_lines.append(line)
        
        # 4. Join and limit length for embedding model
        result = '\n'.join(important_lines)
        
        # Limit to reasonable length for embedding models (typically 512 tokens)
        if len(result) > 2000:  # Rough character limit
            result = result[:2000]
        
        return result
    
    def get_hf_embedding(self, query_text: str, model_type: str = 'unixcoder') -> np.ndarray:
        """
        Generate embedding using HuggingFace AutoModel via OOP embedders.
        
        Args:
            query_text: Text to embed
            model_type: Type of model to use
            
        Returns:
            Normalized embedding vector
        """
        if model_type == 'unixcoder':
            embedder = self.unixcoder_embedder
            
            # Check if we're using OOP embedder or fallback
            if hasattr(embedder, 'embed'):
                # OOP embedder - expects list input
                processed_text = self.preprocess_code_for_unixcoder(query_text)
                embedding = embedder.embed([processed_text], show_progress=False)
                return embedding[0]  # Return first embedding from batch
            else:
                # Fallback to legacy approach
                model = embedder
                tokenizer = self._unixcoder_tokenizer
                processed_text = self.preprocess_code_for_unixcoder(query_text)
                
                # Tokenize and get embeddings
                inputs = tokenizer(processed_text, return_tensors="pt", 
                                  truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use mean pooling of last hidden states
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    # Normalize
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                return embeddings.cpu().numpy().flatten()
        else:
            raise ValueError(f"Unsupported HuggingFace model type: {model_type}")
    
    def get_sbert_embedding(self, query_text: str) -> np.ndarray:
        """
        Generate embedding using Sentence-BERT model via OOP embedders.
        
        Args:
            query_text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        embedder = self.sbert_embedder
        
        # Check if we're using OOP embedder or fallback
        if hasattr(embedder, 'embed'):
            # OOP embedder - expects list input
            embedding = embedder.embed([query_text], show_progress=False)
            return embedding[0]  # Return first embedding from batch
        else:
            # Fallback to legacy approach
            model = embedder
            embedding = model.encode([query_text], convert_to_numpy=True)[0]
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        return embedding
    
    def get_query_embedding(self, query_text: str, model_type: str = 'unixcoder') -> np.ndarray:
        """
        Generate query embedding based on model type.
        
        Args:
            query_text: Query text to embed
            model_type: Type of embedding model ('unixcoder' or 'minilm'/'sbert')
            
        Returns:
            Query embedding vector
        """
        if model_type == 'unixcoder':
            return self.get_hf_embedding(query_text, 'unixcoder')
        elif model_type in ['minilm', 'sbert']:  # Both refer to the same SBERT model
            return self.get_sbert_embedding(query_text)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def search_code(self, query_text: str, model_type: str = 'unixcoder', k: int = 5) -> List[Dict]:
        """
        Search for code using vector similarity.
        
        Args:
            query_text: Search query text
            model_type: Embedding model to use ('unixcoder' or 'minilm'/'sbert')
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            # Get query embedding
            query_embedding = self.get_query_embedding(query_text, model_type)
            
            # Determine collection name based on existing naming convention
            if model_type == 'unixcoder':
                collection_name = "unixcoder_snippets"
            elif model_type in ['minilm', 'sbert']:  # Both refer to the same SBERT collection
                collection_name = "sbert_snippets"
            else:
                collection_name = f"code_chunks_{model_type}"
            
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception:
                print(f"Collection '{collection_name}' not found. Please run the ingestion script first.")
                return []
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # Convert ChromaDB squared L2 distance to similarity score
                # For normalized vectors: squared_L2 = 2 - 2*dot_product
                # Therefore: similarity = 1 - distance/2
                similarity = 1 - distance / 2
                
                result = {
                    'content': doc,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                    'start_line': metadata.get('start_line', 0),
                    'end_line': metadata.get('end_line', 0),
                    'function_name': metadata.get('function_name', ''),
                    'class_name': metadata.get('class_name', ''),
                    'score': float(similarity),
                    'similarity_score': float(similarity),  # True cosine similarity
                    'score_type': 'cosine_similarity',  # Mathematical meaning
                    'distance': float(distance)
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []


# Convenience functions for backward compatibility
def get_hf_embedding(query_text: str, model, tokenizer) -> np.ndarray:
    """Legacy function for HuggingFace embeddings."""
    engine = VectorSearchEngine()
    return engine.get_hf_embedding(query_text, 'unixcoder')


def get_sbert_embedding(query_text: str, model) -> np.ndarray:
    """Legacy function for SBERT embeddings."""
    engine = VectorSearchEngine()
    return engine.get_sbert_embedding(query_text)


def get_query_embedding(query_text: str, model_type: str = 'unixcoder') -> np.ndarray:
    """Legacy function for query embeddings."""
    engine = VectorSearchEngine()
    return engine.get_query_embedding(query_text, model_type)


def search_code(query_text: str, model_type: str = 'unixcoder', k: int = 5) -> List[Dict]:
    """Legacy function for code search."""
    engine = VectorSearchEngine()
    return engine.search_code(query_text, model_type, k)
