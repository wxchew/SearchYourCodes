"""
Object-Oriented Embedder Architecture

This module implements an extensible embedder system following the Open/Closed Principle.
New embedding models can be added by creating new classes without modifying existing code.

Architecture:
- BaseEmbedder: Abstract base class defining the interface
- SentenceTransformerEmbedder: Handles Sentence-BERT models
- HuggingFaceEmbedder: Handles HuggingFace AutoModels
- EmbedderFactory: Creates appropriate embedder instances
- EmbedderRegistry: Manages available embedder types

Benefits:
- Easy to extend with new model types
- Follows SOLID principles
- Maintains separation of concerns
- Type-safe and well-documented
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional
from pathlib import Path
import json

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models.
    
    This class defines the interface that all embedders must implement,
    ensuring consistency and enabling polymorphic usage.
    """
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name/path of the model
            device: Device to use for inference
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.device = device if device != "auto" else self._get_default_device()
        self.model = None
        self._is_loaded = False
        
    @staticmethod
    def _get_default_device() -> str:
        """Get the default device based on availability."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model into memory. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Normalized embedding matrix
        """
        pass
    
    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for texts with automatic batching and progress tracking.
        
        Args:
            texts: List of input texts
            batch_size: Size of processing batches
            show_progress: Whether to show progress bar
            
        Returns:
            Normalized embedding matrix
        """
        # Lazy loading
        if not self._is_loaded:
            self._load_model()
            self._is_loaded = True
        
        all_embeddings = []
        
        # Process in batches
        batch_iterator = range(0, len(texts), batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc=f"Embedding with {self.__class__.__name__}")
        
        for i in batch_iterator:
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # Clear cache to prevent memory buildup
            self._clear_cache()
        
        # Combine all batches
        return np.vstack(all_embeddings)
    
    def _clear_cache(self) -> None:
        """Clear device cache to prevent memory buildup."""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self._is_loaded:
            self._load_model()
            self._is_loaded = True
        # Test with a dummy input to get dimension
        test_embedding = self._embed_batch(["test"])
        return test_embedding.shape[1]
    
    def benchmark(self, texts: List[str], output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Benchmark the embedder's performance.
        
        Args:
            texts: List of texts to benchmark on
            output_file: Optional path to save results
            
        Returns:
            Dictionary containing benchmark metrics
        """
        print(f"\nBenchmarking {self.__class__.__name__} with {self.model_name}...")
        
        start_time = time.time()
        embeddings = self.embed(texts, show_progress=True)
        
        # Synchronize device operations
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        metrics = {
            "embedder_class": self.__class__.__name__,
            "model_name": self.model_name,
            "device": self.device,
            "total_time_seconds": total_time,
            "texts_processed": len(texts),
            "time_per_text": total_time / len(texts),
            "embedding_dimension": embeddings.shape[1],
            "total_embeddings": embeddings.shape[0]
        }
        
        print(f"✓ Processed {len(texts)} texts in {total_time:.2f}s ({total_time/len(texts):.4f}s per text)")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        # Save results if requested
        if output_file:
            # Save embeddings
            np.save(output_file.with_suffix('.npy'), embeddings)
            # Save metadata
            with open(output_file.with_suffix('.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"  Results saved to: {output_file}")
        
        return metrics


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder for Sentence-BERT models via the sentence-transformers library."""
    
    def _load_model(self) -> None:
        """Load the Sentence-BERT model."""
        print(f"Loading Sentence-Transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Sentence-BERT."""
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        # Ensure embeddings are 2D (batch_size, embedding_dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Normalize to unit norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)


class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder for HuggingFace AutoModels with configurable pooling."""
    
    def __init__(self, model_name: str, device: str = "auto", pooling_method: str = "mean", **kwargs):
        """
        Initialize HuggingFace embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for inference
            pooling_method: Pooling strategy ('mean', 'cls', 'pooler')
        """
        super().__init__(model_name, device, **kwargs)
        self.pooling_method = pooling_method
        self.tokenizer = None
    
    def _load_model(self) -> None:
        """Load the HuggingFace model and tokenizer."""
        print(f"Loading HuggingFace AutoModel: {self.model_name} (pooling: {self.pooling_method})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using HuggingFace AutoModel with specified pooling."""
        # Tokenize
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, 
            max_length=512, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Apply pooling strategy
        if self.pooling_method == 'cls':
            embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        elif (self.pooling_method == 'pooler' and hasattr(model_output, 'pooler_output') 
              and model_output.pooler_output is not None):
            embeddings = model_output.pooler_output.cpu().numpy()
        else:  # mean pooling
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        # Normalize to unit norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)


class EmbedderRegistry:
    """Registry for managing available embedder types."""
    
    _embedders: Dict[str, Type[BaseEmbedder]] = {
        "sentence_transformer": SentenceTransformerEmbedder,
        "huggingface_automodel": HuggingFaceEmbedder,
    }
    
    @classmethod
    def register(cls, embedder_type: str, embedder_class: Type[BaseEmbedder]) -> None:
        """
        Register a new embedder type.
        
        Args:
            embedder_type: String identifier for the embedder
            embedder_class: Embedder class that inherits from BaseEmbedder
        """
        if not issubclass(embedder_class, BaseEmbedder):
            raise ValueError(f"Embedder class must inherit from BaseEmbedder")
        cls._embedders[embedder_type] = embedder_class
        print(f"Registered embedder type: {embedder_type}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available embedder types."""
        return list(cls._embedders.keys())
    
    @classmethod
    def create_embedder(cls, embedder_type: str, model_name: str, **kwargs) -> BaseEmbedder:
        """
        Create an embedder instance of the specified type.
        
        Args:
            embedder_type: Type of embedder to create
            model_name: Model name/path
            **kwargs: Additional parameters for the embedder
            
        Returns:
            Configured embedder instance
        """
        if embedder_type not in cls._embedders:
            available = ", ".join(cls.get_available_types())
            raise ValueError(f"Unknown embedder type '{embedder_type}'. Available: {available}")
        
        embedder_class = cls._embedders[embedder_type]
        return embedder_class(model_name, **kwargs)


class EmbedderFactory:
    """Factory for creating embedders from configuration."""
    
    @staticmethod
    def from_config(model_config: Dict[str, Any]) -> BaseEmbedder:
        """
        Create an embedder from a configuration dictionary.
        
        Args:
            model_config: Configuration dictionary containing type, name, and parameters
            
        Returns:
            Configured embedder instance
        """
        embedder_type = model_config['type']
        model_name = model_config['name']
        device = model_config.get('device', 'auto')
        
        # Extract type-specific parameters
        kwargs = {k: v for k, v in model_config.items() 
                 if k not in ['type', 'name', 'device']}
        kwargs['device'] = device
        
        return EmbedderRegistry.create_embedder(embedder_type, model_name, **kwargs)


# Example of how to extend with a new embedder type
class OpenAIEmbedder(BaseEmbedder):
    """
    Example embedder for OpenAI models.
    
    This demonstrates how easy it is to add new embedder types
    without modifying existing code.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, device="api", **kwargs)  # API-based, no local device
        self.api_key = api_key
    
    def _load_model(self) -> None:
        """OpenAI models don't need local loading."""
        print(f"Configuring OpenAI embedder: {self.model_name}")
        # Here you would initialize the OpenAI client
        pass
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings via OpenAI API."""
        # Placeholder implementation
        print(f"Would call OpenAI API for {len(texts)} texts")
        # In real implementation, you'd call the OpenAI API here
        # For demo, return random embeddings
        return np.random.rand(len(texts), 1536)  # OpenAI ada-002 dimension


# Register the new embedder type (demonstrates extensibility)
# EmbedderRegistry.register("openai", OpenAIEmbedder)


def benchmark_all_models(texts: List[str], model_configs: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Benchmark multiple embedder models using the OOP approach.
    
    Args:
        texts: List of texts to benchmark
        model_configs: Dictionary of model configurations
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    for model_key, config in model_configs.items():
        try:
            # Create embedder using factory
            embedder = EmbedderFactory.from_config(config)
            
            # Benchmark
            metrics = embedder.benchmark(texts)
            results[model_key] = metrics
            
        except Exception as e:
            print(f"Error benchmarking {model_key}: {e}")
            continue
    
    return results


# Example usage demonstration
if __name__ == "__main__":
    # Example model configurations
    model_configs = {
        "unixcoder": {
            "type": "huggingface_automodel",
            "name": "microsoft/unixcoder-base",
            "device": "mps",
            "pooling_method": "mean"
        },
        "sbert": {
            "type": "sentence_transformer",
            "name": "all-MiniLM-L6-v2",
            "device": "mps"
        }
    }
    
    # Test texts
    test_texts = [
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "class Rectangle: def __init__(self, width, height): self.width = width",
        "import numpy as np; arr = np.array([1, 2, 3, 4, 5])"
    ]
    
    print("=== Object-Oriented Embedder Architecture Demo ===")
    print(f"Available embedder types: {EmbedderRegistry.get_available_types()}")
    
    # Benchmark all models
    results = benchmark_all_models(test_texts, model_configs)
    
    print("\n=== Benchmark Summary ===")
    for model_key, metrics in results.items():
        print(f"{model_key}: {metrics['texts_processed']} texts in {metrics['total_time_seconds']:.2f}s")
