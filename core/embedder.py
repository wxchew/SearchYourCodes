"""
Embedding Generation System for SearchYourCodes

This module provides embedding generation capabilities using various models:
- HuggingFace AutoModels (e.g., microsoft/unixcoder-base)  
- Sentence Transformers (e.g., all-MiniLM-L6-v2)

The system supports batch processing and different pooling strategies for
optimal performance and accuracy.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Helper Functions for Embedding Generation
def get_hf_embeddings(texts: List[str], model_name: str, device: str = "auto",
                     batch_size: int = 32, pooling_method: str = 'mean') -> np.ndarray:
    """
    Generate embeddings using HuggingFace AutoModel with various pooling strategies.
    
    Args:
        texts: List of input texts
        model_name: HuggingFace model identifier
        device: Device to use for inference
        batch_size: Batch size for processing
        pooling_method: Pooling strategy ('mean', 'cls', 'pooler')
        
    Returns:
        Normalized embeddings array
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    all_embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the batch
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                max_length=512, return_tensors='pt').to(device)
        
        with torch.no_grad():  # Disable gradient calculation for inference
            model_output = model(**encoded_input)
        
        # Apply pooling strategy
        if pooling_method == 'cls':
            # Use [CLS] token representation
            batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        elif (pooling_method == 'pooler' and hasattr(model_output, 'pooler_output') 
              and model_output.pooler_output is not None):
            # Use the pooler_output if available
            batch_embeddings = model_output.pooler_output.cpu().numpy()
        else:
            # Default to mean pooling on the last hidden state
            token_embeddings = model_output.last_hidden_state  # (batch, seq, hidden)
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        # Normalize embeddings to unit norm
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / np.maximum(norms, 1e-8)
        all_embeddings.append(batch_embeddings)
        
        # Clear cache to prevent memory buildup
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        
    # Combine all batches
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings


def get_sbert_embeddings(texts: List[str], model_name: str, device: str = "auto") -> np.ndarray:
    """
    Generate normalized embeddings using Sentence-BERT model.
    
    Args:
        texts: List of input texts
        model_name: Sentence-BERT model identifier
        device: Device to use for inference
        
    Returns:
        Normalized embeddings array
    """
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    # Normalize embeddings to unit norm
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-8)
    return normalized_embeddings


def benchmark_model(texts: List[str], model_config: Dict, output_file: Path) -> Dict:
    """
    Benchmark a model's performance and save results.
    
    Args:
        texts: List of input texts
        model_config: Model configuration dictionary
        output_file: Path to save benchmark results
        
    Returns:
        Dictionary containing benchmark metrics
    """
    model_name = model_config['name']
    model_type = model_config['type']
    device = model_config['device']
    
    print(f"\nBenchmarking {model_name} ({model_type}) on {device}...")
    
    # Use time.time() for timing instead of CUDA events (works on all devices)
    start_time = time.time()
    
    if model_type == "sentence_transformer":
        embeddings = get_sbert_embeddings(texts, model_name, device)
    elif model_type == "huggingface_automodel":
        pooling_method = model_config.get('pooling_method', 'mean')
        embeddings = get_hf_embeddings(texts, model_name, device, batch_size=16, pooling_method=pooling_method)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Synchronize device operations before measuring end time
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    metrics = {
        "model_name": model_name,
        "model_type": model_type,
        "device": device,
        "total_time_seconds": total_time,
        "chunks_processed": len(texts),
        "time_per_chunk": total_time / len(texts),
        "embedding_dimension": embeddings.shape[1],
        "total_embeddings": embeddings.shape[0]
    }
    
    print(f"✓ Processed {len(texts)} chunks in {total_time:.2f}s ({total_time/len(texts):.4f}s per chunk)")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Save embeddings as numpy array (more efficient than JSON)
    numpy_file = output_file.with_suffix('.npy')
    np.save(numpy_file, embeddings)
    print(f"  Saved embeddings to: {numpy_file}")
    
    # Also save just the metadata as JSON for reference
    with open(output_file.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  Results saved to: {output_file}")
    return metrics


def main():
    """Main function for embedding generation and benchmarking."""
    from .config import load_config, get_model_config
    
    parser = argparse.ArgumentParser(description="Generate and save embeddings for code chunks")
    parser.add_argument('--pooling-method', type=str, choices=['mean','cls','pooler'], 
                       default=None, help='Override pooling method for HF models')
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    
    # Load code chunks
    chunks_file = Path(config['data']['processed']) / "code_chunks_clean.json"
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Please run code_parser.py first.")
        return

    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Use all chunks for complete coverage
    sample_texts = [chunk['content'] for chunk in all_chunks]
    
    print(f"\n--- Processing {len(sample_texts)} code chunks ---")

    # Get model configurations from unified config
    try:
        unixcoder_config = get_model_config('unixcoder')
        sbert_config = get_model_config('sbert')
    except ValueError as e:
        print(f"Error loading model configuration: {e}")
        return
    
    # Override pooling method if specified
    if args.pooling_method:
        unixcoder_config['pooling_method'] = args.pooling_method
        print(f"--- Using override pooling method: {args.pooling_method} ---")
    else:
        print(f"--- Using configured pooling method: {unixcoder_config.get('pooling_method', 'mean')} ---")

    # Define models to benchmark based on configuration
    models_to_test = {
        "unixcoder": unixcoder_config,
        "sbert": sbert_config
    }
    
    # Create output directories
    results_dir = Path(config['data']['embeddings'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark each model
    all_results = {}
    for model_key, model_config in models_to_test.items():
        # Use .npy as the primary format, .json for metadata only
        output_file = results_dir / f"{model_key.lower().replace('-', '_')}_embeddings"
        try:
            metrics = benchmark_model(sample_texts, model_config, output_file)
            all_results[model_key] = metrics
        except Exception as e:
            print(f"Error processing {model_key}: {e}")
            continue
    
    # Save consolidated results
    summary_file = results_dir / "embedding_benchmark_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    # Save model state information (separate from configuration)
    saved_models = {}
    for model_key, model_config in models_to_test.items():
        if model_key in all_results:
            metrics = all_results[model_key]
            saved_models[model_key] = {
                "model_name": model_config['name'],
                "model_type": model_config['type'],
                "embedding_file": f"data/embeddings/{model_key.lower().replace('-', '_')}_embeddings.npy",
                "embedding_shape": [metrics['total_embeddings'], metrics['embedding_dimension']],
                "device": model_config['device'],
                "pooling_method": model_config.get('pooling_method', 'built-in')
            }

    # Save model state to data directory (not config directory)
    model_state_path = Path(config['data']['processed']) / "model_state.json"
    with open(model_state_path, 'w') as f:
        json.dump({
            "models": saved_models,
            "sample_info": {
                "sample_size": len(sample_texts),
                "total_chunks": len(all_chunks),
                "chunks_file": str(chunks_file),
                "generated_at": str(Path(__file__).parent.parent)  # Store project root for reference
            }
        }, f, indent=2)
    print(f"\nModel state information saved to {model_state_path}")
    
    print(f"\n=== Benchmark Summary ===")
    for model_key, metrics in all_results.items():
        print(f"{model_key}: {metrics['chunks_processed']} chunks in {metrics['total_time_seconds']:.2f}s")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()