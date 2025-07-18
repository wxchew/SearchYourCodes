"""
Embedding Generation System for Code Chunks

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
def get_hf_embeddings(texts: List[str], model_name: str, batch_size: int = 32, 
                     pooling_method: str = 'mean') -> np.ndarray:
    """
    Generate embeddings using HuggingFace AutoModel with various pooling strategies.
    
    Args:
        texts: List of input texts
        model_name: HuggingFace model identifier
        batch_size: Batch size for processing
        pooling_method: Pooling strategy ('mean', 'cls', 'pooler')
        
    Returns:
        Normalized embeddings array
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    
    all_embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the batch
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                max_length=512, return_tensors='pt').to(DEVICE)
        
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
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
        
    # Combine all batches
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings


def get_sbert_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """
    Generate normalized embeddings using Sentence-BERT model.
    
    Args:
        texts: List of input texts
        model_name: Sentence-BERT model identifier
        
    Returns:
        Normalized embeddings array
    """
    model = SentenceTransformer(model_name, device=DEVICE)
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
    model_name = model_config['model_name']
    model_type = model_config['model_type']
    
    print(f"\nBenchmarking {model_name} ({model_type})...")
    
    # Use time.time() for timing instead of CUDA events (works on all devices)
    start_time = time.time()
    
    if model_type == "sentence_transformer":
        embeddings = get_sbert_embeddings(texts, model_name)
    elif model_type == "huggingface_automodel":
        embeddings = get_hf_embeddings(texts, model_name, batch_size=16)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Synchronize device operations before measuring end time
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    metrics = {
        "model_name": model_name,
        "model_type": model_type,
        "total_time_seconds": total_time,
        "chunks_processed": len(texts),
        "time_per_chunk": total_time / len(texts),
        "embedding_dimension": embeddings.shape[1],
        "total_embeddings": embeddings.shape[0]
    }
    
    print(f"✓ Processed {len(texts)} chunks in {total_time:.2f}s ({total_time/len(texts):.4f}s per chunk)")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Save embeddings and metrics
    results_data = {
        "embeddings": embeddings.tolist(),
        "metadata": metrics
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"  Results saved to: {output_file}")
    return metrics


def main():
    """Main function for embedding generation and benchmarking."""
    parser = argparse.ArgumentParser(description="Generate and save embeddings for code chunks")
    parser.add_argument('--pooling-method', type=str, choices=['mean','cls','pooler'], 
                       default='mean', help='Pooling method for HF models')
    args = parser.parse_args()

    pooling_method = args.pooling_method
    
    # Load code chunks
    chunks_file = Path("data") / "code_chunks_clean.json"
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Please run code_parser_clean.py first.")
        return

    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Use all chunks for complete coverage
    sample_texts = [chunk['content'] for chunk in all_chunks]
    
    print(f"\n--- Experimenting with {len(sample_texts)} code chunks ---")

    # --- Models to evaluate ---
    models_to_test = {
        "Sentence-BERT_MiniLM": "all-MiniLM-L6-v2",
        "UniXcoder": "microsoft/unixcoder-base",
        # "Salesforce_CodeT5P_220M": "Salesforce/codet5p-220m-bpe", # Uncomment if you want to test
        # "BAAI_BGE_Base": "BAAI/bge-base-en-v1.5" # Uncomment if you want to test
    }

    # Store results for comparison
    embedding_results = {}

    for model_alias, model_name in models_to_test.items():
        print(f"\nEvaluating model: {model_alias} ({model_name})")
        
        # Use time.time() for timing instead of CUDA events (works on all devices)
        import time
        start_time = time.time()

        embeddings = None
        if "MiniLM" in model_alias or "mpnet" in model_alias or "BGE" in model_alias:
            embeddings = get_sbert_embeddings(sample_texts, model_name)
        else: # Assume it's a standard AutoModel
            embeddings = get_hf_embeddings(sample_texts, model_name, pooling_method=pooling_method)
        
        # Synchronize device operations before measuring end time
        if DEVICE == "mps":
            torch.mps.synchronize()
        elif DEVICE == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"  Time taken for {len(sample_texts)} chunks: {elapsed_time:.2f} seconds")

        print(f"  Generated embeddings shape: {embeddings.shape}")
        embedding_results[model_alias] = embeddings

        # Basic check of embedding content (first few values of first embedding)
        print(f"  First embedding (first 5 values): {embeddings[0][:5]}")


    # --- Save embeddings from both models ---
    embeddings_dir = Path("data") / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    saved_models = {}
    for model_alias, embeddings in embedding_results.items():
        # Save embeddings as numpy array
        embedding_file = embeddings_dir / f"{model_alias.lower().replace('-', '_')}_embeddings.npy"
        np.save(embedding_file, embeddings)
        print(f"Saved {model_alias} embeddings to {embedding_file}")
        
        # Store model info
        model_name = models_to_test[model_alias]
        if "MiniLM" in model_alias or "mpnet" in model_alias or "BGE" in model_alias:
            model_type = "sentence_transformer"
        else:
            model_type = "huggingface_automodel"
            
        saved_models[model_alias] = {
            "model_name": model_name,
            "model_type": model_type,
            "embedding_file": str(embedding_file),
            "embedding_shape": list(embeddings.shape),
            "device": DEVICE
        }

    # Save model configuration with both models
    model_config_path = Path("config") / "unixcoder_models.json"
    model_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_config_path, 'w') as f:
        json.dump({
            "models": saved_models,
            "sample_info": {
                "sample_size": len(sample_texts),
                "total_chunks": len(all_chunks),
                "chunks_file": str(chunks_file),
                "pooling_method": pooling_method
            }
        }, f, indent=2)
    print(f"\nBoth model configurations saved to {model_config_path}")
    
    print(f"Loaded {len(all_chunks)} code chunks for processing.")
    
    # Define models to benchmark
    models_to_test = {
        "unixcoder": {
            "model_name": "microsoft/unixcoder-base",
            "model_type": "huggingface_automodel",
            "device": DEVICE
        },
        "sbert": {
            "model_name": "all-MiniLM-L6-v2", 
            "model_type": "sentence_transformer",
            "device": DEVICE
        }
    }
    
    # Create output directories
    results_dir = Path("data") / "embeddings"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark each model
    all_results = {}
    for model_key, model_config in models_to_test.items():
        output_file = results_dir / f"{model_key}_embeddings.json"
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
    
    print(f"\n=== Benchmark Summary ===")
    for model_key, metrics in all_results.items():
        print(f"{model_key}: {metrics['chunks_processed']} chunks in {metrics['total_time_seconds']:.2f}s")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()