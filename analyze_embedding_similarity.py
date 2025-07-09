#!/usr/bin/env python3
"""
Embedding Similarity Analysis

Investigate why specific functions (selectR, selectL, SimThread::allHandles) 
consistently appear as top search results by analyzing their embeddings.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def load_embeddings_and_metadata():
    """Load embeddings and chunk metadata."""
    data_dir = Path("data")
    embeddings_dir = data_dir / "embeddings"
    
    # Load CodeBERT embeddings (normalized)
    codebert_file = embeddings_dir / "codebert_embeddings_normalized.npy"
    if not codebert_file.exists():
        codebert_file = embeddings_dir / "codebert_embeddings.npy"
    
    # Load Sentence-BERT embeddings  
    sbert_file = embeddings_dir / "sentence_bert_minilm_embeddings.npy"
    
    # Load metadata
    metadata_file = embeddings_dir / "chunk_metadata.json"
    
    try:
        codebert_embeddings = np.load(codebert_file)
        sbert_embeddings = np.load(sbert_file)
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        print(f"Loaded CodeBERT embeddings: {codebert_embeddings.shape}")
        print(f"Loaded SBERT embeddings: {sbert_embeddings.shape}")
        print(f"Loaded metadata for {len(metadata)} chunks")
        
        return codebert_embeddings, sbert_embeddings, metadata
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def find_function_indices(metadata: List[Dict], function_names: List[str]) -> Dict[str, int]:
    """Find indices of specific functions in the metadata."""
    function_indices = {}
    
    for i, chunk in enumerate(metadata):
        func_name = chunk.get('function_name')
        if func_name in function_names:
            function_indices[func_name] = i
            print(f"Found {func_name} at index {i}")
    
    return function_indices

def analyze_embedding_similarity(embeddings: np.ndarray, 
                                function_indices: Dict[str, int], 
                                metadata: List[Dict],
                                model_name: str) -> Dict:
    """Analyze embedding similarity patterns."""
    results = {}
    
    print(f"\n=== {model_name} Similarity Analysis ===")
    
    # Calculate pairwise similarities between target functions
    target_functions = list(function_indices.keys())
    similarity_matrix = np.zeros((len(target_functions), len(target_functions)))
    
    for i, func1 in enumerate(target_functions):
        for j, func2 in enumerate(target_functions):
            idx1, idx2 = function_indices[func1], function_indices[func2]
            similarity = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0, 0]
            similarity_matrix[i, j] = similarity
            
    print(f"Similarity matrix for target functions:")
    print(f"Functions: {target_functions}")
    print(similarity_matrix)
    
    # Find functions most similar to each target function
    for func_name, func_idx in function_indices.items():
        func_embedding = embeddings[func_idx].reshape(1, -1)
        
        # Calculate similarity with all other functions
        similarities = cosine_similarity(func_embedding, embeddings)[0]
        
        # Get top 10 most similar functions (excluding itself)
        similar_indices = np.argsort(similarities)[::-1][1:11]  # Exclude self
        
        print(f"\nTop 10 functions most similar to {func_name}:")
        for rank, idx in enumerate(similar_indices, 1):
            sim_score = similarities[idx]
            chunk = metadata[idx]
            func = chunk.get('function_name', 'N/A')
            file_path = Path(chunk['file_path']).name
            print(f"  {rank:2d}. {func:<30} (sim: {sim_score:.4f}) in {file_path}")
        
        results[func_name] = {
            'similarities': similarities,
            'top_similar_indices': similar_indices,
            'embedding_norm': np.linalg.norm(embeddings[func_idx])
        }
    
    return results

def analyze_statistical_properties(embeddings: np.ndarray, 
                                 function_indices: Dict[str, int],
                                 metadata: List[Dict],
                                 model_name: str):
    """Analyze statistical properties of target function embeddings."""
    print(f"\n=== {model_name} Statistical Properties ===")
    
    # Overall embedding statistics
    all_norms = np.linalg.norm(embeddings, axis=1)
    print(f"All embeddings - Mean norm: {all_norms.mean():.4f}, Std: {all_norms.std():.4f}")
    print(f"All embeddings - Min norm: {all_norms.min():.4f}, Max norm: {all_norms.max():.4f}")
    
    # Target function statistics
    for func_name, func_idx in function_indices.items():
        embedding = embeddings[func_idx]
        norm = np.linalg.norm(embedding)
        
        # Calculate percentile of this norm
        percentile = (all_norms < norm).mean() * 100
        
        print(f"{func_name}:")
        print(f"  Embedding norm: {norm:.4f} (percentile: {percentile:.1f})")
        print(f"  Mean value: {embedding.mean():.4f}")
        print(f"  Std value: {embedding.std():.4f}")
        print(f"  Min/Max: {embedding.min():.4f}/{embedding.max():.4f}")

def investigate_content_patterns(metadata: List[Dict], function_indices: Dict[str, int]):
    """Investigate content patterns of target functions."""
    print(f"\n=== Content Analysis ===")
    
    # Load actual code chunks
    chunks_file = Path("data/code_chunks_clean.json")
    try:
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        
        for func_name, func_idx in function_indices.items():
            chunk = chunks[func_idx]
            content = chunk['content']
            
            print(f"\n{func_name} content:")
            print(f"  Lines: {chunk['start_line']}-{chunk['end_line']} ({chunk['end_line'] - chunk['start_line'] + 1} lines)")
            print(f"  File: {chunk['file_path']}")
            print(f"  Content:\n{content}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Could not load code chunks: {e}")

def plot_similarity_heatmap(similarity_matrix: np.ndarray, 
                           function_names: List[str], 
                           model_name: str,
                           save_path: str = None):
    """Plot similarity heatmap for target functions."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, 
                xticklabels=function_names,
                yticklabels=function_names,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.3f')
    plt.title(f'{model_name} - Function Similarity Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze embedding similarity patterns")
    parser.add_argument('--target-functions', nargs='+', 
                       default=['selectR', 'selectL', 'SimThread::allHandles'],
                       help='Functions to analyze')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    args = parser.parse_args()
    
    # Load data
    codebert_embeddings, sbert_embeddings, metadata = load_embeddings_and_metadata()
    if codebert_embeddings is None:
        print("Failed to load embeddings. Exiting.")
        return 1
    
    # Find target function indices
    function_indices = find_function_indices(metadata, args.target_functions)
    if not function_indices:
        print("No target functions found in metadata.")
        return 1
    
    print(f"Analyzing functions: {list(function_indices.keys())}")
    
    # Analyze both models
    models = [
        (codebert_embeddings, "CodeBERT"),
        (sbert_embeddings, "Sentence-BERT")
    ]
    
    for embeddings, model_name in models:
        if embeddings is not None:
            # Similarity analysis
            results = analyze_embedding_similarity(embeddings, function_indices, metadata, model_name)
            
            # Statistical properties
            analyze_statistical_properties(embeddings, function_indices, metadata, model_name)
    
    # Content analysis (model-independent)
    investigate_content_patterns(metadata, function_indices)
    
    # Create similarity heatmaps if requested
    if args.save_plots:
        target_functions = list(function_indices.keys())
        
        for embeddings, model_name in models:
            if embeddings is not None and len(target_functions) > 1:
                # Calculate similarity matrix
                similarity_matrix = np.zeros((len(target_functions), len(target_functions)))
                for i, func1 in enumerate(target_functions):
                    for j, func2 in enumerate(target_functions):
                        idx1, idx2 = function_indices[func1], function_indices[func2]
                        similarity = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0, 0]
                        similarity_matrix[i, j] = similarity
                
                save_path = f"data/similarity_heatmap_{model_name.lower().replace('-', '_')}.png"
                plot_similarity_heatmap(similarity_matrix, target_functions, model_name, save_path)
    
    return 0

if __name__ == "__main__":
    exit(main())
