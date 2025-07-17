#!/usr/bin/env python3
"""
Diagnostic script to analyze the CodeBERT embedding similarity issue.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_codebert_issue():
    """Analyze why CodeBERT embeddings are too similar."""
    print("=== CodeBERT Embedding Similarity Issue Analysis ===\n")
    
    # Load embeddings and metadata
    codebert_embeddings = np.load('data/embeddings/codebert_embeddings.npy')
    sbert_embeddings = np.load('data/embeddings/sentence_bert_minilm_embeddings.npy')
    
    with open('data/embeddings/chunk_metadata.json', 'r') as f:
        metadata = json.load(f)
        
    with open('data/code_chunks_clean.json', 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(codebert_embeddings)} CodeBERT embeddings")
    print(f"Loaded {len(sbert_embeddings)} Sentence-BERT embeddings")
    
    # 1. Compare embedding similarity distributions
    print("\n1. EMBEDDING SIMILARITY ANALYSIS")
    print("-" * 50)
    
    # Calculate pairwise similarities
    cb_similarities = cosine_similarity(codebert_embeddings)
    sb_similarities = cosine_similarity(sbert_embeddings)
    
    # Get upper triangular values (excluding diagonal)
    cb_upper = cb_similarities[np.triu_indices_from(cb_similarities, k=1)]
    sb_upper = sb_similarities[np.triu_indices_from(sb_similarities, k=1)]
    
    print(f"CodeBERT similarities:")
    print(f"  Mean: {cb_upper.mean():.6f}")
    print(f"  Std:  {cb_upper.std():.6f}")
    print(f"  Min:  {cb_upper.min():.6f}")
    print(f"  Max:  {cb_upper.max():.6f}")
    print(f"  Median: {np.median(cb_upper):.6f}")
    
    print(f"\nSentence-BERT similarities:")
    print(f"  Mean: {sb_upper.mean():.6f}")
    print(f"  Std:  {sb_upper.std():.6f}")
    print(f"  Min:  {sb_upper.min():.6f}")
    print(f"  Max:  {sb_upper.max():.6f}")
    print(f"  Median: {np.median(sb_upper):.6f}")
    
    # 2. Analyze content patterns
    print("\n2. CONTENT PATTERN ANALYSIS")
    print("-" * 50)
    
    # Check how many chunks start with the same context
    context_starts = {}
    file_purpose_starts = {}
    
    for chunk in chunks:
        content = chunk['content']
        lines = content.split('\n')
        
        # Find context line
        context_line = None
        purpose_line = None
        
        for line in lines:
            if line.startswith('// CONTEXT:'):
                context_line = line
            elif line.startswith('// FILE PURPOSE:'):
                purpose_line = line
        
        if context_line:
            context_starts[context_line] = context_starts.get(context_line, 0) + 1
        if purpose_line:
            file_purpose_starts[purpose_line] = file_purpose_starts.get(purpose_line, 0) + 1
    
    print(f"Number of unique CONTEXT lines: {len(context_starts)}")
    print(f"Number of unique FILE PURPOSE lines: {len(file_purpose_starts)}")
    
    # Show most common context patterns
    print("\nMost common CONTEXT patterns:")
    for context, count in sorted(context_starts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {count:3d}: {context}")
    
    print("\nMost common FILE PURPOSE patterns:")
    for purpose, count in sorted(file_purpose_starts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {count:3d}: {purpose[:80]}...")
    
    # 3. Find the most similar embeddings
    print("\n3. MOST SIMILAR EMBEDDINGS")
    print("-" * 50)
    
    # Find pairs with highest similarity
    max_indices = np.unravel_index(np.argsort(cb_similarities.ravel())[-10:], cb_similarities.shape)
    
    print("Top 10 most similar CodeBERT embedding pairs:")
    for i, (row, col) in enumerate(zip(max_indices[0], max_indices[1])):
        if row != col:  # Skip self-similarity
            sim = cb_similarities[row, col]
            func1 = metadata[row].get('function_name', 'N/A')
            func2 = metadata[col].get('function_name', 'N/A')
            file1 = Path(metadata[row]['file_path']).name
            file2 = Path(metadata[col]['file_path']).name
            print(f"  {sim:.6f}: {func1} ({file1}) <-> {func2} ({file2})")
    
    # 4. Check content length vs similarity
    print("\n4. CONTENT LENGTH VS SIMILARITY")
    print("-" * 50)
    
    content_lengths = [len(chunk['content']) for chunk in chunks]
    print(f"Content length stats:")
    print(f"  Mean: {np.mean(content_lengths):.1f}")
    print(f"  Std:  {np.std(content_lengths):.1f}")
    print(f"  Min:  {np.min(content_lengths)}")
    print(f"  Max:  {np.max(content_lengths)}")
    
    # 5. Proposed solutions
    print("\n5. PROPOSED SOLUTIONS")
    print("-" * 50)
    print("The issue is that CodeBERT is producing nearly identical embeddings due to:")
    print("1. All chunks have similar context structure (// CONTEXT: class X)")
    print("2. All chunks have similar file purpose headers")
    print("3. The actual code content is diluted by this boilerplate")
    print()
    print("Solutions:")
    print("A. Remove context boilerplate from embeddings")
    print("B. Use only the actual code content without context")
    print("C. Use different tokenization/embedding approach")
    print("D. Focus on function/class names and signatures")
    print("E. Use AST-based structural features")
    
    return {
        'codebert_sim_mean': cb_upper.mean(),
        'codebert_sim_std': cb_upper.std(),
        'sbert_sim_mean': sb_upper.mean(),
        'sbert_sim_std': sb_upper.std(),
        'unique_contexts': len(context_starts),
        'unique_purposes': len(file_purpose_starts)
    }

if __name__ == "__main__":
    results = analyze_codebert_issue()
    print(f"\n=== SUMMARY ===")
    print(f"CodeBERT embeddings have {results['codebert_sim_mean']:.3f} mean similarity")
    print(f"This is {results['codebert_sim_mean'] / results['sbert_sim_mean']:.2f}x higher than Sentence-BERT")
    print(f"Only {results['unique_contexts']} unique context patterns for 340 chunks")
