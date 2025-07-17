#!/usr/bin/env python3
"""
Fixed CodeBERT embeddings by removing context boilerplate.
This should solve the issue of all embeddings being too similar.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict
import numpy as np
import re

# Device configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def clean_code_content(content: str) -> str:
    """
    Remove context boilerplate that's causing CodeBERT to produce similar embeddings.
    """
    lines = content.split('\n')
    cleaned_lines = []
    
    skip_patterns = [
        r'^// CONTEXT:',
        r'^// FILE PURPOSE:',
        r'^// CONTEXT END',
        r'^$',  # Empty lines at the start
    ]
    
    started_actual_content = False
    
    for line in lines:
        # Skip boilerplate patterns
        should_skip = any(re.match(pattern, line.strip()) for pattern in skip_patterns)
        
        if should_skip and not started_actual_content:
            continue
        
        # Once we hit actual content, keep everything
        if line.strip() and not should_skip:
            started_actual_content = True
        
        if started_actual_content:
            cleaned_lines.append(line)
    
    # Join and clean up
    cleaned = '\n'.join(cleaned_lines).strip()
    
    # If we end up with very little content, fallback to original
    if len(cleaned) < 50:
        return content
    
    return cleaned

def extract_function_signature(content: str, function_name: str) -> str:
    """
    Extract function signature and nearby code for better embedding.
    """
    if not function_name or function_name == 'N/A':
        return content
    
    lines = content.split('\n')
    
    # Find function definition
    function_lines = []
    found_function = False
    
    for i, line in enumerate(lines):
        if function_name in line and ('(' in line or '::' in line):
            found_function = True
            # Add some context before and after
            start = max(0, i - 2)
            end = min(len(lines), i + 10)
            function_lines = lines[start:end]
            break
    
    if found_function:
        return '\n'.join(function_lines)
    
    return content

def get_hf_embeddings_fixed(texts: List[str], model_name: str, batch_size: int = 32):
    """
    Generate embeddings using HuggingFace model with cleaned content.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=512, return_tensors='pt').to(DEVICE)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Mean pooling
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / np.maximum(norms, 1e-8)
        
        all_embeddings.append(batch_embeddings)
        
        # Clear cache
        if DEVICE == "mps":
            torch.mps.empty_cache()
    
    return np.concatenate(all_embeddings, axis=0)

def generate_fixed_embeddings():
    """Generate fixed CodeBERT embeddings."""
    
    # Load code chunks
    chunks_file = Path("data/code_chunks_clean.json")
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found")
        return
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} code chunks")
    
    # Process chunks with different cleaning strategies
    strategies = {
        'cleaned': [],
        'function_focused': [],
        'minimal': []
    }
    
    for chunk in chunks:
        original_content = chunk['content']
        function_name = chunk.get('function_name')
        
        # Strategy 1: Clean boilerplate
        cleaned_content = clean_code_content(original_content)
        strategies['cleaned'].append(cleaned_content)
        
        # Strategy 2: Function-focused
        if function_name and function_name != 'N/A':
            func_content = extract_function_signature(cleaned_content, function_name)
            strategies['function_focused'].append(func_content)
        else:
            strategies['function_focused'].append(cleaned_content)
        
        # Strategy 3: Minimal (just function name + first few lines)
        if function_name and function_name != 'N/A':
            minimal_content = f"{function_name}\\n{cleaned_content[:200]}"
            strategies['minimal'].append(minimal_content)
        else:
            strategies['minimal'].append(cleaned_content[:200])
    
    # Generate embeddings for each strategy
    results = {}
    
    for strategy_name, texts in strategies.items():
        print(f"\\nGenerating embeddings for strategy: {strategy_name}")
        
        # Show sample of processed content
        print(f"Sample content for {strategy_name}:")
        for i in range(min(3, len(texts))):
            print(f"  Sample {i+1}: {texts[i][:100]}...")
        
        # Generate embeddings
        embeddings = get_hf_embeddings_fixed(texts, 'microsoft/codebert-base')
        
        # Save embeddings
        output_file = Path(f"data/embeddings/codebert_{strategy_name}_embeddings.npy")
        np.save(output_file, embeddings)
        print(f"Saved {strategy_name} embeddings to {output_file}")
        
        # Calculate similarity statistics
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        upper_similarities = similarities[np.triu_indices_from(similarities, k=1)]
        
        print(f"{strategy_name} statistics:")
        print(f"  Mean similarity: {upper_similarities.mean():.6f}")
        print(f"  Std similarity:  {upper_similarities.std():.6f}")
        print(f"  Min similarity:  {upper_similarities.min():.6f}")
        print(f"  Max similarity:  {upper_similarities.max():.6f}")
        
        results[strategy_name] = {
            'embeddings': embeddings,
            'mean_similarity': upper_similarities.mean(),
            'std_similarity': upper_similarities.std(),
            'texts': texts[:5]  # Save first 5 for inspection
        }
    
    # Save results summary
    summary_file = Path("data/embeddings/fixed_codebert_summary.json")
    summary = {
        strategy: {
            'mean_similarity': float(result['mean_similarity']),
            'std_similarity': float(result['std_similarity']),
            'sample_texts': result['texts']
        }
        for strategy, result in results.items()
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nSummary saved to {summary_file}")
    
    # Recommend best strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['mean_similarity'])
    print(f"\\nRecommended strategy: {best_strategy}")
    print(f"This reduces mean similarity from 0.982 to {results[best_strategy]['mean_similarity']:.6f}")

if __name__ == "__main__":
    generate_fixed_embeddings()
