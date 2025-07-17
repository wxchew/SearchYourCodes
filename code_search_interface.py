#!/usr/bin/env python3
"""
Simple Code Search Interface
Demonstrates semantic code search using UniXcoder embeddings on production data.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Tuple, Dict

# Device configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class CodeSearcher:
    def __init__(self, embeddings_path: str = "data/embeddings/unixcoder_embeddings.npy",
                 metadata_path: str = "data/embeddings/unixcoder_chunk_metadata.json"):
        """Initialize the code searcher with precomputed embeddings."""
        
        # Load embeddings and metadata
        self.embeddings = np.load(embeddings_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Load UniXcoder model for query encoding
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        self.model = AutoModel.from_pretrained("microsoft/unixcoder-base").to(DEVICE)
        
        print(f"✅ Loaded {len(self.embeddings)} code chunks")
        print(f"✅ Embedding dimension: {self.embeddings.shape[1]}")
        print(f"✅ Using device: {DEVICE}")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query using the same model as the code chunks."""
        
        # Tokenize query
        encoded_input = self.tokenizer(
            query, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(DEVICE)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Apply mean pooling (same as code chunks)
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        query_embedding = (sum_embeddings / sum_mask).cpu().numpy()
        
        # Normalize to unit norm
        norm = np.linalg.norm(query_embedding)
        return query_embedding / max(norm, 1e-8)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, Dict]]:
        """Search for code chunks similar to the query."""
        
        # Encode the query
        query_embedding = self.encode_query(query)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            chunk_metadata = self.metadata[idx]
            
            results.append((similarity, chunk_metadata))
        
        return results
    
    def display_results(self, results: List[Tuple[float, Dict]], query: str):
        """Display search results in a formatted way."""
        
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 80)
        
        for i, (similarity, chunk) in enumerate(results, 1):
            print(f"\n{i}. Similarity: {similarity:.4f}")
            print(f"   Function: {chunk['function_name'] or 'N/A'}")
            print(f"   File: {chunk['file_path']}")
            print(f"   Lines: {chunk['start_line']}-{chunk['end_line']}")
            
            if chunk.get('class_name'):
                print(f"   Class: {chunk['class_name']}")
            
            print("-" * 60)

def main():
    """Main interactive search interface."""
    
    print("🚀 Code Search Interface")
    print("Using UniXcoder embeddings on production dataset (340 chunks)")
    print("-" * 60)
    
    # Initialize searcher
    try:
        searcher = CodeSearcher()
    except Exception as e:
        print(f"❌ Error initializing searcher: {e}")
        return
    
    # Interactive search loop
    while True:
        print("\n" + "="*60)
        query = input("Enter search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not query:
            continue
            
        try:
            # Perform search
            results = searcher.search(query, top_k=10)
            
            # Display results
            searcher.display_results(results, query)
            
        except Exception as e:
            print(f"❌ Search error: {e}")

def demo_searches():
    """Run some demo searches to showcase the system."""
    
    print("🎯 Demo: Semantic Code Search with UniXcoder")
    print("=" * 60)
    
    searcher = CodeSearcher()
    
    # Demo queries
    demo_queries = [
        "memory allocation",
        "file reading",
        "linked list",
        "matrix multiplication",
        "string processing",
        "error handling"
    ]
    
    for query in demo_queries:
        results = searcher.search(query, top_k=5)
        searcher.display_results(results, query)
        print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_searches()
    else:
        main()
