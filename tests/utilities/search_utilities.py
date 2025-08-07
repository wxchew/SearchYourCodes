#!/usr/bin/env python3
"""
Search System Utilities

Essential utility functions extracted from various debugging and analysis scripts:
1. Distance metric analysis and ChromaDB configuration verification
2. Embedding inspection and normalization checks
3. Database statistics and collection analysis
4. Pooling strategy testing utilities

Usage: python scripts/search_utilities.py [--function FUNC]
    Functions: distances, embeddings, database, pooling, all
"""

import os
import sys
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.search import get_query_embedding
    from transformers import RobertaTokenizer, RobertaModel
    import chromadb
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class SearchUtilities:
    """Essential utilities for search system analysis and debugging."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load UniXcoder for direct analysis
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        self.model = RobertaModel.from_pretrained("microsoft/unixcoder-base")
        self.model.to(self.device)
        self.model.eval()
        
    def analyze_distance_metrics(self):
        """Analyze ChromaDB distance metrics and verify configuration."""
        print("🔬 CHROMADB DISTANCE METRIC ANALYSIS")
        print("=" * 60)
        
        collections = self.client.list_collections()
        print(f"Found {len(collections)} collections:")
        
        for collection in collections:
            print(f"\n📁 Collection: {collection.name}")
            print(f"   Count: {collection.count()}")
            print(f"   Metadata: {collection.metadata}")
            
            # Analyze distance function
            if collection.metadata and 'hnsw:space' in collection.metadata:
                distance_func = collection.metadata['hnsw:space']
                print(f"   Distance Function: {distance_func}")
            else:
                print(f"   Distance Function: squared_l2 (default)")
            
            # Test with sample embeddings if available
            if collection.count() > 1:
                sample = collection.get(limit=2, include=['embeddings'])
                if len(sample['embeddings']) >= 2:
                    emb1 = np.array(sample['embeddings'][0])
                    emb2 = np.array(sample['embeddings'][1])
                    
                    # Query to get actual distance
                    results = collection.query(
                        query_embeddings=[emb1.tolist()],
                        n_results=2,
                        include=['distances']
                    )
                    
                    actual_distance = results['distances'][0][1]
                    
                    # Calculate expected distances
                    cosine_sim = np.dot(emb1, emb2)
                    cosine_dist = 1 - cosine_sim
                    l2_dist = np.linalg.norm(emb1 - emb2)
                    squared_l2_dist = np.sum((emb1 - emb2) ** 2)
                    
                    print(f"   Sample Distance Analysis:")
                    print(f"     Actual ChromaDB distance: {actual_distance:.6f}")
                    print(f"     Expected cosine distance: {cosine_dist:.6f}")
                    print(f"     Expected L2 distance: {l2_dist:.6f}")
                    print(f"     Expected squared L2: {squared_l2_dist:.6f}")
                    
                    # Determine which matches best
                    diffs = {
                        'cosine': abs(actual_distance - cosine_dist),
                        'l2': abs(actual_distance - l2_dist),
                        'squared_l2': abs(actual_distance - squared_l2_dist)
                    }
                    
                    best_match = min(diffs.keys(), key=lambda k: diffs[k])
                    print(f"     Best match: {best_match} (diff: {diffs[best_match]:.6f})")
    
    def analyze_embeddings(self):
        """Analyze embedding properties and normalization."""
        print("🧬 EMBEDDING ANALYSIS")
        print("=" * 60)
        
        collections = self.client.list_collections()
        unixcoder_col = None
        
        for c in collections:
            if 'unixcoder' in c.name:
                unixcoder_col = c
                break
        
        if not unixcoder_col:
            print("❌ No UniXcoder collection found!")
            return
        
        print(f"Analyzing collection: {unixcoder_col.name}")
        
        # Get sample embeddings
        sample_size = min(100, unixcoder_col.count())
        sample = unixcoder_col.get(limit=sample_size, include=['embeddings', 'metadatas'])
        
        embeddings = np.array(sample['embeddings'])
        print(f"Sample size: {len(embeddings)} embeddings")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Analyze normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"\n📊 NORMALIZATION ANALYSIS:")
        print(f"  Norm statistics:")
        print(f"    Mean: {np.mean(norms):.6f}")
        print(f"    Std: {np.std(norms):.6f}")
        print(f"    Min: {np.min(norms):.6f}")
        print(f"    Max: {np.max(norms):.6f}")
        
        # Check if normalized (should be close to 1.0)
        if np.abs(np.mean(norms) - 1.0) < 0.01 and np.std(norms) < 0.01:
            print("  ✅ Embeddings are L2 normalized")
        else:
            print("  ⚠️  Embeddings may not be properly normalized")
        
        # Analyze value distributions
        all_values = embeddings.flatten()
        print(f"\n📈 VALUE DISTRIBUTION:")
        print(f"  Value statistics:")
        print(f"    Mean: {np.mean(all_values):.6f}")
        print(f"    Std: {np.std(all_values):.6f}")
        print(f"    Min: {np.min(all_values):.6f}")
        print(f"    Max: {np.max(all_values):.6f}")
        
        # Check for degenerate patterns
        print(f"\n🔍 DEGENERACY CHECK:")
        
        # Check for identical embeddings
        unique_embeddings = len(set(tuple(emb) for emb in embeddings))
        print(f"  Unique embeddings: {unique_embeddings}/{len(embeddings)}")
        
        if unique_embeddings < len(embeddings):
            print(f"  ⚠️  Found {len(embeddings) - unique_embeddings} duplicate embeddings")
        else:
            print(f"  ✅ All embeddings are unique")
        
        # Check similarity distribution
        sample_pairs = min(1000, len(embeddings) * (len(embeddings) - 1) // 2)
        similarities = []
        
        import random
        pairs = [(i, j) for i in range(len(embeddings)) for j in range(i+1, len(embeddings))]
        random.shuffle(pairs)
        
        for i, j in pairs[:sample_pairs]:
            sim = np.dot(embeddings[i], embeddings[j])
            similarities.append(sim)
        
        similarities = np.array(similarities)
        print(f"\n📊 SIMILARITY DISTRIBUTION (sample of {len(similarities)} pairs):")
        print(f"  Mean similarity: {np.mean(similarities):.6f}")
        print(f"  Std similarity: {np.std(similarities):.6f}")
        print(f"  Min similarity: {np.min(similarities):.6f}")
        print(f"  Max similarity: {np.max(similarities):.6f}")
        
        # Check for CLS token dominance pattern
        if np.std(similarities) < 0.1:
            print(f"  ⚠️  Low similarity variance - possible CLS token dominance")
        else:
            print(f"  ✅ Good similarity distribution")
    
    def analyze_database_statistics(self):
        """Analyze database statistics and collection health."""
        print("📊 DATABASE STATISTICS")
        print("=" * 60)
        
        collections = self.client.list_collections()
        
        total_documents = 0
        for collection in collections:
            count = collection.count()
            total_documents += count
            
            print(f"\n📁 {collection.name}:")
            print(f"   Documents: {count:,}")
            
            if count > 0:
                # Sample analysis
                sample = collection.get(limit=5, include=['metadatas', 'documents'])
                
                # Analyze metadata completeness
                function_names = [m.get('function_name') for m in sample['metadatas']]
                valid_functions = sum(1 for f in function_names if f and f != 'Unknown' and f.strip())
                function_completeness = valid_functions / len(function_names) if function_names else 0
                
                print(f"   Function name completeness: {function_completeness:.1%}")
                
                # Analyze file distribution
                file_paths = [m.get('file_path', '') for m in sample['metadatas']]
                unique_files = len(set(file_paths))
                print(f"   File diversity (sample): {unique_files}/{len(file_paths)}")
                
                # Content length analysis
                if sample['documents']:
                    content_lengths = [len(doc) for doc in sample['documents']]
                    avg_length = sum(content_lengths) / len(content_lengths)
                    print(f"   Average content length: {avg_length:.0f} characters")
        
        print(f"\n📈 TOTAL STATISTICS:")
        print(f"   Total collections: {len(collections)}")
        print(f"   Total documents: {total_documents:,}")
        print(f"   Average docs per collection: {total_documents / len(collections):.0f}")
    
    def test_pooling_strategies(self):
        """Test different pooling strategies with UniXcoder."""
        print("🔄 POOLING STRATEGY TESTING")
        print("=" * 60)
        
        test_queries = [
            "class inheritance with virtual destructor",
            "diffusion coefficient calculation",
            "motor protein stepping mechanism"
        ]
        
        for query in test_queries:
            print(f"\n🎯 Query: '{query[:40]}...'")
            
            # Test with different pooling strategies
            for pooling in ['cls', 'mean']:
                try:
                    embedding = self.get_embedding_with_pooling(query, pooling)
                    
                    print(f"  {pooling.upper()} pooling:")
                    print(f"    Shape: {embedding.shape}")
                    print(f"    Norm: {np.linalg.norm(embedding):.6f}")
                    print(f"    Mean: {np.mean(embedding):.6f}")
                    print(f"    Std: {np.std(embedding):.6f}")
                    
                except Exception as e:
                    print(f"  {pooling.upper()} pooling: ERROR - {e}")
        
        # Compare pooling strategy discrimination
        print(f"\n🔍 POOLING DISCRIMINATION TEST:")
        queries = ["for loop", "class definition", "function call"]
        
        for pooling in ['cls', 'mean']:
            embeddings = []
            for query in queries:
                try:
                    emb = self.get_embedding_with_pooling(query, pooling)
                    embeddings.append(emb)
                except:
                    continue
            
            if len(embeddings) >= 2:
                # Calculate pairwise similarities
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j])
                        similarities.append(sim)
                
                similarities = np.array(similarities)
                print(f"  {pooling.upper()} similarity range: {np.min(similarities):.4f} - {np.max(similarities):.4f}")
                print(f"  {pooling.upper()} similarity std: {np.std(similarities):.4f}")
    
    def get_embedding_with_pooling(self, text: str, pooling_strategy: str = "mean") -> np.ndarray:
        """Generate embedding with specified pooling strategy."""
        inputs = self.tokenizer(text, 
                              return_tensors="pt", 
                              truncation=True, 
                              padding=True, 
                              max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            
            if pooling_strategy == "cls":
                embedding = last_hidden_state[:, 0, :]
            elif pooling_strategy == "mean":
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0]
    
    def run_utilities(self, function: str = "all"):
        """Run specified utility functions."""
        if function in ["distances", "all"]:
            self.analyze_distance_metrics()
        
        if function in ["embeddings", "all"]:
            print("\n" + "="*60)
            self.analyze_embeddings()
        
        if function in ["database", "all"]:
            print("\n" + "="*60)
            self.analyze_database_statistics()
        
        if function in ["pooling", "all"]:
            print("\n" + "="*60)
            self.test_pooling_strategies()

def main():
    parser = argparse.ArgumentParser(description="Search System Utilities")
    parser.add_argument('--function', choices=['distances', 'embeddings', 'database', 'pooling', 'all'], 
                       default='all', help='Utility function to run')
    
    args = parser.parse_args()
    
    utilities = SearchUtilities()
    utilities.run_utilities(args.function)

if __name__ == "__main__":
    main()
