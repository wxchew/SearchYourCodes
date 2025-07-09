"""
Enhanced search with AST-based structural similarity to improve CodeBERT performance.
This combines semantic embeddings with structural code analysis.
"""

import chromadb
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
from ast_analyzer import HybridCodeSearchRanker

# Import embedding functions from embedder.py
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Device configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def get_hf_embeddings(texts: List[str], model_name: str, device: str):
    """Generate embeddings using HuggingFace model (for CodeBERT)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def get_sbert_embeddings(texts: List[str], model_name: str, device: str):
    """Generate embeddings using SentenceTransformer model."""
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def get_query_embedding(query_code: str, model_name: str, model_type: str):
    """Generate embedding for query code based on model type."""
    if model_type == "huggingface_automodel":
        return get_hf_embeddings([query_code], model_name, DEVICE)[0]
    elif model_type == "sentence_transformer":
        return get_sbert_embeddings([query_code], model_name, DEVICE)[0]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_code_chunks(chunks_file: Path) -> Dict[str, Dict]:
    """Load code chunks and create a lookup dictionary."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    chunk_lookup = {}
    for chunk in chunks:
        chunk_lookup[chunk['id']] = chunk
    
    return chunk_lookup

def extract_chunk_id_from_chroma_id(chroma_id: str) -> str:
    """Extract original chunk ID from ChromaDB ID format: 'model_idx_chunkid'."""
    parts = chroma_id.split('_')
    return parts[-1] if len(parts) >= 3 else chroma_id

def search_with_hybrid_ranking(collection, query_code: str, chunk_lookup: Dict, 
                              n_results: int = 10, hybrid_ranker: HybridCodeSearchRanker = None, 
                              query_embedding: np.ndarray = None) -> List[Dict]:
    """
    Search using semantic embeddings and re-rank with structural similarity.
    """
    # First, get semantic search results
    if query_embedding is not None:
        semantic_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results * 2, 50)  # Get more candidates for re-ranking
        )
    else:
        semantic_results = collection.query(
            query_texts=[query_code],
            n_results=min(n_results * 2, 50)  # Get more candidates for re-ranking
        )
    
    if not semantic_results['documents'] or not semantic_results['documents'][0]:
        return []
    
    # Convert ChromaDB results to our format
    candidate_results = []
    for i, (doc_id, document, metadata, distance) in enumerate(zip(
        semantic_results['ids'][0],
        semantic_results['documents'][0], 
        semantic_results['metadatas'][0],
        semantic_results['distances'][0]
    )):
        # Extract original chunk ID
        chunk_id = extract_chunk_id_from_chroma_id(doc_id)
        
        # Get full chunk info
        chunk_info = chunk_lookup.get(chunk_id, {})
        
        # Convert distance to similarity score (cosine distance: 0=identical, 2=opposite)
        semantic_score = max(0.0, 1.0 - distance / 2.0)
        
        candidate_results.append({
            'id': chunk_id,
            'content': document,
            'metadata': metadata,
            'semantic_score': semantic_score,
            'semantic_rank': i + 1,
            'full_chunk_info': chunk_info
        })
    
    # Apply hybrid ranking if ranker is provided
    if hybrid_ranker:
        candidate_results = hybrid_ranker.rank_results(query_code, candidate_results)
    
    # Return top n_results
    return candidate_results[:n_results]

def evaluate_enhanced_search():
    """Evaluate the enhanced search with AST-based re-ranking."""
    
    # Load necessary files
    chunks_file = Path("data/code_chunks_clean.json")
    vector_db_path = Path("data/chroma_db")
    config_file = Path("config/embedding_models.json")
    
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found")
        return
    
    if not config_file.exists():
        print(f"Error: {config_file} not found")
        return
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    chunk_lookup = load_code_chunks(chunks_file)
    print(f"Loaded {len(chunk_lookup)} code chunks")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(vector_db_path))
    
    # Test queries focused on function similarity
    test_queries = [
        {
            'description': 'Vector operations',
            'code': 'void normalize_vector(double* vec, int size) { /* normalize vector */ }'
        },
        {
            'description': 'Matrix multiplication',
            'code': 'void multiply_matrices(double** A, double** B, double** C, int n) { /* multiply */ }'
        },
        {
            'description': 'File I/O operations',
            'code': 'void read_file(const char* filename) { FILE* fp = fopen(filename, "r"); }'
        },
        {
            'description': 'Memory allocation',
            'code': 'void* allocate_memory(size_t size) { return malloc(size); }'
        },
        {
            'description': 'String manipulation',
            'code': 'void copy_string(char* dest, const char* src) { strcpy(dest, src); }'
        }
    ]
    
    # Initialize hybrid ranker with different weight combinations
    weight_combinations = [
        (1.0, 0.0),  # Pure semantic (baseline)
        (0.8, 0.2),  # Mostly semantic
        (0.7, 0.3),  # Balanced
        (0.6, 0.4),  # More structural
        (0.5, 0.5),  # Equal weights
    ]
    
    # Test both models
    model_configs = [
        ('CodeBERT', 'codebert_snippets', config['models']['CodeBERT']),
        ('Sentence-BERT_MiniLM', 'sbert_snippets', config['models']['Sentence-BERT_MiniLM'])
    ]
    
    results = {}
    
    for model_name, collection_name, model_config in model_configs:
        print(f"\n{'='*60}")
        print(f"Testing {model_name} with Enhanced Search")
        print(f"{'='*60}")
        
        try:
            collection = client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' loaded with {collection.count()} documents")
        except Exception as e:
            print(f"Error loading collection {collection_name}: {e}")
            continue
        
        model_results = {}
        
        for semantic_weight, structural_weight in weight_combinations:
            print(f"\nTesting weights: Semantic={semantic_weight:.1f}, Structural={structural_weight:.1f}")
            
            # Initialize ranker
            if structural_weight > 0:
                ranker = HybridCodeSearchRanker(semantic_weight, structural_weight)
            else:
                ranker = None  # Pure semantic baseline
            
            weight_key = f"sem_{semantic_weight:.1f}_struct_{structural_weight:.1f}"
            query_results = []
            
            for query in test_queries:
                print(f"  Query: {query['description']}")
                
                start_time = time.time()
                
                # Generate query embedding
                try:
                    query_embedding = get_query_embedding(
                        query['code'], 
                        model_config['model_name'], 
                        model_config['model_type']
                    )
                    
                    # Perform enhanced search
                    search_results = search_with_hybrid_ranking(
                        collection, query['code'], chunk_lookup, 
                        n_results=5, hybrid_ranker=ranker, query_embedding=query_embedding
                    )
                    
                except Exception as e:
                    print(f"    Error during search: {e}")
                    continue
                
                search_time = time.time() - start_time
                
                # Analyze results
                result_analysis = {
                    'query': query['description'],
                    'query_code': query['code'],
                    'search_time': search_time,
                    'num_results': len(search_results),
                    'results': []
                }
                
                for i, result in enumerate(search_results):
                    result_info = {
                        'rank': i + 1,
                        'chunk_id': result['id'],
                        'semantic_score': result['semantic_score'],
                        'structural_score': result.get('structural_score', 0.0),
                        'combined_score': result.get('combined_score', result['semantic_score']),
                        'function_name': result['metadata'].get('function_name', 'N/A'),
                        'file_path': result['metadata'].get('file_path', 'N/A'),
                        'content_preview': result['content'][:100] + '...' if len(result['content']) > 100 else result['content']
                    }
                    result_analysis['results'].append(result_info)
                
                query_results.append(result_analysis)
                
                # Print top result for inspection
                if search_results:
                    top_result = search_results[0]
                    print(f"    Top result: {top_result['metadata'].get('function_name', 'N/A')} "
                          f"(semantic: {top_result['semantic_score']:.3f}, "
                          f"structural: {top_result.get('structural_score', 0.0):.3f}, "
                          f"combined: {top_result.get('combined_score', top_result['semantic_score']):.3f})")
            
            model_results[weight_key] = query_results
        
        results[model_name] = model_results
    
    # Save detailed results
    results_file = Path("data/enhanced_search_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("ENHANCED SEARCH EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print("\nSUMMARY OF IMPROVEMENTS:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for weight_key, query_results in model_results.items():
            if query_results:
                avg_combined_score = np.mean([
                    np.mean([r['combined_score'] for r in qr['results']])
                    for qr in query_results if qr['results']
                ])
                print(f"  {weight_key}: Avg Combined Score = {avg_combined_score:.3f}")
            else:
                print(f"  {weight_key}: No results")

if __name__ == "__main__":
    evaluate_enhanced_search()
