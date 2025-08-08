"""
ChromaDB Ingestion System for Code Embeddings

This module ingests pre-computed embeddings into ChromaDB collections for
efficient similarity search. It supports multiple embedding models and
creates separate collections for comparison purposes.

The system processes code chunks with their embeddings and metadata,
creating persistent vector databases for the search interface.
"""

import json
from pathlib import Path
from typing import List, Dict, Union

import torch
import numpy as np
import chromadb
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Helper Functions for Embedding Generation
def get_hf_embeddings(texts: List[str], model_name: str, device: str) -> np.ndarray:
    """Generate embeddings using HuggingFace AutoModel."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    encoded_input = tokenizer(texts, padding=True, truncation=True, 
                             max_length=512, return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return (sum_embeddings / sum_mask).cpu().numpy()


def get_sbert_embeddings(texts: List[str], model_name: str, device: str) -> np.ndarray:
    """Generate embeddings using Sentence-BERT model."""
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def get_hf_embeddings(texts: List[str], model_name: str, device: str, pooling_method: str = 'mean') -> np.ndarray:
    """Generate embeddings using HuggingFace AutoModel."""
    # Import from embedder module to avoid duplication
    from .embedder import get_hf_embeddings as _get_hf_embeddings
    return _get_hf_embeddings(texts, model_name, device, pooling_method=pooling_method)


def load_embedding_model(model_config: Dict):
    """Load the chosen embedding model based on configuration."""
    model_name = model_config['name']
    model_type = model_config['type']
    device = model_config['device']

    if model_type == "sentence_transformer":
        print(f"Loading Sentence-Transformer model: {model_name}")
        return lambda texts: get_sbert_embeddings(texts, model_name, device)
    elif model_type == "huggingface_automodel":
        print(f"Loading Hugging Face AutoModel: {model_name}")
        pooling_method = model_config.get('pooling_method', 'mean')
        return lambda texts: get_hf_embeddings(texts, model_name, device, pooling_method=pooling_method)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main ingestion logic."""
    from .config import load_config, get_model_config
    
    # Load configuration
    config = load_config()
    
    # Load configuration and chunks
    chunks_file = Path(config['data']['processed']) / "code_chunks_clean.json"
    vector_db_path = Path(config['data']['chroma_db'])
    model_state_file = Path(config['data']['processed']) / "model_state.json"

    # Check if we have model state from embedder.py
    if not model_state_file.exists():
        print(f"Error: {model_state_file} not found. Please run embedder.py first to generate embeddings.")
        return
        
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Please run code_parser.py first.")
        return

    with open(model_state_file, 'r', encoding='utf-8') as f:
        model_state = json.load(f)
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    if not all_chunks:
        print("No chunks found in code_chunks_clean.json. Exiting.")
        return

    # Process models from state file to create separate collections
    available_models = list(model_state['models'].keys())
    print(f"Available models: {available_models}")
    print(f"Loaded {len(all_chunks)} code chunks.")
    
    # Initialize ChromaDB Client
    vector_db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_db_path))
    print(f"ChromaDB client initialized. Data will be stored in: {vector_db_path}")
    
    # Delete existing model-specific collections to avoid ID duplication
    for coll_name in ["sbert_snippets", "unixcoder_snippets"]:
        try:
            client.delete_collection(name=coll_name)
            print(f"Deleted existing collection: {coll_name}")
        except Exception:
            pass
     
    # Process each available model
    for model_name in available_models:
        model_info = model_state['models'][model_name]
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name} ({model_info['model_name']})")
        print(f"Model type: {model_info['model_type']}")
        
        # Create collection without embedding function (use pre-computed embeddings)
        if model_name == "sbert":
            collection_name = "sbert_snippets"
        elif model_name == "unixcoder":
            collection_name = "unixcoder_snippets"
        else:
            collection_name = f"{model_name.lower()}_snippets"
            
        collection = client.create_collection(name=collection_name)
        
        # Load pre-computed embeddings
        embedding_file = Path(model_info['embedding_file'])
        
        if not embedding_file.exists():
            print(f"Error: Embedding file {embedding_file} not found!")
            continue
            
        print(f"Loading embeddings from: {embedding_file}")
        embeddings = np.load(embedding_file)
        print(f"Loaded embeddings shape: {embeddings.shape}")
        
        # Prepare data for ingestion
        chunk_ids = [chunk["id"] for chunk in all_chunks]
        
        # Debug: Check for duplicates in the data
        print(f"Total chunks loaded: {len(all_chunks)}")
        print(f"Total chunk IDs: {len(chunk_ids)}")
        
        # Find duplicates
        seen_ids = set()
        duplicates = []
        unique_indices = []
        
        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id in seen_ids:
                duplicates.append(chunk_id)
                print(f"Duplicate found at index {i}: {chunk_id}")
            else:
                seen_ids.add(chunk_id)
                unique_indices.append(i)
        
        if duplicates:
            print(f"Found {len(duplicates)} duplicate IDs. Removing duplicates...")
            # Filter to unique chunks only
            all_chunks = [all_chunks[i] for i in unique_indices]
            chunk_ids = [chunk["id"] for chunk in all_chunks]
            embeddings = embeddings[unique_indices]
            print(f"After deduplication: {len(all_chunks)} unique chunks")
        
        chunk_contents = [chunk["content"] for chunk in all_chunks]
        chunk_metadatas = []
        for chunk in all_chunks:
            metadata = {
                "file_path": chunk["file_path"],
                "start_line": str(chunk["start_line"]),
                "end_line": str(chunk["end_line"]),
            }
            # Only add non-None values
            if chunk.get("function_name"):
                metadata["function_name"] = chunk["function_name"]
            if chunk.get("class_name"):
                metadata["class_name"] = chunk["class_name"]
            if chunk.get("namespace"):
                metadata["namespace"] = chunk["namespace"]
            if chunk.get("docstring"):
                metadata["docstring"] = chunk["docstring"]
            
            chunk_metadatas.append(metadata)
        
        # Add chunks to collection in batches with pre-computed embeddings
        batch_size = 50
        total_batches = (len(chunk_ids) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc=f"Ingesting {model_name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunk_ids))
            
            batch_ids = chunk_ids[start_idx:end_idx]
            batch_contents = chunk_contents[start_idx:end_idx]
            batch_metadatas = chunk_metadatas[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx].tolist()
            
            collection.add(
                ids=batch_ids,
                documents=batch_contents,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
        
        print(f"✓ Successfully ingested {len(chunk_ids)} chunks into '{collection_name}'")
        print(f"  Collection count: {collection.count()}")
    
    print(f"\n{'='*60}")
    print("Ingestion complete! Collections created:")
    
    # Verify all collections
    collections = client.list_collections()
    for collection in collections:
        count = client.get_collection(collection.name).count()
        print(f"  - {collection.name}: {count} documents")
    
    print(f"\nVector database ready at: {vector_db_path}")


if __name__ == "__main__":
    main()