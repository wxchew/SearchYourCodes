import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict
import numpy as np
import argparse

# Ensure MPS is available for M1 Mac
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Helper Function to get embeddings ---
def get_hf_embeddings(texts: List[str], model_name: str, batch_size: int = 32, pooling_method: str = 'mean'):
    """
    Generates embeddings using a Hugging Face AutoModel (e.g., CodeBERT).
    Applies mean pooling to get sentence-level embeddings.
    Processes texts in batches to avoid memory issues.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    
    all_embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the batch
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        
        with torch.no_grad(): # Disable gradient calculation for inference
            model_output = model(**encoded_input)
        
        # Pooling strategies
        if pooling_method == 'cls':
            # Use [CLS] token representation
            batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        elif pooling_method == 'pooler' and hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
            # Use the pooler_output if available
            batch_embeddings = model_output.pooler_output.cpu().numpy()
        else:
            # Default to mean pooling on the last hidden state
            token_embeddings = model_output.last_hidden_state # (batch, seq, hidden)
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        # Normalize embeddings to unit norm
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / np.maximum(norms, 1e-8)
        all_embeddings.append(batch_embeddings)
        
        # Clear cache to free memory
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    return np.concatenate(all_embeddings, axis=0)

def get_sbert_embeddings(texts: List[str], model_name: str):
    """
    Generates embeddings using a Sentence-BERT model.
    """
    model = SentenceTransformer(model_name, device=DEVICE)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # Normalize SBERT embeddings too
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


# --- Main Experimentation Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save embeddings for code chunks")
    parser.add_argument('--pooling-method', type=str, choices=['mean','cls','pooler'], default='mean', help='Pooling method for HF models')
    args = parser.parse_args()

    pooling_method = args.pooling_method
    # Load your generated code chunks
    chunks_file = Path("data") / "code_chunks_clean.json"
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Please run src/code_parser.py first.")
        exit(1)

    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Process all chunks for complete coverage
    sample_size = len(all_chunks)  # Use all chunks instead of limiting to 100
    sample_chunks = all_chunks
    sample_texts = [chunk['content'] for chunk in sample_chunks]
    
    print(f"\n--- Experimenting with {len(sample_texts)} code chunks ---")

    # --- Models to evaluate ---
    models_to_test = {
        "Sentence-BERT_MiniLM": "all-MiniLM-L6-v2",
        "CodeBERT": "microsoft/codebert-base",
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
    model_config_path = Path("config") / "embedding_models.json"
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
    
    # Also save chunk metadata for reference
    chunk_metadata_file = embeddings_dir / "chunk_metadata.json"
    chunk_metadata = [
        {
            "index": i,
            "chunk_id": chunk["id"],
            "file_path": chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "function_name": chunk.get("function_name"),
            "class_name": chunk.get("class_name")
        }
        for i, chunk in enumerate(sample_chunks)
    ]
    with open(chunk_metadata_file, 'w') as f:
        json.dump(chunk_metadata, f, indent=2)
    print(f"Chunk metadata saved to {chunk_metadata_file}")