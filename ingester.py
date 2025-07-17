import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict, Union
import numpy as np

# --- Imports for ChromaDB ---
import chromadb
from chromadb.utils import embedding_functions

# --- Device Configuration (from embedder.py) ---
# Ensure MPS is available for M1 Mac
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Helper Function to get embeddings (copied/adapted from embedder.py) ---
# It's good practice to centralize this, but for this learning sprint,
# copying it here makes the ingester self-contained.
def get_hf_embeddings(texts: List[str], model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return (sum_embeddings / sum_mask).cpu().numpy()

def get_sbert_embeddings(texts: List[str], model_name: str, device: str):
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def load_embedding_model(model_config: Dict):
    """Loads the chosen embedding model based on configuration."""
    model_name = model_config['model_name']
    model_type = model_config['model_type']
    device = model_config['device']

    if model_type == "sentence_transformer":
        # For SentenceTransformer, we need a callable function that takes texts and returns embeddings
        # ChromaDB's default SentenceTransformerEF can be used, or we wrap our own.
        # For simplicity and direct control, we'll use our `get_sbert_embeddings` directly.
        print(f"Loading Sentence-Transformer model: {model_name}")
        return lambda texts: get_sbert_embeddings(texts, model_name, device)
    elif model_type == "huggingface_automodel":
        print(f"Loading Hugging Face AutoModel: {model_name}")
        return lambda texts: get_hf_embeddings(texts, model_name, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'sentence_transformer' or 'huggingface_automodel'.")


# --- Main Ingestion Logic ---
if __name__ == "__main__":
    # --- 1. Load Configuration and Chunks ---
    model_config_path = Path("config") / "unixcoder_models.json"
    chunks_file = Path("data") / "code_chunks_clean.json"
    vector_db_path = Path("data") / "chroma_db" # Directory for ChromaDB persistence

    if not model_config_path.exists():
        print(f"Error: {model_config_path} not found. Please run src/embedder.py first.")
        exit(1)
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Please run src/code_parser.py first.")
        exit(1)

    with open(model_config_path, 'r', encoding='utf-8') as f:
        models_config = json.load(f)
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    if not all_chunks:
        print("No chunks found in code_chunks_clean.json. Exiting.")
        exit(0)

    # Process BOTH models to create separate collections for comparison
    available_models = list(models_config['models'].keys())
    print(f"Available models: {available_models}")
    print(f"Loaded {len(all_chunks)} code chunks.")
    
    # --- 2. Initialize ChromaDB Client ---
    vector_db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_db_path))
    print(f"ChromaDB client initialized. Data will be stored in: {vector_db_path}")
    
    # Delete existing model-specific collections to avoid ID duplication
    for coll_name in ["codebert_snippets", "sbert_snippets", "unixcoder_snippets"]:
        try:
            client.delete_collection(name=coll_name)
            print(f"Deleted existing collection: {coll_name}")
        except Exception:
            pass
     
    # Process each available model
    for model_name in available_models:
        model_config = models_config['models'][model_name]
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name} ({model_config['model_name']})")
        print(f"Model type: {model_config['model_type']}")
        
        # Initialize embedding function for this model
        embedding_function_callable = load_embedding_model(model_config)
        
        # Create collection name based on model
        if model_name == "CodeBERT":
            collection_name = "codebert_snippets"
        elif model_name == "Sentence-BERT_MiniLM":
            collection_name = "sbert_snippets"
        elif model_name == "UniXcoder":
            collection_name = "unixcoder_snippets"
        else:
            collection_name = f"{model_name.lower()}_snippets"
        
        collection = client.get_or_create_collection(name=collection_name)
        print(f"ChromaDB collection '{collection_name}' ready.")

        # --- Prepare Data for Ingestion ---
        documents = []  # The actual code content
        metadatas = []  # The metadata dictionaries
        ids = []        # Unique IDs for each chunk

        for idx, chunk in enumerate(all_chunks):
            documents.append(chunk['content'])
            # Ensure metadata is a flat dictionary suitable for Chroma (no None values)
            metadata = {}
            for k, v in chunk.items():
                if k not in ['content', 'id']:
                    if v is not None:
                        metadata[k] = str(v)
            metadatas.append(metadata)
            # Use index to ensure unique IDs
            unique_id = f"{model_name}_{idx}_{chunk['id']}"
            ids.append(unique_id)

        print(f"Prepared {len(documents)} documents for {model_name}.")

        # --- Generate Embeddings and Ingest into ChromaDB ---
        # Determine the correct embedding file path based on model
        if model_name == "CodeBERT":
            embedding_file = Path("data/embeddings/codebert_embeddings.npy")
        elif model_name == "Sentence-BERT_MiniLM":
            embedding_file = Path("data/embeddings/sentence_bert_minilm_embeddings.npy")
        elif model_name == "UniXcoder":
            embedding_file = Path("data/embeddings/unixcoder_embeddings.npy")
        else:
            # Fallback to config file path if available
            embedding_file = Path(model_config.get('embedding_file', ''))
        
        # Always load pre-computed normalized embeddings
        if not embedding_file.exists():
            print(f"Error: Normalized embedding file not found: {embedding_file}")
            exit(1)
        print(f"Loading normalized embeddings from {embedding_file}")
        all_embeddings = np.load(embedding_file)
        if len(all_embeddings) != len(documents):
            print(f"Error: Embedding count ({len(all_embeddings)}) != document count ({len(documents)})")
            exit(1)

        print(f"Ingesting {model_name} embeddings and metadata into ChromaDB...")
        try:
            # ChromaDB's add method takes lists of embeddings, metadatas, and ids
            collection.add(
                embeddings=[e.tolist() for e in all_embeddings], # Convert numpy array to list for Chroma
                metadatas=metadatas,
                documents=documents, # Optionally store the original document content as well
                ids=ids
            )
            print(f"✅ Successfully ingested {len(all_chunks)} chunks into '{collection_name}' collection.")
            print(f"   Collection count: {collection.count()}")

        except Exception as e:
            print(f"❌ Error during {model_name} ingestion: {e}")
    
    # --- Summary ---
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE - SUMMARY")
    print(f"{'='*60}")
    collections = client.list_collections()
    for col in collections:
        print(f"📦 Collection: {col.name} - Count: {col.count()}")
    print(f"✅ Both models processed successfully!")
    print("\nYou can now run search.py to compare model performance.")