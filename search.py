import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import chromadb

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path="./data/chroma_db")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Load Models and Tokenizers ONCE ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# CodeBERT
CODEBERT_MODEL_NAME = 'microsoft/codebert-base'
tokenizer_codebert = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
model_codebert = AutoModel.from_pretrained(CODEBERT_MODEL_NAME).to(DEVICE)
model_codebert.eval()

# Sentence-BERT
MINILM_MODEL_NAME = 'all-MiniLM-L6-v2'
model_minilm = SentenceTransformer(MINILM_MODEL_NAME, device=DEVICE)

print("Models loaded successfully!")

def get_hf_embedding(query_text: str, model, tokenizer):
    """
    Generate embedding using a pre-loaded HuggingFace model and tokenizer.
    """
    encoded_input = tokenizer([query_text], padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling (consistent with embedder.py)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Get mean pooled embeddings
    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
    
    # Normalize embeddings for consistent distance calculation
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norm, 1e-8)
    
    return normalized_embeddings

def get_sbert_embedding(query_text: str, model):
    """
    Generate embedding using a pre-loaded Sentence-BERT model.
    """
    embeddings = model.encode([query_text], convert_to_numpy=True)
    # Normalize to match stored embeddings (consistent with embedder.py)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-8)
    return normalized_embeddings

def get_query_embedding(query_text: str, model_type: str = 'codebert'):
    """
    Generate query embedding using specified model type.
    """
    if model_type.lower() == 'codebert':
        return get_hf_embedding(query_text, model_codebert, tokenizer_codebert)
    elif model_type.lower() in ['minilm', 'sentence_bert', 'sbert']:
        return get_sbert_embedding(query_text, model_minilm)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'codebert' or 'minilm'.")

def search_code(query_text: str, model_type: str = 'codebert', k: int = 5):
    """
    Search code chunks using the collection created by ingester.py
    
    Args:
        query_text (str): Natural language query
        model_type (str): 'codebert' or 'minilm' - determines which collection to use
        k (int): Number of results to return
    
    Returns:
        list: List of dictionaries containing search results
    """
    try:
        # Map model type to collection name (matches ingester.py naming)
        if model_type.lower() == 'codebert':
            collection_name = "codebert_snippets"
        elif model_type.lower() in ['minilm', 'sentence_bert', 'sbert']:
            collection_name = "sbert_snippets"
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'codebert' or 'minilm'.")
        
        # Get the appropriate collection
        collection = client.get_collection(name=collection_name)
        
        # Generate query embedding using the same model type
        query_vector = get_query_embedding(query_text, model_type)
        
        # Perform search
        results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        retrieved_chunks = []
        if results['documents'] and results['documents'][0]:
            ids = results.get('ids', [[]])[0]
            documents = results.get('documents', [[]])[0] 
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            for i, doc in enumerate(documents):
                raw_distance = distances[i]
                
                # Handle different distance metrics:
                # Both models use normalized embeddings, so we expect distances in [0, 2] range
                # ChromaDB likely uses L2 distance on normalized vectors
                
                if model_type.lower() == 'codebert':
                    # CodeBERT: normalized embeddings with L2 distance
                    # Convert L2 distance to cosine-like similarity
                    # For normalized vectors: L2_dist = sqrt(2 * (1 - cosine_similarity))
                    # So: cosine_similarity = 1 - (L2_dist^2 / 2)
                    similarity = max(0.0, 1 - (raw_distance ** 2) / 2)
                else:
                    # Sentence-BERT: normalized embeddings with L2 distance
                    # Same formula as CodeBERT since both are normalized
                    similarity = max(0.0, 1 - (raw_distance ** 2) / 2)
                
                retrieved_chunks.append({
                    'id': ids[i],
                    'content': doc,
                    'metadata': metadatas[i],
                    'distance': distances[i],
                    'similarity_score': similarity,
                    'raw_distance': raw_distance,
                    'model_type': model_type
                })
        
        return retrieved_chunks
        
    except Exception as e:
        print(f"Search error for {model_type}: {e}")
        return []

def print_results(results, query, model_type):
    """
    Print search results in a formatted way
    """
    if not results:
        print(f"No results found for {model_type.upper()}.")
        return
    
    print(f"\n🔍 Search Results for: '{query}' [{model_type.upper()}]")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        score = result['similarity_score']
        
        print(f"\n📄 Result {i} (Similarity: {score:.3f}, Distance: {result.get('raw_distance', 0):.3f})")
        print(f"📁 File: {metadata.get('file_path', 'Unknown')}")
        print(f"📍 Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}")
        
        if metadata.get('function_name') and metadata['function_name'] != 'None':
            print(f"⚙️  Function: {metadata['function_name']}")
        if metadata.get('class_name') and metadata['class_name'] != 'None':
            print(f"🏛️  Class: {metadata['class_name']}")
        
        # Show code preview
        content = result['content']
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"📝 Code Preview:")
        print("```cpp")
        print(preview)
        print("```")
        print("-" * 60)

def compare_models(query: str, k: int = 3):
    """
    Compare search results from both models side by side
    """
    print(f"\n{'='*80}")
    print(f"🔀 MODEL COMPARISON FOR QUERY: '{query}'")
    print(f"{'='*80}")
    
    # Search with both models
    codebert_results = search_code(query, model_type='codebert', k=k)
    sbert_results = search_code(query, model_type='minilm', k=k)
    
    # Display results side by side (summary)
    print(f"\n📊 QUICK COMPARISON (Top {k} results):")
    print("-" * 80)
    print(f"{'RANK':<4} {'CODEBERT':<35} {'SENTENCE-BERT':<35}")
    print("-" * 80)
    
    for i in range(max(len(codebert_results), len(sbert_results))):
        codebert_info = ""
        sbert_info = ""
        
        if i < len(codebert_results):
            cb_meta = codebert_results[i]['metadata']
            cb_score = codebert_results[i]['similarity_score']
            codebert_info = f"{cb_meta.get('function_name', 'Unknown')[:25]} ({cb_score:.3f})"
        
        if i < len(sbert_results):
            sb_meta = sbert_results[i]['metadata']
            sb_score = sbert_results[i]['similarity_score']
            sbert_info = f"{sb_meta.get('function_name', 'Unknown')[:25]} ({sb_score:.3f})"
        
        print(f"{i+1:<4} {codebert_info:<35} {sbert_info:<35}")
    
    # Detailed results for each model
    print_results(codebert_results[:2], query, 'codebert')  # Show top 2 detailed
    print_results(sbert_results[:2], query, 'sentence-bert')  # Show top 2 detailed
    
    return codebert_results, sbert_results

if __name__ == "__main__":
    # --- All your startup checks remain the same ---
    try:
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        print(f"Available collections: {collection_names}")
        
        required_collections = ["codebert_snippets", "sbert_snippets"]
        missing_collections = [col for col in required_collections if col not in collection_names]
        
        if missing_collections:
            print(f"❌ Missing collections: {missing_collections}")
            exit(1)
        
        for col_name in required_collections:
            collection = client.get_collection(col_name)
            print(f"✅ Collection '{col_name}' loaded with {collection.count()} chunks")
            
        # --- NEW: Interactive Search Loop ---
        print("\n\n🚀 **Interactive Semantic Code Search** 🚀")
        print("Enter your query below. Type 'exit' or 'quit' to end.")
        
        while True:
            print("\n" + "="*80)
            query = input("Enter your search query: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting search. Goodbye!")
                break
            
            if not query.strip():
                print("Query cannot be empty.")
                continue

            compare_models(query, k=5)

    except Exception as e:
        print(f"❌ A critical error occurred: {e}")