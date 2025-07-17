import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import chromadb
import requests
import json

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path="./data/chroma_db")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Load Models and Tokenizers ONCE ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# UniXcoder
UNIXCODER_MODEL_NAME = 'microsoft/unixcoder-base'
tokenizer_unixcoder = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)
model_unixcoder = AutoModel.from_pretrained(UNIXCODER_MODEL_NAME).to(DEVICE)
model_unixcoder.eval()

# Sentence-BERT
MINILM_MODEL_NAME = 'all-MiniLM-L6-v2'
model_minilm = SentenceTransformer(MINILM_MODEL_NAME, device=DEVICE)

print("Models loaded successfully!")

# --- LLM Query Refinement Configuration ---
QUERY_REFINEMENT_CONFIG = {
    "enabled": True,
    "ollama_url": "http://localhost:11434",
    "model": "deepseek-coder:latest",
    "fallback_model": "llama3.2:latest",
    "show_refinement": True,
    "max_tokens": 50,  # Reduced for shorter responses
    "temperature": 0.1,
    "timeout": 10  # Reduced timeout for faster responses
}

def call_ollama(prompt: str, model: str = None) -> str:
    """
    Call Ollama API to generate text
    """
    model = model or QUERY_REFINEMENT_CONFIG["model"]
    
    try:
        response = requests.post(
            f"{QUERY_REFINEMENT_CONFIG['ollama_url']}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": QUERY_REFINEMENT_CONFIG["temperature"],
                    "num_predict": QUERY_REFINEMENT_CONFIG["max_tokens"]
                }
            },
            timeout=QUERY_REFINEMENT_CONFIG["timeout"]
        )
        
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None



def refine_query_code_specific(query: str) -> str:
    """
    Refine query using code-specific enhancement
    """
    prompt = f"""Add programming terms to this query while keeping ALL original words:

"{query}"

Rules:
- Keep every original word exactly
- Add 1-3 programming terms: function, method, implementation, algorithm, behavior, code
- Maximum 15 words total

Enhanced query:"""
    
    refined = call_ollama(prompt, QUERY_REFINEMENT_CONFIG["fallback_model"])
    if refined:
        # Clean up the response - take first line that looks reasonable
        lines = refined.split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines and instructional text
            if (line and 
                not line.startswith(('Enhanced', 'Query:', 'Here', 'The', 'Original:', 'CRITICAL:', 'Rules:', 'Add', 'Keep', 'Maximum')) and
                len(line) > len(query) * 0.8):  # Must be reasonably similar length
                
                refined = line.replace('"', '').replace("'", "").strip()
                # Remove parenthetical explanations
                if '(' in refined:
                    refined = refined.split('(')[0].strip()
                
                # Basic validation
                if len(refined.split()) <= 15 and len(refined) >= len(query):
                    # Quick check that key domain terms are preserved
                    original_lower = query.lower()
                    refined_lower = refined.lower()
                    
                    # Check for key terms like 'ase walker', 'microtubule', etc.
                    key_phrases = []
                    if 'ase walker' in original_lower:
                        key_phrases.append('ase walker')
                    if 'microtubule' in original_lower:
                        key_phrases.append('microtubule')
                    if 'plus end' in original_lower:
                        key_phrases.append('plus end')
                    
                    # Ensure key phrases are preserved
                    if all(phrase in refined_lower for phrase in key_phrases):
                        return refined
    
    return query

def refine_query_with_llm(query: str, method: str = "code") -> tuple:
    """
    Refine query using LLM with code-specific enhancement
    
    Args:
        query: Original query
        method: "code" or "none"
    
    Returns:
        tuple: (original_query, refined_query, method_used)
    """
    if not QUERY_REFINEMENT_CONFIG["enabled"] or method == "none":
        return query, query, "none"
    
    original_query = query
    
    if method == "code":
        refined = refine_query_code_specific(query)
        return original_query, refined, "code-specific"
    
    return original_query, query, "none"

def get_hf_embedding(query_text: str, model, tokenizer):
    """
    Generate embedding using a pre-loaded HuggingFace model and tokenizer.
    """
    encoded_input = tokenizer([query_text], padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
    
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

def get_query_embedding(query_text: str, model_type: str = 'unixcoder'):
    """
    Generate query embedding using specified model type.
    """
    if model_type.lower() == 'unixcoder':
        return get_hf_embedding(query_text, model_unixcoder, tokenizer_unixcoder)
    elif model_type.lower() in ['minilm', 'sentence_bert', 'sbert']:
        return get_sbert_embedding(query_text, model_minilm)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'unixcoder' or 'minilm'.")

def search_code(query_text: str, model_type: str = 'unixcoder', k: int = 5, enable_refinement: bool = True):
    """
    Search code chunks using the collection created by ingester.py
    
    Args:
        query_text (str): Natural language query
        model_type (str): 'unixcoder' or 'minilm' - determines which collection to use
        k (int): Number of results to return
        enable_refinement (bool): Whether to use LLM query refinement
    
    Returns:
        list: List of dictionaries containing search results
    """
    try:
        # Refine query if enabled
        original_query = query_text
        refined_query = query_text
        refinement_method = "none"
        
        if enable_refinement and QUERY_REFINEMENT_CONFIG["enabled"]:
            original_query, refined_query, refinement_method = refine_query_with_llm(query_text, "code")
            
            # Show refinement for transparency
            if QUERY_REFINEMENT_CONFIG["show_refinement"] and refined_query != original_query:
                print(f"🔍 Original query: '{original_query}'")
                print(f"🤖 Refined query:  '{refined_query}' [{refinement_method}]")
                print("-" * 60)
        
        # Use refined query for search
        search_query = refined_query
        
        # Map model type to collection name (matches ingester.py naming)
        if model_type.lower() == 'unixcoder':
            collection_name = "unixcoder_snippets"
        elif model_type.lower() in ['minilm', 'sentence_bert', 'sbert']:
            collection_name = "sbert_snippets"
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'unixcoder' or 'minilm'.")
        
        # Get the appropriate collection
        collection = client.get_collection(name=collection_name)
        
        # Generate query embedding using the refined query
        query_vector = get_query_embedding(search_query, model_type)
        
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
                # All models use normalized embeddings, so we expect distances in [0, 2] range
                # ChromaDB likely uses L2 distance on normalized vectors
                
                if model_type.lower() == 'unixcoder':
                    # UniXcoder: normalized embeddings with L2 distance
                    # Convert L2 distance to cosine-like similarity
                    # For normalized vectors: L2_dist = sqrt(2 * (1 - cosine_similarity))
                    # So: cosine_similarity = 1 - (L2_dist^2 / 2)
                    similarity = max(0.0, 1 - (raw_distance ** 2) / 2)
                else:
                    # Sentence-BERT: normalized embeddings with L2 distance
                    # Same formula as UniXcoder since both are normalized
                    similarity = max(0.0, 1 - (raw_distance ** 2) / 2)
                
                retrieved_chunks.append({
                    'id': ids[i],
                    'content': doc,
                    'metadata': metadatas[i],
                    'distance': distances[i],
                    'similarity_score': similarity,
                    'raw_distance': raw_distance,
                    'model_type': model_type,
                    'original_query': original_query,
                    'refined_query': refined_query,
                    'refinement_method': refinement_method
                })
        
        return retrieved_chunks
        
    except Exception as e:
        print(f"Search error for {model_type}: {e}")
        # Fallback to original query if refinement fails
        if enable_refinement and query_text != original_query:
            print(f"🔄 Falling back to original query: '{original_query}'")
            return search_code(original_query, model_type, k, enable_refinement=False)
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

def compare_models(query: str, k: int = 3, enable_refinement: bool = True):
    """
    Compare search results from UniXcoder and Sentence-BERT models side by side
    """
    print(f"\n{'='*80}")
    refinement_note = " (with LLM refinement)" if enable_refinement else " (original query)"
    print(f"🔀 MODEL COMPARISON FOR QUERY: '{query}'{refinement_note}")
    print(f"{'='*80}")
    
    # Search with both models
    unixcoder_results = search_code(query, model_type='unixcoder', k=k, enable_refinement=enable_refinement)
    sbert_results = search_code(query, model_type='minilm', k=k, enable_refinement=enable_refinement)
    
    # Display results side by side (summary)
    print(f"\n📊 QUICK COMPARISON (Top {k} results):")
    print("-" * 70)
    print(f"{'RANK':<4} {'UNIXCODER':<30} {'SENTENCE-BERT':<30}")
    print("-" * 70)
    
    for i in range(max(len(unixcoder_results), len(sbert_results))):
        unixcoder_info = ""
        sbert_info = ""
        
        if i < len(unixcoder_results):
            ux_meta = unixcoder_results[i]['metadata']
            ux_score = unixcoder_results[i]['similarity_score']
            unixcoder_info = f"{ux_meta.get('function_name', 'Unknown')[:22]} ({ux_score:.3f})"
        
        if i < len(sbert_results):
            sb_meta = sbert_results[i]['metadata']
            sb_score = sbert_results[i]['similarity_score']
            sbert_info = f"{sb_meta.get('function_name', 'Unknown')[:22]} ({sb_score:.3f})"
        
        print(f"{i+1:<4} {unixcoder_info:<30} {sbert_info:<30}")
    
    # Show refinement info if used
    if enable_refinement and unixcoder_results:
        result = unixcoder_results[0]
        if result.get('refined_query') != result.get('original_query'):
            print(f"\n🤖 Query refinement applied: {result.get('refinement_method', 'unknown')}")
    
    # Detailed results for each model
    print_results(unixcoder_results[:2], query, 'unixcoder')  # Show top 2 detailed
    print_results(sbert_results[:2], query, 'sentence-bert')  # Show top 2 detailed
    
    return unixcoder_results, sbert_results

if __name__ == "__main__":
    # --- All your startup checks remain the same ---
    try:
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        print(f"Available collections: {collection_names}")
        
        required_collections = ["unixcoder_snippets", "sbert_snippets"]
        missing_collections = [col for col in required_collections if col not in collection_names]
        
        if missing_collections:
            print(f"❌ Missing collections: {missing_collections}")
            exit(1)
        
        for col_name in required_collections:
            collection = client.get_collection(col_name)
            print(f"✅ Collection '{col_name}' loaded with {collection.count()} chunks")
            
        # --- NEW: Interactive Search Loop ---
        print("\n\n🚀 **Interactive Semantic Code Search with LLM Query Refinement** 🚀")
        print("Features:")
        print("  - 🤖 LLM-powered code-specific query refinement using Ollama")
        print("  - 🔍 Comparison between original and refined queries")
        print("  - 📊 Side-by-side model comparison (UniXcoder vs Sentence-BERT)")
        print("  - 🎯 Transparent refinement process")
        print("\nCommands:")
        print("  - Type your search query to see both comparisons")
        print("  - 'exit' or 'quit' - End session")
        
        while True:
            print("\n" + "="*80)
            query = input(f"Enter search query: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting search. Goodbye!")
                break
            
            if not query.strip():
                print("Query cannot be empty.")
                continue

            # Compare with and without LLM refinement
            print(f"\n🔀 Comparing: No Refinement vs Code-Specific Refinement")
            print("=" * 80)
            
            # Test without refinement
            print("\n1️⃣ Without LLM Refinement:")
            compare_models(query, k=5, enable_refinement=False)
            
            # Test with code-specific refinement
            print("\n2️⃣ With Code-Specific LLM Refinement:")
            original, code_refined, method = refine_query_with_llm(query, "code")
            if code_refined != original:
                print(f"   📝 Original: '{original}'")
                print(f"   🔧 Refined:  '{code_refined}' [{method}]")
            compare_models(query, k=5, enable_refinement=True)

    except Exception as e:
        print(f"❌ A critical error occurred: {e}")