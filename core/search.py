"""
Comprehensive Code Search System

This module provides a multi-method search interface for C++ codebases using:
1. Keyword Search: Exact text matching with fuzzy search capabilities
2. UniXcoder: Code structure and programming pattern search
3. SBERT: Semantic understanding and conceptual search

The system integrates ChromaDB for vector storage and provides an interactive
comparison interface for evaluating different search approaches.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np
import chromadb
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Local imports with fallback
try:
    from config import resolve_codebase_path, get_file_extensions
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        # Try importing from core module when running from project root
        from core.config import resolve_codebase_path, get_file_extensions
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False
        print("Warning: config.py not found, using default settings")

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: fuzzywuzzy not available. Fuzzy matching disabled.")

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
UNIXCODER_MODEL_NAME = 'microsoft/unixcoder-base'
MINILM_MODEL_NAME = 'all-MiniLM-L6-v2'

# Initialize ChromaDB and Models
print(f"Using device: {DEVICE}")
# Use configuration system for ChromaDB path
try:
    from .config import get_chroma_db_path
    chroma_db_path = str(get_chroma_db_path())
except ImportError:
    # Fallback for standalone execution
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_db_path = os.path.join(project_root, "data", "chroma_db")

client = chromadb.PersistentClient(path=chroma_db_path)

# IMPORTANT: ChromaDB uses SQUARED L2 distance by default (not cosine!)
# Our collections were created without specifying hnsw:space, so they use Squared L2.
# For normalized vectors: squared_L2 = 2 - 2*dot_product
# Therefore: similarity = 1 - distance/2 (see docs/CHROMADB_DISTANCE_NOTES.md)

# Load models once at startup
tokenizer_unixcoder = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)
model_unixcoder = AutoModel.from_pretrained(UNIXCODER_MODEL_NAME).to(DEVICE)
model_unixcoder.eval()

model_minilm = SentenceTransformer(MINILM_MODEL_NAME, device=DEVICE)
print("Models loaded successfully!")

# Fallback function for when config is not available
if not CONFIG_AVAILABLE:
    def resolve_codebase_path():
        """Fallback codebase path resolution when config is not available"""
        import yaml
        import os
        try:
            # Try to find project root
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            
            config_path = os.path.join(project_root, "config.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            codebase_path = config['codebase']['path']
            if not os.path.isabs(codebase_path):
                codebase_path = os.path.join(project_root, codebase_path)
            return Path(codebase_path)
        except Exception as e:
            print(f"Warning: Could not load codebase path from config: {e}")
            # Default fallback to the known test codebase
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            return Path(project_root) / "data" / "codebases" / "manuel_natcom" / "src" / "sim"

@dataclass
class KeywordMatch:
    """Represents a keyword search match with scoring information."""
    file_path: Path
    content: str
    start_line: int
    end_line: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    score: float = 0.0
    match_type: str = "content"
    matched_keywords: List[str] = None
    context: str = ""
    
    def __post_init__(self):
        if self.matched_keywords is None:
            self.matched_keywords = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility with other search methods."""
        return {
            "file_path": str(self.file_path),
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "score": self.score,
            "match_type": self.match_type,
            "matched_keywords": self.matched_keywords,
            "context": self.context,
            "similarity_score": self.score
        }

class KeywordSearchEngine:
    """Keyword-based search engine for C++ code files with fuzzy matching support."""
    
    # File extensions and common stop words
    CPP_EXTENSIONS = {'.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx', '.c'}
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those'
    }

    def __init__(self, verbose: bool = False, fuzzy_threshold: int = 80, context_lines: int = 2):
        """Initialize the keyword search engine with configuration options."""
        self.verbose = verbose
        self.fuzzy_threshold = fuzzy_threshold
        self.context_lines = context_lines
        self.file_cache = {}
        
        # Regex patterns for C++ parsing
        self.function_pattern = re.compile(
            r'^\s*(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?'
            r'(?:const\s+)?[\w:*&<>\s]+\s+'
            r'(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?[{;]',
            re.MULTILINE
        )
        
        self.class_pattern = re.compile(
            r'^\s*(?:template\s*<[^>]*>\s*)?'
            r'(?:class|struct)\s+(\w+)',
            re.MULTILINE
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query, filtering out stop words."""
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words 
                   if len(word) > 2 and word not in self.STOP_WORDS]
        if len(keywords) > 1:
            keywords.append(query.lower().strip())
        return keywords

    def _parse_file(self, file_path: Path) -> Dict:
        """Parse C++ file to extract functions and classes with caching."""
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            functions = []
            for match in self.function_pattern.finditer(content):
                func_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                end_line = min(start_line + 20, len(lines))
                
                functions.append({
                    'name': func_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'content': '\n'.join(lines[start_line-1:end_line])
                })
            
            classes = []
            for match in self.class_pattern.finditer(content):
                class_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                classes.append({
                    'name': class_name,
                    'start_line': start_line
                })
            
            file_info = {
                'content': content,
                'lines': lines,
                'functions': functions,
                'classes': classes
            }
            
            self.file_cache[file_path] = file_info
            return file_info
            
        except Exception as e:
            if self.verbose:
                print(f"Error parsing {file_path}: {e}")
            return {'content': '', 'lines': [], 'functions': [], 'classes': []}

    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """Calculate match score for text against keywords with fuzzy matching."""
        text_lower = text.lower()
        matched_keywords = []
        total_score = 0.0
        
        for keyword in keywords:
            # Exact match scoring
            exact_matches = text_lower.count(keyword)
            if exact_matches > 0:
                matched_keywords.append(keyword)
                total_score += exact_matches * 10
                continue
            
            # Fuzzy match scoring (if available)
            if FUZZY_AVAILABLE and len(keyword) > 3:
                words_in_text = re.findall(r'\b\w+\b', text_lower)
                best_fuzzy_score = 0
                
                for word in words_in_text:
                    if len(word) > 2:
                        fuzzy_score = fuzz.ratio(keyword, word)
                        if fuzzy_score >= self.fuzzy_threshold:
                            matched_keywords.append(f"{keyword}~{word}")
                            best_fuzzy_score = max(best_fuzzy_score, fuzzy_score)
                
                if best_fuzzy_score > 0:
                    total_score += (best_fuzzy_score / 100.0) * 5
        
        return total_score, matched_keywords

    def search_directory(self, directory: Path, query: str, max_results: int = 20) -> List[KeywordMatch]:
        """Search directory for keyword matches in C++ files."""
        keywords = self._extract_keywords(query)
        all_matches = []
        
        for root, _, files in os.walk(directory):
            for filename in files:
                if Path(filename).suffix.lower() in self.CPP_EXTENSIONS:
                    file_path = Path(root) / filename
                    try:
                        file_info = self._parse_file(file_path)
                        
                        # Search functions
                        for func in file_info['functions']:
                            name_score, name_matches = self._calculate_keyword_score(func['name'], keywords)
                            content_score, content_matches = self._calculate_keyword_score(func['content'], keywords)
                            
                            if name_score > 0 or content_score > 0:
                                all_matches.append(KeywordMatch(
                                    file_path=file_path,
                                    content=func['content'][:500] + "..." if len(func['content']) > 500 else func['content'],
                                    start_line=func['start_line'],
                                    end_line=func['end_line'],
                                    function_name=func['name'],
                                    score=name_score * 2 + content_score,
                                    match_type="function_name" if name_score > content_score else "content",
                                    matched_keywords=name_matches + content_matches
                                ))
                        
                        # Search classes
                        for cls in file_info['classes']:
                            name_score, name_matches = self._calculate_keyword_score(cls['name'], keywords)
                            if name_score > 0:
                                start_line = cls['start_line']
                                end_line = min(start_line + 20, len(file_info['lines']))
                                class_content = '\n'.join(file_info['lines'][start_line-1:end_line])
                                
                                all_matches.append(KeywordMatch(
                                    file_path=file_path,
                                    content=class_content,
                                    start_line=start_line,
                                    end_line=end_line,
                                    class_name=cls['name'],
                                    score=name_score * 1.5,
                                    match_type="class_name",
                                    matched_keywords=name_matches
                                ))
                                
                    except Exception as e:
                        if self.verbose:
                            print(f"Error searching {file_path}: {e}")
        
        all_matches.sort(key=lambda x: x.score, reverse=True)
        return all_matches[:max_results]


# Embedding and Search Functions
def preprocess_code_for_unixcoder(code_text: str) -> str:
    """Preprocess code to focus on distinctive patterns for better UniXcoder embeddings."""
    import re
    
    # Remove common boilerplate patterns
    processed = code_text
    
    # 1. Remove copyright headers
    processed = re.sub(r'//.*?Copyright.*?EMBL\..*?\n', '', processed, flags=re.MULTILINE)
    
    # 2. Remove common include patterns (but keep unique ones)
    common_includes = ['#include "dim.h"', '#include "sim.h"', '#include "glossary.h"', 
                      '#include "exceptions.h"', '#include "iowrapper.h"']
    for include in common_includes:
        processed = processed.replace(include, '')
    
    # 3. Remove excessive whitespace and empty lines
    processed = re.sub(r'\n\s*\n\s*\n', '\n\n', processed)
    processed = re.sub(r'^\s*\n', '', processed, flags=re.MULTILINE)
    
    # 4. Focus on function bodies and class definitions
    lines = processed.split('\n')
    important_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Keep lines with actual code logic
        if (stripped and not stripped.startswith('//') and 
            ('{' in stripped or '}' in stripped or '=' in stripped or 
             'if' in stripped or 'for' in stripped or 'while' in stripped or
             'return' in stripped or 'class' in stripped or 'struct' in stripped or
             ('(' in stripped and ')' in stripped and len(stripped) > 10))):
            important_lines.append(line)
    
    # If we filtered too much, keep original
    if len(important_lines) < 3:
        return code_text
    
    return '\n'.join(important_lines)

def get_hf_embedding(query_text: str, model, tokenizer) -> np.ndarray:
    """Generate normalized embeddings using HuggingFace model with Mean pooling for better semantic understanding."""
    # Preprocess code to focus on distinctive patterns
    processed_text = preprocess_code_for_unixcoder(query_text)
    
    encoded_input = tokenizer([processed_text], padding=True, truncation=True, 
                             max_length=512, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling (average all token embeddings, weighted by attention mask)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    
    # Expand attention mask to match embedding dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum embeddings weighted by mask, then divide by sum of mask
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
    
    # L2 normalize
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norm, 1e-8)


def get_sbert_embedding(query_text: str, model) -> np.ndarray:
    """Generate normalized embeddings using Sentence-BERT model."""
    embeddings = model.encode([query_text], convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def get_query_embedding(query_text: str, model_type: str = 'unixcoder') -> np.ndarray:
    """Get query embedding based on model type."""
    if model_type.lower() == 'unixcoder':
        return get_hf_embedding(query_text, model_unixcoder, tokenizer_unixcoder)
    elif model_type.lower() in ['minilm', 'sentence_bert', 'sbert']:
        return get_sbert_embedding(query_text, model_minilm)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# Initialize search engine
keyword_engine = KeywordSearchEngine(verbose=False)

def search_code(query_text: str, model_type: str = 'unixcoder', k: int = 5) -> List[Dict]:
    """Search code using vector embeddings (UniXcoder or SBERT)."""
    try:
        collection_name = ("unixcoder_snippets" if model_type.lower() == 'unixcoder' 
                          else "sbert_snippets")
        
        collection = client.get_collection(name=collection_name)
        query_vector = get_query_embedding(query_text, model_type)
        
        # Ensure query_vector is flattened for ChromaDB
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()
        
        results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_chunks = []
        if results['documents'] and results['documents'][0]:
            ids = results.get('ids', [[]])[0]
            documents = results.get('documents', [[]])[0] 
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            for i, doc in enumerate(documents):
                raw_distance = distances[i]
                
                # Use consistent similarity calculation for both models
                # Convert ChromaDB squared L2 distance to similarity score (higher is better)
                # For normalized vectors: squared_L2 = 2 - 2*dot_product, so dot_product = 1 - squared_L2/2
                similarity = max(0.0, 1 - raw_distance / 2)
                
                retrieved_chunks.append({
                    'id': ids[i],
                    'content': doc,
                    'metadata': metadatas[i],
                    'distance': distances[i],
                    'similarity_score': similarity,
                    'raw_distance': raw_distance,
                    'model_type': model_type,
                    'query': query_text
                })
        
        return retrieved_chunks
        
    except Exception as e:
        print(f"Search error for {model_type}: {e}")
        return []


def search_keyword(query_text: str, k: int = 5) -> List[Dict]:
    """Search using keyword matching."""
    try:
        directory = resolve_codebase_path()
        
        if not directory.exists():
            print(f"Warning: Directory {directory} not found for keyword search")
            return []
        
        matches = keyword_engine.search_directory(directory, query_text, max_results=k)
        
        results = []
        for match in matches:
            normalized_score = min(1.0, match.score / 50.0)
            
            result = {
                'id': f"kw_{match.start_line}_{match.file_path.name}",
                'content': match.content,
                'metadata': {
                    'file_path': str(match.file_path),
                    'start_line': match.start_line,
                    'end_line': match.end_line,
                    'function_name': match.function_name,
                    'class_name': match.class_name,
                    'match_type': match.match_type,
                    'matched_keywords': match.matched_keywords
                },
                'distance': 1.0 - normalized_score,
                'similarity_score': normalized_score,
                'raw_distance': 1.0 - normalized_score,
                'model_type': 'keyword',
                'query': query_text
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Keyword search error: {e}")
        return []


# Display and Utility Functions
def get_model_display_name(model_type: str) -> str:
    """Get meaningful display names that highlight each model's strengths."""
    model_names = {
        'keyword': 'EXACT MATCH (Literal Text)',
        'unixcoder': 'CODE STRUCTURE (Programming Patterns)', 
        'minilm': 'SEMANTIC (Conceptual Understanding)',
        'sbert': 'SEMANTIC (Conceptual Understanding)',
        'sentence-bert': 'SEMANTIC (Conceptual Understanding)'
    }
    return model_names.get(model_type.lower(), model_type.upper())

def print_results(results: List[Dict], query: str, model_type: str) -> None:
    """Print search results in a clean, formatted layout."""
    if not results:
        print(f"No results found for {get_model_display_name(model_type)}.")
        return
    
    print(f"\n{get_model_display_name(model_type)} Results for: '{query}'")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        score = result['similarity_score']
        
        file_path = metadata.get('file_path', 'Unknown')
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        
        # Only show score for semantic methods, not for exact match
        if model_type.lower() != 'keyword':
            print(f"\n{i}. Score: {score:.3f}")
        else:
            print(f"\n{i}.")
        print(f"   File: {file_name} (lines {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')})")
        print(f"   Path: {file_path}")
        
        if metadata.get('function_name') and metadata['function_name'] != 'None':
            print(f"   Function: {metadata['function_name']}")
        if metadata.get('class_name') and metadata['class_name'] != 'None':
            print(f"   Class: {metadata['class_name']}")
        
        # Additional info for keyword search
        if model_type.lower() == 'keyword':
            if metadata.get('match_type'):
                print(f"   Match Type: {metadata['match_type']}")
            if metadata.get('matched_keywords'):
                keywords = [kw.split('~')[0] for kw in metadata['matched_keywords']]
                print(f"   Matched: {', '.join(set(keywords))}")
        
        # Code preview
        content = result['content']
        lines = content.split('\n')
        preview_lines = []
        for line in lines[:5]:
            clean_line = line.strip()
            if clean_line and not clean_line.startswith('//'):
                preview_lines.append(clean_line)
                if len(preview_lines) >= 3:
                    break
        
        if preview_lines:
            print(f"   Code: {' | '.join(preview_lines)}")
        
        print("-" * 70)

def compare_models(query: str, k: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Compare search results from all three search methods."""
    print(f"\n{'='*80}")
    print(f"SEARCH COMPARISON: '{query}'")
    print(f"{'='*80}")
    
    # Execute all searches
    keyword_results = search_keyword(query, k=k)
    unixcoder_results = search_code(query, model_type='unixcoder', k=k)
    sbert_results = search_code(query, model_type='minilm', k=k)
    
    # Quick comparison table
    print(f"\nQUICK COMPARISON (Top 5 results):")
    print("-" * 110)
    print(f"{'RANK':<4} {'EXACT MATCH':<35} {'CODE STRUCTURE':<35} {'SEMANTIC':<35}")
    print("-" * 110)
    
    for i in range(5):
        keyword_info = ""
        unixcoder_info = ""
        sbert_info = ""
        
        if i < len(keyword_results):
            kw_meta = keyword_results[i]['metadata']
            kw_name = kw_meta.get('function_name') or kw_meta.get('class_name') or 'content'
            kw_file = kw_meta.get('file_path', '').split('/')[-1] if kw_meta.get('file_path') else ''
            keyword_info = f"{kw_file[:15]} - {kw_name[:15]}"
        
        if i < len(unixcoder_results):
            ux_meta = unixcoder_results[i]['metadata']
            ux_score = unixcoder_results[i]['similarity_score']
            ux_name = ux_meta.get('function_name', 'Unknown')
            ux_file = ux_meta.get('file_path', '').split('/')[-1] if ux_meta.get('file_path') else ''
            unixcoder_info = f"{ux_file[:15]} - {ux_name[:15]} ({ux_score:.3f})"
        
        if i < len(sbert_results):
            sb_meta = sbert_results[i]['metadata']
            sb_score = sbert_results[i]['similarity_score']
            sb_name = sb_meta.get('function_name', 'Unknown')
            sb_file = sb_meta.get('file_path', '').split('/')[-1] if sb_meta.get('file_path') else ''
            sbert_info = f"{sb_file[:15]} - {sb_name[:15]} ({sb_score:.3f})"
        
        print(f"{i+1:<4} {keyword_info:<35} {unixcoder_info:<35} {sbert_info:<35}")
    
    # Top result summary
    print(f"\nTOP RESULT FROM EACH METHOD:")
    print("-" * 80)
    
    if keyword_results:
        kw_best = keyword_results[0]
        kw_meta = kw_best['metadata']
        kw_file = kw_meta.get('file_path', 'Unknown').split('/')[-1]
        print(f"EXACT MATCH: {kw_meta.get('function_name', 'Unknown')} in {kw_file}")
    
    if unixcoder_results:
        ux_best = unixcoder_results[0]
        ux_meta = ux_best['metadata']
        ux_file = ux_meta.get('file_path', 'Unknown').split('/')[-1]
        print(f"CODE STRUCTURE: {ux_meta.get('function_name', 'Unknown')} (score: {ux_best['similarity_score']:.3f}) in {ux_file}")
    
    if sbert_results:
        sb_best = sbert_results[0]
        sb_meta = sb_best['metadata']
        sb_file = sb_meta.get('file_path', 'Unknown').split('/')[-1]
        print(f"SEMANTIC: {sb_meta.get('function_name', 'Unknown')} (score: {sb_best['similarity_score']:.3f}) in {sb_file}")
    
    # Detailed results (top 2 from each method)
    if keyword_results:
        print_results(keyword_results[:2], query, 'keyword')
    if unixcoder_results:
        print_results(unixcoder_results[:2], query, 'unixcoder')
    if sbert_results:
        print_results(sbert_results[:2], query, 'sentence-bert')
    
    return keyword_results, unixcoder_results, sbert_results


def search_all_methods(query: str, k: int = 5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Search using all three methods and return individual results."""
    keyword_results = search_keyword(query, k=k)
    unixcoder_results = search_code(query, model_type='unixcoder', k=k)
    sbert_results = search_code(query, model_type='minilm', k=k)
    
    return keyword_results, unixcoder_results, sbert_results


def main():
    """Main interactive search interface."""
    try:
        # Verify collections exist
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        print(f"Available collections: {collection_names}")
        
        required_collections = ["unixcoder_snippets", "sbert_snippets"]
        missing_collections = [col for col in required_collections if col not in collection_names]
        
        if missing_collections:
            print(f"Missing collections: {missing_collections}")
            return
        
        for col_name in required_collections:
            collection = client.get_collection(col_name)
            print(f"Collection '{col_name}' loaded with {collection.count()} chunks")
            
        print("\n\nInteractive Code Search System")
        print("=" * 50)
        print("Search Methods:")
        print("  - EXACT MATCH: Literal text and keyword matching")
        print("  - CODE STRUCTURE: Programming patterns and syntax")
        print("  - SEMANTIC: Conceptual understanding and meaning")
        print("\nCommands:")
        print("  - Type your search query")
        print("  - 'exit' or 'quit' - End session")
        
        while True:
            print("\n" + "="*80)
            query = input("Enter search query: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting search. Goodbye!")
                break
            
            if not query.strip():
                print("Query cannot be empty.")
                continue

            compare_models(query, k=5)

    except Exception as e:
        print(f"A critical error occurred: {e}")


if __name__ == "__main__":
    main()
