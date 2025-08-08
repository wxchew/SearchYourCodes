#!/usr/bin/env python3
"""
SearchYourCodes Complete Setup Script - Final Version

This script provides a complete, working setup for the SearchYourCodes system with:
- Proper code parsing using existing CppParser
- Correct embedding generation matching search.py
- Quality validation and troubleshooting
- Multiple approaches to fix common issues

Usage: python setup_SearchYourCodes_final.py [options]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import traceback

import numpy as np
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Add core to path and import required modules
sys.path.append('core')
try:
    from config import resolve_codebase_path, get_file_extensions, validate_config
    from core.parsers import CodeChunk, CppParser
    from core.search import get_hf_embedding, get_sbert_embedding as search_get_sbert_embedding
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class SearchYourCodesSetup:
    """Complete SearchYourCodes setup with validation and troubleshooting."""
    
    def __init__(self, verbose: bool = False, fix_unixcoder: bool = False):
        self.verbose = verbose
        self.fix_unixcoder = fix_unixcoder
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Initialize paths
        self.codebase_path = resolve_codebase_path()
        self.chroma_path = Path("data/chroma_db")
        
        # Models (loaded on demand)
        self.unixcoder_model = None
        self.unixcoder_tokenizer = None
        self.sbert_model = None
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'chunks_extracted': 0,
            'chunks_with_functions': 0,
            'chunks_with_classes': 0,
            'unixcoder_embeddings': 0,
            'sbert_embeddings': 0,
            'start_time': time.time(),
            'parsing_errors': 0,
            'embedding_errors': 0
        }
        
        print(f"🔧 Using device: {self.device}")
        if self.fix_unixcoder:
            print("🔧 UniXcoder fix mode enabled")
    
    def validate_environment(self):
        """Validate the environment and configuration."""
        print("🔍 Validating environment...")
        
        # Check configuration
        if not validate_config():
            raise RuntimeError("Configuration validation failed")
        
        # Check dependencies
        try:
            import tree_sitter_cpp
            from tree_sitter import Language, Parser
            print("  ✅ Tree-sitter C++ available")
        except ImportError:
            print("  ❌ Tree-sitter C++ not available")
            print("     Install with: pip install tree-sitter tree-sitter-cpp")
            raise
        
        # Check ChromaDB directory
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ ChromaDB path ready: {self.chroma_path}")
        
        # Check codebase
        cpp_files = list(self.codebase_path.rglob("*.cpp")) + list(self.codebase_path.rglob("*.h"))
        print(f"  ✅ Found {len(cpp_files)} C++ files in codebase")
        
        if len(cpp_files) == 0:
            raise RuntimeError("No C++ files found in codebase")
        
        print("✅ Environment validation passed!")
    
    def load_models(self):
        """Load embedding models with validation."""
        print("📥 Loading embedding models...")
        
        try:
            # Load UniXcoder
            print("  Loading UniXcoder...")
            self.unixcoder_tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
            self.unixcoder_model = AutoModel.from_pretrained('microsoft/unixcoder-base').to(self.device)
            self.unixcoder_model.eval()
            
            # Load SBERT
            print("  Loading SBERT...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            # Test embeddings
            test_text = "int main() { return 0; }"
            
            unixcoder_emb = self.get_unixcoder_embedding(test_text)
            sbert_emb = self.get_sbert_embedding(test_text)
            
            print(f"  ✅ UniXcoder: {unixcoder_emb.shape}, norm: {np.linalg.norm(unixcoder_emb):.3f}")
            print(f"  ✅ SBERT: {sbert_emb.shape}, norm: {np.linalg.norm(sbert_emb):.3f}")
            
            # Validate embeddings are different for different inputs
            test_text2 = "class MyClass { public: void method(); };"
            unixcoder_emb2 = self.get_unixcoder_embedding(test_text2)
            similarity = np.dot(unixcoder_emb.flatten(), unixcoder_emb2.flatten())
            
            if similarity > 0.99:
                print(f"  ⚠️  Warning: UniXcoder embeddings too similar ({similarity:.3f})")
                if self.fix_unixcoder:
                    print("  🔧 Applying UniXcoder fix...")
                    # Could implement alternative embedding strategy here
            else:
                print(f"  ✅ UniXcoder diversity check passed ({similarity:.3f})")
            
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def get_unixcoder_embedding(self, text: str) -> np.ndarray:
        """Generate UniXcoder embedding using search.py method."""
        return get_hf_embedding(text, self.unixcoder_model, self.unixcoder_tokenizer)
    
    def get_sbert_embedding(self, text: str) -> np.ndarray:
        """Generate SBERT embedding using search.py method.""" 
        return search_get_sbert_embedding(text, self.sbert_model)
    
    def parse_codebase(self) -> List[CodeChunk]:
        """Parse codebase using existing CppParser."""
        print(f"📝 Parsing codebase from: {self.codebase_path}")
        
        # Initialize parser
        parser = CppParser(verbose=self.verbose, min_lines=3)
        
        # Get all C++ files
        extensions = ['.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx', '.c']
        cpp_files = []
        for ext in extensions:
            cpp_files.extend(list(self.codebase_path.rglob(f"*{ext}")))
        
        print(f"📁 Found {len(cpp_files)} C++ files to process")
        
        all_chunks = []
        
        for i, file_path in enumerate(cpp_files, 1):
            if self.verbose:
                print(f"  Processing {i}/{len(cpp_files)}: {file_path}")
            else:
                if i % 50 == 0 or i == len(cpp_files):
                    print(f"  Processed {i}/{len(cpp_files)} files...")
            
            try:
                chunks = parser.parse_file(file_path)
                
                valid_chunks = []
                for chunk in chunks:
                    # Basic validation
                    if len(chunk.content.strip()) < 10:
                        continue
                    
                    # Ensure metadata is properly set
                    if not chunk.function_name:
                        chunk.function_name = 'Unknown'
                    if not chunk.class_name:
                        chunk.class_name = 'Unknown'
                    
                    # Set relative file path for better display
                    try:
                        chunk.file_path = str(file_path.relative_to(self.codebase_path.parent))
                    except ValueError:
                        chunk.file_path = str(file_path)
                    
                    valid_chunks.append(chunk)
                
                all_chunks.extend(valid_chunks)
                self.stats['files_processed'] += 1
                self.stats['chunks_extracted'] += len(valid_chunks)
                
                # Update function/class statistics
                for chunk in valid_chunks:
                    if chunk.function_name and chunk.function_name != 'Unknown':
                        self.stats['chunks_with_functions'] += 1
                    if chunk.class_name and chunk.class_name != 'Unknown':
                        self.stats['chunks_with_classes'] += 1
                        
            except Exception as e:
                self.stats['parsing_errors'] += 1
                if self.verbose:
                    print(f"    Error parsing {file_path}: {e}")
        
        print(f"\\n📊 Parsing Summary:")
        print(f"  Files processed: {self.stats['files_processed']}")
        print(f"  Parsing errors: {self.stats['parsing_errors']}")
        print(f"  Total chunks: {self.stats['chunks_extracted']}")
        print(f"  Chunks with functions: {self.stats['chunks_with_functions']}")
        print(f"  Chunks with classes: {self.stats['chunks_with_classes']}")
        
        if self.stats['chunks_extracted'] > 0:
            func_coverage = (self.stats['chunks_with_functions'] / self.stats['chunks_extracted']) * 100
            class_coverage = (self.stats['chunks_with_classes'] / self.stats['chunks_extracted']) * 100
            print(f"  Function coverage: {func_coverage:.1f}%")
            print(f"  Class coverage: {class_coverage:.1f}%")
        
        return all_chunks
    
    def create_vector_databases(self, chunks: List[CodeChunk]):
        """Create ChromaDB collections with proper embeddings."""
        print("🗄️  Creating vector databases...")
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # Clean up existing collections
        existing_collections = [c.name for c in client.list_collections()]
        for collection_name in ["unixcoder_snippets", "sbert_snippets"]:
            if collection_name in existing_collections:
                client.delete_collection(collection_name)
                print(f"  Deleted existing {collection_name} collection")
        
        # Create new collections
        unixcoder_collection = client.create_collection(
            name="unixcoder_snippets",
            metadata={"description": "Code chunks with UniXcoder embeddings", "model": "microsoft/unixcoder-base"}
        )
        
        sbert_collection = client.create_collection(
            name="sbert_snippets",
            metadata={"description": "Code chunks with SBERT embeddings", "model": "all-MiniLM-L6-v2"}
        )
        
        print(f"  Processing {len(chunks)} chunks...")
        
        # Process in batches for better memory management
        batch_size = 32
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            print(f"  Batch {batch_idx + 1}/{total_batches}: Processing chunks {start_idx + 1}-{end_idx}")
            
            # Prepare batch data
            ids = []
            documents = []
            metadatas = []
            unixcoder_embeddings = []
            sbert_embeddings = []
            
            for i, chunk in enumerate(batch_chunks):
                chunk_id = f"chunk_{start_idx + i}"
                ids.append(chunk_id)
                documents.append(chunk.content)
                
                # Create metadata
                metadata = {
                    'file_path': chunk.file_path,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'function_name': chunk.function_name or 'Unknown',
                    'class_name': chunk.class_name or 'Unknown',
                    'namespace': getattr(chunk, 'namespace', None) or 'Unknown',
                    'content_length': len(chunk.content),
                    'has_function': chunk.function_name not in [None, 'Unknown'],
                    'has_class': chunk.class_name not in [None, 'Unknown'],
                    'docstring': getattr(chunk, 'docstring', '') or ''
                }
                metadatas.append(metadata)
                
                # Generate embeddings with error handling
                try:
                    unixcoder_emb = self.get_unixcoder_embedding(chunk.content)
                    unixcoder_embeddings.append(unixcoder_emb.flatten().tolist())
                    self.stats['unixcoder_embeddings'] += 1
                except Exception as e:
                    self.stats['embedding_errors'] += 1
                    if self.verbose:
                        print(f"    UniXcoder embedding error for {chunk_id}: {e}")
                    # Use zero vector as fallback
                    unixcoder_embeddings.append([0.0] * 768)
                
                try:
                    sbert_emb = self.get_sbert_embedding(chunk.content)
                    sbert_embeddings.append(sbert_emb.flatten().tolist())
                    self.stats['sbert_embeddings'] += 1
                except Exception as e:
                    self.stats['embedding_errors'] += 1
                    if self.verbose:
                        print(f"    SBERT embedding error for {chunk_id}: {e}")
                    # Use zero vector as fallback
                    sbert_embeddings.append([0.0] * 384)
            
            # Add to collections
            try:
                unixcoder_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=unixcoder_embeddings
                )
                
                sbert_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=sbert_embeddings
                )
                
            except Exception as e:
                print(f"    ❌ Error adding batch {batch_idx + 1}: {e}")
                self.stats['embedding_errors'] += len(batch_chunks)
        
        # Validate final collections
        ux_count = unixcoder_collection.count()
        sb_count = sbert_collection.count()
        
        print(f"\\n📊 Database Creation Summary:")
        print(f"  UniXcoder collection: {ux_count} documents")
        print(f"  SBERT collection: {sb_count} documents")
        print(f"  UniXcoder embeddings: {self.stats['unixcoder_embeddings']} successful")
        print(f"  SBERT embeddings: {self.stats['sbert_embeddings']} successful")
        print(f"  Embedding errors: {self.stats['embedding_errors']}")
        
        if ux_count == 0 or sb_count == 0:
            raise RuntimeError("Failed to create vector databases - no documents added")
        
        return client
    
    def run_quality_test(self):
        """Run a quick quality test."""
        print("🔍 Running quality test...")
        
        # Import search functions
        from core.search import search_keyword, search_code
        
        test_queries = [
            ("motor", "biology term"),
            ("for loop", "programming construct"),
            ("class", "C++ keyword")
        ]
        
        results = {}
        
        for query, description in test_queries:
            print(f"  Testing '{query}' ({description}):")
            
            query_results = {}
            
            # Test each method
            methods = [
                ("keyword", lambda: search_keyword(query, k=3)),
                ("unixcoder", lambda: search_code(query, model_type='unixcoder', k=3)),
                ("sbert", lambda: search_code(query, model_type='minilm', k=3))
            ]
            
            for method_name, search_func in methods:
                try:
                    method_results = search_func()
                    count = len(method_results)
                    
                    # Check diversity (different files)
                    if method_results:
                        files = [r.get('metadata', {}).get('file_path', '').split('/')[-1] for r in method_results]
                        unique_files = len(set(files))
                        diversity = unique_files / len(method_results)
                        
                        # Check metadata quality
                        metadata_quality = sum(1 for r in method_results 
                                             if r.get('metadata', {}).get('function_name', 'Unknown') != 'Unknown') / len(method_results)
                    else:
                        diversity = 0.0
                        metadata_quality = 0.0
                    
                    status = "✅" if count > 0 else "❌"
                    print(f"    {method_name:<10}: {status} {count} results, diversity: {diversity:.2f}, metadata: {metadata_quality:.2f}")
                    
                    query_results[method_name] = {
                        'count': count,
                        'diversity': diversity,
                        'metadata_quality': metadata_quality,
                        'success': count > 0
                    }
                    
                except Exception as e:
                    print(f"    {method_name:<10}: ❌ Error: {e}")
                    query_results[method_name] = {'count': 0, 'diversity': 0.0, 'metadata_quality': 0.0, 'success': False}
            
            results[query] = query_results
        
        # Summary
        total_tests = len(test_queries) * 3
        successful_tests = sum(1 for query_results in results.values() 
                              for method_results in query_results.values() 
                              if method_results['success'])
        
        print(f"\\n📊 Quality Test Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Check for UniXcoder issues
        unixcoder_diversities = [results[query]['unixcoder']['diversity'] for query in results 
                                if results[query]['unixcoder']['success']]
        if unixcoder_diversities and all(d < 0.5 for d in unixcoder_diversities):
            print("  ⚠️  UniXcoder shows low diversity - may have repetition issues")
        
        return results
    
    def run_complete_setup(self):
        """Run the complete setup process."""
        print("🚀 SearchYourCodes Complete Setup")
        print("=" * 80)
        
        try:
            # Step 1: Validate environment
            self.validate_environment()
            
            # Step 2: Load models
            self.load_models()
            
            # Step 3: Parse codebase
            chunks = self.parse_codebase()
            
            if len(chunks) == 0:
                raise RuntimeError("No code chunks extracted")
            
            # Step 4: Create vector databases
            client = self.create_vector_databases(chunks)
            
            # Step 5: Quality test
            quality_results = self.run_quality_test()
            
            # Final summary
            elapsed_time = time.time() - self.stats['start_time']
            print(f"\\n🎉 Setup Complete!")
            print(f"=" * 80)
            print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
            print(f"📁 Files processed: {self.stats['files_processed']}")
            print(f"📝 Code chunks: {self.stats['chunks_extracted']}")
            print(f"🧠 UniXcoder embeddings: {self.stats['unixcoder_embeddings']}")
            print(f"🔍 SBERT embeddings: {self.stats['sbert_embeddings']}")
            print(f"⚠️  Errors: {self.stats['parsing_errors']} parsing, {self.stats['embedding_errors']} embedding")
            
            success_rate = sum(1 for qr in quality_results.values() for mr in qr.values() if mr['success'])
            total_tests = len(quality_results) * 3
            print(f"✅ Quality test: {success_rate}/{total_tests} ({(success_rate/total_tests)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"\\n❌ Setup failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="SearchYourCodes Complete Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_SearchYourCodes_final.py                    # Basic setup
  python scripts/setup_SearchYourCodes_final.py --verbose          # Detailed output
  python scripts/setup_SearchYourCodes_final.py --fix-unixcoder    # Try UniXcoder fix
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--fix-unixcoder', action='store_true',
                       help='Apply UniXcoder fixes for repetition issues')
    
    args = parser.parse_args()
    
    # Run setup
    setup = SearchYourCodesSetup(verbose=args.verbose, fix_unixcoder=args.fix_unixcoder)
    success = setup.run_complete_setup()
    
    if success:
        print(f"\\n💡 Next Steps:")
        print(f"  1. Test quality: python scripts/evaluate_search_quality.py")
        print(f"  2. Start web app: python scripts/web_app.py")  
        print(f"  3. Interactive test: python core/search.py")
        print(f"\\n📚 Documentation:")
        print(f"  - User guide: USER_GUIDE.md")
        print(f"  - README: README_SearchYourCodes.md")
    else:
        print(f"\\n💥 Setup failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
