#!/usr/bin/env python3
"""
Test script to verify chunk ID handling fix.

This test validates that:
1. Chunk IDs are properly stored in metadata during ingestion
2. Chunk IDs are correctly retrieved during search
3. No fabricated fallback IDs (chunk_0, chunk_1, etc.) are generated
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.vector_search import VectorSearchEngine, search_code

def test_chunk_id_handling():
    """Test that chunk IDs are consistently handled in the existing database."""
    
    print("🧪 TESTING CHUNK ID HANDLING")
    print("=" * 50)
    
    try:
        # Load configuration
        from core.config import load_config
        config = load_config()
        
        # Use the search_code function instead of initializing a class
        print("✅ Using search_code function")
        
        # Test search and examine chunk IDs
        test_queries = [
            "class definition",
            "function implementation", 
            "diffusion coefficient"
        ]
        
        print("\n🔍 TESTING SEARCH RESULTS...")
        
        all_valid = True
        for query in test_queries:
            print(f"\n🎯 Query: '{query}'")
            
            # Search with UniXcoder using the search_code function
            results = search_code(query, "unixcoder", k=3)
            
            if not results:
                print("  ⚠️ No results found")
                continue
            
            for i, result in enumerate(results):
                chunk_id = result.get('chunk_id', 'NOT_FOUND')
                
                print(f"  Result {i+1}:")
                print(f"    Chunk ID: {chunk_id}")
                print(f"    Function: {result.get('function_name', 'N/A')}")
                print(f"    Score: {result.get('score', 0):.4f}")
                
                # Check if chunk ID looks like a fallback
                if chunk_id.startswith('chunk_') and chunk_id[6:].isdigit():
                    print(f"    ❌ FALLBACK ID DETECTED: {chunk_id}")
                    all_valid = False
                elif chunk_id == 'NOT_FOUND':
                    print(f"    ❌ NO CHUNK ID FOUND")
                    all_valid = False
                else:
                    print(f"    ✅ Valid chunk ID: {chunk_id}")
        
        if all_valid:
            print("\n✅ ALL CHUNK IDS VALID - Test passed!")
            return True
        else:
            print("\n❌ SOME CHUNK IDS INVALID - Test failed!")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def inspect_existing_data():
    """Inspect existing database to check current chunk ID status."""
    
    print("\n🔍 INSPECTING EXISTING DATABASE")
    print("=" * 50)
    
    try:
        from core.config import load_config
        import chromadb
        
        config = load_config()
        client = chromadb.PersistentClient(path=config['data']['chroma_db'])
        
        collections = client.list_collections()
        print(f"Found {len(collections)} collections")
        
        for collection in collections:
            print(f"\n📁 Collection: {collection.name}")
            count = collection.count()
            print(f"   Document count: {count}")
            
            if count > 0:
                # Get a few samples to examine
                sample = collection.get(limit=3, include=['metadatas'])
                
                print("   Sample results:")
                for i, metadata in enumerate(sample['metadatas']):
                    chunk_id = metadata.get('chunk_id', 'NOT_IN_METADATA')
                    print(f"     {i+1}. Metadata: {list(metadata.keys())}")
                    print(f"        Chunk ID in metadata: {chunk_id}")
                    
                    if chunk_id == 'NOT_IN_METADATA':
                        print(f"        ❌ No chunk_id in metadata")
                    elif chunk_id.startswith('chunk_') and chunk_id[6:].isdigit():
                        print(f"        ⚠️ Possible fallback ID")
                    else:
                        print(f"        ✅ Valid chunk ID")
                        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")

if __name__ == "__main__":
    print("CHUNK ID HANDLING TEST")
    print("=" * 60)
    
    # First inspect existing data
    inspect_existing_data()
    
    # Then run the fix test
    print("\n" + "=" * 60)
    success = test_chunk_id_handling()
    
    if success:
        print("\n🎉 CHUNK ID HANDLING FIX VERIFIED!")
    else:
        print("\n💥 CHUNK ID HANDLING STILL HAS ISSUES!")
        
    print("=" * 60)
