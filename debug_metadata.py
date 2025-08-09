#!/usr/bin/env python3
"""
Debug script to examine what's actually in the database metadata
"""

import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_database_metadata():
    """Debug what's actually stored in the database metadata."""
    
    print("🔍 DEBUGGING DATABASE METADATA")
    print("=" * 60)
    
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
                # Get a sample to examine in detail
                sample = collection.get(limit=1, include=['ids', 'metadatas', 'documents'])
                
                if sample['ids']:
                    print(f"\n   Sample document:")
                    print(f"     ID: {sample['ids'][0]}")
                    print(f"     Content preview: {sample['documents'][0][:100]}...")
                    
                    # Print metadata in detail
                    metadata = sample['metadatas'][0]
                    print(f"     Metadata keys: {list(metadata.keys())}")
                    print(f"     Metadata content:")
                    for key, value in metadata.items():
                        print(f"       {key}: {value}")
                    
                    # Specifically check for chunk_id
                    if 'chunk_id' in metadata:
                        print(f"     ✅ chunk_id found: {metadata['chunk_id']}")
                    else:
                        print(f"     ❌ chunk_id NOT found in metadata")
                        
                        # Let's see if it might be under a different key
                        for key in metadata.keys():
                            if 'chunk' in key.lower() or 'id' in key.lower():
                                print(f"       Possible related key: {key} = {metadata[key]}")
                
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def check_chunks_data():
    """Check what's in the chunks data file."""
    
    print("\n🔍 DEBUGGING CHUNKS DATA FILE")
    print("=" * 60)
    
    try:
        from core.config import load_config
        from pathlib import Path
        
        config = load_config()
        chunks_file = Path(config['data']['processed']) / "code_chunks_clean.json"
        
        if chunks_file.exists():
            print(f"✅ Found chunks file: {chunks_file}")
            
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
            
            print(f"Chunks data structure: {type(chunks_data)}")
            if isinstance(chunks_data, list) and chunks_data:
                sample_chunk = chunks_data[0]
                print(f"Sample chunk keys: {list(sample_chunk.keys())}")
                print(f"Sample chunk:")
                for key, value in sample_chunk.items():
                    if key == 'content':
                        print(f"  {key}: {str(value)[:100]}...")
                    else:
                        print(f"  {key}: {value}")
                        
                # Check if chunk has an 'id' field
                if 'id' in sample_chunk:
                    print(f"✅ Chunk has 'id' field: {sample_chunk['id']}")
                else:
                    print(f"❌ Chunk missing 'id' field")
                    print(f"Available keys: {list(sample_chunk.keys())}")
            else:
                print(f"❌ Unexpected chunks data format")
        else:
            print(f"❌ Chunks file not found: {chunks_file}")
            
    except Exception as e:
        print(f"❌ Chunks debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("DATABASE METADATA DEBUG")
    print("=" * 60)
    
    debug_database_metadata()
    check_chunks_data()
