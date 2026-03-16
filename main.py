#!/usr/bin/env python3
"""
SearchYourCodes - Intelligent Code Search Platform
Main Entry Point

A unified launcher for the SearchYourCodes system that provides:
- Intelligent code search with multiple AI models
- Web-based interface for interactive searching
- Support for multiple programming languages
- Vector-based semantic search capabilities

Usage:
    python main.py                    # Start web application
    python main.py --setup          # Run initial setup
    python main.py --clean          # Clean setup (remove databases/processed files)
    python main.py --config         # Show configuration
    python main.py --test           # Run comprehensive tests (system + web app)
    python main.py --test-webapp    # Test web app functionality only
"""

import os
import sys
import argparse
import yaml
from pathlib import Path


def load_config():
    """Load configuration using the unified config system"""
    try:
        from core.config import load_config as core_load_config
        return core_load_config()
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


def show_config(config):
    """Display current configuration"""
    print("🔧 SearchYourCodes Configuration")
    print("=" * 50)
    print(f"📁 Codebase: {config['codebase']['path']}")
    print(f"📄 Extensions: {', '.join(config['codebase']['extensions'])}")
    print(f"🗄️ ChromaDB: {config['data']['chroma_db']}")
    print(f"🌐 Web App: http://localhost:{config['app']['port']}")
    print(f"🔍 Search Methods: {', '.join([k for k, v in config['search']['methods'].items() if v])}")
    
    # Display model configurations
    print(f"🤖 Models:")
    for model_name, model_config in config['models'].items():
        device = model_config.get('device', 'auto')
        print(f"  • {model_name}: {model_config['name']} ({model_config['type']}) on {device}")


def check_setup(config):
    """Check if setup is required"""
    chroma_db_path = Path(config['data']['chroma_db'])
    
    if not chroma_db_path.exists():
        print("⚠️  Setup required: ChromaDB not found")
        return False
    
    # Check if collections exist
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_db_path))
        collections = client.list_collections()
        
        required_collections = ['unixcoder_snippets', 'sbert_snippets']
        existing_collections = [c.name for c in collections]
        
        missing = [c for c in required_collections if c not in existing_collections]
        if missing:
            print(f"⚠️  Setup required: Missing collections: {missing}")
            return False
            
        print("✅ Setup complete - all collections found")
        return True
        
    except Exception as e:
        print(f"⚠️  Setup check failed: {e}")
        return False


def clean_setup(config):
    """Clean/reverse the setup to allow fresh re-setup"""
    print("🧹 Cleaning SearchYourCodes Setup...")
    
    import shutil
    from pathlib import Path
    
    try:
        items_to_clean = []
        
        # 1. ChromaDB directory
        chroma_db_path = Path(config['data']['chroma_db'])
        if chroma_db_path.exists():
            items_to_clean.append(('ChromaDB', chroma_db_path))
        
        # 2. Processed files
        processed_dir = Path(config['data']['processed'])
        if processed_dir.exists():
            processed_files = [
                processed_dir / "code_chunks.json",
                processed_dir / "model_state.json"
            ]
            for file_path in processed_files:
                if file_path.exists():
                    items_to_clean.append(('Processed file', file_path))
        
        # 3. Embeddings directory
        embeddings_dir = Path(config['data']['embeddings'])
        if embeddings_dir.exists() and any(embeddings_dir.iterdir()):
            items_to_clean.append(('Embeddings directory', embeddings_dir))
        
        if not items_to_clean:
            print("✅ Nothing to clean - setup already clean")
            return True
        
        # Show what will be cleaned
        print(f"\n📋 Items to be removed:")
        for item_type, item_path in items_to_clean:
            print(f"  • {item_type}: {item_path}")
        
        # Confirm with user
        confirm = input(f"\n⚠️  This will remove {len(items_to_clean)} items. Continue? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Cleanup cancelled by user")
            return False
        
        # Perform cleanup
        cleaned_count = 0
        for item_type, item_path in items_to_clean:
            try:
                if item_path.is_file():
                    item_path.unlink()
                    print(f"  ✅ Removed {item_type}: {item_path.name}")
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
                    print(f"  ✅ Removed {item_type}: {item_path.name}")
                cleaned_count += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {item_type} {item_path}: {e}")
        
        print(f"\n🎉 Cleanup complete! Removed {cleaned_count}/{len(items_to_clean)} items")
        print("💡 You can now run 'python main.py --setup' for a fresh setup")
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_setup(config):
    """Run the setup process"""
    print("🔧 Running SearchYourCodes Setup...")
    
    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        # Import and run setup
        from setup_SearchYourCodes_final import SearchYourCodesSetup
        
        setup = SearchYourCodesSetup(verbose=True)
        success = setup.run_complete_setup()
        
        if success:
            print("✅ Setup completed successfully!")
            return True
        else:
            print("❌ Setup failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Setup module not found: {e}")
        print("   Make sure setup_SearchYourCodes_final.py exists in project root")
        return False


def run_reindex(config):
    """Run incremental re-indexing of changed files"""
    print("Incremental Re-indexing...")

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        import json
        import time
        from core.parsers.parser_factory import MultiLanguageParser

        codebase_path = Path(config['codebase']['path'])
        processed_dir = Path(config['data']['processed'])
        mtime_file = processed_dir / "file_mtimes.json"
        chunks_file = processed_dir / "code_chunks_clean.json"

        # Load previous mtimes
        old_mtimes = {}
        if mtime_file.exists():
            with open(mtime_file, 'r') as f:
                old_mtimes = json.load(f)

        # Scan current files
        extensions = set(config['codebase']['extensions'])
        current_files = {}
        for ext in extensions:
            for fp in codebase_path.rglob(f"*{ext}"):
                rel = str(fp.relative_to(codebase_path))
                current_files[rel] = fp.stat().st_mtime

        # Find changed/new/deleted files
        changed = []
        for rel, mtime in current_files.items():
            if rel not in old_mtimes or old_mtimes[rel] != mtime:
                changed.append(rel)

        deleted = [rel for rel in old_mtimes if rel not in current_files]

        if not changed and not deleted:
            print("No files changed since last index. Nothing to do.")
            return True

        print(f"  Changed/new files: {len(changed)}")
        print(f"  Deleted files: {len(deleted)}")

        # Parse changed files
        parser = MultiLanguageParser(verbose=True)
        new_chunks = []
        for rel in changed:
            fp = codebase_path / rel
            try:
                file_chunks = parser.parse_file(fp)
                for chunk in file_chunks:
                    new_chunks.append(chunk.to_dict())
            except Exception as e:
                print(f"  Error parsing {rel}: {e}")

        print(f"  Parsed {len(new_chunks)} chunks from changed files")

        # Load existing chunks, remove stale ones, add new ones
        existing_chunks = []
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                existing_chunks = json.load(f)

        # Remove chunks from changed or deleted files
        changed_or_deleted = set(changed) | set(deleted)
        filtered_chunks = [
            c for c in existing_chunks
            if not any(c.get('file_path', '').endswith(rel) for rel in changed_or_deleted)
        ]
        filtered_chunks.extend(new_chunks)

        # Save updated chunks
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_file, 'w') as f:
            json.dump(filtered_chunks, f, indent=2)

        # Save updated mtimes
        with open(mtime_file, 'w') as f:
            json.dump(current_files, f, indent=2)

        print(f"  Updated chunks file: {len(filtered_chunks)} total chunks")
        print("  Re-run --setup to regenerate embeddings and re-ingest into ChromaDB.")
        print("  Incremental re-indexing complete!")
        return True

    except Exception as e:
        print(f"Re-indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search(config):
    """Test search functionality using pytest and basic smoke tests"""
    print("Testing Search Functionality...")

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from core.search import compare_models

        # Smoke test: run a query through all search methods
        test_query = "motor"
        print(f"\nSmoke test with query: '{test_query}'")

        keyword_results, unixcoder_results, sbert_results = compare_models(test_query, k=2)

        print(f"  Keyword search: {len(keyword_results)} results")
        print(f"  UniXcoder search: {len(unixcoder_results)} results")
        print(f"  SBERT search: {len(sbert_results)} results")

        # Run pytest suite
        print("\nRunning pytest suite...")
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-v'],
            cwd=str(Path(__file__).parent)
        )

        if result.returncode == 0:
            print("\nAll tests passed!")
            return True
        else:
            print("\nSome tests failed.")
            return False

    except Exception as e:
        print(f"Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webapp_only(config):
    """Test web application functionality only"""
    print("Testing Web Application Functionality...")

    sys.path.insert(0, str(Path(__file__).parent))

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/test_app.py', '-v'],
            cwd=str(Path(__file__).parent)
        )

        if result.returncode == 0:
            print("\nWeb app tests passed!")
            return True
        else:
            print("\nSome web app tests failed.")
            return False

    except Exception as e:
        print(f"Web app test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def start_web_app(config):
    """Start the web application"""
    print("🚀 Starting SearchYourCodes Web Application...")
    
    # Add paths to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Set configuration in environment for the app to use
        os.environ['SearchYourCodes_CONFIG'] = str(project_root / "config.yaml")
        
        # Import and run the Flask app
        from app.main import app
        
        print(f"🌐 Access the interface at: http://localhost:{config['app']['port']}")
        print("💡 Press Ctrl+C to shutdown gracefully")
        
        app.run(
            debug=config['app']['debug'],
            host=config['app']['host'],
            port=config['app']['port']
        )
        
    except ImportError as e:
        print(f"❌ Web application not found: {e}")
        print("   Make sure app/main.py exists")
    except KeyboardInterrupt:
        print("\n🛑 SearchYourCodes stopped by user")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="SearchYourCodes - Intelligent Code Search Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start web application
  python main.py --setup          # Run initial setup  
  python main.py --clean          # Clean setup for fresh re-setup
  python main.py --config         # Show configuration
  python main.py --test           # Run comprehensive tests (system + web app)
  python main.py --test-webapp    # Test web app functionality only
        """
    )
    
    parser.add_argument('--setup', action='store_true',
                       help='Run initial setup (parse code, create embeddings)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean/reverse setup (remove databases and processed files)')
    parser.add_argument('--config', action='store_true', 
                       help='Show current configuration')
    parser.add_argument('--test', action='store_true',
                       help='Run comprehensive tests (includes system and web app tests)')
    parser.add_argument('--test-webapp', action='store_true',
                       help='Test web application functionality only')
    parser.add_argument('--force-setup', action='store_true',
                       help='Force setup even if already configured')
    parser.add_argument('--reindex', action='store_true',
                       help='Incremental re-index: only re-process changed files')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Handle different modes
    if args.config:
        show_config(config)
        
    elif args.clean:
        success = clean_setup(config)
        if not success:
            sys.exit(1)
            
    elif args.setup or args.force_setup:
        success = run_setup(config)
        if not success:
            sys.exit(1)
            
    elif args.reindex:
        success = run_reindex(config)
        if not success:
            sys.exit(1)

    elif args.test:
        success = test_search(config)
        if not success:
            sys.exit(1)
            
    elif args.test_webapp:
        success = test_webapp_only(config)
        if not success:
            sys.exit(1)
            
    else:
        # Default: start web application
        # Check if setup is needed
        if not check_setup(config):
            print("\n💡 Run 'python main.py --setup' first to initialize the system")
            sys.exit(1)
            
        start_web_app(config)


if __name__ == "__main__":
    main()
