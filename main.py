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


def test_search(config):
    """Test search functionality"""
    print("🧪 Testing Search Functionality...")
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from core.search import compare_models, _run_comprehensive_tests, _test_webapp_functionality
        
        # Run comprehensive system tests
        print("\n" + "="*60)
        print("1. RUNNING COMPREHENSIVE SYSTEM TESTS")
        print("="*60)
        _run_comprehensive_tests()
        
        # Run web app tests
        print("\n" + "="*60)
        print("2. RUNNING WEB APPLICATION TESTS")
        print("="*60)
        _test_webapp_functionality()
        
        # Run legacy compatibility test
        print("\n" + "="*60)
        print("3. RUNNING LEGACY COMPATIBILITY TEST")
        print("="*60)
        
        # Test with a query that should exist in the biological simulation codebase
        test_query = "motor"  # Changed from "main" to a term that exists in this codebase
        print(f"Testing legacy compare_models with query: '{test_query}'")
        
        keyword_results, unixcoder_results, sbert_results = compare_models(test_query, k=2)
        
        print(f"✅ Keyword search: {len(keyword_results)} results")
        print(f"✅ UniXcoder search: {len(unixcoder_results)} results") 
        print(f"✅ SBERT search: {len(sbert_results)} results")
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webapp_only(config):
    """Test web application functionality only"""
    print("🌐 Testing Web Application Functionality...")
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from core.search import _test_webapp_functionality
        
        _test_webapp_functionality()
        
        print("\n🎉 Web app tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Web app test failed: {e}")
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
  python main.py --config         # Show configuration
  python main.py --test           # Run comprehensive tests (system + web app)
  python main.py --test-webapp    # Test web app functionality only
        """
    )
    
    parser.add_argument('--setup', action='store_true',
                       help='Run initial setup (parse code, create embeddings)')
    parser.add_argument('--config', action='store_true', 
                       help='Show current configuration')
    parser.add_argument('--test', action='store_true',
                       help='Run comprehensive tests (includes system and web app tests)')
    parser.add_argument('--test-webapp', action='store_true',
                       help='Test web application functionality only')
    parser.add_argument('--force-setup', action='store_true',
                       help='Force setup even if already configured')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Handle different modes
    if args.config:
        show_config(config)
        
    elif args.setup or args.force_setup:
        success = run_setup(config)
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
