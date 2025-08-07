"""
Configuration Manager for CodeFinder

Unified configuration system that loads from YAML files with hardcoded fallbacks.
Handles absolute path resolution to avoid fragile relative path issues.
Edit config.yaml in the project root to customize settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Set


# Global configuration cache
_config_cache = None
_project_root_cache = None


def get_project_root() -> Path:
    """Get the project root directory with caching for performance"""
    global _project_root_cache
    
    if _project_root_cache is not None:
        return _project_root_cache
    
    # Start from current file location
    current_path = Path(__file__).resolve()
    
    # Look for project markers (config.yaml, main.py, requirements.txt, .git)
    project_markers = ['config.yaml', 'main.py', 'requirements.txt', '.git', 'README.md']
    
    # Search up the directory tree
    for parent in [current_path.parent.parent] + list(current_path.parent.parent.parents):
        if any((parent / marker).exists() for marker in project_markers):
            _project_root_cache = parent
            return parent
    
    # Fallback to parent of core directory
    _project_root_cache = current_path.parent.parent
    return _project_root_cache


def resolve_path(path_str: str, relative_to: Path = None) -> Path:
    """Resolve a path string to an absolute Path object
    
    Args:
        path_str: Path string (can be relative or absolute)
        relative_to: Base directory for relative paths (defaults to project root)
    
    Returns:
        Absolute Path object
    """
    if relative_to is None:
        relative_to = get_project_root()
    
    path = Path(path_str)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path.resolve()
    
    # Resolve relative to base directory
    return (relative_to / path).resolve()


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file or environment"""
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    # Try to find config file using project root
    project_root = get_project_root()
    config_paths = [
        os.environ.get('CODEFINDER_CONFIG'),
        str(project_root / 'config.yaml'),
        str(Path.cwd() / 'config.yaml')
    ]
    
    config = None
    for config_path in config_paths:
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                break
            except Exception as e:
                print(f"Warning: Error loading config from {config_path}: {e}")
                continue
    
    # Fallback to default configuration
    if config is None:
        print("Using default hardcoded configuration")
        config = get_default_config()
    
    # Resolve relative paths
    config = resolve_config_paths(config)
    
    _config_cache = config
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default hardcoded configuration when YAML file is not found"""
    project_root = get_project_root()
    
    return {
        'codebase': {
            'path': 'data/codebases/manuel_natcom/src/sim',  # Relative to project root
            'extensions': ['.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx', '.c']
        },
        'search': {
            'max_results_default': 5,
            'fuzzy_threshold': 80,
            'min_chunk_lines': 3,
            'methods': {
                'keyword_search': True,
                'unixcoder': True,
                'sbert': True
            }
        },
        'models': {
            'unixcoder': {
                'name': 'microsoft/unixcoder-base',
                'type': 'huggingface_automodel',
                'device': 'auto',
                'pooling_method': 'mean'
            },
            'sbert': {
                'name': 'all-MiniLM-L6-v2',
                'type': 'sentence_transformer',
                'device': 'auto'
            }
        },
        'data': {
            'chroma_db': 'data/chroma_db',      # Relative to project root
            'embeddings': 'data/embeddings',    # Relative to project root
            'processed': 'data/processed'       # Relative to project root
        },
        'app': {
            'host': '0.0.0.0',
            'port': 8081,
            'debug': True,
            'max_content_length': 33554432,
            'max_code_display_lines': 50,
            'max_code_height': 600
        }
    }


def resolve_config_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve relative paths in configuration to absolute paths"""
    project_root = get_project_root()
    
    # Resolve codebase path
    config['codebase']['path'] = str(resolve_path(config['codebase']['path'], project_root))
    
    # Resolve data paths
    for key in config['data']:
        config['data'][key] = str(resolve_path(config['data'][key], project_root))
    
    return config


# Backward compatibility functions for existing code
def resolve_codebase_path() -> Path:
    """Get the codebase path from configuration"""
    config = load_config()
    return Path(config['codebase']['path'])


def get_file_extensions() -> Set[str]:
    """Get file extensions from configuration"""
    config = load_config()
    return set(config['codebase']['extensions'])


def get_data_path(data_type: str) -> Path:
    """Get absolute path for data directory
    
    Args:
        data_type: Type of data ('chroma_db', 'embeddings', 'processed')
    
    Returns:
        Absolute Path object for the data directory
    """
    config = load_config()
    if data_type not in config['data']:
        raise ValueError(f"Unknown data type '{data_type}'. Available: {list(config['data'].keys())}")
    
    return Path(config['data'][data_type])


def get_chroma_db_path() -> Path:
    """Get absolute path to ChromaDB directory"""
    return get_data_path('chroma_db')


def get_embeddings_path() -> Path:
    """Get absolute path to embeddings directory"""
    return get_data_path('embeddings')


def get_processed_data_path() -> Path:
    """Get absolute path to processed data directory"""
    return get_data_path('processed')


def ensure_data_directories() -> None:
    """Ensure all data directories exist"""
    config = load_config()
    for data_type in config['data']:
        data_path = get_data_path(data_type)
        data_path.mkdir(parents=True, exist_ok=True)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    config = load_config()
    if model_name not in config['models']:
        raise ValueError(f"Model '{model_name}' not found in configuration. Available models: {list(config['models'].keys())}")
    
    model_config = config['models'][model_name].copy()
    
    # Resolve device setting
    if model_config.get('device') == 'auto':
        model_config['device'] = _resolve_device()
    
    return model_config


def get_all_model_configs() -> Dict[str, Dict[str, Any]]:
    """Get configurations for all models"""
    config = load_config()
    models = {}
    for model_name in config['models']:
        models[model_name] = get_model_config(model_name)
    return models


def _resolve_device() -> str:
    """Automatically resolve the best available device"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def get_app_config() -> Dict[str, Any]:
    """Get web application configuration"""
    config = load_config()
    return config.get('app', {
        'host': '0.0.0.0',
        'port': 8081,
        'debug': True,
        'max_content_length': 33554432,
        'max_code_display_lines': 50,
        'max_code_height': 600
    })


# Legacy constants for backward compatibility (now using absolute paths)
def _get_legacy_constants():
    """Get legacy constants from config for backward compatibility"""
    config = load_config()
    project_root = get_project_root()
    
    return {
        'CODEBASE_PATH': config['codebase']['path'],
        'CPP_EXTENSIONS': set(config['codebase']['extensions']),
        'MAX_RESULTS_DEFAULT': config['search']['max_results_default'],
        'FUZZY_THRESHOLD': config['search']['fuzzy_threshold'],
        'MIN_CHUNK_LINES': config['search']['min_chunk_lines'],
        'USE_KEYWORD_SEARCH': config['search']['methods']['keyword_search'],
        'USE_UNIXCODER': config['search']['methods']['unixcoder'],
        'USE_SBERT': config['search']['methods']['sbert'],
        # Use absolute paths for legacy compatibility
        'CHUNKS_OUTPUT_FILE': str(get_processed_data_path() / "code_chunks_clean.json"),
        'EMBEDDINGS_DIR': str(get_embeddings_path()),
        'VECTOR_DB_PATH': str(get_chroma_db_path())
    }


# Initialize legacy constants on import for backward compatibility
_legacy = _get_legacy_constants()
CODEBASE_PATH = _legacy['CODEBASE_PATH']
CPP_EXTENSIONS = _legacy['CPP_EXTENSIONS']
MAX_RESULTS_DEFAULT = _legacy['MAX_RESULTS_DEFAULT']
FUZZY_THRESHOLD = _legacy['FUZZY_THRESHOLD']
MIN_CHUNK_LINES = _legacy['MIN_CHUNK_LINES']
USE_KEYWORD_SEARCH = _legacy['USE_KEYWORD_SEARCH']
USE_UNIXCODER = _legacy['USE_UNIXCODER']
USE_SBERT = _legacy['USE_SBERT']

# Legacy paths - now properly resolved to absolute paths
CHUNKS_OUTPUT_FILE = _legacy['CHUNKS_OUTPUT_FILE']
EMBEDDINGS_DIR = _legacy['EMBEDDINGS_DIR']
VECTOR_DB_PATH = _legacy['VECTOR_DB_PATH']


def validate_config() -> bool:
    """Validate the configuration settings and path resolution."""
    try:
        config = load_config()
        project_root = get_project_root()
        errors = []
        warnings = []
        
        print(f"🔧 Validating configuration from project root: {project_root}")
        
        # Check codebase path
        codebase_path = Path(config['codebase']['path'])
        if not codebase_path.exists():
            errors.append(f"Codebase path does not exist: {codebase_path}")
        elif not codebase_path.is_dir():
            errors.append(f"Codebase path is not a directory: {codebase_path}")
        else:
            # Check if codebase has files with configured extensions
            extensions = set(config['codebase']['extensions'])
            found_files = []
            for ext in extensions:
                found_files.extend(list(codebase_path.rglob(f"*{ext}")))
            
            if not found_files:
                warnings.append(f"No files found with configured extensions {extensions} in {codebase_path}")
            else:
                print(f"  ✅ Found {len(found_files)} source files in codebase")
        
        # Check file extensions
        extensions = config['codebase']['extensions']
        if not extensions or not isinstance(extensions, list):
            errors.append("No file extensions configured")
        
        # Validate model configurations
        if 'models' in config:
            for model_name, model_config in config['models'].items():
                if 'name' not in model_config:
                    errors.append(f"Model '{model_name}' missing 'name' field")
                if 'type' not in model_config:
                    errors.append(f"Model '{model_name}' missing 'type' field")
                elif model_config['type'] not in ['sentence_transformer', 'huggingface_automodel']:
                    errors.append(f"Model '{model_name}' has invalid type: {model_config['type']}")
                if 'device' not in model_config:
                    errors.append(f"Model '{model_name}' missing 'device' field")
        else:
            errors.append("No models configuration found")
        
        # Check and create data directories
        try:
            for key in config['data']:
                data_path = Path(config['data'][key])
                
                # Verify path is absolute after resolution
                if not data_path.is_absolute():
                    warnings.append(f"Data path '{key}' is not absolute after resolution: {data_path}")
                
                # Create directory if it doesn't exist
                data_path.mkdir(parents=True, exist_ok=True)
                
                # Verify it's writable
                if not os.access(data_path, os.W_OK):
                    errors.append(f"Data directory '{key}' is not writable: {data_path}")
                else:
                    print(f"  ✅ Data directory '{key}': {data_path}")
        except Exception as e:
            errors.append(f"Error creating/validating data directories: {e}")
        
        # Check ChromaDB path accessibility
        try:
            chroma_path = get_chroma_db_path()
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Try importing chromadb to check if path works
            import chromadb
            test_client = chromadb.PersistentClient(path=str(chroma_path))
            collections = test_client.list_collections()
            print(f"  ✅ ChromaDB accessible: {len(collections)} collections found")
        except Exception as e:
            warnings.append(f"ChromaDB path validation failed: {e}")
        
        # Print warnings
        if warnings:
            print("⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        # Print errors and return result
        if errors:
            print("❌ Configuration errors found:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ Configuration validation passed!")
        print(f"  - Project Root: {project_root}")
        print(f"  - Codebase: {codebase_path}")
        print(f"  - Extensions: {set(extensions)}")
        search_methods = config['search']['methods']
        print(f"  - Search Methods: Keyword={search_methods['keyword_search']}, UniXcoder={search_methods['unixcoder']}, SBERT={search_methods['sbert']}")
        
        # Display model configurations
        print(f"  - Models:")
        for model_name, model_config in config['models'].items():
            device = model_config.get('device', 'auto')
            if device == 'auto':
                device = f"auto -> {_resolve_device()}"
            pooling = model_config.get('pooling_method', 'default')
            print(f"    * {model_name}: {model_config['name']} ({model_config['type']}) on {device}" + 
                  (f" with {pooling} pooling" if pooling != 'default' else ""))
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    validate_config()
