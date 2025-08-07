# CodeFinder User Guide

## 🚀 Quick Start

### 1. Setup Your Codebase
```bash
# Configure your codebase path in core/config.py
# Update CODEBASE_PATH to point to your source code directory
```

### 2. Run Initial Setup
```bash
# Complete system setup (parses code, generates embeddings, creates vector DB)
python scripts/setup_codefinder_final.py

# This will:
# - Parse your C++ codebase using Tree-sitter
# - Generate UniXcoder and SBERT embeddings
# - Create ChromaDB vector database
# - Run quality validation tests
```

### 3. Launch CodeFinder
```bash
# Primary method (recommended)
python run_codefinder.py

# Alternative: Direct execution
python app/views.py
```

### 4. Access the Interface
- Open your browser to: **http://localhost:8081**
- The interface provides three search methods:
  - **Exact Match**: Direct keyword/text search
  - **Code Structure**: UniXcoder AI semantic search
  - **Semantic Search**: SBERT natural language search

## 🔍 Search Methods Explained

### Exact Match (Keyword Search)
- **Best for**: Finding specific function names, variable names, or code patterns
- **Example queries**: `"filament"`, `"vector<int>"`, `"class MyClass"`
- **How it works**: Direct text matching with fuzzy search capabilities

### Code Structure (UniXcoder)
- **Best for**: Finding similar code patterns and programming structures
- **Example queries**: `"loop through array"`, `"initialize variables"`, `"error handling"`
- **How it works**: Microsoft's UniXcoder model understands code semantics

### Semantic Search (SBERT)
- **Best for**: Natural language descriptions of what you're looking for
- **Example queries**: `"file reading function"`, `"data structure for storing user info"`
- **How it works**: Sentence transformers for natural language understanding

## 📁 File Structure

```
llm_code_search/
├── 🚀 run_codefinder.py                    # Main launcher
├── 🌐 app/views.py                         # Primary web application
├── 🧠 core/                               # Core search engine
│   ├── search.py                          # Multi-method search
│   ├── config.py                         # Configuration (EDIT THIS)
│   ├── ingester.py                       # Data processing
│   └── code_parser_clean.py              # Code parsing
├── 💾 data/                               # Generated data
│   ├── chroma_db/                        # Vector database
│   └── codebases/                        # Your source code
├── ⚙️ config/                             # Model configurations
└── 🔧 scripts/setup_codefinder_final.py   # Setup script
```

## ⚙️ Configuration

Edit `core/config.py` to customize:

```python
# 1. Set your codebase path
CODEBASE_PATH = "/path/to/your/codebase"

# 2. Choose file extensions to search
CPP_EXTENSIONS = {'.cpp', '.h', '.hpp', '.cc', '.cxx'}

# 3. Enable/disable search methods
USE_KEYWORD_SEARCH = True
USE_UNIXCODER = True
USE_SBERT = True
```

## 🔧 Troubleshooting

### Setup Issues
```bash
# If Tree-sitter fails, use the working setup
python bin/setup_working.py

# Check dependencies
pip install flask transformers sentence-transformers chromadb torch numpy tree-sitter tree-sitter-cpp
```

### Search Issues
```bash
# Test search functionality
python -c "
from core.search import search_keyword
results = search_keyword('filament', k=3)
print(f'Found {len(results)} results')
"
```

### Path Issues
- Ensure you're running from the project root directory
- Check that `core/config.py` has the correct `CODEBASE_PATH`
- Verify ChromaDB was created: `ls -la data/chroma_db/`

## 🎯 Best Practices

1. **Configure before setup**: Edit `core/config.py` first
2. **Run setup once**: Only run setup when you change codebases
3. **Use appropriate search**: Different methods for different query types
4. **File viewer**: Click on results to view full file with syntax highlighting

## 📚 Advanced Features

- **File Viewer**: Click search results to open files with line highlighting
- **Desktop Optimized**: Larger results, better display for desktop users
- **Batch Processing**: Handles large codebases efficiently
- **Multiple Models**: Compare results across different AI models
- **Metadata Rich**: Function names, classes, namespaces automatically extracted

## 🆘 Support

- Check `CLEANUP_ANALYSIS.md` for known issues
- Review `README_CODEFINDER.md` for technical details
- Test individual components in `core/` directory