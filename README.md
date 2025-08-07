# CodeFinder - Intelligent Code Search Platform

🔍 **AI-powered code search** with multiple search methods: exact matching, code structure analysis (UniXcoder), and semantic understanding (SBERT).

## 🚀 Quick Start for New Users

### Prerequisites
- Python 3.8+ 
- 4GB+ RAM (for AI models)
- macOS/Linux (Windows with WSL)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Target Codebase
**IMPORTANT**: Edit `config.yaml` to point to your source code directory:

```yaml
codebase:
  path: "data/codebases/manuel_natcom/src/sim"  # 👈 CHANGE THIS PATH
  extensions: [.cpp, .h, .py, .js]  # 👈 ADJUST FOR YOUR LANGUAGE
```

**Default Test Setup**: The project comes with a C++ codebase at:
`data/codebases/manuel_natcom/src/sim` - Use this for testing!

### Step 3: Run Complete Setup (Clean State)
```bash
python main.py --setup
```
This will:
- ✅ Parse your codebase (~2-10 minutes depending on size)
- ✅ Generate AI embeddings with UniXcoder & SBERT
- ✅ Create ChromaDB vector database
- ✅ Verify all search methods work

### Step 4: Launch Web Interface  
```bash
python main.py
```
Access at: **http://localhost:8081**

### Step 5: Test Search Quality
```bash
python main.py --test
```

## 🛠️ Testing Pipeline from Complete Clean State

**For New Users**: Want to test the complete pipeline from scratch?

### Complete Clean Start Instructions

**Step 1: Environment Cleanup (Optional)**
```bash
# Remove existing environment to simulate new user experience
rm -rf .venv __pycache__ data/chroma_db/*
```

**Step 2: Prerequisites Check**
```bash
python3 --version  # Should be 3.8+
pip3 --version     # Package manager available
```

**Step 3: Fresh Virtual Environment**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

**Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: Complete Setup**
```bash
# Check configuration first
python main.py --config

# Run complete setup (downloads models, generates embeddings)
python main.py --setup
```

**What setup does:**
- 🔍 Scans codebase for files
- 🧠 Downloads AI models (UniXcoder, SBERT) ~2GB first time
- 📊 Generates embeddings for code chunks
- 💾 Creates ChromaDB vector database
- ✅ Runs verification tests

**Step 6: Test & Launch**
```bash
# Test search functionality
python main.py --test

# Launch web interface
python main.py
# Access at: http://localhost:8081
```

### Quick Clean Test (One-liner)
```bash
rm -rf .venv __pycache__ data/chroma_db/* && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python main.py --setup
```

**Current Status**: Clean ChromaDB (no embeddings) ✅
**Target Codebase**: `data/codebases/manuel_natcom/src/sim` (C++ simulation code) ✅
**Expected Setup Time**: ~10-20 minutes (includes downloads)

### Success Criteria
- ✅ Setup completes without errors  
- ✅ All 3 search methods work (Exact, Code Structure, Semantic)
- ✅ Web interface loads and displays results
- ✅ Database contains code chunks from your codebase

### Troubleshooting
**Virtual environment issues:** Try `python3.9 -m venv .venv` or `python3.8 -m venv .venv`
**Dependency issues:** Run `pip install --upgrade pip` first
**Setup failures:** Check Python 3.8+, 4GB+ RAM, sufficient disk space (~3GB)

### Test Commands:
```bash
# 1. Check configuration
python main.py --config

# 2. Run complete setup from scratch  
python main.py --setup

# 3. Test functionality
python main.py --test

# 4. Start web interface
python main.py
```

## 🔍 Search Methods

| Method | Best For | Example Query |
|--------|----------|---------------|
| **Exact Match** | Function names, keywords | `"main"`, `"vector<int>"` |
| **Code Structure** | Similar code patterns | `"loop through array"` |
| **Semantic** | Natural language | `"file reading function"` |

## 📁 Project Structure

```
codefinder/
├── main.py              # 🚀 Main entry point
├── config.yaml          # ⚙️ Configuration
├── app/                 # 🌐 Web interface
├── core/                # 🧠 Search engine
├── data/                # 💾 Databases & embeddings
└── docs/                # 📚 Documentation
```

## 🛠️ Commands

```bash
python main.py              # Start web app
python main.py --setup      # Initial setup
python main.py --config     # Show config
python main.py --test       # Test search
```

## 📚 Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Detailed usage instructions
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Advanced setup options  
- **[API Reference](docs/API_REFERENCE.md)** - For developers

## 🎯 Features

- ✅ **Multiple AI Models**: UniXcoder + SBERT for comprehensive search
- ✅ **Vector Database**: ChromaDB for fast similarity search  
- ✅ **Multi-language**: C++, Python, JavaScript, Java support
- ✅ **Web Interface**: Modern, responsive design
- ✅ **File Viewer**: Syntax highlighting with line numbers
- ✅ **Desktop Optimized**: Large displays, more results

## ⚙️ Requirements

- Python 3.8+
- PyTorch (for AI models)
- ChromaDB (for vector storage)
- Flask (for web interface)

## 📝 License

MIT License - see LICENSE file for details.
