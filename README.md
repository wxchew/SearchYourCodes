# SearchYourCodes - Intelligent Code Search Platform

🔍 **AI-powered code search** with multiple search methods: exact matching, code structure analysis (UniXcoder), and semantic understanding (SBERT).

## 🚀 Quick Start for New Users

### Prerequisites
- Python 3.8+ 
- 4GB+ RAM (for AI models)
- macOS/Linux (Windows with WSL)

**Step 0: Fresh Virtual Environment**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

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


**Step 3: Complete Setup**
```bash
# Check configuration first
python main.py --config

# Run complete setup (downloads models, generates embeddings)
python main.py --setup
```

This will:
- ✅ Parse your codebase (~2-10 minutes depending on size)
- ✅ Generate AI embeddings with UniXcoder & SBERT
- ✅ Create ChromaDB vector database
- ✅ Verify all search methods work

### Step 4: Test Search Quality
```bash
python main.py --test
```

### Step 5: Launch Web Interface  
```bash
python main.py
```
Access at: **http://localhost:8081**


## 🔍 Search Methods

| Method | Best For | Example Query |
|--------|----------|---------------|
| **Exact Match** | Function names, keywords | `"couple"`, `"vector<int>"` |
| **Code Structure** | Similar code patterns | `"loop through array"` |
| **Semantic** | Natural language | `"file reading function"` |

## 📁 Project Structure

```
SearchYourCodes/
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

## ⚙️ Requirements

- Python 3.8+
- PyTorch (for AI models)
- ChromaDB (for vector storage)
- Flask (for web interface)

## 📝 License

MIT License - see LICENSE file for details.
