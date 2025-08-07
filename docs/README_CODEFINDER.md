# SearchYourCodes - Intelligent Code Search Platform

## 🚀 Quick Start

### Option 1: Using the Launcher (Recommended)
```bash
# From project root
python3 run_SearchYourCodes.py
```

### Option 2: Direct Execution
```bash
# Activate virtual environment (if using one)
source .venv/bin/activate

# Run from src directory
cd src/
python3 web_app.py
```

## 📁 Project Structure

```
llm_code_search/
├── run_SearchYourCodes.py          # 🚀 Main launcher script
├── requirements.txt           # 📦 Python dependencies
├── data/                      # 💾 Data files and embeddings
├── config/                    # ⚙️ Configuration files
└── src/                       # 📂 Source code
    ├── web_app.py            # 🌐 Flask web application
    ├── search.py             # 🔍 Search functionality
    ├── templates/            # 🎨 HTML templates
    └── ...                   # Other modules
```

## 🌐 Access

Once running, access SearchYourCodes at: **http://localhost:8081**

## 🔧 Features

- **Exact Match**: Direct keyword search in code
- **Code Structure**: UniXcoder AI semantic matching
- **Semantic Search**: SBERT natural language understanding
- **Desktop-optimized UI** with larger displays and more results

## 🛠️ Troubleshooting

If you encounter issues:

1. **Install Flask**: `pip install flask`
2. **Activate virtual environment**: `source .venv/bin/activate`
3. **Install all requirements**: `pip install -r requirements.txt`

## 📝 Notes

- The launcher automatically detects and uses virtual environments
- All paths are properly resolved for the new structure
- Web application runs from `src/` directory with correct path resolution
