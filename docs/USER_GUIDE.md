# 🔍 Code Search System - Quick Start Guide

## How to Test Your Own Codebase

This system can search any codebase using three powerful methods:
- **Keyword Search**: Exact text matching with fuzzy search
- **UniXcoder**: Code structure and programming patterns  
- **SBERT**: Semantic understanding and concepts

### 🚀 Quick Setup (2 minutes)

1. **Run the setup script:**
   ```bash
   python setup_for_my_code.py
   ```

2. **Follow the prompts:**
   - Enter your code directory path
   - Choose your programming language
   - Let it run the pipeline

3. **Start searching:**
   ```bash
   streamlit run streamlit_interface.py
   ```

### 📁 Manual Configuration

If you prefer manual setup, edit `config.py`:

```python
# Change this to your codebase path
CODEBASE_PATH = "/path/to/your/project/src"

# Change this to your file extensions
FILE_EXTENSIONS = {'.py', '.js', '.cpp', '.java'}  # Your languages
```

Then run:
```bash
python code_parser_clean.py    # Parse your code
python embedder.py             # Generate embeddings  
python ingester.py             # Build search database
```

### 🔒 Privacy & Security

**✅ 100% Local & Private:**
- All processing happens on your machine
- No code is sent to external servers
- Streamlit interface runs locally only
- Your code never leaves your computer

**✅ Safe for Corporate/Sensitive Code:**
- No internet connection required for search
- No API calls to external services
- Complete data sovereignty

### 🎯 Search Examples

Once set up, try these searches:

**For C++ Code:**
- "memory allocation"
- "class constructor" 
- "error handling"
- "template function"

**For Python Code:**
- "async function"
- "exception handling"
- "class method"
- "data validation"

**For JavaScript:**
- "arrow function"
- "event handler" 
- "API call"
- "state management"

### 📊 Compare Search Methods

The system shows you results from all three methods side-by-side:

- **Keyword**: Fast, exact matches
- **UniXcoder**: Understands code structure  
- **SBERT**: Finds conceptually related code

### 🛠️ Supported Languages

**Fully Tested:**
- C/C++ (.cpp, .h, .cc, .cxx)
- Python (.py)
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- Java (.java)

**Easily Configurable:**
- Any text-based programming language
- Mixed-language projects
- Custom file extensions

### 📈 Performance

**Typical Processing Times:**
- Small project (< 1000 files): 1-2 minutes
- Medium project (1000-5000 files): 3-5 minutes  
- Large project (> 5000 files): 5-10 minutes

**Search Speed:**
- Keyword search: Instant
- Vector search: < 1 second
- All methods combined: < 2 seconds

### 🆘 Troubleshooting

**"No results found":**
- Check if your file extensions are correct in config.py
- Verify your codebase path exists
- Try broader search terms

**"Model download errors":**
- Ensure internet connection for first-time model download
- Models are cached locally after first download

**"Memory errors":**
- Reduce batch size in embedder.py
- Process smaller code chunks

### 🔄 Re-running for Different Projects

To switch to a new codebase:
1. Run `python setup_for_my_code.py` again
2. Or manually update `CODEBASE_PATH` in config.py
3. Re-run the pipeline (parser → embedder → ingester)

### 📞 Need Help?

The system is designed to be self-contained and user-friendly. Most issues can be resolved by:
1. Re-running the setup script
2. Checking file paths in config.py
3. Ensuring your code directory has the expected file types
