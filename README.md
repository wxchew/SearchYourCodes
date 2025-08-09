# SearchYourCodes - An Intelligent Code Search Platform

SearchYourCodes is an AI-powered search engine that helps you find code in your local repositories using multiple advanced search methods. Instead of just matching keywords, it understands the meaning and structure of your code, allowing for more intuitive and powerful discovery.

![Search Interface](docs/screenshot.png) <!-- Assuming a screenshot will be placed in docs/ -->

## 🌟 Key Features

- **Multi-Method Search**: Combines three search techniques for comprehensive results:
  1.  **Keyword Search**: Fast and precise for finding specific variable or function names.
  2.  **Code-Structure Search (UniXcoder)**: Finds code that is structurally similar to your query.
  3.  **Semantic Search (SBERT)**: Understands natural language queries to find conceptually related code.
- **User-Friendly Web Interface**: A clean, three-column layout to easily compare results from all search methods at once.
- **Local First**: Your code never leaves your machine. All processing and searching happens locally.
- **Extensible**: Designed to support more programming languages and embedding models in the future.

---

## 🚀 Getting Started

Follow these steps to set up and run SearchYourCodes on your own codebase.

### Prerequisites

- Python 3.8+
- `git` for cloning the repository.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/SearchYourCodes.git
cd SearchYourCodes
```

### Step 2: Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### Step 3: Install Dependencies

Install all the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Step 4: Configure Your Codebase

Tell SearchYourCodes where to find your source code.

1.  **Open `config.yaml`** in the project root.
2.  **Update the `path`** under the `codebase` section to the absolute or relative path of the code you want to index.

```yaml
codebase:
  # ✍️ CHANGE THIS to the path of your local code repository
  path: "data/codebases/your-project/src" 
  
  # You can also adjust the file extensions to scan
  extensions:
    - ".cpp"
    - ".h"
    - ".py"
```

### Step 5: Run the Setup Process

This is a one-time process that will parse your code, generate AI embeddings, and build the search database. This may take several minutes depending on the size of your codebase and your computer's hardware.

```bash
python main.py --setup
```

This command will:
- 🧠 Download the required AI models.
- 📝 Parse all supported files in your codebase.
- 🔢 Generate vector embeddings for each code chunk.
- 🗄️ Store everything in a local ChromaDB database inside the `data/chroma_db` directory.

### Step 6: Launch the Web App

Once the setup is complete, you can start the web application.

```bash
python main.py
```

Now, open your web browser and navigate to **http://localhost:8081** to start searching!

---

## 🛠️ Usage and Commands

All commands are run through `main.py`.

- **Run the Web App**:
  ```bash
  python main.py
  ```

- **Run the Setup Process**:
  (Only needed once, or again if your code changes significantly).
  ```bash
  python main.py --setup
  ```

- **Clean the Database and Processed Files**:
  (Use this if you want to do a fresh setup).
  ```bash
  python main.py --clean
  ```

- **Run System Tests**:
  (Verifies that all components are working correctly).
  ```bash
  python main.py --test
  ```

- **View Current Configuration**:
  ```bash
  python main.py --config
  ```

---

## 📁 Project Structure

```
/
├── app/                  # 🌐 Flask Web Application source
├── core/                 # 🧠 Core search logic, parsers, and embedders
│   ├── parsers/          # Language-specific code parsers (e.g., CppParser)
│   └── embedders_oop.py  # Manages AI embedding models
├── data/                 # 🗄️ Default location for codebases and ChromaDB
├── docs/                 # 📚 Documentation files
├── main.py               # 🚀 Main entry point for all commands
├── setup_SearchYourCodes_final.py # ⚙️ The main setup and ingestion script
└── config.yaml           # 🔧 Project configuration
```
