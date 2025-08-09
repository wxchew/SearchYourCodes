# SearchYourCodes - Technical Documentation

This document provides a technical overview of the SearchYourCodes platform, detailing its architecture, workflow, and key components.

## 1. System Architecture

The system is designed with a modular architecture, separating concerns into distinct components:

-   **Main Entry Point (`main.py`)**: A command-line interface (CLI) that orchestrates all high-level operations like setup, cleaning, testing, and launching the web app. It acts as a facade, delegating tasks to the appropriate modules.
-   **Configuration (`config.yaml`)**: A central YAML file that defines all system parameters, including codebase paths, database locations, and model settings.
-   **Setup & Ingestion (`setup_SearchYourCodes_final.py`)**: The core script responsible for the entire data processing pipeline. It handles parsing the codebase, generating embeddings, and populating the vector database.
-   **Core Search Logic (`core/`)**: This directory contains the heart of the search engine.
    -   `search.py`: Provides a unified API for querying all search methods (keyword, UniXcoder, SBERT).
    -   `keyword_search.py`: Implements exact-match keyword searching.
    -   `vector_search.py`: Handles semantic search by querying the ChromaDB vector database.
    -   `embedders_oop.py`: An object-oriented framework for managing and using different AI embedding models (e.g., UniXcoder, SBERT).
    -   `parsers/`: Contains language-specific parsers (like `CppParser`) responsible for extracting code chunks from source files.
-   **Web Application (`app/`)**: A Flask-based web server that provides the user interface for interacting with the search engine.
-   **Data Storage (`data/`)**:
    -   `codebases/`: The default location for the source code to be indexed.
    -   `chroma_db/`: The directory where the ChromaDB vector database is stored.

## 2. The Ingestion Workflow

The setup process (`main.py --setup`) triggers the ingestion workflow, which is orchestrated by `setup_SearchYourCodes_final.py`. This workflow consists of several key stages:

1.  **Environment Validation**: The script first checks for necessary dependencies (like `tree-sitter`), validates the `config.yaml` file, and ensures the target codebase directory exists.

2.  **Model Loading**: The required AI models (UniXcoder and SBERT) are loaded into memory using the `EmbedderFactory` from `core/embedders_oop.py`. This factory pattern allows for easy extension with new models in the future.

3.  **Code Parsing**:
    -   The script iterates through all files in the configured codebase path that match the specified file extensions.
    -   For each file, a language-specific parser (e.g., `CppParser`) is used. These parsers are built on top of the `tree-sitter` library, which creates a concrete syntax tree (CST) of the code.
    -   The parser traverses the syntax tree to identify and extract meaningful **code chunks**. A chunk can be a function, a class, a method, or a standalone block of code.
    -   Each chunk is stored as a `CodeChunk` object, which contains the code content as well as rich metadata (file path, start/end lines, function/class names).

4.  **Embedding Generation**:
    -   The extracted code chunks are processed in batches for efficiency.
    -   For each batch, the content of the chunks is passed to the pre-loaded AI models (UniXcoder and SBERT) to generate vector embeddings.
    -   **UniXcoder** produces embeddings that capture the structural and syntactic properties of the code.
    -   **SBERT** produces embeddings that capture the semantic meaning, making it effective for natural language queries.

5.  **Database Population**:
    -   The system uses **ChromaDB** as its vector database. Two separate collections are created: `unixcoder_snippets` and `sbert_snippets`.
    -   For each code chunk, the generated embeddings are stored in the corresponding collection.
    -   Crucially, each entry in the database includes the chunk's content, its vector embedding, and all its associated metadata. The chunk's unique ID (e.g., `trapper.cc::~TrapperLong:21-23`) is used as the primary identifier.

## 3. The Search Workflow

When a user submits a query through the web interface or CLI, the following occurs:

1.  **Query Reception**: The Flask app (`app/main.py`) receives the search query.

2.  **Parallel Search Execution**: The `core/search.py` module's `compare_models` function is called, which executes all three search methods in parallel:
    -   **Keyword Search**: The query is used to perform a direct text search against the documents stored in ChromaDB.
    -   **Vector Search (UniXcoder & SBERT)**:
        -   The user's query is first converted into a vector embedding using the same AI model as the target collection (UniXcoder or SBERT).
        -   This query vector is then used to perform a similarity search (specifically, an Approximate Nearest Neighbor search) against the vectors in the corresponding ChromaDB collection.
        -   ChromaDB returns the `k` most similar code chunks based on vector distance.

3.  **Result Formatting**:
    -   The results from all three methods are collected.
    -   For vector search results, the distance metric (Squared L2) is converted into a more intuitive similarity score (0.0 to 1.0).
    -   The results, including code content, file paths, line numbers, and scores, are formatted into a standardized JSON structure.

4.  **Display**: The formatted JSON is sent to the frontend, where the web interface renders the results in three distinct columns for easy comparison.

## 4. Key Technical Decisions

-   **Tree-sitter for Parsing**: Chosen for its high performance, error-resilience, and broad language support. It allows for precise extraction of code structure.
-   **ChromaDB for Vector Storage**: Selected for its simplicity, local-first operation, and efficient similarity search capabilities, making it ideal for a desktop application.
-   **OOP Embedder Framework**: The `embedders_oop.py` module was designed to make the system modular. Adding a new embedding model is as simple as creating a new class that inherits from `BaseEmbedder` and registering it with the `EmbedderFactory`.
-   **Facade Pattern in `main.py` and `core/search.py`**: These scripts act as simple, high-level interfaces to the more complex underlying logic, making the system easier to use and maintain.
