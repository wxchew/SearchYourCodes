"""
Singleton ChromaDB Client

Provides a single shared ChromaDB PersistentClient instance to avoid
duplicate connections across modules.
"""

import chromadb
from typing import Optional

_client: Optional[chromadb.ClientAPI] = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create the singleton ChromaDB PersistentClient."""
    global _client
    if _client is None:
        from core.config import get_chroma_db_path
        _client = chromadb.PersistentClient(path=str(get_chroma_db_path()))
    return _client


def reset_client() -> None:
    """Reset the singleton client (useful for testing or re-initialization)."""
    global _client
    _client = None
