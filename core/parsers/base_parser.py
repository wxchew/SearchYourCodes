"""
Base Parser Interface

This module defines the abstract base class for all language-specific parsers.
It provides a unified interface and common data structures for code parsing.

Features:
- Abstract parser interface for consistent behavior
- Common CodeChunk data structure
- Language-agnostic parsing contract
- Extensible design for new language support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class CodeChunk:
    """
    Represents a parsed code chunk with metadata.
    
    A code chunk can be a function, method, class, or file section that has been
    extracted from a source file. Each chunk contains the source code content and
    metadata such as the source location, names, and associated documentation.
    
    Attributes:
        content: Raw code content
        file_path: Path to source file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (inclusive)
        function_name: Name of the function/method (if applicable)
        class_name: Name of the class (if applicable)
        namespace: Namespace information (if applicable)
        docstring: Associated documentation comments
        parent_summary: Summary from parent class or file
        language: Programming language of the code
    """
    content: str
    file_path: Path
    start_line: int
    end_line: int
    language: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    namespace: Optional[str] = None
    docstring: Optional[str] = None
    parent_summary: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        # Create enriched content with hierarchical context, docstring, and parent summary
        enriched_content = self._create_enriched_content()
        
        return {
            "id": self._generate_id(),
            "content": enriched_content,  # Use enriched content for embedding
            "raw_content": self.content,  # Keep original content for reference
            "file_path": str(self.file_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "namespace": self.namespace,
            "docstring": self.docstring,
            "parent_summary": self.parent_summary,
            "language": self.language,
            "num_lines": self.end_line - self.start_line + 1
        }
    
    def _generate_id(self) -> str:
        """Generate a unique identifier for the chunk."""
        file_name = self.file_path.name
        if self.function_name:
            base = f"{file_name}::{self.function_name}"
        elif self.class_name:
            base = f"{file_name}::{self.class_name}"
        else:
            base = f"{file_name}::chunk"
        
        return f"{base}:{self.start_line}-{self.end_line}"
    
    def _create_enriched_content(self) -> str:
        """Create enriched content with context and documentation."""
        parts = []
        
        # Add namespace context
        if self.namespace:
            parts.append(f"namespace: {self.namespace}")
        
        # Add class context
        if self.class_name:
            parts.append(f"class: {self.class_name}")
        
        # Add function context
        if self.function_name:
            parts.append(f"function: {self.function_name}")
        
        # Add docstring if available
        if self.docstring and self.docstring.strip():
            parts.append(f"documentation: {self.docstring.strip()}")
        
        # Add parent summary if available
        if self.parent_summary and self.parent_summary.strip():
            parts.append(f"context: {self.parent_summary.strip()}")
        
        # Add the actual code content
        parts.append(f"code: {self.content}")
        
        return "\n\n".join(parts)


class BaseParser(ABC):
    """
    Abstract base class for language-specific code parsers.
    
    This class defines the interface that all language parsers must implement
    to ensure consistent behavior and integration with the code search system.
    """
    
    def __init__(self, 
                 verbose: bool = False, 
                 min_lines: int = 3, 
                 context_window: int = 5):
        """
        Initialize the parser with common configuration.
        
        Args:
            verbose: Enable verbose logging
            min_lines: Minimum lines for a chunk to be considered
            context_window: Lines of context to check for docstrings
        """
        self.verbose = verbose
        self.min_lines = min_lines
        self.context_window = context_window
    
    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the name of the programming language."""
        pass
    
    @property
    @abstractmethod
    def file_extensions(self) -> Set[str]:
        """Return the set of file extensions supported by this parser."""
        pass
    
    @property
    @abstractmethod
    def ignore_patterns(self) -> List[str]:
        """Return list of patterns to ignore during parsing."""
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Parse a single file and extract code chunks.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            List of CodeChunk objects extracted from the file
            
        Raises:
            ParsingError: If the file cannot be parsed
        """
        pass
    
    @abstractmethod
    def is_supported_file(self, file_path: Path) -> bool:
        """
        Check if a file is supported by this parser.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the file can be parsed by this parser
        """
        pass
    
    def parse_directory(self, 
                       directory: Path, 
                       recursive: bool = True,
                       max_files: Optional[int] = None) -> List[CodeChunk]:
        """
        Parse all supported files in a directory.
        
        Args:
            directory: Directory to parse
            recursive: Whether to parse subdirectories
            max_files: Maximum number of files to parse (None for no limit)
            
        Returns:
            List of all CodeChunk objects from the directory
        """
        chunks = []
        file_count = 0
        
        # Get all supported files
        if recursive:
            files = [f for f in directory.rglob("*") if f.is_file() and self.is_supported_file(f)]
        else:
            files = [f for f in directory.iterdir() if f.is_file() and self.is_supported_file(f)]
        
        for file_path in files:
            if max_files and file_count >= max_files:
                break
                
            try:
                file_chunks = self.parse_file(file_path)
                chunks.extend(file_chunks)
                file_count += 1
                
                if self.verbose:
                    print(f"Parsed {file_path}: {len(file_chunks)} chunks")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error parsing {file_path}: {e}")
                continue
        
        return chunks
    
    def get_statistics(self, chunks: List[CodeChunk]) -> Dict:
        """
        Get statistics about parsed chunks.
        
        Args:
            chunks: List of CodeChunk objects
            
        Returns:
            Dictionary with parsing statistics
        """
        stats = {
            'total_chunks': len(chunks),
            'files_processed': len(set(chunk.file_path for chunk in chunks)),
            'languages': [self.language_name],
            'chunks_by_type': {},
            'average_chunk_size': 0,
            'total_lines': 0
        }
        
        # Count chunks by type
        function_chunks = sum(1 for chunk in chunks if chunk.function_name)
        class_chunks = sum(1 for chunk in chunks if chunk.class_name and not chunk.function_name)
        other_chunks = len(chunks) - function_chunks - class_chunks
        
        stats['chunks_by_type'] = {
            'functions': function_chunks,
            'classes': class_chunks,
            'other': other_chunks
        }
        
        # Calculate size statistics
        if chunks:
            total_lines = sum(chunk.end_line - chunk.start_line + 1 for chunk in chunks)
            stats['total_lines'] = total_lines
            stats['average_chunk_size'] = total_lines / len(chunks)
        
        return stats


class ParsingError(Exception):
    """Exception raised when parsing fails."""
    
    def __init__(self, message: str, file_path: Optional[Path] = None):
        """
        Initialize parsing error.
        
        Args:
            message: Error message
            file_path: File that caused the error (if applicable)
        """
        self.file_path = file_path
        if file_path:
            super().__init__(f"Error parsing {file_path}: {message}")
        else:
            super().__init__(message)
