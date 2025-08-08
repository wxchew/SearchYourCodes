"""
Parser Factory

This module provides a factory for creating language-specific parsers.
It supports automatic parser selection based on file extensions and
explicit parser creation by language name.

Features:
- Automatic parser detection by file extension
- Parser registration system
- Extensible architecture for new languages
- Language detection utilities
"""

from pathlib import Path
from typing import Dict, Optional, Set, Type, List

from .base_parser import BaseParser
from .cpp_parser import CppParser


class ParserFactory:
    """
    Factory class for creating language-specific parsers.
    
    This factory manages parser registration and provides utilities for
    automatic parser selection based on file characteristics.
    """
    
    # Registry of available parsers
    _parsers: Dict[str, Type[BaseParser]] = {}
    _extension_map: Dict[str, str] = {}
    
    @classmethod
    def register_parser(cls, language: str, parser_class: Type[BaseParser]):
        """
        Register a parser for a specific language.
        
        Args:
            language: Language identifier (e.g., 'cpp', 'python')
            parser_class: Parser class implementing BaseParser
        """
        cls._parsers[language] = parser_class
        
        # Create a temporary instance to get file extensions
        temp_parser = parser_class()
        for ext in temp_parser.file_extensions:
            cls._extension_map[ext.lower()] = language
    
    @classmethod
    def get_parser(cls, 
                   language: str, 
                   verbose: bool = False, 
                   min_lines: int = 3, 
                   context_window: int = 5) -> BaseParser:
        """
        Create a parser for the specified language.
        
        Args:
            language: Language identifier
            verbose: Enable verbose logging
            min_lines: Minimum lines for a chunk
            context_window: Context window for docstring extraction
            
        Returns:
            Parser instance for the language
            
        Raises:
            ValueError: If language is not supported
        """
        if language not in cls._parsers:
            available = list(cls._parsers.keys())
            raise ValueError(f"Unsupported language '{language}'. Available: {available}")
        
        parser_class = cls._parsers[language]
        return parser_class(verbose=verbose, min_lines=min_lines, context_window=context_window)
    
    @classmethod
    def list_available_parsers(cls) -> List[str]:
        """
        List all available parser languages.
        
        Returns:
            List of language identifiers
        """
        return list(cls._parsers.keys())
    
    @classmethod
    def create_parser(cls, language: str, verbose: bool = False) -> BaseParser:
        """
        Create a parser for the specified language.
        
        Args:
            language: Language identifier
            verbose: Enable verbose logging
            
        Returns:
            Parser instance for the language
            
        Raises:
            ValueError: If language is not supported
        """
        return cls.get_parser(language, verbose=verbose)
    
    @classmethod
    def get_parser_for_file(cls, 
                           file_path: Path, 
                           verbose: bool = False, 
                           min_lines: int = 3, 
                           context_window: int = 5) -> Optional[BaseParser]:
        """
        Create a parser based on file extension.
        
        Args:
            file_path: Path to the file
            verbose: Enable verbose logging
            min_lines: Minimum lines for a chunk
            context_window: Context window for docstring extraction
            
        Returns:
            Parser instance if file type is supported, None otherwise
        """
        extension = file_path.suffix.lower()
        
        if extension not in cls._extension_map:
            return None
        
        language = cls._extension_map[extension]
        return cls.get_parser(language, verbose, min_lines, context_window)
    
    @classmethod
    def detect_language(cls, file_path: Path) -> Optional[str]:
        """
        Detect the programming language of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language identifier if detected, None otherwise
        """
        extension = file_path.suffix.lower()
        return cls._extension_map.get(extension)
    
    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """
        Get all supported file extensions.
        
        Returns:
            Set of supported file extensions
        """
        return set(cls._extension_map.keys())
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """
        Get all supported languages.
        
        Returns:
            List of supported language identifiers
        """
        return list(cls._parsers.keys())
    
    @classmethod
    def is_supported_file(cls, file_path: Path) -> bool:
        """
        Check if a file is supported by any parser.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the file can be parsed
        """
        return cls.detect_language(file_path) is not None


# Register built-in parsers
ParserFactory.register_parser('cpp', CppParser)


class MultiLanguageParser:
    """
    Multi-language parser that can handle multiple programming languages.
    
    This class automatically selects the appropriate parser for each file
    based on file extension and language detection.
    """
    
    def __init__(self, 
                 verbose: bool = False, 
                 min_lines: int = 3, 
                 context_window: int = 5):
        """
        Initialize multi-language parser.
        
        Args:
            verbose: Enable verbose logging
            min_lines: Minimum lines for a chunk
            context_window: Context window for docstring extraction
        """
        self.verbose = verbose
        self.min_lines = min_lines
        self.context_window = context_window
        self._parser_cache = {}
    
    def parse_file(self, file_path: Path):
        """
        Parse a file using the appropriate language parser.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of CodeChunk objects
            
        Raises:
            ValueError: If file language is not supported
        """
        language = ParserFactory.detect_language(file_path)
        if not language:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Get or create parser for this language
        if language not in self._parser_cache:
            self._parser_cache[language] = ParserFactory.get_parser(
                language, self.verbose, self.min_lines, self.context_window
            )
        
        parser = self._parser_cache[language]
        return parser.parse_file(file_path)
    
    def parse_directory(self, 
                       directory: Path, 
                       recursive: bool = True,
                       max_files: Optional[int] = None):
        """
        Parse all supported files in a directory.
        
        Args:
            directory: Directory to parse
            recursive: Whether to parse subdirectories
            max_files: Maximum number of files to parse
            
        Returns:
            List of all CodeChunk objects from the directory
        """
        from .base_parser import CodeChunk
        
        chunks = []
        file_count = 0
        
        # Get all files
        if recursive:
            files = [f for f in directory.rglob("*") if f.is_file()]
        else:
            files = [f for f in directory.iterdir() if f.is_file()]
        
        for file_path in files:
            if max_files and file_count >= max_files:
                break
            
            if not ParserFactory.is_supported_file(file_path):
                continue
            
            try:
                file_chunks = self.parse_file(file_path)
                chunks.extend(file_chunks)
                file_count += 1
                
                if self.verbose:
                    language = ParserFactory.detect_language(file_path)
                    print(f"Parsed {file_path} ({language}): {len(file_chunks)} chunks")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error parsing {file_path}: {e}")
                continue
        
        return chunks
    
    def get_statistics(self, chunks):
        """
        Get statistics about parsed chunks.
        
        Args:
            chunks: List of CodeChunk objects
            
        Returns:
            Dictionary with parsing statistics
        """
        from collections import defaultdict
        
        stats = {
            'total_chunks': len(chunks),
            'files_processed': len(set(chunk.file_path for chunk in chunks)),
            'languages': set(),
            'chunks_by_language': defaultdict(int),
            'chunks_by_type': defaultdict(int),
            'average_chunk_size': 0,
            'total_lines': 0
        }
        
        # Analyze chunks
        for chunk in chunks:
            stats['languages'].add(chunk.language)
            stats['chunks_by_language'][chunk.language] += 1
            
            if chunk.function_name:
                stats['chunks_by_type']['functions'] += 1
            elif chunk.class_name:
                stats['chunks_by_type']['classes'] += 1
            else:
                stats['chunks_by_type']['other'] += 1
        
        # Convert sets to lists for JSON serialization
        stats['languages'] = list(stats['languages'])
        stats['chunks_by_language'] = dict(stats['chunks_by_language'])
        stats['chunks_by_type'] = dict(stats['chunks_by_type'])
        
        # Calculate size statistics
        if chunks:
            total_lines = sum(chunk.end_line - chunk.start_line + 1 for chunk in chunks)
            stats['total_lines'] = total_lines
            stats['average_chunk_size'] = total_lines / len(chunks)
        
        return stats
