#!/usr/bin/env python3
"""
Multi-Language Code Parser

Tree-sitter based parser that extracts functions and classes from various programming
language codebases for code analysis and search applications.

This module provides functionality to parse source files and extract
structured code chunks with rich contextual information including function
definitions, class definitions, namespaces, and documentation.

Features:
    - Multi-language support through modular parser architecture
    - Tree-sitter based parsing for robust AST analysis
    - Extracts functions, classes, and their documentation
    - Handles multiple programming languages
    - Provides enriched content with contextual prefixes
    - Configurable chunking parameters

Usage:
    python code_parser.py [--repo-path PATH] [--output PATH] [--verbose] [--language LANG]

Dependencies:
    pip install tree-sitter tree-sitter-cpp tqdm

Authors:
    Code Search System

Version:
    2.0.0 - Multi-language support
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Import the new modular parser system
from .parsers import ParserFactory, MultiLanguageParser, BaseParser

# Local imports with fallback
try:
    from .config import resolve_codebase_path, get_file_extensions
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config.py not found, using default settings")


def parse_repository(repo_path: Path, 
                    language: Optional[str] = None,
                    verbose: bool = False,
                    min_lines: int = 3,
                    context_window: int = 5) -> List[Dict]:
    """
    Parse a repository and extract code chunks.
    
    Args:
        repo_path: Path to the repository
        language: Specific language to parse (None for all supported)
        verbose: Enable verbose output
        min_lines: Minimum lines for a chunk
        context_window: Context window for docstring extraction
        
    Returns:
        List of code chunks as dictionaries
    """
    if verbose:
        print(f"Parsing repository: {repo_path}")
        if language:
            print(f"Language filter: {language}")
        else:
            print("Languages: all supported")
        print(f"Supported languages: {ParserFactory.get_supported_languages()}")
        print(f"Supported extensions: {ParserFactory.get_supported_extensions()}")
    
    chunks = []
    
    if language:
        # Use specific language parser
        try:
            parser = ParserFactory.get_parser(language, verbose, min_lines, context_window)
            repo_chunks = parser.parse_directory(repo_path, recursive=True)
            
            # Convert to dictionaries
            chunks = [chunk.to_dict() for chunk in repo_chunks]
            
            if verbose:
                print(f"Extracted {len(chunks)} chunks using {language} parser")
                
        except ValueError as e:
            print(f"Error: {e}")
            return []
            
    else:
        # Use multi-language parser
        parser = MultiLanguageParser(verbose, min_lines, context_window)
        repo_chunks = parser.parse_directory(repo_path, recursive=True)
        
        # Convert to dictionaries
        chunks = [chunk.to_dict() for chunk in repo_chunks]
        
        if verbose:
            print(f"Extracted {len(chunks)} chunks using multi-language parser")
            
            # Show statistics
            stats = parser.get_statistics(repo_chunks)
            print(f"Statistics:")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Languages detected: {stats['languages']}")
            print(f"  Chunks by language: {stats['chunks_by_language']}")
            print(f"  Chunks by type: {stats['chunks_by_type']}")
            print(f"  Average chunk size: {stats['average_chunk_size']:.1f} lines")
    
    return chunks


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Parse code repositories and extract structured chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Parse all supported languages in a repository
    python code_parser.py --repo-path /path/to/repo --output chunks.json
    
    # Parse only C++ files
    python code_parser.py --repo-path /path/to/repo --language cpp --output cpp_chunks.json
    
    # Verbose parsing with custom parameters
    python code_parser.py --repo-path /path/to/repo --verbose --min-lines 5 --context-window 3
        """
    )
    
    parser.add_argument(
        '--repo-path', 
        type=Path,
        help='Path to the repository to parse (default: from config)'
    )
    
    parser.add_argument(
        '--output', 
        type=Path,
        help='Output JSON file path (default: data/processed/code_chunks.json)'
    )
    
    parser.add_argument(
        '--language',
        choices=ParserFactory.get_supported_languages(),
        help='Specific language to parse (default: all supported languages)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--min-lines',
        type=int,
        default=3,
        help='Minimum number of lines for a chunk (default: 3)'
    )
    
    parser.add_argument(
        '--context-window',
        type=int,
        default=5,
        help='Context window for docstring extraction (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        # Determine repository path
        if args.repo_path:
            repo_path = args.repo_path
        elif CONFIG_AVAILABLE:
            repo_path = resolve_codebase_path()
        else:
            # Fallback to default test repository
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            repo_path = Path(project_root) / "data" / "codebases" / "manuel_natcom" / "src" / "sim"
        
        if not repo_path.exists():
            print(f"Error: Repository path does not exist: {repo_path}")
            return 1
        
        if args.verbose:
            print(f"Repository path: {repo_path}")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            output_path = Path(project_root) / "data" / "processed" / "code_chunks.json"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.verbose:
            print(f"Output path: {output_path}")
        
        # Parse repository
        chunks = parse_repository(
            repo_path=repo_path,
            language=args.language,
            verbose=args.verbose,
            min_lines=args.min_lines,
            context_window=args.context_window
        )
        
        if not chunks:
            print("No chunks extracted. Check repository path and supported file types.")
            return 1
        
        # Save results
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2)
            
            print(f"Saved {len(chunks)} chunks to {output_path}")
        except IOError as e:
            print(f"Error writing output file: {e}")
            return 1
        
        # Show sample output if verbose
        if chunks and args.verbose:
            print(f"\n--- Sample of {min(2, len(chunks))} Chunks ---")
            for i, chunk in enumerate(chunks[:2]):
                print(f"Chunk {i+1}: {chunk['id']}")
                print(f"  Language: {chunk['language']}")
                print(f"  Function: {chunk['function_name'] or 'N/A'}")
                print(f"  Class: {chunk['class_name'] or 'N/A'}")
                print(f"  Lines: {chunk['start_line']}-{chunk['end_line']}")
                print(f"  Content preview: {chunk['content'][:80]}...")
                print("-" * 40)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
