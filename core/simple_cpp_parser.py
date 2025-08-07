#!/usr/bin/env python3
"""
Simple C++ Parser - Tree-sitter Compatibility Fix

This replaces the complex Tree-sitter queries with regex-based parsing
that works reliably across different Tree-sitter versions.
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class CodeChunk:
    """Simple code chunk without Tree-sitter dependencies."""
    content: str
    file_path: Path
    start_line: int
    end_line: int
    function_name: str = 'Unknown'
    class_name: str = 'Unknown'
    namespace: str = 'Unknown'
    docstring: str = ''
    parent_summary: str = ''

class CppParser:
    """Simple regex-based C++ parser."""
    
    def __init__(self, verbose: bool = False, min_lines: int = 3):
        self.verbose = verbose
        self.min_lines = min_lines
        
        # Regex patterns for basic C++ constructs
        self.patterns = {
            'class': re.compile(r'^\s*(?:class|struct)\s+(\w+)', re.MULTILINE),
            'function': re.compile(r'^\s*(?:[\w:*&<>]+\s+)*?(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?[{;]', re.MULTILINE),
            'namespace': re.compile(r'^\s*namespace\s+(\w+)', re.MULTILINE),
            'comment_block': re.compile(r'/\*\*(.*?)\*/', re.DOTALL),
            'comment_line': re.compile(r'///(.*?)$', re.MULTILINE)
        }
    
    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a C++ file into code chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if len(content.strip()) < 20:
                return []
            
            # Extract metadata
            classes = self._extract_names(content, 'class')
            functions = self._extract_names(content, 'function')
            namespaces = self._extract_names(content, 'namespace')
            
            # Create chunks
            chunks = self._create_chunks(content, file_path, classes, functions, namespaces)
            
            if self.verbose and chunks:
                print(f"    Extracted {len(chunks)} chunks, {len(functions)} functions, {len(classes)} classes")
            
            return chunks
            
        except Exception as e:
            if self.verbose:
                print(f"    Error parsing {file_path}: {e}")
            return []
    
    def _extract_names(self, content: str, pattern_type: str) -> List[str]:
        """Extract names using regex patterns."""
        pattern = self.patterns[pattern_type]
        return [match.group(1) for match in pattern.finditer(content)]
    
    def _create_chunks(self, content: str, file_path: Path, classes: List[str], 
                      functions: List[str], namespaces: List[str]) -> List[CodeChunk]:
        """Create code chunks from content."""
        lines = content.split('\n')
        chunks = []
        
        # Determine chunk strategy based on file size
        if len(lines) <= 100:
            # Small file - single chunk
            chunks.append(self._create_chunk(
                content, file_path, 1, len(lines), 
                classes, functions, namespaces
            ))
        else:
            # Large file - multiple chunks
            chunk_size = max(50, len(lines) // 8)  # Adaptive chunking
            
            for i in range(0, len(lines), chunk_size):
                end_idx = min(i + chunk_size, len(lines))
                chunk_content = '\n'.join(lines[i:end_idx])
                
                if len(chunk_content.strip()) >= self.min_lines * 15:
                    chunks.append(self._create_chunk(
                        chunk_content, file_path, i + 1, end_idx,
                        classes, functions, namespaces
                    ))
        
        return [chunk for chunk in chunks if chunk is not None]
    
    def _create_chunk(self, content: str, file_path: Path, start_line: int, end_line: int,
                     classes: List[str], functions: List[str], namespaces: List[str]) -> Optional[CodeChunk]:
        """Create a single code chunk with metadata."""
        if len(content.strip()) < 20:
            return None
        
        # Find relevant metadata for this chunk
        chunk_classes = [c for c in classes if c in content]
        chunk_functions = [f for f in functions if f in content and len(f) > 2]
        chunk_namespaces = [n for n in namespaces if n in content]
        
        return CodeChunk(
            content=content.strip(),
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            function_name=chunk_functions[0] if chunk_functions else 'Unknown',
            class_name=chunk_classes[0] if chunk_classes else 'Unknown',
            namespace=chunk_namespaces[0] if chunk_namespaces else 'Unknown',
            docstring='',
            parent_summary=''
        )
