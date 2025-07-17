#!/usr/bin/env python3
"""
C++ Code Parser

Tree-sitter based parser that extracts functions and classes from C++ codebases
for code analysis and search applications.

Usage:
    python code_parser_clean.py [--repo-path PATH] [--output PATH] [--verbose]

Dependencies:
    pip install tree-sitter tree-sitter-cpp tqdm
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from tree_sitter import Language, Parser
import tree_sitter_cpp
from tqdm import tqdm


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
    """
    content: str
    file_path: Path
    start_line: int
    end_line: int
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
        }

    def _generate_id(self) -> str:
        """Generate unique ID for the chunk."""
        identifier = self.function_name or self.class_name or "file_chunk"
        # Include namespace in ID if available
        if self.namespace and self.function_name:
            identifier = f"{self.namespace}::{self.function_name}"
        return f"{self.file_path.name}:{self.start_line}-{self.end_line}:{identifier}"
        
    def _create_enriched_content(self) -> str:
        """Create enriched content with context prefixes, docstrings, and parent summaries."""
        parts = []
        
        # 1. Add hierarchical context prefix
        context_prefix = ""
        if self.namespace:
            context_prefix += f"namespace {self.namespace} {{ "
        if self.class_name and self.function_name:
            context_prefix += f"class {self.class_name} {{ "
        if context_prefix:
            parts.append(f"// CONTEXT: {context_prefix}")
            
        # 2. Add parent summary if available
        if self.parent_summary:
            parts.append(f"// FILE PURPOSE: {self.parent_summary.strip()}")
            
        # 3. Add docstring if available
        if self.docstring:
            parts.append(self.docstring)
            
        # 4. Add the actual code content
        parts.append(self.content)
        
        # 5. Close any opened context brackets
        if context_prefix:
            parts.append("// CONTEXT END")
            
        return "\n\n".join(parts)

class CppParser:
    """
    C++ code parser using Tree-sitter.
    
    This class provides functionality to parse C++ source files and extract
    structured code chunks with rich contextual information such as:
    - Function definitions with their docstrings
    - Class definitions with their docstrings
    - Namespace information
    - File-level documentation
    
    The parser handles both implementation files (.cpp, .cc) and header files (.h, .hpp),
    and attempts to correlate information between them.
    """
    
    # File extensions to process
    CPP_EXTENSIONS = {'.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx'}
    
    # Directories and files to ignore
    IGNORE_PATTERNS = [
        '.git', '.venv', '__pycache__', 'node_modules', '.DS_Store',
        'build/', 'dist/', '.egg-info', '.pytest_cache', '.idea',
        '*.min.js', '*.css.map', '*.wasm', '.inc'
    ]

    def __init__(self, verbose: bool = False, min_lines: int = 3, context_window: int = 5):
        """Initialize parser with Tree-sitter C++ language."""
        self.verbose = verbose
        self.min_lines = min_lines
        self.context_window = context_window  # Lines of context to check for docstrings
        self.parser = Parser()
        
        try:
            self.language = Language(tree_sitter_cpp.language())
            self.parser.language = self.language
            self._init_queries()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tree-sitter C++ parser: {e}")

    def _init_queries(self):
        """Initialize Tree-sitter queries for functions, classes, namespaces, and comments."""
        # Define all queries in a dictionary for better organization
        queries = {
            'function': '''
                (function_definition) @function.def
                (template_declaration (function_definition) @function.def)
            ''',
            'class': '(class_specifier) @class.def',
            'namespace': '(namespace_definition) @namespace.def',
            'comment': '(comment) @comment'
        }
        
        try:
            # Initialize all queries at once
            self.function_query = self.language.query(queries['function'])
            self.class_query = self.language.query(queries['class'])
            self.namespace_query = self.language.query(queries['namespace'])
            self.comment_query = self.language.query(queries['comment'])
        except Exception as e:
            raise RuntimeError(f"Failed to compile Tree-sitter queries: {e}")

    def _extract_node_info(self, node, source_bytes: bytes) -> tuple:
        """Extract content and line numbers from AST node."""
        content = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
        start_line = node.start_point[0] + 1  # Convert to 1-indexed
        end_line = node.end_point[0] + 1
        return content, start_line, end_line

    def _extract_function_name(self, node) -> str:
        """Extract function name from function_definition node.
        
        For methods in implementation files (with Class::method syntax),
        this returns the full function name including scope.
        """
        # Try structured extraction first
        for child in node.children:
            if child.type == 'function_declarator':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        return subchild.text.decode('utf-8', errors='ignore')
                    # For qualified identifiers like Class::method
                    if subchild.type in ('qualified_identifier', 'scoped_identifier'):
                        return subchild.text.decode('utf-8', errors='ignore')
        
        # Fallback: regex on code content
        content = node.text.decode('utf-8', errors='ignore')
        content = re.sub(r'template\s*<[^>]+>\s*', '', content)  # Remove template declaration
        
        # Check for operator overload
        if op_match := re.search(r'operator\s*([\w<>\[\]]+)', content):
            return 'operator' + op_match.group(1)
            
        # General function name before '('
        if m := re.search(r'([A-Za-z_:~][A-Za-z0-9_:<>~]+)\s*\(', content):
            return m.group(1).strip()
            
        return "unknown_function"
        
    def _extract_scope_parts(self, function_name: str) -> tuple:
        """Extract class name and method name from scoped function name (e.g., Class::method)."""
        parts = function_name.split('::')
        if len(parts) > 1:
            class_name = '::'.join(parts[:-1])  # Handle nested namespaces/classes
            method_name = parts[-1]
            return class_name, method_name
        return None, function_name

    def _extract_class_name(self, node) -> str:
        """Extract class name from class_specifier node."""
        for child in node.children:
            if child.type == 'type_identifier':
                return child.text.decode('utf-8', errors='ignore')
        return "unknown_class"

    def _extract_file_docstring(self, source_bytes: bytes, tree) -> str:
        """Extract file-level docstring/header comments."""
        # Get comments from the file
        comment_matches = self.comment_query.matches(tree.root_node)
        comments = []
        
        for match_id, captures_dict in comment_matches:
            for capture_name, nodes in captures_dict.items():
                if capture_name == 'comment':
                    for node in nodes:
                        comment_text = node.text.decode('utf-8', errors='ignore')
                        comments.append((node.start_point[0], comment_text))
        
        # Sort comments by line number
        comments.sort()
        
        # Define patterns for categorizing comments
        copyright_pattern = re.compile(r'copyright|license|\(c\)|©|all rights reserved', re.IGNORECASE)
        doc_pattern = re.compile(r'/\*\*|\*\s+@|@file|@brief|@author|@description', re.IGNORECASE)
        
        # Collect meaningful comments
        file_comments = []
        doc_comments = []
        
        # Process first 50 lines of comments
        for line_num, comment in comments:
            if line_num > 50:
                break
                
            # Clean up the comment
            cleaned = re.sub(r'^/\*+|\*+/$', '', comment)
            cleaned = re.sub(r'^//+\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            # Skip empty or separator comments
            if not cleaned or cleaned.startswith('---') or cleaned.startswith('==='):
                continue
                
            # Skip copyright notices
            if copyright_pattern.search(cleaned):
                continue
                
            # Collect doc comments and regular comments
            if doc_pattern.search(comment) or comment.startswith('/**') or comment.startswith('///'):
                doc_comments.append(cleaned)
            elif line_num < 30 and len(cleaned) > 15:
                file_comments.append(cleaned)
        
        # Select the best comments for summary
        summary = ""
        if doc_comments:
            summary = "\n".join(doc_comments)
        elif file_comments:
            summary = "\n".join(file_comments)
        
        # Final cleanup
        if summary:
            summary = re.sub(r'^\s*\*+\s*', '', summary, flags=re.MULTILINE)
            summary = re.sub(r'^\s*//+\s*', '', summary, flags=re.MULTILINE)
            summary = re.sub(r'\n\s*\n', '\n', summary)
        
        return summary
        


    def _find_docstring(self, node, source_bytes: bytes, source_lines: List[str]) -> str:
        """Find docstring/comments that precede a node."""
        start_line = node.start_point[0]
        docstring_lines = []
        
        # Look backwards from the function/class definition
        for i in range(start_line - 1, max(-1, start_line - self.context_window - 1), -1):
            if i < 0 or i >= len(source_lines):
                continue
                
            line = source_lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Handle comment patterns
            if line.startswith('///'):
                # Doxygen triple slash comment
                comment_text = re.sub(r'^///\s*', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('//'):
                # Regular double slash comment
                comment_text = re.sub(r'^//\s*', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('/*') and line.endswith('*/'):
                # Single line comment
                comment_text = re.sub(r'^/\*+\s*|\s*\*+/$', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('*') and docstring_lines:
                # Continuation of multi-line comment
                comment_text = re.sub(r'^\*\s*', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('/*'):
                # Multi-line comment
                comment_start = i
                comment_end = i
                
                # Find comment end
                while comment_end < len(source_lines) and '*/' not in source_lines[comment_end]:
                    comment_end += 1
                
                if comment_end < len(source_lines):
                    # Process multi-line comment
                    comment_block = []
                    for j in range(comment_start, comment_end + 1):
                        comment_line = source_lines[j].strip()
                        if j == comment_start:
                            comment_line = re.sub(r'^/\*+\s*', '', comment_line)
                        if j == comment_end:
                            comment_line = re.sub(r'\s*\*+/$', '', comment_line)
                        if comment_line.startswith('*'):
                            comment_line = re.sub(r'^\*\s*', '', comment_line)
                        
                        if comment_line:
                            comment_block.append(comment_line)
                    
                    docstring_lines = comment_block + docstring_lines
                    break
                else:
                    break
            else:
                # Non-comment line, stop looking
                break
        
        # Clean up and return the docstring
        if docstring_lines:
            # Remove empty lines
            while docstring_lines and not docstring_lines[0].strip():
                docstring_lines.pop(0)
            while docstring_lines and not docstring_lines[-1].strip():
                docstring_lines.pop()
            
            if docstring_lines:
                docstring = "\n".join(docstring_lines).strip()
                
                # Filter out separator lines and copyright notices
                if (docstring and 
                    not all(c in '-=_*~#' for c in docstring.replace(' ', '')) and
                    not docstring.lower().startswith('copyright') and
                    not docstring.lower().startswith('license') and
                    len(docstring) > 5):
                    return docstring
        
        return None

    def _find_parent_context(self, node, source_bytes: bytes) -> tuple:
        """Find parent class and namespace for a node."""
        class_name = None
        namespace = None
        
        # Walk up the tree to find parent nodes
        current = node.parent
        while current:
            if current.type == 'class_specifier':
                for child in current.children:
                    if child.type == 'type_identifier':
                        class_name = child.text.decode('utf-8', errors='ignore')
                        break
            
            if current.type == 'namespace_definition':
                for child in current.children:
                    if child.type in ('identifier', 'namespace_identifier'):
                        ns_part = child.text.decode('utf-8', errors='ignore')
                        namespace = ns_part if namespace is None else f"{ns_part}::{namespace}"
                        break
            
            current = current.parent
        
        return class_name, namespace

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single C++ file and extract code chunks."""
        chunks = []
        
        try:
            with open(file_path, 'rb') as f:
                source_bytes = f.read()

            if not source_bytes.strip():
                return chunks

            # Decode source for line-by-line processing
            source_text = source_bytes.decode('utf-8', errors='ignore')
            source_lines = source_text.splitlines()
            
            tree = self.parser.parse(source_bytes)
            
            if self.verbose:
                print(f"Parsing {file_path.name}...")
                
            # Extract file-level docstring/summary
            file_summary = self._extract_file_docstring(source_bytes, tree)
            if self.verbose and file_summary:
                print(f"  Found file summary: {file_summary[:50]}...")

            # Extract classes with their docstrings
            class_info = {}  # Store class info for context
            class_matches = self.class_query.matches(tree.root_node)
            for match_id, captures_dict in class_matches:
                for capture_name, nodes in captures_dict.items():
                    if capture_name == 'class.def':
                        for node in nodes:
                            # Find the class name in the node
                            class_name = self._extract_class_name(node)
                            if class_name and node.type == 'class_specifier':
                                content, start_line, end_line = self._extract_node_info(node, source_bytes)
                                
                                # Find class docstring
                                class_docstring = self._find_docstring(node, source_bytes, source_lines)
                                
                                # Find class namespace
                                _, class_namespace = self._find_parent_context(node, source_bytes)
                                
                                # Store class info
                                class_info[class_name] = {
                                    'docstring': class_docstring,
                                    'namespace': class_namespace
                                }
                                
                                # For header files, also extract individual function declarations with their docstrings
                                if file_path.suffix.lower() in ['.h', '.hpp', '.hxx']:
                                    # Extract function declarations from within the class
                                    self._extract_function_declarations_from_class(
                                        node, source_bytes, source_lines, file_path, class_name, 
                                        class_namespace, class_docstring or file_summary, chunks
                                    )
                                
                                # Skip tiny classes or don't add whole class as chunk for header files
                                if end_line - start_line + 1 < self.min_lines:
                                    continue
                                
                                # For implementation files, add the class as a chunk
                                if file_path.suffix.lower() not in ['.h', '.hpp', '.hxx']:
                                    chunks.append(CodeChunk(
                                        content=content,
                                        file_path=file_path,
                                        start_line=start_line,
                                        end_line=end_line,
                                        class_name=class_name,
                                        namespace=class_namespace,
                                        docstring=class_docstring,
                                        parent_summary=file_summary
                                    ))

            # Extract namespaces for context
            namespace_dict = {}
            namespace_matches = self.namespace_query.matches(tree.root_node)
            for match_id, captures_dict in namespace_matches:
                for capture_name, nodes in captures_dict.items():
                    if capture_name == 'namespace.def':
                        for node in nodes:
                            # TODO: Add namespace tracking if needed
                            pass

            # Extract functions with their docstrings and context
            function_matches = self.function_query.matches(tree.root_node)
            
            # Initialize data for storing class documentation
            class_docs = {}
            
            for match_id, captures_dict in function_matches:
                for capture_name, nodes in captures_dict.items():
                    if capture_name == 'function.def':
                        for node in nodes:
                            content, start_line, end_line = self._extract_node_info(node, source_bytes)
                            
                            # Skip trivial or destructor chunks
                            line_count = end_line - start_line + 1
                            full_function_name = self._extract_function_name(node)
                            
                            # Parse class name from scoped function name (Class::method)
                            class_from_scope, method_name = self._extract_scope_parts(full_function_name)
                            
                            # Skip destructors and tiny methods
                            if line_count < self.min_lines or method_name.startswith('~'):
                                continue
                                
                            # Find function docstring
                            docstring = self._find_docstring(node, source_bytes, source_lines)
                            
                            # Find parent class and namespace from AST if not already found from scope
                            parent_class, namespace = self._find_parent_context(node, source_bytes)
                            
                            # Use class name from scope resolution if available
                            if class_from_scope:
                                parent_class = class_from_scope
                                function_name = method_name  # Use method name without class prefix
                            else:
                                function_name = full_function_name
                            
                            # If no docstring found in implementation file and we have a class method,
                            # try to find it in the header file
                            if not docstring and parent_class and file_path.suffix.lower() in ['.cc', '.cpp', '.cxx']:
                                header_path = self._find_header_file(file_path)
                                if header_path and header_path.exists():
                                    docstring = self._find_function_docstring_in_header(
                                        header_path, function_name, parent_class)
                            
                            # If still no docstring and it's a standalone function, try header lookup
                            if not docstring and not parent_class and file_path.suffix.lower() in ['.cc', '.cpp', '.cxx']:
                                header_path = self._find_header_file(file_path)
                                if header_path and header_path.exists():
                                    docstring = self._find_function_docstring_in_header(
                                        header_path, function_name, None)
                                
                            # For class methods, get class docstring as parent summary
                            parent_summary = file_summary
                            
                            # If we have a class name (either from scope or AST), try to get class info
                            if parent_class:
                                class_docstring = None
                                
                                # Check if we already have class info
                                if parent_class in class_info:
                                    class_doc = class_info[parent_class]['docstring']
                                    if class_doc:
                                        class_docstring = class_doc
                                
                                # If no class docstring yet, try to find it in the header file
                                if not class_docstring and file_path.suffix.lower() in ['.cc', '.cpp', '.cxx']:
                                    header_path = self._find_header_file(file_path)
                                    if header_path and header_path.exists():
                                        class_summary = self._extract_class_summary_from_header(
                                            header_path, parent_class)
                                        if class_summary:
                                            class_docstring = class_summary
                                            # Cache it for later use
                                            class_docs[parent_class] = class_summary
                                
                                # If we found a class docstring, use it instead of file summary
                                if class_docstring and class_docstring.strip():
                                    parent_summary = class_docstring
                                    if self.verbose:
                                        print(f"  Using class docstring for {parent_class}::{function_name}")
                                        print(f"  {class_docstring[:50]}..." if len(class_docstring) > 50 else f"  {class_docstring}")
                                            
                            # Ensure parent_summary is never just a copyright notice
                            if parent_summary and ("copyright" in parent_summary.lower() or "license" in parent_summary.lower()) and len(parent_summary) < 100:
                                parent_summary = ""
                                    
                            chunks.append(CodeChunk(
                                content=content,
                                file_path=file_path,
                                start_line=start_line,
                                end_line=end_line,
                                function_name=function_name,
                                class_name=parent_class,
                                namespace=namespace,
                                docstring=docstring,
                                parent_summary=parent_summary
                            ))

            # Fallback: whole file if no structured elements found
            if not chunks:
                content, start_line, end_line = self._extract_node_info(tree.root_node, source_bytes)
                # Include only if file has enough lines
                if end_line - start_line + 1 >= self.min_lines:
                    chunks.append(CodeChunk(
                        content=content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        parent_summary=file_summary
                    ))

        except Exception as e:
            if self.verbose:
                print(f"Error parsing {file_path}: {e}")
            
            # Fallback to simple file read
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Try to extract a meaningful file summary from comments
                file_summary = None
                lines = content.splitlines()[:50]  # Look at more lines
                
                # Extract all comment lines
                comment_lines = []
                in_multiline = False
                doc_comment_lines = []
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    
                    # Handle multiline comments
                    if '/*' in line and '*/' in line:
                        # Single line comment block
                        comment_lines.append(line)
                    elif '/*' in line:
                        # Start of multiline comment
                        in_multiline = True
                        comment_lines.append(line)
                        if '/**' in line:  # Documentation comment
                            doc_comment_lines.append(line)
                    elif '*/' in line and in_multiline:
                        # End of multiline comment
                        comment_lines.append(line)
                        in_multiline = False
                        if doc_comment_lines:  # If we were collecting doc comments
                            doc_comment_lines.append(line)
                    elif in_multiline:
                        # Middle of multiline comment
                        comment_lines.append(line)
                        if doc_comment_lines:  # If we were collecting doc comments
                            doc_comment_lines.append(line)
                    elif line.startswith('//'):
                        # Single line comment
                        comment_lines.append(line)
                
                # Filter out copyright notices and prioritize documentation comments
                copyright_pattern = re.compile(r'copyright|license|\(c\)|©|all rights reserved', re.IGNORECASE)
                
                # Clean up comments
                doc_comments = []
                for line in doc_comment_lines:
                    cleaned = re.sub(r'^/\*+|\*+/$', '', line)
                    cleaned = re.sub(r'^\s*\*\s*', '', cleaned)
                    if cleaned and not copyright_pattern.search(cleaned):
                        doc_comments.append(cleaned)
                
                # Clean regular comments if no doc comments
                regular_comments = []
                if not doc_comments:
                    for line in comment_lines:
                        cleaned = re.sub(r'^/\*+|\*+/$', '', line)
                        cleaned = re.sub(r'^//+\s*', '', cleaned)
                        cleaned = cleaned.strip()
                        if cleaned and not copyright_pattern.search(cleaned) and len(cleaned) > 10:
                            regular_comments.append(cleaned)
                
                # Use doc comments if available, otherwise use regular comments
                if doc_comments:
                    file_summary = "\n".join(doc_comments)
                elif regular_comments:
                    file_summary = "\n".join(regular_comments)
                else:
                    # Fallback to first comment if nothing else available
                    file_summary = "\n".join([re.sub(r'^/\*+|\*+/$|^//+\s*', '', line.strip()) 
                                             for line in comment_lines[:3] if line.strip()])
                    
                if content.strip():
                    chunks.append(CodeChunk(
                        content=content,
                        file_path=file_path,
                        start_line=1,
                        end_line=len(content.splitlines()),
                        parent_summary=file_summary
                    ))
            except Exception as read_error:
                if self.verbose:
                    print(f"Could not read file {file_path}: {read_error}")

        return chunks

    def _find_header_file(self, source_file_path: Path) -> Optional[Path]:
        """Find corresponding header file for a source file."""
        # Map of source extensions to header extensions
        source_to_header = {
            '.cpp': ['.h', '.hpp'],
            '.cc': ['.h', '.hpp'],
            '.cxx': ['.h', '.hpp'],
            '.c': ['.h']
        }
        
        # Check if this is a source file
        source_ext = source_file_path.suffix.lower()
        if source_ext not in source_to_header:
            return None
            
        # Get all possible header extensions for this source file
        header_exts = source_to_header[source_ext]
                
        # Try to find header file in the same directory
        for header_ext in header_exts:
            header_path = source_file_path.with_suffix(header_ext)
            if header_path.exists():
                return header_path
                
        # Try to find header in sibling include/headers directory
        parent_dir = source_file_path.parent
        stem = source_file_path.stem
        
        for header_ext in header_exts:
            # Check in "../include" directory
            include_path = parent_dir.parent / "include" / f"{stem}{header_ext}"
            if include_path.exists():
                return include_path
                
            # Check in "../headers" directory
            headers_path = parent_dir.parent / "headers" / f"{stem}{header_ext}"
            if headers_path.exists():
                return headers_path
                
        return None
        
    def _extract_class_summary_from_header(self, header_path: Path, class_name: str) -> Optional[str]:
        """Extract the docstring for a specific class from a header file."""
        try:
            with open(header_path, 'rb') as f:
                source_bytes = f.read()
                
            tree = self.parser.parse(source_bytes)
            source_text = source_bytes.decode('utf-8', errors='ignore')
            source_lines = source_text.splitlines()
            
            # Get file-level documentation
            file_summary = self._extract_file_docstring(source_bytes, tree)
            
            # Find class definition
            class_def_pattern = re.compile(r'class\s+' + re.escape(class_name) + r'\s*[:{]')
            class_line = -1
            
            for i, line in enumerate(source_lines):
                if class_def_pattern.search(line):
                    class_line = i
                    break
            
            # Look for comments before class definition
            if class_line > 0:
                # Look for continuous comments above the class
                comment_lines = []
                line_idx = class_line - 1
                
                # Go back until we find non-comment lines
                while line_idx >= 0:
                    line = source_lines[line_idx].strip()
                    
                    # Skip empty lines
                    if not line:
                        line_idx -= 1
                        continue
                    
                    if line.startswith('///') or line.startswith('//') or line.startswith('/*') or line.startswith('*'):
                        # Clean up comment markers
                        cleaned = re.sub(r'^///\s*|^//\s*|^/\*+\s*|\*+/\s*|^\*\s*', '', line)
                        comment_lines.insert(0, cleaned)
                        line_idx -= 1
                    else:
                        break
                
                if comment_lines:
                    # Join comments and clean up
                    docstring = '\n'.join(comment_lines).strip()
                    if docstring and len(docstring) > 10:
                        return docstring
            
            # If no direct class comment found, use file summary
            if file_summary:
                return file_summary
            
            return None
                                
        except Exception as e:
            if self.verbose:
                print(f"Error extracting class summary from {header_path}: {e}")
            return None
    
    def _extract_function_declarations_from_class(self, class_node, source_bytes: bytes, source_lines: List[str], 
                                                  file_path: Path, class_name: str, namespace: Optional[str], 
                                                  class_summary: str, chunks: List[CodeChunk]):
        """Extract individual function declarations from within a class node in header files."""
        # Get the class content
        class_content = class_node.text.decode('utf-8', errors='ignore')
        class_start_line = class_node.start_point[0]
        
        # Split class content into lines
        class_lines = class_content.split('\n')
        
        # Pattern to match function declarations with optional docstrings
        func_decl_pattern = re.compile(r'^\s*(?:virtual\s+)?(?:static\s+)?(?:inline\s+)?(?:explicit\s+)?'
                                      r'(?:const\s+)?(?:[\w:]+\s+)?'  # return type
                                      r'(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?;?\s*$')
        
        i = 0
        while i < len(class_lines):
            line = class_lines[i].strip()
            
            # Skip empty lines, comments, and access specifiers
            if not line or line.startswith('//') or line.startswith('/*') or \
               line in ['public:', 'private:', 'protected:', '{', '}']:
                i += 1
                continue
            
            # Check if this line looks like a function declaration
            func_match = func_decl_pattern.match(line)
            if func_match:
                function_name = func_match.group(1)
                
                # Skip constructors, destructors, and operators for now
                if function_name == class_name or function_name.startswith('~') or function_name == 'operator':
                    i += 1
                    continue
                
                # Look for preceding docstring
                docstring = None
                docstring_lines = []
                
                # Look backwards for comments
                j = i - 1
                while j >= 0:
                    prev_line = class_lines[j].strip()
                    if not prev_line:
                        j -= 1
                        continue
                    
                    if prev_line.startswith('///'):
                        comment_text = re.sub(r'^///\s*', '', prev_line)
                        docstring_lines.insert(0, comment_text)
                        j -= 1
                    elif prev_line.startswith('//'):
                        comment_text = re.sub(r'^//\s*', '', prev_line)
                        docstring_lines.insert(0, comment_text)
                        j -= 1
                    else:
                        break
                
                if docstring_lines:
                    docstring = '\n'.join(docstring_lines).strip()
                
                # Calculate line numbers in the original file
                func_line_in_file = class_start_line + i + 1  # +1 for 1-based indexing
                
                # Create a chunk for this function declaration
                if docstring:  # Only create chunks for functions with docstrings
                    chunks.append(CodeChunk(
                        content=line,
                        file_path=file_path,
                        start_line=func_line_in_file,
                        end_line=func_line_in_file,
                        function_name=function_name,
                        class_name=class_name,
                        namespace=namespace,
                        docstring=docstring,
                        parent_summary=class_summary
                    ))
                    
                    if self.verbose:
                        print(f"  Found function declaration: {class_name}::{function_name} with docstring: {docstring[:30]}...")
            
            i += 1
            
    def _find_function_docstring_in_header(self, header_path: Path, function_name: str, class_name: Optional[str] = None) -> Optional[str]:
        """Find docstring for a specific function in a header file."""
        if self.verbose:
            print(f"Looking for docstring for {function_name} in {header_path}")
            
        try:
            with open(header_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_lines = f.readlines()
            
            # Create pattern to match the function declaration
            if class_name:
                # For class methods, look for just the method name (without class prefix)
                # since header files typically don't include the class scope in method declarations
                # Look for function declarations, not function calls - must start with return type
                method_pattern = re.compile(rf'^\s*(virtual\s+)?(\w+|\w+\s*\*|\w+\s*&)\s+{re.escape(function_name)}\s*\(')
            else:
                # For standalone functions
                func_pattern = re.compile(rf'^\s*(virtual\s+)?(\w+|\w+\s*\*|\w+\s*&)\s+{re.escape(function_name)}\s*\(')
            
            # Search for the function declaration
            for i, line in enumerate(source_lines):
                line_stripped = line.strip()
                
                # Check if this line contains our function
                if class_name:
                    if method_pattern.search(line_stripped):
                        # Found the method, now look for preceding docstring
                        return self._extract_preceding_comment(source_lines, i)
                else:
                    if func_pattern.search(line_stripped):
                        # Found the function, now look for preceding docstring
                        return self._extract_preceding_comment(source_lines, i)
            
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"Error finding function docstring in {header_path}: {e}")
            return None
    
    def _extract_preceding_comment(self, source_lines: List[str], function_line: int) -> Optional[str]:
        """Extract comment block that precedes a function declaration."""
        docstring_lines = []
        
        # Look backwards from the function line to find documentation
        for i in range(function_line - 1, max(-1, function_line - 10), -1):
            if i < 0:
                break
                
            line = source_lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for various comment patterns
            if line.startswith('///'):
                # Doxygen triple slash comment
                comment_text = re.sub(r'^///\s*', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('//'):
                # Regular double slash comment
                comment_text = re.sub(r'^//\s*', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('/*') and line.endswith('*/'):
                # Single line /* ... */ comment
                comment_text = re.sub(r'^/\*\s*|\s*\*/$', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('/**') and line.endswith('*/'):
                # Single line /** ... */ comment
                comment_text = re.sub(r'^/\*\*\s*|\s*\*/$', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('*'):
                # Continuation of multi-line comment
                comment_text = re.sub(r'^\*\s*', '', line)
                docstring_lines.insert(0, comment_text)
            elif line.startswith('/*') or line.startswith('/**'):
                # Start of multi-line comment - need to look for the end
                comment_start = i
                comment_end = i
                
                # Find the end of the multi-line comment
                while comment_end < len(source_lines) and '*/' not in source_lines[comment_end]:
                    comment_end += 1
                
                if comment_end < len(source_lines):
                    # Extract the entire multi-line comment
                    comment_block = []
                    for j in range(comment_start, comment_end + 1):
                        comment_line = source_lines[j].strip()
                        # Clean up comment markers
                        if j == comment_start:
                            comment_line = re.sub(r'^/\*+\s*', '', comment_line)
                        if j == comment_end:
                            comment_line = re.sub(r'\s*\*+/$', '', comment_line)
                        if comment_line.startswith('*'):
                            comment_line = re.sub(r'^\*\s*', '', comment_line)
                        
                        if comment_line:  # Only add non-empty lines
                            comment_block.append(comment_line)
                    
                    # Add this block to the beginning of docstring_lines
                    docstring_lines = comment_block + docstring_lines
                    break  # Found a multi-line comment, stop looking
                else:
                    break  # Unterminated comment, stop looking
            else:
                # Hit a non-comment line, stop looking
                break
        
        # Clean up and return the docstring
        if docstring_lines:
            # Remove empty lines at the beginning and end
            while docstring_lines and not docstring_lines[0].strip():
                docstring_lines.pop(0)
            while docstring_lines and not docstring_lines[-1].strip():
                docstring_lines.pop()
            
            if docstring_lines:
                docstring = "\n".join(docstring_lines).strip()
                return docstring if docstring else None
        
        return None
        
    def _should_ignore(self, path: str) -> bool:
        """Check if path should be ignored based on patterns."""
        return any(pattern in path for pattern in self.IGNORE_PATTERNS)

    def parse_repository(self, repo_path: Path) -> List[Dict]:
        """Parse all C++ files in repository and return code chunks."""
        chunks = []
        cpp_files = []
        
        # Find all C++ files in repository
        for root, _, files in os.walk(repo_path):
            if self._should_ignore(str(root)):
                continue
                
            for filename in files:
                if not self._should_ignore(filename) and Path(filename).suffix in self.CPP_EXTENSIONS:
                    file_path = Path(root) / filename
                    cpp_files.append(file_path)
        
        print(f"Found {len(cpp_files)} C++ files to process in {repo_path}")
        
        # Process files with progress bar
        with tqdm(total=len(cpp_files), desc="Processing files", disable=not self.verbose) as pbar:
            for file_path in cpp_files:
                try:
                    file_chunks = self.parse_file(file_path)
                    chunks.extend([chunk.to_dict() for chunk in file_chunks])
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to process {file_path}: {e}")
                
                pbar.update(1)

        print(f"Processed {len(cpp_files)} files, created {len(chunks)} code chunks")
        return chunks


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Parse C++ codebase into structured chunks")
    parser.add_argument(
        '--repo-path', 
        type=Path,
        default=Path('manuel_natcom/src/sim/hands'),
        help='Path to repository root (default: manuel_natcom/src/sim/hands)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/code_chunks_clean.json'),
        help='Output file path (default: data/code_chunks_clean.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--min-lines',
        type=int,
        default=3,
        help='Minimum number of lines per chunk (default: 3)'
    )
    parser.add_argument(
        '--context-window',
        type=int,
        default=5,
        help='Lines of context to check for docstrings (default: 5)'
    )
    
    args = parser.parse_args()
    
    if not args.repo_path.exists():
        print(f"Error: Repository path '{args.repo_path}' does not exist.")
        return 1

    try:
        # Initialize parser and process repository
        cpp_parser = CppParser(
            verbose=args.verbose, 
            min_lines=args.min_lines, 
            context_window=args.context_window
        )
        chunks = cpp_parser.parse_repository(args.repo_path)
        
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2)
            
            print(f"Saved {len(chunks)} chunks to {args.output}")
        except IOError as e:
            print(f"Error writing output file: {e}")
            return 1
        
        # Show sample output if verbose
        if chunks and args.verbose:
            print(f"\n--- Sample of {min(2, len(chunks))} Chunks ---")
            for i, chunk in enumerate(chunks[:2]):
                print(f"Chunk {i+1}: {chunk['id']}")
                print(f"  Function: {chunk['function_name'] or 'N/A'}")
                print(f"  Class: {chunk['class_name'] or 'N/A'}")
                print(f"  Lines: {chunk['start_line']}-{chunk['end_line']}")
                print(f"  Content preview: {chunk['content'][:80]}...")
                print("-" * 40)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
