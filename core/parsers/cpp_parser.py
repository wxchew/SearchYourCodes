"""
C++ Parser Implementation

This module provides a C++ language parser using Tree-sitter for robust AST analysis.
It implements the BaseParser interface and extracts functions, classes, and documentation
from C++ source files.

Features:
- Tree-sitter based C++ parsing
- Function and class extraction
- Documentation parsing
- Header/implementation file correlation
- Rich context generation
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import tree_sitter_cpp
from tree_sitter import Language, Parser, Query, QueryCursor

from .base_parser import BaseParser, CodeChunk, ParsingError


class CppParser(BaseParser):
    """C++ parser implementation using tree-sitter."""

    @property
    def language_name(self) -> str:
        """Return the language name for this parser."""
        return "cpp"

    @property
    def file_extensions(self) -> Set[str]:
        """Return the set of file extensions supported by this parser."""
        return {'.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx'}
    
    @property
    def ignore_patterns(self) -> List[str]:
        """Return list of patterns to ignore during parsing."""
        return [
            '.git', '.venv', '__pycache__', 'node_modules', '.DS_Store',
            'build/', 'dist/', '.egg-info', '.pytest_cache', '.idea',
            '*.min.js', '*.css.map', '*.wasm', '.inc'
        ]

    def __init__(self, verbose: bool = False, min_lines: int = 3, context_window: int = 5):
        """Initialize the C++ parser."""
        super().__init__(verbose, min_lines, context_window)
        
        # Load tree-sitter library and language
        try:
            language_lib = tree_sitter_cpp.language()
            self.language = Language(language_lib)
            self.parser = Parser()
            self.parser.language = self.language
            
        except ImportError:
            raise ImportError("tree-sitter-cpp is required for C++ parsing. Install with: pip install tree-sitter-cpp")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize C++ parser: {e}")

    def _execute_query(self, tree, query_string: str):
        """Helper to execute a tree-sitter query and return captures."""
        try:
            query = Query(self.language, query_string)
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)
            
            # Extract all nodes from all matches - modern API format
            nodes = []
            for pattern_index, captures_dict in matches:
                for capture_name, node_list in captures_dict.items():
                    for node in node_list:
                        nodes.append((node, capture_name))
            
            return nodes
        except Exception as e:
            if self.verbose:
                print(f"Error executing query '{query_string[:30]}...': {e}")
            return []

    def is_supported_file(self, file_path: Path) -> bool:
        """
        Check if a file is supported by this parser.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the file can be parsed by this parser
        """
        if not file_path.is_file():
            return False
        
        # Check extension
        if file_path.suffix.lower() not in self.file_extensions:
            return False
        
        # Check ignore patterns
        path_str = str(file_path)
        for pattern in self.ignore_patterns:
            if pattern in path_str:
                return False
        
        return True

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Parse a single C++ file and extract code chunks.
        
        Args:
            file_path: Path to the C++ source file
            
        Returns:
            List of CodeChunk objects extracted from the file
            
        Raises:
            ParsingError: If the file cannot be parsed
        """
        if not self.is_supported_file(file_path):
            raise ParsingError(f"File not supported: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                if self.verbose:
                    print(f"Warning: Empty file {file_path}")
                return []
            
            # Parse with Tree-sitter
            tree = self.parser.parse(content.encode('utf-8'))
            
            if not tree.root_node:
                if self.verbose:
                    print(f"Warning: Failed to parse AST for {file_path}")
                return []
            
            # Extract chunks
            chunks = []
            
            # Extract function chunks
            chunks.extend(self._extract_functions(tree, content, file_path))
            
            # Extract class chunks
            chunks.extend(self._extract_classes(tree, content, file_path))
            
            # Filter chunks by minimum size
            chunks = [chunk for chunk in chunks if (chunk.end_line - chunk.start_line + 1) >= self.min_lines]
            
            if self.verbose:
                print(f"Extracted {len(chunks)} chunks from {file_path}")
            
            return chunks
            
        except UnicodeDecodeError as e:
            raise ParsingError(f"Encoding error: {e}", file_path)
        except IOError as e:
            raise ParsingError(f"File I/O error: {e}", file_path)
        except Exception as e:
            raise ParsingError(f"Failed to parse file: {e}", file_path)

    def _extract_functions(self, tree, content: str, file_path: Path) -> List[CodeChunk]:
        """Extract function definitions from the parsed tree."""
        chunks = []
        lines = content.split('\n')
        
        query_string = '''
            (function_definition) @function.def
            (template_declaration (function_definition) @function.def)
        '''
        captures = self._execute_query(tree, query_string)
        
        for node, capture_name in captures:
            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                function_name = self._extract_function_name(node)
                
                # Extract class name from the function name if it's a qualified name
                class_name = None
                if function_name and '::' in self._get_node_text(node, content):
                    # This is likely a member function
                    class_name = self._extract_class_from_qualified_name(node, content)
                
                namespace = self._get_namespace_context(node, content)
                function_content = self._get_node_text(node, content)
                docstring = self._extract_docstring(node, lines, start_line)
                
                chunk = CodeChunk(
                    content=function_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language_name,
                    function_name=function_name,
                    class_name=class_name,
                    namespace=namespace,
                    docstring=docstring
                )
                chunks.append(chunk)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing function capture: {e}")
                continue
        
        return chunks

    def _extract_classes(self, tree, content: str, file_path: Path) -> List[CodeChunk]:
        """Extract class definitions from the parsed tree."""
        chunks = []
        lines = content.split('\n')
        
        query_string = '(class_specifier) @class.def'
        captures = self._execute_query(tree, query_string)
        
        for node, capture_name in captures:
            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                class_name = self._extract_class_name(node)
                namespace = self._get_namespace_context(node, content)
                class_content = self._get_node_text(node, content)
                docstring = self._extract_docstring(node, lines, start_line)
                
                chunk = CodeChunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language_name,
                    class_name=class_name,
                    namespace=namespace,
                    docstring=docstring
                )
                chunks.append(chunk)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing class capture: {e}")
                continue
        
        return chunks

    def _extract_function_name(self, node) -> Optional[str]:
        """Extract function name from a function definition node."""
        try:
            # For function_definition nodes, look for the declarator
            for child in node.children:
                if child.type == 'function_declarator':
                    # Look for identifier in function_declarator
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            return subchild.text.decode('utf-8')
                        elif subchild.type == 'qualified_identifier':
                            # Handle qualified names like Bridge::Bridge
                            for identifier in subchild.children:
                                if identifier.type == 'identifier':
                                    # Return the last identifier (the actual function name)
                                    pass
                            # Get the last identifier
                            identifiers = [c for c in subchild.children if c.type == 'identifier']
                            if identifiers:
                                return identifiers[-1].text.decode('utf-8')
                        elif subchild.type == 'destructor_name':
                            # Handle destructors like ~Bridge
                            return subchild.text.decode('utf-8')
                elif child.type == 'template_declaration':
                    # For template functions, recurse into the declaration
                    for template_child in child.children:
                        if template_child.type == 'function_definition':
                            return self._extract_function_name(template_child)
            
            # Alternative approach: look in the node text for function patterns
            node_text = node.text.decode('utf-8') if hasattr(node, 'text') else ''
            if '::~' in node_text:
                # This is a destructor
                parts = node_text.split('::~')
                if len(parts) > 1:
                    destructor_part = parts[1].split('(')[0].strip()
                    return f'~{destructor_part}'
            
            # Fallback: look for any identifier that could be the function name
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
                    
        except Exception as e:
            if self.verbose:
                print(f"Error extracting function name: {e}")
        return None

    def _extract_class_name(self, node) -> Optional[str]:
        """Extract class name from a class definition node."""
        try:
            # Look for type_identifier in class_specifier
            for child in node.children:
                if child.type == 'type_identifier':
                    return child.text.decode('utf-8')
        except Exception:
            pass
        return None

    def _extract_class_from_qualified_name(self, node, content: str) -> Optional[str]:
        """Extract class name from qualified function names like Bridge::Bridge."""
        try:
            node_text = self._get_node_text(node, content)
            # Look for patterns like ClassName::functionName
            if '::' in node_text:
                # Find the first line to get the function signature
                first_line = node_text.split('\n')[0]
                if '::' in first_line:
                    # Extract the class name before ::
                    parts = first_line.split('::')
                    if len(parts) >= 2:
                        # Get the part before ::, removing any return type
                        class_part = parts[0].strip()
                        # Remove return type if present (e.g., "Vector Bridge" -> "Bridge")
                        class_tokens = class_part.split()
                        if class_tokens:
                            return class_tokens[-1]  # Last token is usually the class name
        except Exception:
            pass
        return None

    def _get_namespace_context(self, node, content: str) -> Optional[str]:
        """Get namespace context for a node."""
        try:
            # Walk up the tree to find namespace_definition
            current = node.parent
            namespaces = []
            
            while current:
                if current.type == 'namespace_definition':
                    # Extract namespace name
                    for child in current.children:
                        if child.type == 'identifier':
                            namespaces.append(child.text.decode('utf-8'))
                            break
                current = current.parent
            
            if namespaces:
                return '::'.join(reversed(namespaces))
        except Exception:
            pass
        return None

    def _get_class_context(self, node, content: str) -> Optional[str]:
        """Get class context for a function node."""
        try:
            # Walk up the tree to find class_specifier
            current = node.parent
            
            while current:
                if current.type == 'class_specifier':
                    return self._extract_class_name(current)
                current = current.parent
        except Exception:
            pass
        return None

    def _get_node_text(self, node, content: str) -> str:
        """Extract text content from a tree-sitter node."""
        try:
            start_byte = node.start_byte
            end_byte = node.end_byte
            return content[start_byte:end_byte]
        except Exception:
            return ""

    def _extract_docstring(self, node, lines: List[str], start_line: int) -> Optional[str]:
        """Extract documentation/comments for a function or class."""
        try:
            # Look for comments before the function/class definition
            comments = []
            
            # Check lines before the definition within context window
            for i in range(max(0, start_line - self.context_window - 1), start_line - 1):
                line = lines[i].strip()
                
                # C++ style comments
                if line.startswith('//'):
                    comments.append(line[2:].strip())
                # C style comments (single line)
                elif line.startswith('/*') and line.endswith('*/'):
                    comments.append(line[2:-2].strip())
                # Doxygen style comments
                elif line.startswith('///') or line.startswith('/**'):
                    comments.append(line[3:].strip() if line.startswith('///') else line[3:-2].strip())
                # Empty line - keep collecting if we have comments
                elif line == '' and comments:
                    continue
                # Non-comment line - stop if we have comments
                elif comments:
                    break
            
            if comments:
                return '\n'.join(comments)
        except Exception:
            pass
        return None
