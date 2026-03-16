"""
Python Parser Implementation

Tree-sitter based Python parser that extracts functions, classes, and methods
from Python source files. Implements the BaseParser interface.
"""

from pathlib import Path
from typing import List, Optional, Set

try:
    import tree_sitter_python
except ImportError:
    tree_sitter_python = None

from tree_sitter import Language, Parser, Query, QueryCursor

from .base_parser import BaseParser, CodeChunk, ParsingError


class PythonParser(BaseParser):
    """Python parser implementation using tree-sitter."""

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> Set[str]:
        return {'.py', '.pyx', '.pyi'}

    @property
    def ignore_patterns(self) -> List[str]:
        return [
            '.git', '.venv', 'venv', '__pycache__', 'node_modules',
            '.DS_Store', 'build/', 'dist/', '.egg-info', '.pytest_cache',
            '.tox', '.mypy_cache', 'site-packages',
        ]

    def __init__(self, verbose: bool = False, min_lines: int = 3, context_window: int = 5):
        super().__init__(verbose, min_lines, context_window)

        if tree_sitter_python is None:
            raise ImportError(
                "tree-sitter-python is required for Python parsing. "
                "Install with: pip install tree-sitter-python"
            )

        try:
            language_lib = tree_sitter_python.language()
            self.language = Language(language_lib)
            self.parser = Parser()
            self.parser.language = self.language
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Python parser: {e}")

    def _execute_query(self, tree, query_string: str):
        """Execute a tree-sitter query and return captures."""
        try:
            query = Query(self.language, query_string)
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)
            nodes = []
            for pattern_index, captures_dict in matches:
                for capture_name, node_list in captures_dict.items():
                    for node in node_list:
                        nodes.append((node, capture_name))
            return nodes
        except Exception as e:
            if self.verbose:
                print(f"Error executing query: {e}")
            return []

    def is_supported_file(self, file_path: Path) -> bool:
        if not file_path.is_file():
            return False
        if file_path.suffix.lower() not in self.file_extensions:
            return False
        path_str = str(file_path)
        for pattern in self.ignore_patterns:
            if pattern in path_str:
                return False
        return True

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        if not self.is_supported_file(file_path):
            raise ParsingError(f"File not supported: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return []

            tree = self.parser.parse(content.encode('utf-8'))
            if not tree.root_node:
                return []

            chunks = []
            chunks.extend(self._extract_functions(tree, content, file_path))
            chunks.extend(self._extract_classes(tree, content, file_path))

            # Filter by minimum size
            chunks = [c for c in chunks if (c.end_line - c.start_line + 1) >= self.min_lines]

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
        """Extract top-level and nested function definitions."""
        chunks = []
        lines = content.split('\n')

        query_string = '(function_definition) @function.def'
        captures = self._execute_query(tree, query_string)

        for node, capture_name in captures:
            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                function_name = self._extract_function_name(node)
                class_name = self._get_enclosing_class(node)
                func_content = self._get_node_text(node, content)
                docstring = self._extract_docstring_from_body(node, content)

                chunks.append(CodeChunk(
                    content=func_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language_name,
                    function_name=function_name,
                    class_name=class_name,
                    docstring=docstring,
                ))
            except Exception as e:
                if self.verbose:
                    print(f"Error processing function: {e}")
                continue

        return chunks

    def _extract_classes(self, tree, content: str, file_path: Path) -> List[CodeChunk]:
        """Extract class definitions (the class header + body, not individual methods)."""
        chunks = []
        lines = content.split('\n')

        query_string = '(class_definition) @class.def'
        captures = self._execute_query(tree, query_string)

        for node, capture_name in captures:
            try:
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                class_name = self._extract_class_name(node)
                class_content = self._get_node_text(node, content)
                docstring = self._extract_docstring_from_body(node, content)

                chunks.append(CodeChunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language_name,
                    class_name=class_name,
                    docstring=docstring,
                ))
            except Exception as e:
                if self.verbose:
                    print(f"Error processing class: {e}")
                continue

        return chunks

    def _extract_function_name(self, node) -> Optional[str]:
        """Extract function name from a function_definition node."""
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
        return None

    def _extract_class_name(self, node) -> Optional[str]:
        """Extract class name from a class_definition node."""
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
        return None

    def _get_enclosing_class(self, node) -> Optional[str]:
        """Walk up the tree to find an enclosing class_definition."""
        current = node.parent
        while current:
            if current.type == 'class_definition':
                return self._extract_class_name(current)
            current = current.parent
        return None

    def _get_node_text(self, node, content: str) -> str:
        """Extract text content from a tree-sitter node."""
        try:
            return content[node.start_byte:node.end_byte]
        except Exception:
            return ""

    def _extract_docstring_from_body(self, node, content: str) -> Optional[str]:
        """Extract docstring from the first expression_statement in a function/class body."""
        try:
            # Find the block (body) child
            for child in node.children:
                if child.type == 'block':
                    # First child of block might be expression_statement with string
                    if child.children:
                        first_stmt = child.children[0]
                        if first_stmt.type == 'expression_statement':
                            for expr_child in first_stmt.children:
                                if expr_child.type == 'string':
                                    raw = self._get_node_text(expr_child, content)
                                    # Strip triple quotes
                                    if raw.startswith('"""') or raw.startswith("'''"):
                                        return raw[3:-3].strip()
                                    elif raw.startswith('"') or raw.startswith("'"):
                                        return raw[1:-1].strip()
        except Exception:
            pass
        return None
