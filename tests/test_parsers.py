"""Tests for the parser modules."""

import tempfile
from pathlib import Path

import pytest


class TestCppParser:
    def test_parse_cpp_file(self, sample_cpp_code, tmp_path):
        from core.parsers.cpp_parser import CppParser

        cpp_file = tmp_path / "test.cpp"
        cpp_file.write_text(sample_cpp_code)

        parser = CppParser(verbose=False, min_lines=2)
        chunks = parser.parse_file(cpp_file)

        assert len(chunks) > 0
        assert all(c.language == "cpp" for c in chunks)

    def test_cpp_file_extensions(self):
        from core.parsers.cpp_parser import CppParser

        parser = CppParser()
        assert '.cpp' in parser.file_extensions
        assert '.h' in parser.file_extensions
        assert '.hpp' in parser.file_extensions

    def test_cpp_unsupported_file(self, tmp_path):
        from core.parsers.cpp_parser import CppParser

        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')")

        parser = CppParser()
        assert not parser.is_supported_file(py_file)

    def test_empty_file(self, tmp_path):
        from core.parsers.cpp_parser import CppParser

        empty_file = tmp_path / "empty.cpp"
        empty_file.write_text("")

        parser = CppParser()
        chunks = parser.parse_file(empty_file)
        assert chunks == []


class TestPythonParser:
    def test_parse_python_file(self, sample_python_code, tmp_path):
        try:
            from core.parsers.python_parser import PythonParser
        except ImportError:
            pytest.skip("tree-sitter-python not installed")

        py_file = tmp_path / "test.py"
        py_file.write_text(sample_python_code)

        parser = PythonParser(verbose=False, min_lines=2)
        chunks = parser.parse_file(py_file)

        assert len(chunks) > 0
        assert all(c.language == "python" for c in chunks)

        # Should find the Calculator class and fibonacci function
        names = [c.function_name or c.class_name for c in chunks]
        assert "Calculator" in names or "fibonacci" in names

    def test_python_docstring_extraction(self, tmp_path):
        try:
            from core.parsers.python_parser import PythonParser
        except ImportError:
            pytest.skip("tree-sitter-python not installed")

        code = '''
def greet(name):
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
        py_file = tmp_path / "greet.py"
        py_file.write_text(code)

        parser = PythonParser(verbose=False, min_lines=2)
        chunks = parser.parse_file(py_file)

        assert len(chunks) >= 1
        func_chunk = [c for c in chunks if c.function_name == "greet"]
        assert len(func_chunk) == 1
        assert func_chunk[0].docstring == "Say hello to someone."

    def test_python_file_extensions(self):
        try:
            from core.parsers.python_parser import PythonParser
        except ImportError:
            pytest.skip("tree-sitter-python not installed")

        parser = PythonParser()
        assert '.py' in parser.file_extensions
        assert '.pyi' in parser.file_extensions


class TestParserFactory:
    def test_list_parsers(self):
        from core.parsers.parser_factory import ParserFactory

        parsers = ParserFactory.list_available_parsers()
        assert 'cpp' in parsers

    def test_get_cpp_parser(self):
        from core.parsers.parser_factory import ParserFactory

        parser = ParserFactory.get_parser('cpp')
        assert parser.language_name == 'cpp'

    def test_unsupported_language(self):
        from core.parsers.parser_factory import ParserFactory

        with pytest.raises(ValueError, match="Unsupported language"):
            ParserFactory.get_parser('fortran')

    def test_detect_language(self):
        from core.parsers.parser_factory import ParserFactory

        assert ParserFactory.detect_language(Path("test.cpp")) == 'cpp'
        assert ParserFactory.detect_language(Path("test.h")) == 'cpp'
        assert ParserFactory.detect_language(Path("test.rs")) is None


class TestMultiLanguageParser:
    def test_parse_directory(self, sample_cpp_code, tmp_path):
        from core.parsers.parser_factory import MultiLanguageParser

        cpp_file = tmp_path / "test.cpp"
        cpp_file.write_text(sample_cpp_code)

        # Also write a non-supported file
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("This is a readme")

        parser = MultiLanguageParser(verbose=False, min_lines=2)
        chunks = parser.parse_directory(tmp_path)

        assert len(chunks) > 0
        assert all(c.language == "cpp" for c in chunks)
