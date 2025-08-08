"""
Keyword Search Module

This module provides keyword-based search functionality with fuzzy matching
capabilities for code search applications.

Features:
- Exact text matching
- Fuzzy search using fuzzywuzzy
- File content parsing and indexing
- Configurable scoring and context extraction
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: fuzzywuzzy not available. Fuzzy matching disabled.")

# Local imports with fallback
try:
    from config import resolve_codebase_path, get_file_extensions
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        from core.config import resolve_codebase_path, get_file_extensions
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False
        print("Warning: config.py not found, using default settings")


@dataclass
class KeywordMatch:
    """Represents a keyword search match with associated metadata."""
    file_path: str
    line_number: int
    matched_text: str
    context: str
    score: float
    matched_keywords: List[str]
    
    def __post_init__(self):
        """Ensure score is within valid range."""
        self.score = max(0.0, min(1.0, self.score))
    
    def to_dict(self) -> Dict:
        """Convert match to dictionary format."""
        return {
            'file_path': self.file_path,
            'line_number': self.line_number,
            'matched_text': self.matched_text,
            'context': self.context,
            'score': self.score,
            'matched_keywords': self.matched_keywords
        }


class KeywordSearchEngine:
    """
    Keyword-based search engine with fuzzy matching capabilities.
    
    This engine provides exact text matching and fuzzy search functionality
    for code files, with configurable scoring and context extraction.
    """
    
    def __init__(self, verbose: bool = False, fuzzy_threshold: int = 80, context_lines: int = 2):
        """
        Initialize the keyword search engine.
        
        Args:
            verbose: Enable verbose logging
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            context_lines: Number of context lines to include around matches
        """
        self.verbose = verbose
        self.fuzzy_threshold = fuzzy_threshold
        self.context_lines = context_lines
        
        # Set default file extensions if config is not available
        if CONFIG_AVAILABLE:
            self.file_extensions = get_file_extensions()
        else:
            self.file_extensions = {'.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx', '.py', '.java', '.js', '.ts'}
        
        if self.verbose:
            print(f"KeywordSearchEngine initialized with fuzzy_threshold={fuzzy_threshold}")
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from search query.
        
        Args:
            query: Search query string
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction - split on whitespace and remove empty strings
        keywords = [word.strip() for word in query.split() if word.strip()]
        return keywords
    
    def _parse_file(self, file_path: Path) -> Dict:
        """
        Parse a file and extract content with line numbers.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Dictionary with file content and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            return {
                'file_path': str(file_path),
                'lines': lines,
                'total_lines': len(lines)
            }
        except Exception as e:
            if self.verbose:
                print(f"Error parsing file {file_path}: {e}")
            return None
    
    def _extract_context(self, lines: List[str], line_number: int) -> str:
        """
        Extract context around a matched line.
        
        Args:
            lines: List of file lines
            line_number: Target line number (1-indexed)
            
        Returns:
            Context string with surrounding lines
        """
        start = max(0, line_number - self.context_lines - 1)
        end = min(len(lines), line_number + self.context_lines)
        
        context_lines = []
        for i in range(start, end):
            marker = ">>> " if i == line_number - 1 else "    "
            context_lines.append(f"{marker}{i + 1:4d}: {lines[i].rstrip()}")
        
        return "\n".join(context_lines)
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """
        Calculate relevance score for text based on keywords.
        
        Args:
            text: Text to score
            keywords: List of keywords to match
            
        Returns:
            Tuple of (score, matched_keywords)
        """
        text_lower = text.lower()
        matched_keywords = []
        total_score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact match (highest score)
            if keyword_lower in text_lower:
                matched_keywords.append(keyword)
                total_score += 1.0
            # Fuzzy match if enabled
            elif FUZZY_AVAILABLE:
                fuzzy_score = fuzz.partial_ratio(keyword_lower, text_lower)
                if fuzzy_score >= self.fuzzy_threshold:
                    matched_keywords.append(keyword)
                    total_score += fuzzy_score / 100.0
        
        # Normalize score by number of keywords
        if keywords:
            final_score = total_score / len(keywords)
        else:
            final_score = 0.0
        
        return final_score, matched_keywords
    
    def search_directory(self, directory: Path, query: str, max_results: int = 20) -> List[KeywordMatch]:
        """
        Search for keywords in all files within a directory.
        
        Args:
            directory: Directory to search
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of KeywordMatch objects sorted by relevance
        """
        keywords = self._extract_keywords(query)
        if not keywords:
            return []
        
        if self.verbose:
            print(f"Searching for keywords: {keywords}")
        
        matches = []
        
        # Find all relevant files
        files_to_search = []
        for ext in self.file_extensions:
            files_to_search.extend(directory.rglob(f"*{ext}"))
        
        for file_path in files_to_search:
            if not file_path.is_file():
                continue
                
            file_data = self._parse_file(file_path)
            if not file_data:
                continue
            
            # Search each line in the file
            for line_num, line in enumerate(file_data['lines'], 1):
                score, matched_kw = self._calculate_keyword_score(line, keywords)
                
                if score > 0 and matched_kw:
                    context = self._extract_context(file_data['lines'], line_num)
                    
                    match = KeywordMatch(
                        file_path=str(file_path),
                        line_number=line_num,
                        matched_text=line.strip(),
                        context=context,
                        score=score,
                        matched_keywords=matched_kw
                    )
                    matches.append(match)
        
        # Sort by score (descending) and limit results
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:max_results]
    
    def search_files(self, file_paths: List[Path], query: str, max_results: int = 20) -> List[KeywordMatch]:
        """
        Search for keywords in specific files.
        
        Args:
            file_paths: List of file paths to search
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of KeywordMatch objects sorted by relevance
        """
        keywords = self._extract_keywords(query)
        if not keywords:
            return []
        
        matches = []
        
        for file_path in file_paths:
            if not file_path.is_file():
                continue
            
            file_data = self._parse_file(file_path)
            if not file_data:
                continue
            
            # Search each line in the file
            for line_num, line in enumerate(file_data['lines'], 1):
                score, matched_kw = self._calculate_keyword_score(line, keywords)
                
                if score > 0 and matched_kw:
                    context = self._extract_context(file_data['lines'], line_num)
                    
                    match = KeywordMatch(
                        file_path=str(file_path),
                        line_number=line_num,
                        matched_text=line.strip(),
                        context=context,
                        score=score,
                        matched_keywords=matched_kw
                    )
                    matches.append(match)
        
        # Sort by score (descending) and limit results
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:max_results]


def search_keyword(query_text: str, k: int = 5) -> List[Dict]:
    """
    Convenience function for keyword search using default configuration.
    
    Args:
        query_text: Search query string
        k: Maximum number of results to return
        
    Returns:
        List of search results as dictionaries
    """
    try:
        if CONFIG_AVAILABLE:
            codebase_path = resolve_codebase_path()
        else:
            # Fallback path resolution
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            codebase_path = Path(project_root) / "data" / "codebases" / "manuel_natcom" / "src" / "sim"
        
        engine = KeywordSearchEngine(verbose=False)
        matches = engine.search_directory(codebase_path, query_text, max_results=k)
        
        # Convert to dictionary format
        results = []
        for match in matches:
            result = {
                'file_path': match.file_path,
                'line_number': match.line_number,
                'content': match.matched_text,
                'context': match.context,
                'score': match.score,
                'matched_keywords': match.matched_keywords
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error in keyword search: {e}")
        return []
