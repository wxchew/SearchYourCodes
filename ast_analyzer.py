"""
AST-based structural similarity analyzer for code search enhancement.
This module provides functionality to extract structural features from code
and compute similarity metrics based on Abstract Syntax Trees.
"""

import ast
import json
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import re

class ASTFeatureExtractor:
    """Extract structural features from code using AST analysis."""
    
    def __init__(self):
        self.node_types = set()
        self.function_signatures = {}
        
    def extract_features(self, code: str, language: str = "cpp") -> Dict:
        """
        Extract structural features from code.
        For C++, we'll use regex-based parsing since Python's AST only works for Python.
        """
        features = {
            'function_calls': self._extract_function_calls(code),
            'control_structures': self._extract_control_structures(code),
            'data_types': self._extract_data_types(code),
            'operators': self._extract_operators(code),
            'complexity_metrics': self._compute_complexity_metrics(code),
            'structural_patterns': self._extract_structural_patterns(code)
        }
        return features
    
    def _extract_function_calls(self, code: str) -> List[str]:
        """Extract function calls from C++ code."""
        # Pattern to match function calls: identifier followed by parentheses
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, code)
        # Filter out keywords that aren't function calls
        keywords = {'if', 'while', 'for', 'switch', 'catch', 'sizeof', 'typeof'}
        return [match for match in matches if match not in keywords]
    
    def _extract_control_structures(self, code: str) -> List[str]:
        """Extract control flow structures from code."""
        structures = []
        patterns = {
            'if': r'\bif\s*\(',
            'while': r'\bwhile\s*\(',
            'for': r'\bfor\s*\(',
            'switch': r'\bswitch\s*\(',
            'try': r'\btry\s*\{',
            'catch': r'\bcatch\s*\('
        }
        
        for structure, pattern in patterns.items():
            count = len(re.findall(pattern, code))
            structures.extend([structure] * count)
        
        return structures
    
    def _extract_data_types(self, code: str) -> List[str]:
        """Extract data types used in the code."""
        # Common C++ data types
        cpp_types = [
            'int', 'float', 'double', 'char', 'bool', 'void', 'string', 'size_t',
            'uint32_t', 'uint64_t', 'int32_t', 'int64_t', 'vector', 'map', 'set',
            'list', 'array', 'pair', 'tuple', 'auto', 'const', 'static', 'unsigned'
        ]
        
        found_types = []
        for data_type in cpp_types:
            pattern = r'\b' + re.escape(data_type) + r'\b'
            count = len(re.findall(pattern, code))
            found_types.extend([data_type] * count)
        
        return found_types
    
    def _extract_operators(self, code: str) -> List[str]:
        """Extract operators used in the code."""
        operators = []
        operator_patterns = {
            'arithmetic': r'[+\-*/]',
            'comparison': r'[<>=!]=?',
            'logical': r'&&|\|\|',
            'bitwise': r'[&|^~]',
            'assignment': r'[+\-*/]?=',
            'increment': r'\+\+|--',
            'pointer': r'[*&]',
            'member': r'[.\->]+'
        }
        
        for op_type, pattern in operator_patterns.items():
            count = len(re.findall(pattern, code))
            operators.extend([op_type] * count)
        
        return operators
    
    def _compute_complexity_metrics(self, code: str) -> Dict[str, int]:
        """Compute basic complexity metrics."""
        return {
            'line_count': len(code.split('\n')),
            'brace_depth': self._compute_brace_depth(code),
            'function_count': len(re.findall(r'\b\w+\s*\([^)]*\)\s*\{', code)),
            'loop_count': len(re.findall(r'\b(for|while)\s*\(', code)),
            'conditional_count': len(re.findall(r'\bif\s*\(', code))
        }
    
    def _compute_brace_depth(self, code: str) -> int:
        """Compute maximum brace nesting depth."""
        depth = 0
        max_depth = 0
        for char in code:
            if char == '{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == '}':
                depth -= 1
        return max_depth
    
    def _extract_structural_patterns(self, code: str) -> List[str]:
        """Extract common structural patterns."""
        patterns = []
        
        # Class definitions
        if re.search(r'\bclass\s+\w+', code):
            patterns.append('class_definition')
        
        # Constructor patterns
        if re.search(r'\w+\s*\([^)]*\)\s*:', code):
            patterns.append('constructor')
        
        # Destructor patterns
        if re.search(r'~\w+\s*\(\)', code):
            patterns.append('destructor')
        
        # Template patterns
        if re.search(r'template\s*<', code):
            patterns.append('template')
        
        # Namespace patterns
        if re.search(r'namespace\s+\w+', code):
            patterns.append('namespace')
        
        # Exception handling
        if re.search(r'\btry\s*\{', code):
            patterns.append('exception_handling')
        
        return patterns

class ASTSimilarityCalculator:
    """Calculate similarity between code snippets based on AST features."""
    
    def __init__(self):
        self.feature_extractor = ASTFeatureExtractor()
        self.weights = {
            'function_calls': 0.25,
            'control_structures': 0.20,
            'data_types': 0.15,
            'operators': 0.15,
            'complexity_metrics': 0.15,
            'structural_patterns': 0.10
        }
    
    def compute_similarity(self, code1: str, code2: str) -> float:
        """Compute structural similarity between two code snippets."""
        features1 = self.feature_extractor.extract_features(code1)
        features2 = self.feature_extractor.extract_features(code2)
        
        total_similarity = 0.0
        
        for feature_type, weight in self.weights.items():
            if feature_type in features1 and feature_type in features2:
                if feature_type == 'complexity_metrics':
                    sim = self._compute_complexity_similarity(
                        features1[feature_type], features2[feature_type]
                    )
                else:
                    sim = self._compute_list_similarity(
                        features1[feature_type], features2[feature_type]
                    )
                total_similarity += weight * sim
        
        return total_similarity
    
    def _compute_list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Compute similarity between two lists using Jaccard similarity."""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
        
        # Convert to Counter for frequency-based similarity
        counter1 = Counter(list1)
        counter2 = Counter(list2)
        
        # Compute intersection and union
        intersection = sum((counter1 & counter2).values())
        union = sum((counter1 | counter2).values())
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_complexity_similarity(self, metrics1: Dict, metrics2: Dict) -> float:
        """Compute similarity between complexity metrics."""
        if not metrics1 or not metrics2:
            return 0.0
        
        similarities = []
        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            elif val1 == 0 or val2 == 0:
                similarities.append(0.0)
            else:
                # Normalized similarity based on ratio
                ratio = min(val1, val2) / max(val1, val2)
                similarities.append(ratio)
        
        return np.mean(similarities) if similarities else 0.0

class HybridCodeSearchRanker:
    """Hybrid ranker combining semantic embeddings with structural similarity."""
    
    def __init__(self, semantic_weight: float = 0.7, structural_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        self.ast_calculator = ASTSimilarityCalculator()
    
    def rank_results(self, query_code: str, candidate_results: List[Dict]) -> List[Dict]:
        """
        Re-rank search results by combining semantic and structural similarity.
        
        Args:
            query_code: The search query code
            candidate_results: List of dicts with 'content', 'metadata', and 'semantic_score'
        
        Returns:
            Re-ranked list of results with combined scores
        """
        enhanced_results = []
        
        for result in candidate_results:
            candidate_code = result['content']
            semantic_score = result.get('semantic_score', 0.0)
            
            # Compute structural similarity
            structural_score = self.ast_calculator.compute_similarity(
                query_code, candidate_code
            )
            
            # Combine scores
            combined_score = (
                self.semantic_weight * semantic_score +
                self.structural_weight * structural_score
            )
            
            enhanced_result = result.copy()
            enhanced_result.update({
                'structural_score': structural_score,
                'combined_score': combined_score,
                'semantic_score': semantic_score
            })
            enhanced_results.append(enhanced_result)
        
        # Sort by combined score (descending)
        enhanced_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return enhanced_results

def analyze_code_chunk_features(chunks_file: Path, output_file: Path):
    """Analyze all code chunks and extract their AST features."""
    print("Analyzing code chunks for AST features...")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    feature_extractor = ASTFeatureExtractor()
    
    chunk_features = {}
    for chunk in chunks:
        chunk_id = chunk['id']
        features = feature_extractor.extract_features(chunk['content'])
        chunk_features[chunk_id] = features
    
    # Save features to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_features, f, indent=2, default=str)
    
    print(f"AST features saved to {output_file}")
    print(f"Analyzed {len(chunks)} code chunks")

if __name__ == "__main__":
    # Analyze code chunks and extract features
    chunks_file = Path("data/code_chunks_clean.json")
    features_file = Path("data/ast_features.json")
    
    if chunks_file.exists():
        analyze_code_chunk_features(chunks_file, features_file)
    else:
        print(f"Code chunks file not found: {chunks_file}")
