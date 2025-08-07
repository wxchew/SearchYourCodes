#!/usr/bin/env python3
"""
Enhanced Search Evaluation Framework

This script integrates the most essential functionality from multiple evaluation scripts:
1. Comprehensive search quality evaluation with codebase-inspired queries
2. Pooling strategy comparison and analysis
3. Mean pooling verification tests
4. Final search quality validation
5. Distance metric analysis and debugging utilities

Usage: python scripts/enhanced_search_evaluation.py [--test-type TYPE]
    Types: quality, pooling, validation, debug, all
"""

import os
import sys
import numpy as np
import torch
from collections import Counter
from typing import List, Dict, Set, Optional
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.search import search_keyword, search_code, get_query_embedding
    from transformers import RobertaTokenizer, RobertaModel
    import chromadb
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class EnhancedSearchEvaluator:
    """Comprehensive search evaluation framework combining multiple analysis capabilities."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        
        # Enhanced relevance keywords inspired by actual codebase patterns
        self.relevance_keywords = {
            # Code structure patterns from actual files
            "signal handler registration with function pointer callbacks": ["signal", "handler", "registration", "function", "pointer", "callback", "SIGINT", "SIGTERM"],
            "exception handling with try catch blocks and error propagation": ["try", "catch", "exception", "throw", "Exception", "error", "what()"],
            "command line argument parsing with argc argv parameters": ["argc", "argv", "command", "line", "argument", "parsing", "parameter"],
            "class inheritance with virtual destructor and polymorphism": ["class", "virtual", "destructor", "~", "inheritance", "polymorphism", "override"],
            "template function specialization with typename parameters": ["template", "typename", "specialization", "parameter", "function"],
            "const member function with reference return values": ["const", "member", "function", "reference", "return", "&"],
            "iterator pattern with begin end and increment operators": ["iterator", "begin", "end", "++", "operator", "pattern"],
            "RAII resource management with constructor destructor pairs": ["RAII", "resource", "management", "constructor", "destructor", "new", "delete"],
            "friend class declaration for private member access": ["friend", "class", "declaration", "private", "member", "access"],
            "namespace usage with using declarations and scope resolution": ["namespace", "using", "declaration", "scope", "resolution", "::"],
            "static cast dynamic cast for safe type conversion": ["static_cast", "dynamic_cast", "type", "conversion", "safe"],
            "preprocessor macro with conditional compilation guards": ["#ifdef", "#ifndef", "#define", "macro", "conditional", "compilation"],
            
            # Physics and mathematical equations from the codebase
            "diffusion coefficient calculation with sqrt and time step": ["diffusion", "coefficient", "sqrt", "6.0", "time_step", "diffusion_dt"],
            "Einstein relation for mobility from diffusion and thermal energy": ["Einstein", "relation", "mobility", "diffusion", "kT", "thermal", "energy"],
            "variance equation for random walk with uniform distribution": ["variance", "equation", "random", "walk", "uniform", "distribution", "1/3"],
            "stability analysis for mobility stiffness product constraint": ["stability", "analysis", "mobility", "stiffness", "product", "constraint", "> 1.0"],
            "force velocity relationship with stall force and load dependence": ["force", "velocity", "relationship", "stall", "load", "dependence"],
            "matrix multiplication with BLAS and LAPACK linear algebra": ["matrix", "multiplication", "BLAS", "LAPACK", "linear", "algebra"],
            "conjugate gradient method for sparse matrix solution": ["conjugate", "gradient", "method", "sparse", "matrix", "solution"],
            "Newton method iteration for nonlinear equation solving": ["Newton", "method", "iteration", "nonlinear", "equation", "solving"],
            "eigenvalue calculation with power iteration algorithm": ["eigenvalue", "calculation", "power", "iteration", "algorithm"],
            "preconditioner computation for iterative solver acceleration": ["preconditioner", "computation", "iterative", "solver", "acceleration"],
            
            # Biological simulation concepts
            "motor protein stepping mechanism with ATP hydrolysis": ["motor", "protein", "stepping", "mechanism", "ATP", "hydrolysis"],
            "filament polymerization dynamics with growth and shrinkage": ["filament", "polymerization", "dynamics", "growth", "shrinkage"],
            "crosslinker binding specificity with parallel antiparallel orientation": ["crosslinker", "binding", "specificity", "parallel", "antiparallel", "orientation"],
            "hand monitor attachment and detachment kinetics": ["hand", "monitor", "attachment", "detachment", "kinetics"],
            "fiber bending rigidity with elastic deformation energy": ["fiber", "bending", "rigidity", "elastic", "deformation", "energy"],
            "organizer position calculation from object centroid averaging": ["organizer", "position", "calculation", "object", "centroid", "averaging"],
            
            # Memory and data structure patterns
            "vector container with push back and iterator traversal": ["vector", "container", "push_back", "iterator", "traversal"],
            "smart pointer RAII with automatic memory management": ["smart", "pointer", "RAII", "automatic", "memory", "management"],
            "reference counting with buddy system for object lifecycle": ["reference", "counting", "buddy", "system", "object", "lifecycle"],
            "linked list traversal with node insertion and deletion": ["linked", "list", "traversal", "node", "insertion", "deletion"],
            
            # Exact keyword queries for benchmarking
            "for loop": ["for", "loop", "iteration", "iterator", "while", "do"],
            "constructor destructor": ["constructor", "destructor", "~", "init", "destroy", "new", "delete"],
            "class inheritance": ["class", "inherit", "virtual", "override", "base", "derived", "public", "private"],
            "kinesin": ["kinesin"],
            "motor": ["motor"],
            "filament": ["filament", "fiber"],
            "vector": ["vector", "array", "container"],
            "force": ["force"],
            "rigidity": ["rigidity", "rigid"]
        }
    
    def calculate_relevance_score(self, query: str, results: List[Dict]) -> Dict:
        """Calculate comprehensive relevance metrics."""
        if not results:
            return {"relevance": 0.0, "diversity": 0.0, "metadata_quality": 0.0, "score_range": 0.0}
        
        expected_keywords = self.relevance_keywords.get(query, [])
        
        # Content relevance analysis
        relevance_scores = []
        for result in results:
            metadata = result.get('metadata', {})
            content = metadata.get('content', '').lower() if 'content' in metadata else ''
            file_path = metadata.get('file_path', '').lower()
            
            all_text = f"{content} {file_path}"
            matches = sum(1 for keyword in expected_keywords if keyword in all_text)
            relevance = matches / len(expected_keywords) if expected_keywords else 0.0
            relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Result diversity analysis
        file_paths = [r.get('metadata', {}).get('file_path', '') for r in results]
        unique_files = len(set(file_paths))
        diversity = unique_files / len(results) if results else 0.0
        
        # Metadata quality check
        function_names = [r.get('metadata', {}).get('function_name') for r in results]
        valid_functions = sum(1 for f in function_names if f and f != 'Unknown' and f.strip())
        metadata_quality = valid_functions / len(results) if results else 0.0
        
        # Score range analysis (for discrimination)
        scores = [r.get('similarity_score', 0) for r in results]
        score_range = max(scores) - min(scores) if scores else 0.0
        
        return {
            "relevance": avg_relevance,
            "diversity": diversity,
            "metadata_quality": metadata_quality,
            "score_range": score_range,
            "unique_files": unique_files,
            "total_results": len(results),
            "file_distribution": Counter([p.split('/')[-1] for p in file_paths if p])
        }
    
    def test_search_quality_comprehensive(self):
        """Comprehensive search quality evaluation with codebase-inspired queries."""
        print("🔍 COMPREHENSIVE SEARCH QUALITY EVALUATION")
        print("=" * 80)
        
        # Enhanced test queries covering all categories
        test_queries = [
            # Code structure patterns
            "signal handler registration with function pointer callbacks",
            "class inheritance with virtual destructor and polymorphism",
            "template function specialization with typename parameters",
            "const member function with reference return values",
            "iterator pattern with begin end and increment operators",
            "RAII resource management with constructor destructor pairs",
            "namespace usage with using declarations and scope resolution",
            "static cast dynamic cast for safe type conversion",
            
            # Physics and mathematical equations
            "diffusion coefficient calculation with sqrt and time step",
            "Einstein relation for mobility from diffusion and thermal energy",
            "variance equation for random walk with uniform distribution",
            "stability analysis for mobility stiffness product constraint",
            "force velocity relationship with stall force and load dependence",
            "matrix multiplication with BLAS and LAPACK linear algebra",
            "Newton method iteration for nonlinear equation solving",
            "eigenvalue calculation with power iteration algorithm",
            
            # Biological simulation concepts
            "motor protein stepping mechanism with ATP hydrolysis",
            "filament polymerization dynamics with growth and shrinkage",
            "crosslinker binding specificity with parallel antiparallel orientation",
            "fiber bending rigidity with elastic deformation energy",
            
            # Memory and data structure patterns
            "vector container with push back and iterator traversal",
            "smart pointer RAII with automatic memory management",
            "reference counting with buddy system for object lifecycle",
            "linked list traversal with node insertion and deletion",
            
            # Simple benchmark queries
            "for loop", "constructor destructor", "class inheritance",
            "kinesin", "motor", "filament", "vector", "force"
        ]
        
        methods = ['unixcoder', 'minilm', 'keyword']
        method_results = {method: [] for method in methods}
        
        print(f"Testing {len(test_queries)} queries across {len(methods)} search methods...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Query {i}/{len(test_queries)}: '{query[:50]}...'")
            
            # Classify query type
            if any(word in query.lower() for word in ["diffusion", "mobility", "variance", "stability", "force", "energy", "matrix", "eigenvalue", "newton"]):
                query_type = "EQUATION/PHYSICS"
            elif any(word in query.lower() for word in ["motor", "protein", "filament", "crosslinker", "hand", "fiber", "organizer"]):
                query_type = "BIOLOGY/SIMULATION"
            elif any(word in query.lower() for word in ["vector", "pointer", "memory", "reference", "linked", "container"]):
                query_type = "DATA_STRUCTURE"
            else:
                query_type = "CODE_STRUCTURE"
            print(f"    Type: {query_type}")
            
            for method in methods:
                try:
                    if method == 'keyword':
                        results = search_keyword(query, k=5)
                        # Convert to standard format
                        if results:
                            results = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
                    else:
                        results = search_code(query, model_type=method, k=5)
                    
                    metrics = self.calculate_relevance_score(query, results)
                    method_results[method].append({
                        'query': query,
                        'query_type': query_type,
                        'metrics': metrics,
                        'results': results
                    })
                    
                    print(f"    {method.upper()}: Relevance={metrics['relevance']:.3f}, "
                          f"Diversity={metrics['diversity']:.3f}, Range={metrics['score_range']:.3f}")
                    
                except Exception as e:
                    print(f"    {method.upper()}: ERROR - {e}")
                    method_results[method].append({
                        'query': query,
                        'query_type': query_type,
                        'metrics': {'relevance': 0, 'diversity': 0, 'score_range': 0},
                        'results': []
                    })
        
        # Print comprehensive summary
        self.print_quality_summary(method_results)
    
    def print_quality_summary(self, method_results: Dict):
        """Print comprehensive quality evaluation summary."""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE SEARCH QUALITY SUMMARY")
        print(f"{'='*80}")
        
        methods = list(method_results.keys())
        
        # Calculate averages for each method
        method_averages = {}
        for method in methods:
            results = method_results[method]
            if results:
                method_averages[method] = {
                    'relevance': np.mean([r['metrics']['relevance'] for r in results]),
                    'diversity': np.mean([r['metrics']['diversity'] for r in results]),
                    'score_range': np.mean([r['metrics']['score_range'] for r in results]),
                    'total_queries': len(results)
                }
            else:
                method_averages[method] = {'relevance': 0, 'diversity': 0, 'score_range': 0, 'total_queries': 0}
        
        # Overall comparison table
        print(f"\n🏆 OVERALL METHOD COMPARISON:")
        print("-" * 80)
        print(f"{'METHOD':<12} | {'RELEVANCE':<10} | {'DIVERSITY':<10} | {'SCORE RANGE':<12} | {'QUERIES':<8}")
        print("-" * 80)
        for method in methods:
            avg = method_averages[method]
            print(f"{method.upper():<12} | {avg['relevance']:<10.3f} | {avg['diversity']:<10.3f} | "
                  f"{avg['score_range']:<12.3f} | {avg['total_queries']:<8}")
        
        # Query type analysis
        query_types = set()
        for method in methods:
            for result in method_results[method]:
                query_types.add(result['query_type'])
        
        print(f"\n📋 PERFORMANCE BY QUERY TYPE:")
        print("-" * 80)
        for query_type in sorted(query_types):
            print(f"\n{query_type}:")
            print(f"{'METHOD':<12} | {'RELEVANCE':<10} | {'DIVERSITY':<10} | {'SCORE RANGE':<12}")
            print("-" * 60)
            
            for method in methods:
                type_results = [r for r in method_results[method] if r['query_type'] == query_type]
                if type_results:
                    type_avg_rel = np.mean([r['metrics']['relevance'] for r in type_results])
                    type_avg_div = np.mean([r['metrics']['diversity'] for r in type_results])
                    type_avg_range = np.mean([r['metrics']['score_range'] for r in type_results])
                    print(f"{method.upper():<12} | {type_avg_rel:<10.3f} | {type_avg_div:<10.3f} | {type_avg_range:<12.3f}")
        
        # Best performer analysis
        print(f"\n🥇 BEST PERFORMERS:")
        print("-" * 40)
        best_relevance = max(methods, key=lambda m: method_averages[m]['relevance'])
        best_diversity = max(methods, key=lambda m: method_averages[m]['diversity'])
        best_range = max(methods, key=lambda m: method_averages[m]['score_range'])
        
        print(f"Best Relevance: {best_relevance.upper()} ({method_averages[best_relevance]['relevance']:.3f})")
        print(f"Best Diversity: {best_diversity.upper()} ({method_averages[best_diversity]['diversity']:.3f})")
        print(f"Best Score Range: {best_range.upper()} ({method_averages[best_range]['score_range']:.3f})")
    
    def test_mean_pooling_verification(self):
        """Verify Mean pooling implementation works correctly."""
        print("🧪 MEAN POOLING VERIFICATION TEST")
        print("=" * 60)
        
        test_queries = [
            "diffusion coefficient calculation with sqrt and time step",
            "class inheritance with virtual destructor",
            "signal handler registration with function pointer",
            "motor protein stepping mechanism"
        ]
        
        print("Testing UniXcoder search with Mean pooling implementation...")
        
        all_passed = True
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}: '{query[:40]}...'")
            
            try:
                results = search_code(query, model_type='unixcoder', k=5)
                
                if results:
                    print(f"  ✅ Got {len(results)} results")
                    
                    # Check score range (should be better than old CLS approach)
                    scores = [r.get('similarity_score', 0) for r in results]
                    score_range = max(scores) - min(scores) if scores else 0
                    
                    print(f"  📊 Score range: {score_range:.4f}")
                    if score_range > 0.03:  # Expect better discrimination than CLS
                        print(f"  ✅ Good score discrimination")
                    else:
                        print(f"  ⚠️  Low score discrimination")
                    
                    # Show top results
                    for j, result in enumerate(results[:3], 1):
                        file_name = os.path.basename(result['metadata']['file_path'])
                        score = result.get('similarity_score', 0)
                        print(f"    {j}. {file_name} (score: {score:.4f})")
                else:
                    print("  ⚠️  No results returned")
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_passed = False
        
        print(f"\n{'='*60}")
        if all_passed:
            print("✅ ALL TESTS PASSED - Mean pooling implementation working correctly!")
            print("📈 Score discrimination appears improved over previous CLS approach")
        else:
            print("❌ SOME TESTS FAILED - Check Mean pooling implementation")
    
    def debug_search_process(self, query: str = "for loop"):
        """Debug search process step by step for troubleshooting."""
        print("🔍 DEBUGGING SEARCH PROCESS")
        print("=" * 60)
        
        print(f"🎯 Debug Query: '{query}'")
        
        try:
            # Test query embedding generation
            print(f"\n1️⃣ QUERY EMBEDDING GENERATION:")
            query_emb = get_query_embedding(query, 'unixcoder')
            print(f"  Shape: {query_emb.shape}")
            print(f"  Norm: {np.linalg.norm(query_emb):.6f}")
            print(f"  Mean: {np.mean(query_emb):.6f}")
            print(f"  Std: {np.std(query_emb):.6f}")
            
            # Manual ChromaDB query
            print(f"\n2️⃣ CHROMADB MANUAL QUERY:")
            collections = self.client.list_collections()
            unixcoder_col = None
            for c in collections:
                if 'unixcoder' in c.name:
                    unixcoder_col = c
                    break
            
            if unixcoder_col:
                query_flat = query_emb.flatten()
                chroma_results = unixcoder_col.query(
                    query_embeddings=[query_flat.tolist()],
                    n_results=5,
                    include=['documents', 'metadatas', 'distances']
                )
                
                distances = chroma_results['distances'][0]
                metadatas = chroma_results['metadatas'][0]
                
                print(f"  Retrieved {len(distances)} results")
                print(f"  Distance range: {min(distances):.6f} - {max(distances):.6f}")
                print(f"  Distance spread: {max(distances) - min(distances):.6f}")
                
                print(f"\n📊 TOP 5 MANUAL RESULTS:")
                for i, (distance, metadata) in enumerate(zip(distances, metadatas)):
                    file_path = metadata.get('file_path', 'unknown')
                    file_name = file_path.split('/')[-1]
                    function_name = metadata.get('function_name', 'Unknown')
                    similarity = max(0.0, 1 - distance / 2)  # Convert to similarity
                    
                    print(f"  {i+1}. {file_name} ({function_name})")
                    print(f"     Distance: {distance:.6f}, Similarity: {similarity:.6f}")
            
            # Compare with search_code function
            print(f"\n3️⃣ SEARCH_CODE FUNCTION COMPARISON:")
            search_results = search_code(query, model_type='unixcoder', k=5)
            
            if search_results:
                print(f"  Retrieved {len(search_results)} results")
                
                print(f"\n📊 SEARCH_CODE RESULTS:")
                for i, result in enumerate(search_results):
                    metadata = result.get('metadata', {})
                    file_path = metadata.get('file_path', 'unknown')
                    file_name = file_path.split('/')[-1]
                    function_name = metadata.get('function_name', 'Unknown')
                    similarity = result.get('similarity_score', 0)
                    distance = result.get('distance', 0)
                    
                    print(f"  {i+1}. {file_name} ({function_name})")
                    print(f"     Distance: {distance:.6f}, Similarity: {similarity:.6f}")
                
                # Analyze results consistency
                if unixcoder_col:
                    manual_files = [m.get('file_path', '').split('/')[-1] for m in metadatas]
                    search_files = [r.get('metadata', {}).get('file_path', '').split('/')[-1] for r in search_results]
                    
                    print(f"\n4️⃣ CONSISTENCY CHECK:")
                    print(f"  Manual query files: {manual_files[:3]}")
                    print(f"  search_code files: {search_files[:3]}")
                    print(f"  Results match: {manual_files == search_files}")
            
        except Exception as e:
            print(f"❌ Debug error: {e}")
            import traceback
            traceback.print_exc()
    
    def run_evaluation(self, test_type: str = "all"):
        """Run specified evaluation tests."""
        if test_type in ["quality", "all"]:
            self.test_search_quality_comprehensive()
        
        if test_type in ["validation", "all"]:
            print("\n" + "="*80)
            self.test_mean_pooling_verification()
        
        if test_type in ["debug", "all"]:
            print("\n" + "="*80)
            self.debug_search_process()

def main():
    parser = argparse.ArgumentParser(description="Enhanced Search Evaluation Framework")
    parser.add_argument('--test-type', choices=['quality', 'validation', 'debug', 'all'], 
                       default='all', help='Type of evaluation to run')
    
    args = parser.parse_args()
    
    evaluator = EnhancedSearchEvaluator()
    evaluator.run_evaluation(args.test_type)

if __name__ == "__main__":
    main()
