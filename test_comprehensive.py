#!/usr/bin/env python3

from search import compare_all_refinement_methods

def main():
    # Test query
    query = "what happen when ase walker reaches the plus end of microtubule"
    
    print("Testing comprehensive refinement comparison...")
    print(f"Query: '{query}'")
    
    # Run comprehensive comparison
    all_results = compare_all_refinement_methods(query, k=3)
    
    # Extract and display top results summary
    print(f"\n{'='*100}")
    print(f"🏆 TOP RESULTS SUMMARY")
    print(f"{'='*100}")
    
    methods = [
        ("none", "No Refinement"),
        ("intent", "Intent-Based"),
        ("code", "Code-Specific"), 
        ("both", "Combined")
    ]
    
    for method_key, method_name in methods:
        if method_key in all_results:
            results = all_results[method_key]
            ux_results = results['unixcoder']
            sb_results = results['sbert']
            refined_query = results['refined_query']
            
            print(f"\n🔍 {method_name}:")
            if method_key != "none":
                print(f"   Query: '{refined_query}'")
            else:
                print(f"   Query: '{query}' (original)")
            
            # UniXcoder top result
            if ux_results:
                ux_top = ux_results[0]
                ux_meta = ux_top['metadata']
                ux_score = ux_top['similarity_score']
                print(f"   🔵 UniXcoder Top: {ux_meta.get('function_name', 'Unknown')} ({ux_score:.3f})")
                print(f"      File: {ux_meta.get('file_path', 'Unknown')}")
            else:
                print(f"   🔵 UniXcoder Top: No results")
            
            # Sentence-BERT top result
            if sb_results:
                sb_top = sb_results[0]
                sb_meta = sb_top['metadata']
                sb_score = sb_top['similarity_score']
                print(f"   🟢 SBERT Top: {sb_meta.get('function_name', 'Unknown')} ({sb_score:.3f})")
                print(f"      File: {sb_meta.get('file_path', 'Unknown')}")
            else:
                print(f"   🟢 SBERT Top: No results")

if __name__ == "__main__":
    main()
