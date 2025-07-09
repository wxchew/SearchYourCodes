#!/usr/bin/env python3
"""
Enhanced Search Results Analysis

A script to analyze the enhanced search results for model performance insights.
"""

import json
from collections import defaultdict

def analyze_results(file_path):
    """
    Analyzes the enhanced search results to provide insights on model performance.
    """
    with open(file_path, 'r') as f:
        results_data = json.load(f)

    analysis = defaultdict(lambda: defaultdict(list))

    for model, weighted_results in results_data.items():
        for weight_config, query_results in weighted_results.items():
            for result in query_results:
                query = result['query']
                top_5_chunks = [res['chunk_id'] for res in result['results']]
                
                analysis[model][query].append({
                    'weight_config': weight_config,
                    'top_5_chunks': top_5_chunks
                })

    return analysis

def display_analysis(analysis):
    """
    Displays the analysis in a readable format.
    """
    for model, queries in analysis.items():
        print(f"--- Model: {model} ---")
        for query, results in queries.items():
            print(f"  Query: \"{query}\"")
            for res in results:
                print(f"    Weight Config: {res['weight_config']}")
                print(f"      Top 5 Chunks: {res['top_5_chunks']}")
            print()

if __name__ == "__main__":
    analysis = analyze_results('data/enhanced_search_results.json')
    display_analysis(analysis)
