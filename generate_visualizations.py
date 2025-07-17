#!/usr/bin/env python3
"""
Model Comparison Visualization Script
Generates charts and visualizations for the embedding model comparison results.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_data():
    """Load embedding data and metrics."""
    # Load embeddings
    unixcoder_embeddings = np.load('data/embeddings/test_raw_unixcoder_embeddings.npy')
    codebert_embeddings = np.load('data/embeddings/test_raw_codebert_embeddings.npy')
    sbert_embeddings = np.load('data/embeddings/test_raw_sentence_bert_minilm_embeddings.npy')
    
    # Load metrics
    with open('data/model_comparison_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    return {
        'unixcoder': unixcoder_embeddings,
        'codebert': codebert_embeddings,
        'sentence_bert': sbert_embeddings
    }, metrics

def create_performance_comparison_chart(metrics):
    """Create a comprehensive performance comparison chart."""
    models = ['UniXcoder', 'Sentence-BERT', 'CodeBERT']
    
    # Extract metrics
    mean_similarities = [
        metrics['performance_metrics']['unixcoder']['mean_similarity'],
        metrics['performance_metrics']['sentence_bert']['mean_similarity'],
        metrics['performance_metrics']['codebert']['mean_similarity']
    ]
    
    diversity_stds = [
        metrics['performance_metrics']['unixcoder']['diversity_std'],
        metrics['performance_metrics']['sentence_bert']['diversity_std'],
        metrics['performance_metrics']['codebert']['diversity_std']
    ]
    
    uniqueness_ratios = [
        metrics['performance_metrics']['unixcoder']['uniqueness_ratio'],
        metrics['performance_metrics']['sentence_bert']['uniqueness_ratio'],
        metrics['performance_metrics']['codebert']['uniqueness_ratio']
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Mean Similarity (lower is better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, mean_similarities, color=['#2E8B57', '#4169E1', '#DC143C'])
    ax1.set_title('Mean Similarity (Lower = Better Discrimination)', fontweight='bold')
    ax1.set_ylabel('Mean Cosine Similarity')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars1, mean_similarities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Diversity (higher is better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, diversity_stds, color=['#2E8B57', '#4169E1', '#DC143C'])
    ax2.set_title('Diversity (Higher = More Varied Embeddings)', fontweight='bold')
    ax2.set_ylabel('Standard Deviation')
    
    for bar, value in zip(bars2, diversity_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Uniqueness Ratio
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, uniqueness_ratios, color=['#2E8B57', '#4169E1', '#DC143C'])
    ax3.set_title('Uniqueness Ratio (Higher = Less Embedding Collapse)', fontweight='bold')
    ax3.set_ylabel('Unique Embeddings / Total Embeddings')
    ax3.set_ylim(0.8, 1.0)
    
    for bar, value in zip(bars3, uniqueness_ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Overall Ranking
    ax4 = axes[1, 1]
    overall_ranks = [
        metrics['performance_metrics']['unixcoder']['overall_rank'],
        metrics['performance_metrics']['sentence_bert']['overall_rank'],
        metrics['performance_metrics']['codebert']['overall_rank']
    ]
    
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    bars4 = ax4.bar(models, [4-rank for rank in overall_ranks], color=colors)
    ax4.set_title('Overall Ranking (Higher = Better)', fontweight='bold')
    ax4.set_ylabel('Rank Score')
    ax4.set_ylim(0, 4)
    
    rank_labels = ['1st', '2nd', '3rd']
    for bar, rank in zip(bars4, overall_ranks):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                rank_labels[rank-1], ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_similarity_distribution_plots(embeddings):
    """Create similarity distribution plots for each model."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Similarity Distribution Analysis', fontsize=16, fontweight='bold')
    
    models = ['UniXcoder', 'CodeBERT', 'Sentence-BERT']
    colors = ['#2E8B57', '#DC143C', '#4169E1']
    
    for i, (model_name, emb) in enumerate(embeddings.items()):
        sim_matrix = cosine_similarity(emb)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
        
        ax = axes[i]
        ax.hist(upper_triangle, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        ax.set_title(f'{models[i]} Similarity Distribution', fontweight='bold')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(upper_triangle), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(upper_triangle):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_performance_radar_chart(metrics):
    """Create a radar chart comparing model performance."""
    categories = ['Discrimination\\n(1/mean_sim)', 'Diversity\\n(std)', 'Uniqueness\\n(ratio)', 
                  'Speed\\n(1/time)', 'Dimensions\\n(normalized)']
    
    # Normalize metrics for radar chart
    models_data = {
        'UniXcoder': [
            1 / metrics['performance_metrics']['unixcoder']['mean_similarity'] / 10,  # Scale discrimination
            metrics['performance_metrics']['unixcoder']['diversity_std'] * 5,  # Scale diversity
            metrics['performance_metrics']['unixcoder']['uniqueness_ratio'],
            1 / metrics['models_evaluated']['unixcoder']['processing_time_seconds'] * 10,  # Scale speed
            metrics['models_evaluated']['unixcoder']['dimensions'] / 1000  # Scale dimensions
        ],
        'Sentence-BERT': [
            1 / metrics['performance_metrics']['sentence_bert']['mean_similarity'] / 10,
            metrics['performance_metrics']['sentence_bert']['diversity_std'] * 5,
            metrics['performance_metrics']['sentence_bert']['uniqueness_ratio'],
            1 / metrics['models_evaluated']['sentence_bert']['processing_time_seconds'] * 10,
            metrics['models_evaluated']['sentence_bert']['dimensions'] / 1000
        ],
        'CodeBERT': [
            1 / metrics['performance_metrics']['codebert']['mean_similarity'] / 10,
            metrics['performance_metrics']['codebert']['diversity_std'] * 5,
            metrics['performance_metrics']['codebert']['uniqueness_ratio'],
            1 / metrics['models_evaluated']['codebert']['processing_time_seconds'] * 10,
            metrics['models_evaluated']['codebert']['dimensions'] / 1000
        ]
    }
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#2E8B57', '#4169E1', '#DC143C']
    
    for i, (model, values) in enumerate(models_data.items()):
        values += values[:1]  # Complete the circle
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles_plot, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles_plot, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True)
    
    return fig

def generate_summary_table(metrics):
    """Generate a summary table of all metrics."""
    data = []
    
    for model_key, model_name in [('unixcoder', 'UniXcoder'), 
                                  ('sentence_bert', 'Sentence-BERT'), 
                                  ('codebert', 'CodeBERT')]:
        perf = metrics['performance_metrics'][model_key]
        model_info = metrics['models_evaluated'][model_key]
        
        data.append({
            'Model': model_name,
            'Mean Similarity': f"{perf['mean_similarity']:.4f}",
            'Diversity (std)': f"{perf['diversity_std']:.4f}",
            'Unique Embeddings': f"{perf['unique_embeddings']}/65",
            'Uniqueness Ratio': f"{perf['uniqueness_ratio']:.3f}",
            'Dimensions': model_info['dimensions'],
            'Processing Time (s)': f"{model_info['processing_time_seconds']:.2f}",
            'Overall Rank': perf['overall_rank']
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    """Main function to generate all visualizations and reports."""
    print("Loading data...")
    embeddings, metrics = load_data()
    
    # Create output directory
    output_dir = Path('data/visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating performance comparison chart...")
    fig1 = create_performance_comparison_chart(metrics)
    fig1.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    
    print("Generating similarity distribution plots...")
    fig2 = create_similarity_distribution_plots(embeddings)
    fig2.savefig(output_dir / 'similarity_distributions.png', dpi=300, bbox_inches='tight')
    
    print("Generating radar chart...")
    fig3 = create_performance_radar_chart(metrics)
    fig3.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
    
    print("Generating summary table...")
    summary_df = generate_summary_table(metrics)
    summary_df.to_csv(output_dir / 'model_comparison_summary.csv', index=False)
    
    # Save table as formatted text
    with open(output_dir / 'model_comparison_table.txt', 'w') as f:
        f.write("MODEL COMPARISON SUMMARY TABLE\\n")
        f.write("=" * 80 + "\\n")
        f.write(summary_df.to_string(index=False))
        f.write("\\n\\n")
        f.write("PERFORMANCE RANKINGS:\\n")
        f.write("-" * 40 + "\\n")
        f.write("1. Best Overall: UniXcoder (Rank 1)\\n")
        f.write("2. Runner-up: Sentence-BERT (Rank 2)\\n")
        f.write("3. Poor Performance: CodeBERT (Rank 3)\\n")
    
    print("\\nVisualization files generated:")
    print(f"- {output_dir / 'performance_comparison.png'}")
    print(f"- {output_dir / 'similarity_distributions.png'}")
    print(f"- {output_dir / 'performance_radar.png'}")
    print(f"- {output_dir / 'model_comparison_summary.csv'}")
    print(f"- {output_dir / 'model_comparison_table.txt'}")
    
    print("\\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Close all figures to free memory
    plt.close('all')
    
    print("\\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()
