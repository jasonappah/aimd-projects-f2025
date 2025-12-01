"""Compare all trained models side-by-side."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def compare_models(results_dir: str = "results"):
    """Compare all model results."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist.")
        return
    
    # Load all result files
    results = {}
    for result_file in results_path.glob("*.json"):
        import json
        with open(result_file, 'r') as f:
            results[result_file.stem] = json.load(f)
    
    if not results:
        print("No results found.")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'PR-AUC': metrics.get('pr_auc', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            'F1 Score': metrics.get('f1_at_threshold', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'Brier Score': metrics.get('brier_score', 0),
            'FN per 10k hours': metrics.get('false_negatives_per_10k_hours', 0),
            'False Alarms/day': metrics.get('false_alarms_per_day', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('PR-AUC', ascending=False)
    
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    
    # Save to CSV
    output_path = results_path / "model_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results",
                      help="Directory containing model results")
    args = parser.parse_args()
    
    compare_models(args.results_dir)

