import csv
import argparse
from pathlib import Path
from scipy import stats
import numpy as np


def load_experiment_results(csv_path):
    mlp_results = []
    cnn_results = []
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"].lower()
            val_acc = float(row["val_acc"])
            
            if model == "mlp":
                mlp_results.append({
                    "experiment_id": row["experiment_id"],
                    "val_acc": val_acc,
                    "epochs": int(row["epochs"]),
                    "learning_rate": row["learning_rate"],
                    "hidden_dim": row["hidden_dim"],
                    "num_layers": row["num_layers"]
                })
            elif model == "cnn":
                cnn_results.append({
                    "experiment_id": row["experiment_id"],
                    "val_acc": val_acc,
                    "epochs": int(row["epochs"]),
                    "learning_rate": row["learning_rate"],
                    "hidden_dim": row["hidden_dim"],
                    "num_layers": row["num_layers"]
                })
    
    return mlp_results, cnn_results


def compute_statistics(results, name):
    if not results:
        return None
    
    accuracies = [r["val_acc"] for r in results]
    
    return {
        "name": name,
        "count": len(accuracies),
        "mean": np.mean(accuracies),
        "std": np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0,
        "min": np.min(accuracies),
        "max": np.max(accuracies),
        "accuracies": accuracies
    }


def paired_t_test(mlp_accs, cnn_accs):
    # uses independent t-test if sample sizes differ
    if len(mlp_accs) == len(cnn_accs) and len(mlp_accs) > 1:
        # Paired t-test (same experiments run on both models)
        t_stat, p_value = stats.ttest_rel(cnn_accs, mlp_accs)
        test_type = "Paired"
    elif len(mlp_accs) > 1 and len(cnn_accs) > 1:
        # Independent t-test (different number of experiments)
        t_stat, p_value = stats.ttest_ind(cnn_accs, mlp_accs)
        test_type = "Independent"
    else:
        return None, None, "Insufficient data"
    
    return t_stat, p_value, test_type


def analyze_experiments(csv_path):
    print("=" * 70)
    print("EXPERIMENT ANALYSIS")
    print(f"Source: {csv_path}")
    print("=" * 70)
    
    mlp_results, cnn_results = load_experiment_results(csv_path)
    
    print(f"\nLoaded {len(mlp_results)} MLP experiments")
    print(f"Loaded {len(cnn_results)} CNN experiments")
    
    if not mlp_results and not cnn_results:
        print("\nNo experiments found in log file.")
        return
    
    mlp_stats = compute_statistics(mlp_results, "MLP")
    cnn_stats = compute_statistics(cnn_results, "CNN")
    
    print("\n" + "-" * 70)
    print("MODEL STATISTICS (Validation Accuracy)")
    print("-" * 70)
    print(f"{'Model':<10} {'N':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    if mlp_stats:
        print(f"{'MLP':<10} {mlp_stats['count']:<6} {mlp_stats['mean']:<10.4f} "
              f"{mlp_stats['std']:<10.4f} {mlp_stats['min']:<10.4f} {mlp_stats['max']:<10.4f}")
    else:
        print(f"{'MLP':<10} {'N/A':<6}")
    
    if cnn_stats:
        print(f"{'CNN':<10} {cnn_stats['count']:<6} {cnn_stats['mean']:<10.4f} "
              f"{cnn_stats['std']:<10.4f} {cnn_stats['min']:<10.4f} {cnn_stats['max']:<10.4f}")
    else:
        print(f"{'CNN':<10} {'N/A':<6}")
    
    print("-" * 70)
    
    if mlp_stats and cnn_stats:
        print("\n" + "-" * 70)
        print("STATISTICAL COMPARISON: CNN vs MLP")
        print("-" * 70)
        
        mean_diff = cnn_stats['mean'] - mlp_stats['mean']
        print(f"Mean difference (CNN - MLP): {mean_diff:+.4f}")
        
        t_stat, p_value, test_type = paired_t_test(
            mlp_stats['accuracies'], 
            cnn_stats['accuracies']
        )
        
        if t_stat is not None:
            print(f"\n{test_type} t-test:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            
            # Interpret results
            alpha = 0.05
            print(f"\nSignificance level: α = {alpha}")
            
            if p_value < alpha:
                if mean_diff > 0:
                    conclusion = "CNN significantly outperforms MLP"
                else:
                    conclusion = "MLP significantly outperforms CNN"
                print(f"Result: SIGNIFICANT (p < {alpha})")
                print(f"Conclusion: {conclusion}")
            else:
                print(f"Result: NOT SIGNIFICANT (p >= {alpha})")
                print("Conclusion: No significant difference between CNN and MLP")
            
            pooled_std = np.sqrt((mlp_stats['std']**2 + cnn_stats['std']**2) / 2)
            if pooled_std > 0:
                cohens_d = mean_diff / pooled_std
                print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
                
                if abs(cohens_d) < 0.2:
                    effect_interp = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_interp = "small"
                elif abs(cohens_d) < 0.8:
                    effect_interp = "medium"
                else:
                    effect_interp = "large"
                print(f"Interpretation: {effect_interp} effect")
        else:
            print(f"\n{test_type}")
    
    print("\n" + "-" * 70)
    print("INDIVIDUAL EXPERIMENTS")
    print("-" * 70)
    
    if mlp_results:
        print("\nMLP Experiments:")
        print(f"  {'ID':<20} {'Val Acc':<10} {'Epochs':<8} {'LR':<10} {'Hidden':<10} {'Layers':<8}")
        for r in mlp_results:
            print(f"  {r['experiment_id']:<20} {r['val_acc']:<10.4f} {r['epochs']:<8} "
                  f"{r['learning_rate']:<10} {r['hidden_dim']:<10} {r['num_layers']:<8}")
    
    if cnn_results:
        print("\nCNN Experiments:")
        print(f"  {'ID':<20} {'Val Acc':<10} {'Epochs':<8} {'LR':<10} {'Filters':<10} {'Layers':<8}")
        for r in cnn_results:
            print(f"  {r['experiment_id']:<20} {r['val_acc']:<10.4f} {r['epochs']:<8} "
                  f"{r['learning_rate']:<10} {r['hidden_dim']:<10} {r['num_layers']:<8}")
    
    print("\n" + "=" * 70)
    
    return mlp_stats, cnn_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--log-path", type=str, 
                        default="experiments/experiment_log.csv",
                        help="Path to experiment log CSV")
    
    args = parser.parse_args()
    
    if not Path(args.log_path).exists():
        print(f"Error: Log file not found: {args.log_path}")
        print("Run some experiments first using models/train.py")
    else:
        analyze_experiments(args.log_path)






