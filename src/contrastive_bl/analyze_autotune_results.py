import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
import numpy as np

def analyze_results(results_dir):
    print(f"Analyzing results in {results_dir}...")

    summary_path = os.path.join(results_dir, "autotune_results.csv")
    if not os.path.exists(summary_path):
        print("Error: autotune_results.csv not found.")
        return

    df_summary = pd.read_csv(summary_path)
    print("\n>>> Top 5 Configurations by Sharpe Ratio:")
    print(df_summary.head(5)[['d_model', 'n_bins', 'dropout', 'sharpe', 'ann_ret']])

    # 1. Hyperparameter Sensitivity Heatmap
    try:
        pivot_table = df_summary.pivot_table(
            values='sharpe',
            index='d_model',
            columns='n_bins',
            aggfunc='mean'
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt=".3f")
        plt.title('Avg Sharpe Ratio: d_model vs n_bins')
        plt.savefig(os.path.join(results_dir, "heatmap_sharpe.png"))
        print("\nGenerated heatmap_sharpe.png")
    except Exception as e:
        print(f"Skipping Heatmap: {e}")

    # 2. Risk-Return Scatter (Pareto Frontier)
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_summary, x='ann_vol', y='ann_ret', hue='d_model', style='n_bins', s=100)

        # Add labels for top points
        top_points = df_summary.head(3)
        for _, row in top_points.iterrows():
            plt.text(row['ann_vol'], row['ann_ret'],
                     f" S:{row['sharpe']:.2f}\n(d{int(row['d_model'])},n{int(row['n_bins'])})",
                     fontsize=9)

        plt.title('Risk-Return Profile (Annualized)')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, "scatter_risk_return.png"))
        print("Generated scatter_risk_return.png")
    except Exception as e:
        print(f"Skipping Scatter: {e}")

    # 3. Equity Curve Comparison (Top 3 vs Benchmark/Avg)
    print("\n>>> Generating Equity Curve Comparison...")
    plt.figure(figsize=(12, 7))

    trial_dirs = sorted(glob.glob(os.path.join(results_dir, "trial_*")))

    # Map trial_dir to params for legend
    # We can match by index if sorted same way, but better use trial_dir from summary if available
    # Our run_autotune_orca implementation saves 'trial_dir' in csv.

    if 'trial_dir' in df_summary.columns:
        # Sort by sharpe descending
        top_trials = df_summary.head(5)

        for _, row in top_trials.iterrows():
            trial_path = row['trial_dir']
            if not os.path.isabs(trial_path):
                 # Try to resolve relative to cwd or results_dir
                 # The script saved relative path usually.
                 pass

            # Check if file exists
            # We ran from root, so path in csv is likely "results/autotune/run_.../trial_..."
            # Adjust if we are running this script from root
            full_path = os.path.join(trial_path, "rolling_results_full.csv")

            if os.path.exists(full_path):
                df_ts = pd.read_csv(full_path)
                df_ts['date'] = pd.to_datetime(df_ts['date'])
                df_ts = df_ts.sort_values('date').drop_duplicates('date', keep='last')
                df_ts['cum_ret'] = (1 + df_ts['ret']).cumprod()

                label = f"S:{row['sharpe']:.2f} (d{int(row['d_model'])}, n{int(row['n_bins'])})"
                plt.plot(df_ts['date'], df_ts['cum_ret'], label=label)
            else:
                print(f"Warning: Missing time series for {trial_path}")

    plt.title('Top 5 Models Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "equity_curves_comparison.png"))
    print("Generated equity_curves_comparison.png")

    # 4. Return Distribution Analysis (Kernel Density Estimation)
    # Compare Best Model distribution vs Mean of all models (as a proxy for random)
    try:
        plt.figure(figsize=(10, 6))

        # Best Model
        best_row = df_summary.iloc[0]
        best_path = os.path.join(best_row['trial_dir'], "rolling_results_full.csv")

        if os.path.exists(best_path):
            df_best = pd.read_csv(best_path)
            sns.kdeplot(df_best['ret'], label=f"Best Model (S:{best_row['sharpe']:.2f})", fill=True, alpha=0.3)

            # Add mean/std lines
            mean_ret = df_best['ret'].mean()
            plt.axvline(mean_ret, color='blue', linestyle='--', alpha=0.5, label=f"Mean: {mean_ret:.2%}")

        plt.title('Monthly Return Distribution (Best Model)')
        plt.xlabel('Monthly Return')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(results_dir, "return_distribution_kde.png"))
        print("Generated return_distribution_kde.png")

    except Exception as e:
        print(f"Skipping KDE: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Path to the autotune output directory (e.g. results/autotune/run_2025...)")
    args = parser.parse_args()

    analyze_results(args.results_dir)
