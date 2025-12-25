
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze():
    csv_path = "results/baseline/grid_search_summary.csv"
    if not os.path.exists(csv_path):
        print("No grid search summary found.")
        return

    df = pd.read_csv(csv_path)

    print("\n=== Grid Search Analysis ===")
    print(df[['Method', 'Param', 'Value', 'sharpe', 'total_return', 'max_drawdown']].to_string(index=False))

    # Plotting
    methods = df['Method'].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1: axes = [axes]

    for i, method in enumerate(methods):
        subset = df[df['Method'] == method]
        ax = axes[i]

        # Sort by Value
        subset = subset.sort_values('Value')

        ax.plot(subset['Value'], subset['sharpe'], marker='o', label='Sharpe')
        ax.set_title(f"{method}\nSharpe vs {subset['Param'].iloc[0]}")
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True)

        # Add values
        for x, y in zip(subset['Value'], subset['sharpe']):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    out_img = "results/baseline/analysis_plot.png"
    plt.savefig(out_img)
    print(f"\nSaved analysis plot to {out_img}")

    # --- Best Model Comparison (New) ---
    print("\n--- Generating Best Model Comparison ---")
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for method in methods:
        subset = df[df['Method'] == method]
        # Identify Best Param (Max Sharpe)
        # If Sharpe is identical (short test), fallback to Total Return
        best_row = subset.sort_values(by=['sharpe', 'total_return'], ascending=False).iloc[0]

        # Reconstruct Subdir
        param_name = best_row['Param']
        val = best_row['Value']

        # Naming convention must match run_grid_search.py
        # float formatting might be tricky? usually just str(val) unless it was formatted
        # In grid search: f"kmeans_k_{k}" (k is int), f"agglo_q_{q}" (q is float)

        if method == 'K-Means':
             subdir = f"kmeans_k_{int(val)}"
        elif method == 'Agglomerative':
             subdir = f"agglo_q_{val}" # Float string might differ if truncated?
             # q was [0.05, 0.1 ...]
        elif method == 'DBSCAN':
             subdir = f"dbscan_q_{val}"
        else:
             continue

        pnl_path = f"results/baseline/{subdir}/rolling_pnl.csv"

        if os.path.exists(pnl_path):
            pnl_df = pd.read_csv(pnl_path, index_col=0, parse_dates=True)
            # Cum Ret
            pnl_df['cum_ret'] = (1 + pnl_df['pnl']).cumprod() - 1

            label = f"{method} (Best {param_name}={val})"
            ax2.plot(pnl_df.index, pnl_df['cum_ret'], label=label)
        else:
            print(f"Warning: Results not found for {method} at {pnl_path}")

    ax2.set_title("Cumulative Returns of Best Configurations")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    ax2.grid(True)

    out_img2 = "results/baseline/best_models_comparison.png"
    plt.savefig(out_img2)
    print(f"Saved comparison plot to {out_img2}")

if __name__ == "__main__":
    analyze()
