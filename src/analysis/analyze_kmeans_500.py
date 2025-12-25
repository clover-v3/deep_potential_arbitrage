
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Path to the results folder
RESULTS_DIR = 'results/baseline/FINAL_K_MEANS_500'

def load_and_merge_data(results_dir):
    files = glob.glob(os.path.join(results_dir, "rolling_pnl_*.csv"))
    if not files:
        print(f"No files found in {results_dir}")
        return None

    # Load and concat
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        dfs.append(df)

    full_df = pd.concat(dfs)
    full_df.sort_index(inplace=True)
    return full_df

def calculate_metrics(returns):
    # Assuming daily returns
    total_ret = (1 + returns).prod() - 1
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.0) / ann_vol # Risk free rate assumed 0 for simplicity or excess return provided

    # Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()

    return {
        'Total Return': total_ret,
        'Annualized Return': ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }

def plot_cumulative_returns(returns, output_path):
    cum_ret = (1 + returns).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(cum_ret.index, cum_ret.values, label='K-Means (K=500)')
    plt.title('Cumulative Returns: K-Means (K=500) Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (1 + R)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Log scale is often better for long term

    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    print(f"Analyzing results from {RESULTS_DIR}...")
    full_pnl = load_and_merge_data(RESULTS_DIR)

    if full_pnl is None:
        return

    # Assuming the structure is either a Series of returns or a DataFrame with a 'ret' column
    # Let's inspect columns if it's a DF
    if isinstance(full_pnl, pd.DataFrame):
        # Locate return column. Usually 'daily_ret' or 'ret'
        if 'daily_ret' in full_pnl.columns:
            returns = full_pnl['daily_ret']
        elif 'ret' in full_pnl.columns:
            returns = full_pnl['ret']
        elif '0' in full_pnl.columns: # Sometimes headerless
             returns = full_pnl['0']
        else:
            # Fallback: assume single column is returns
            returns = full_pnl.iloc[:, 0]
    else:
        returns = full_pnl

    metrics = calculate_metrics(returns)
    print("\nXXX PERFORMANCE METRICS XXX")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")

    # Plot
    plot_output = os.path.join(RESULTS_DIR, 'cumulative_return_plot.png')
    plot_cumulative_returns(returns, plot_output)

if __name__ == "__main__":
    main()
