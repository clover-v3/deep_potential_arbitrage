"""
Script to validation Laplacian Dynamics Assumption on Real Market Data (CRSP Daily).

1. Loads Real CRSP Daily Data (Prices/Returns)
2. Filters for Liquid Universe (Top N by Dollar Volume)
3. Computes Daily Log Returns
4. Builds Correlation Graph (Proxy for Laplacian)
5. Runs Dynamics Tests (Force-Return Correlation, etc.)
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path to import sibling modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.validation.dynamics_test import (
    test_force_return_correlation,
    test_dynamics_regression,
    test_prediction_accuracy,
    visualize_dynamics_test_results
)
from src.validation.graph_learning import correlation_graph

def load_real_data(data_dir: str, start_year: int = 2015, end_year: int = 2023, top_n: int = 50) -> pd.DataFrame:
    """
    Load CRSP Daily Data, filter by year and liquidity.
    Returns:
        pivot_returns: (T, N) DataFrame of daily log returns for Top N stocks.
    """
    print(f"Loading CRSP data from {data_dir} ({start_year}-{end_year})...")

    # Load all year files
    dfs = []
    # Assuming structure: crsp_dsf/dsf_YYYY.parquet or similar
    # The pull_wrds.py saves as f"dsf_{year}.parquet" inside crsp_dsf dir?
    # Let's check listing logic: usually it is flattened.
    # We will iterate and try to find matching files.

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])

    for f in files:
        # Extract year from filename (assuming dsf_YYYY.parquet or similar)
        # Simple check: if year in filename
        try:
            # fast parse
            y_str = ''.join(filter(str.isdigit, f))
            if not y_str: continue
            y = int(y_str[:4]) # First 4 digits

            if start_year <= y <= end_year:
                path = os.path.join(data_dir, f)
                df = pd.read_parquet(path)
                # Keep only necessary cols to save memory
                keep_cols = [c for c in ['permno', 'date', 'prc', 'vol', 'ret'] if c in df.columns]
                dfs.append(df[keep_cols])
                print(f"  Loaded {f}")
        except Exception as e:
            print(f"  Skipping {f}: {e}")

    if not dfs:
        raise ValueError("No data loaded.")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df['date'] = pd.to_datetime(full_df['date'])

    # Clean Prices/Returns
    full_df['prc'] = full_df['prc'].abs() # Handle negative prices (mid-quotes)

    # Calculate Dollar Volume for Filtering
    full_df['dvol'] = full_df['prc'] * full_df['vol']

    # Select Top N Liquid Stocks *consistently* over the period?
    # Or just select based on average rank.
    # Group by PERMNO, sum DVOL.
    print("Filtering Top Liquid Stocks...")
    total_dvol = full_df.groupby('permno')['dvol'].sum()
    top_permnos = total_dvol.nlargest(top_n).index

    df_filtered = full_df[full_df['permno'].isin(top_permnos)].copy()

    # Pivot to (Date, Permno) -> Return
    # Using 'ret' column (Total Return)
    print("Pivoting Returns...")
    pivot_ret = df_filtered.pivot(index='date', columns='permno', values='ret')

    # Enforce numeric
    pivot_ret = pivot_ret.apply(pd.to_numeric, errors='coerce')

    # Fill missing
    pivot_ret = pivot_ret.fillna(0.0)

    # Clean and Log
    # Clip to avoid log(0) or log(negative)
    pivot_ret = np.log1p(pivot_ret.clip(lower=-0.99))

    # Clean Infs
    pivot_ret = pivot_ret.replace([np.inf, -np.inf], 0.0)

    # Cast to numpy float64 explicitly
    pivot_ret = pivot_ret.astype(np.float64)

    print(f"Data Shape: {pivot_ret.shape}, Dtype: {pivot_ret.dtypes.iloc[0]}")
    return pivot_ret

def run_real_dynamics_test():
    # Paths
    # Adjust path relative to execution or absolute
    # Assuming running from project root or src
    # Try multiple common paths for robustness
    candidates = [
        'data/raw_ghz/crsp_dsf',
        '../../data/raw_ghz/crsp_dsf',
        '/Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/data/raw_ghz/crsp_dsf'
    ]

    data_dir = None
    for p in candidates:
        if os.path.exists(p):
            data_dir = p
            break

    if not data_dir:
        print("Error: Could not find CRSP DSF directory.")
        return

    # 1. Load Data
    try:
        # Load recent 3 years for quick validaton
        returns_df = load_real_data(data_dir, start_year=2020, end_year=2023, top_n=50)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    f_series = returns_df.values # (T, N)
    dates = returns_df.index

    # 2. Build Graph (Correlation)
    # Use first 50% for graph, test on last 50%? Or rolling?
    # For validating "Existence of Dynamics", using the whole period graph is acceptable
    # to check if *static* structure explains dynamics.
    # Even better: Rolling test. But let's start with static for simplicity (Assumption Check).
    print("\nBuilding Correlation Graph...")
    # Compute correlation on first half
    split = len(f_series) // 2
    train_data = f_series[:split]
    test_data = f_series[split:]

    A_corr = correlation_graph(train_data, threshold=0.1) # Sparse threshold

    # Compute Laplacian
    # Normalized Laplacian or Standard?
    # Dynamics: df/dt = -L f. Usually L is Combinatorial (D - A) or Normalized.
    # Let's use Normalized Logic: D - A usually.
    degree = A_corr.sum(axis=1)
    L = np.diag(degree) - A_corr
    # Normalize trace/scale?
    # L = L / np.linalg.norm(L) # Scale doesn't matter for correlation, but matters for regression beta

    print(f"Graph Density: {(A_corr > 0).mean():.2%}")

    # 3. Run Tests (On Test Data)
    print("\nRunning Dynamics Tests on Out-of-Sample Data...")

    # Test 1: Force-Return
    res_ic = test_force_return_correlation(test_data, L, lag=1)
    print(f"1. Force-Return IC: {res_ic['ic_mean']:.4f} (p={res_ic['p_value']:.4e})")

    # Test 2: Regression
    # Note: Returns are inherently "diff".
    # Dynamics Equation: df/dt = -L f.
    # If f is Price, df/dt is Return.
    # So Return(t) = -L * Price(t-1).
    # BUT we passed Returns as f_series!
    # If f_series is Returns, then df/dt is Change in Returns (Acceleration).
    # Is our assumption Price Mean Reversion or Return Mean Reversion?
    # Usually: Prices mean revert to equilibrium.
    # So we need Cumulative Returns (Prices) as state 'f', and Returns as 'df/dt'.

    print("\n[CRITICAL ADJUSTMENT] Transforming Returns to Cumulative Returns (Pseudo-Prices)...")
    # Gumulative sum of log returns = Log Price Profile
    # De-mean to remove market drift?
    # Remove market mode?
    f_prices_test = np.cumsum(test_data, axis=0)
    # Remove cross-sectional mean (Market Mode) - essential for pair trading / relative potential
    f_prices_test = f_prices_test - f_prices_test.mean(axis=1, keepdims=True)

    # Now f = Prices. df = Returns.
    # Re-run IC with Prices as State
    res_ic_price = test_force_return_correlation(f_prices_test, L, lag=1)
    print(f"1b. Price-Restoring Force IC: {res_ic_price['ic_mean']:.4f} (p={res_ic_price['p_value']:.4e})")

    # Test 2: Regression
    res_reg = test_dynamics_regression(f_prices_test, L)
    print(f"2. Dynamics Regression R2: {res_reg['r_squared']:.4f}, Beta: {res_reg['beta']:.4f}")

    # Visualization
    save_path = os.path.join(os.path.dirname(__file__), 'real_data_dynamics.png')
    visualize_dynamics_test_results(res_ic_price, save_path=save_path)

if __name__ == "__main__":
    run_real_dynamics_test()
