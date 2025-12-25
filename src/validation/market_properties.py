"""
Market Properties Validation Script

Directly tests for the existence of "Potential Wells" (Mean Reversion) in the market
by analyzing the statistical properties of Eigen-portfolios (PCA modes).

Theory:
- If Laplacian Dynamics exist, there are stable "modes" (eigenvectors of L) that act as harmonic oscillators.
- These modes should exhibit Mean Reversion (Hurst Exponent < 0.5).
- They should have stationary distributions (ADF Test).
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Add parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import data loader from previous script (create a shared util if needed, but for now specific import)
from run_real_data_dynamics import load_real_data

def get_hurst_exponent(time_series, max_lag=20):
    """
    Returns the Hurst Exponent of the time series vector ts.
    H < 0.5: Mean Reverting (Potential Well)
    H = 0.5: Random Walk (Free Particle)
    H > 0.5: Trending (Unstable)
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def run_market_physics_test():
    # 1. Load Data
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
        print("Error: CRSP data not found.")
        return

    # Load 3 years
    returns_df = load_real_data(data_dir, start_year=2020, end_year=2023, top_n=50)

    # 2. De-mean returns (Remove Market Mode? Or let PCA find it?)
    # Let PCA find it. It should be the first component.
    X = returns_df.values # (T, N)

    # 3. PCA / Eigen-decomposition
    # Covariance Matrix of Returns
    cov_mat = np.cov(X.T)
    evals, evecs = np.linalg.eigh(cov_mat)

    # Sort descending
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    # 4. Project Data onto Eigenvectors -> "Eigen-portfolios"
    # P = X @ E
    eigen_portfolios = X @ evecs # (T, N)

    # 5. Analyze Physics of Each Mode
    print("\nAnalyzing Physics of Eigen-portfolios (Potential Wells)...")
    print(f"{'Mode':<5} {'Explained Var':<15} {'Hurst Exp':<10} {'ADF p-val':<10} {'Interpretation'}")
    print("-" * 65)

    hursts = []
    adfs = []

    # Check top 20 modes
    for i in range(min(20, X.shape[1])):
        # Cumulative Sum = Price Path of the Portfolio
        price_path = np.cumsum(eigen_portfolios[:, i])

        # Hurst
        h = get_hurst_exponent(price_path, max_lag=100)
        hursts.append(h)

        # ADF Test (p-value < 0.05 => Stationary)
        try:
            adf_res = adfuller(price_path)
            p_val = adf_res[1]
        except:
            p_val = 1.0
        adfs.append(p_val)

        # Interpretation
        if h < 0.45: interp = "✅ Mean Rev (Well)"
        elif h > 0.55: interp = "❌ Trending"
        else: interp = "⚪ Random Walk"

        var_expl = evals[i] / np.sum(evals)
        print(f"{i:<5} {var_expl:<15.2%} {h:<10.4f} {p_val:<10.4f} {interp}")

    # Plot Hurst Distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hursts)), hursts, color=['red' if h > 0.5 else 'green' for h in hursts])
    plt.axhline(0.5, color='black', linestyle='--', label='Random Walk (0.5)')
    plt.xlabel('Eigen-mode Index (0=Market)')
    plt.ylabel('Hurst Exponent')
    plt.title('Physics of Market Modes: Finding the Potential Wells')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'market_physics_hurst.png'))
    print(f"\nPlot saved to market_physics_hurst.png")

if __name__ == "__main__":
    run_market_physics_test()
