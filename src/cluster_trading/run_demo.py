import torch
import pandas as pd
import numpy as np
from src.cluster_trading.system import CoopTradingSystem

def run_demo():
    print("=== Cluster-Based Trading System: Small Scale Demo ===")

    # 1. Generate Synthetic Minute Data
    # 50 Tickers, 500 Minutes
    print("Generating synthetic minute-level data (50 tickers, 500 mins)...")
    N = 50
    T = 500
    tickers = [f"TICKER_{i:02d}" for i in range(N)]

    # Random Walk Prices
    returns = torch.randn(N, T) * 0.001
    prices = 100 * torch.exp(torch.cumsum(returns, dim=1)) # (N, T)

    # Add Batch Dim
    prices_input = prices.unsqueeze(0) # (1, N, T)

    # 2. Initialize System
    print("\nInitializing CoopTradingSystem...")
    # Window=60 mins, 5 Clusters
    system = CoopTradingSystem(n_clusters=5, feature_window=60, temp=0.5)

    # 3. Data-Driven Initialization (Clustering)
    print("Initialize clusters using K-Means++ on first chunk of data...")
    system.initialize_clusters(prices_input)

    # 4. Run Forward Pass (Simulation)
    print("\nRunning Forward Pass (Generating Signals)...")
    with torch.no_grad():
        # returns dict with 'features', 'assignments', 'z_scores', 'positions'
        out = system(prices_input, hard=True)

    positions = out['positions'].squeeze(0) # (N, T') T'=T-window+1
    assignments = out['assignments'] # (1, N, T', K)

    # 5. Display Results
    valid_steps = positions.shape[1]
    print(f"\nSimulation Complete. Valid Time Steps: {valid_steps}")

    # Show Snapshot at last timestamp
    print("\nSnapshot at Last Timestamp (t=-1):")

    # Get cluster assignments for last step
    # assignments is soft/hard one-hot. Get argmax.
    # shape (1, N, T', K)
    last_assign = assignments[0, :, -1, :].argmax(dim=-1) # (N,)
    last_pos = positions[:, -1] # (N,)
    last_prices = prices[:, -1] # (N,)

    # Create DataFrame for view
    df_view = pd.DataFrame({
        'Ticker': tickers,
        'Price': last_prices.numpy(),
        'Cluster': last_assign.numpy(),
        'Signal': last_pos.numpy()
    })

    # Filter for active signals
    active = df_view[df_view['Signal'].abs() > 0.1].sort_values('Signal')

    print(f"Total Active Signals: {len(active)} / {N}")
    print("\nTop Long Candidates:")
    print(active[active['Signal'] > 0].head(5).to_string(index=False))

    print("\nTop Short Candidates:")
    print(active[active['Signal'] < 0].head(5).to_string(index=False))

    print("\n=== Demo Finished Successfully ===")

if __name__ == "__main__":
    run_demo()
