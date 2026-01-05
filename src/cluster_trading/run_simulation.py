import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.cluster_trading.system import CoopTradingSystem

def run_simulation():
    print("=== Cluster-Based Trading System: Cointegration Simulation ===")

    # 1. Generate Cointegrated Data
    # 5 Clusters, 10 Stocks each = 50 Stocks
    n_clusters = 5
    stocks_per_cluster = 10
    N = n_clusters * stocks_per_cluster
    T = 1000

    print(f"Generating data for {N} stocks over {T} time steps...")

    prices_list = []
    tickers = []
    ground_truth_clusters = []

    # Random Seed
    torch.manual_seed(42)
    np.random.seed(42)

    for c in range(n_clusters):
        # Common Factor for this cluster (Random Walk)
        # Start at 100
        factor_returns = torch.randn(T) * 0.001
        factor_price = 100 * torch.exp(torch.cumsum(factor_returns, dim=0))

        for s in range(stocks_per_cluster):
            ticker_id = c * stocks_per_cluster + s
            tickers.append(f"Stock_{c}_{s}")
            ground_truth_clusters.append(c)

            # Idiosyncratic Noise (Ornstein-Uhlenbeck-like or just white noise)
            noise = torch.randn(T) * 0.5

            # Stock Price = Factor + Noise
            stock_price = factor_price + noise

            # Inject Arbitrage Opportunity to Stock 0 in Cluster 0
            if c == 0 and s == 0:
                print(f"Injecting Massive Step Shock to Stock_{c}_{s} at t=800...")
                shock = torch.zeros(T)
                shock[800:900] = 20.0 # Huge Shock (20%)
                stock_price = stock_price + shock

            prices_list.append(stock_price)

    prices = torch.stack(prices_list) # (N, T)

    # View correlation of first few
    # print("Correlation matrix of Cluster 0:", torch.corrcoef(prices[:5, :]))

    prices_input = prices.unsqueeze(0) # (1, N, T)

    # 2. Init System
    print("\nInitializing System...")
    # Init System with new parameters
    # Threshold 1.5, Scaling Factor 5.0 (Sharper activation)
    # Stop Threshold 50.0 (Temporarily disable stop loss for this test or make it high enough)
    # We want to detect the shock of 20.0 sigma?
    # Our shock is +20.0 price? No, returns are 0.001. Price is ~100.
    # A step shock of +20.0 on price 100 is 20%.
    # With std of returns ~0.001, sigma is small. Feature extraction uses rolling Z-score.
    # The Z-score will be huge. A 20% jump vs 0.1% vol is 200 sigma.
    # If stop_threshold is default 4.0, this signal will confirm be KILLED by the stop loss.
    # To verify the SIGNAL detection first, we should set stop_threshold very high (e.g. 1000).
    # Or, we want to verify that the system DOES kill it if we set it low.
    print("Initialize System with High Stop Threshold and Normal Similarity Threshold...")
    system = CoopTradingSystem(
        n_clusters=n_clusters,
        feature_window=60,
        temp=0.1,
        entry_threshold=1.5,
        stop_threshold=1000.0, # Disable stop loss for this massive shock test
        similarity_threshold=0.5, # Normal threshold
        scaling_factor=5.0
    )

    # 3. Simulation
    print("Initializing Clusters...")
    system.initialize_clusters(prices_input[..., :200])

    print("Running Forward Pass...")
    with torch.no_grad():
        out = system(prices_input, hard=True)

    positions = out['positions'].squeeze(0) # (N, T)
    assignments = out['assignments'].squeeze(0) # (N, T, K)
    z_scores = out['z_scores'].squeeze(0) # (N, T, W)
    gate = out['gate'].squeeze(0) # (N, T, 1)

    # 4. Analyze Stock 0
    # Note on Alignment:
    # FeatureExtractor output length is T - W + 1.
    # index 't' in output corresponds to window ending at 't + W - 1'.
    # So to find signal at time T_sim, we need index i = T_sim - (W - 1).
    window = 60
    offset = window - 1

    start_idx = 800 - offset
    end_idx = 900 - offset

    # Ensure valid bounds
    start_idx = max(0, start_idx)
    end_idx = min(positions.shape[1], end_idx)

    idx = 0 # Stock_0_0
    pos_series = positions[idx, start_idx:end_idx]

    # DEBUG Level 2
    # Z-score at time 800 (window ending at 800) is at index 800-offset
    z_idx = 800 - offset
    # z_scores last dim is window position, -1 is current time
    print(f"\nDEBUG: Stock 0 Z-scores at t=800 (idx={z_idx}): {z_scores[idx, z_idx, -1]:.4f}")
    print(f"DEBUG: Stock 0 Signal at t=800: {positions[idx, z_idx]:.4f}")
    print(f"DEBUG: Stock 0 Outlier Gate at t=800: {gate[idx, z_idx].item():.4f}")

    print("\nAnalysis of Shocked Stock (Stock_0_0) during Event (t=800-900):")
    print(f"Mean Signal Value: {pos_series.mean().item():.4f}")

    n_shorts = (pos_series < -0.5).sum().item()
    print(f"Number of Short Signals triggered (< -0.5): {n_shorts} / 100 steps")

    # Check for ANY activation
    n_active = (pos_series.abs() > 0.01).sum().item()

    # Cluster Assignment Stability
    # Did it stay in Cluster 0?
    sim_assigns = assignments[idx, 800:900].argmax(dim=-1)
    mode_cluster = torch.mode(sim_assigns).values.item()
    print(f"Dominant Cluster Assignment: {mode_cluster} (Ground Truth: 0)")

    # 5. Global Stats
    print("\nGlobal Statistics:")
    active_mask = positions.abs() > 0.5
    print(f"Total Active Signals across all tickers/time: {active_mask.sum().item()}")

    # 6. Verify Outlier Logic
    outliers_detected = (gate < 0.1).sum().item()
    print(f"Total Outlier Steps detected (Gate < 0.1): {outliers_detected} / {gate.numel()}")

    # Optional: Save plot?
    # Not plotting, just text output.

    if n_shorts > 0:
        print("\nSUCCESS: System detected the arbitrage opportunity!")
    else:
        print("\nFAILURE: System missed the shock.")

if __name__ == "__main__":
    run_simulation()
