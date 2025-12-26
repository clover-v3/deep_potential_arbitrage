import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm
from src.contrastive_bl.data_loader import ORCADataLoader
from src.contrastive_bl.orca import ORCAModel

def backtest_orca(data_root, model_path, start_year=2000, end_year=2023, gamma=1.0, device=None):
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
             device = 'mps'
        else:
            device = 'cpu'

    print(f"Running Backtest on {device}...")

    # 1. Load Data
    loader = ORCADataLoader(data_root)
    # Load all data for backtest
    loader.load_data(start_year, end_year)
    df = loader.build_orca_features()

    if df.empty:
        print("No data.")
        return

    # 2. Load Model
    # Determine features from data
    # Exclude IDs and metadata (Must match train.py)
    exclude_cols = ['permno', 'date', 'gvkey', 'cusip', 'valid_from',
                    'datadate', 'cusip6', 'ncusip6', 'namedt', 'nameendt']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    n_features = len(feature_cols)
    print(f"Features: {n_features}")

    model = ORCAModel(n_features=n_features, n_clusters=30).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Predict Clusters (Rolling / Monthly)
    # We iterate by month to simulate 'rebalancing'
    # Actually, we can pre-calculate clusters for all rows if model is static.
    # The paper implies using the trained model to assign clusters.

    # Prepare Tensor
    # Normalize with same stats as training?
    # Ideal: fit scaler on training period. Here we use global stats for simplicity/demo.
    # Robust conversion
    X_raw = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
     # X_raw = np.nan_to_num(X_raw) # Already handled by fillna(0)
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0) + 1e-6
    X_norm = (X_raw - mean) / std

    # Inference in batches
    batch_size = 4096
    all_clusters = []

    print("Predicting Clusters...")
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = torch.tensor(X_norm[i:i+batch_size]).to(device)
            # Forward through backbone -> cluster head
            h = model.backbone(batch)
            probs = model.cluster_head(h)
            clusters = torch.argmax(probs, dim=1).cpu().numpy()
            all_clusters.append(clusters)

    df['cluster'] = np.concatenate(all_clusters)

    # 4. Strategy Execution
    # Monthly Rebalance
    # Group by Date, then Cluster.

    df['date'] = pd.to_datetime(df['date'])
    dates = sorted(df['date'].unique())

    portfolio_returns = []

    print("Running Backtest...")
    for t in tqdm(dates):
        # Current Month Data
        period_df = df[df['date'] == t].copy()

        # We need Next Month Return for PnL
        # In this DF, 'ret' is valid for month t (the one ending at date t).
        # We trade at end of t, hold for t+1.
        # So we need outcome: ret_{t+1}.
        # Simple way: shift the 'ret' column in the full df?
        # Or look ahead.

        # Let's pivot returns to get lookahead easily.
        pass # Too slow inside loop.

    # Vectorized Lookahead
    # Sort by permno, date
    df = df.sort_values(['permno', 'date'])
    df['next_ret'] = df.groupby('permno')['ret'].shift(-1)

    # Drop rows where next_ret is missing (e.g. last month)
    df = df.dropna(subset=['next_ret'])

    # Re-loop
    total_pnl = 0
    daily_returns = [] # actually monthly

    for t in tqdm(dates):
        # At time t, we observe 'mom_1' (ret_t), 'cluster', etc.
        # We form portfolio.
        # Outcome is 'next_ret'.

        period_df = df[df['date'] == t]

        # Iterate Clusters
        longs = []
        shorts = []

        for k, cluster_df in period_df.groupby('cluster'):
            if len(cluster_df) < 5: continue # Skip tiny clusters

            # Ranking within cluster by mom_1 (Prior 1-month return)
            # "Sort assets ... by prior one-month return ... to identify top/bottom"
            # mom_1 is ret_{t-1}?
            # In loader: mom_1 = shift(1) ret. So it is ret_{t-1}. Correct.

            # Compute Momentum Spread
            # "The return differential ... defines a momentum spread"
            # Wrapper: Standardize rank?
            # "based on its rank ... diff between top and bottom"
            # Actually Algorithm 1 says:
            # 2: Momentum Spread: ... based on its rank of R_{t-1}.
            # 6: if Spread < -gamma * sigma ...

            # Interpretation: Spread(x) = Rank(x) centered?
            # Or Spread(x) = R_{t-1}(x) - ClusterMean(R_{t-1})?
            # Ranking usually implies Uniformity.
            # Let's assuming Z-score of R_{t-1} within cluster?
            # Paper says "based on its rank ...".
            # Let's use Z-score of return.

            vals = cluster_df['mom_1'].values
            mu = vals.mean()
            sig = vals.std() + 1e-8
            z_score = (vals - mu) / sig

            # Signals
            # Long Losers (Mean Reversion) -> Spread < -gamma
            # Short Winners -> Spread > gamma

            # If mom_1 is high (Winner), spread is high. -> Short.
            # If mom_1 is low (Loser), spread is low. -> Long.

            long_mask = z_score < -gamma
            short_mask = z_score > gamma

            longs.extend(cluster_df[long_mask]['next_ret'].values)
            shorts.extend(cluster_df[short_mask]['next_ret'].values) # Short -> -1 * ret

        # Portfolio Return (Equal Weighted)
        pnl = 0
        count = len(longs) + len(shorts)

        if count > 0:
            # Long: sum(ret)
            # Short: sum(-ret)
            r_long = np.sum(longs)
            r_short = np.sum(shorts) * -1

            # Avg return
            avg_ret = (r_long + r_short) / count
            portfolio_returns.append({'date': t, 'ret': avg_ret, 'positions': count})
        else:
            portfolio_returns.append({'date': t, 'ret': 0.0, 'positions': 0})

    res_df = pd.DataFrame(portfolio_returns)
    res_df['cum_ret'] = (1 + res_df['ret']).cumprod()

    # --- Metrics Logic ---
    print("\n" + "="*40)
    print("       BACKTEST PERFORMANCE SUMMARY       ")
    print("="*40)

    # 1. Total Cumulative Return
    if not res_df.empty:
        total_ret = res_df['cum_ret'].iloc[-1] - 1
    else:
        total_ret = 0.0

    # 2. Annualized Return & Vol
    # Assuming monthly rebalancing (12 periods/year)
    ann_ret = res_df['ret'].mean() * 12
    ann_vol = res_df['ret'].std() * np.sqrt(12) + 1e-9

    # 3. Sharpe Ratio
    sharpe = ann_ret / ann_vol

    # 4. Max Drawdown
    cum_series = res_df['cum_ret']
    running_max = cum_series.cummax()
    drawdown = (cum_series - running_max) / running_max
    max_dd = drawdown.min()

    # 5. Calmar Ratio (Ann Return / Abs Max DD)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # 6. Win Rate (Months with > 0 return)
    win_months = (res_df['ret'] > 0).sum()
    total_months = len(res_df)
    win_rate = win_months / total_months if total_months > 0 else 0.0

    print(f"Total Return:      {total_ret*100:.2f}%")
    print(f"Annualized Return: {ann_ret*100:.2f}%")
    print(f"Annualized Vol:    {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.4f}")
    print(f"Max Drawdown:      {max_dd*100:.2f}%")
    print(f"Calmar Ratio:      {calmar:.4f}")
    print(f"Win Rate:          {win_rate*100:.2f}%")
    print("="*40)

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # 1. Cumulative Return
        plt.subplot(2, 1, 1)
        plt.plot(res_df['date'], res_df['cum_ret'], label='Portfolio')
        plt.title('Cumulative Return')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 2. Drawdown
        plt.subplot(2, 1, 2)
        plt.plot(res_df['date'], drawdown, label='Drawdown', color='red')
        plt.fill_between(res_df['date'], drawdown, 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("backtest_plots.png")
        print("Plots saved to backtest_plots.png")
        plt.close()
    except ImportError:
        print("Matplotlib not installed. Skipping plots.")
    except Exception as e:
        print(f"Error plotting: {e}")

    print("="*40)

    res_df.to_csv("backtest_results.csv")
    print(f"Detailed results saved to backtest_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/raw_ghz")
    parser.add_argument("--model_path", type=str, default="orca_model.pth")
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")
    args = parser.parse_args()

    backtest_orca(args.data_root, args.model_path, start_year=args.start_year, end_year=args.end_year, device=args.device)
