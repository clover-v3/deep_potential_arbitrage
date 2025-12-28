import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.contrastive_bl.data_loader import ORCADataLoader
from src.contrastive_bl.orca import ORCAModel
from src.baseline.strategy import BaselineStrategy

def load_universe(model_path):
    """Load Saved Universe (.npy) if available"""
    uni_path = model_path.replace('.pth', '_universe.npy')
    if os.path.exists(uni_path):
        print(f"Loading Universe from {uni_path}")
        return np.load(uni_path, allow_pickle=True)
    else:
        print("Warning: Universe file not found. Trading all stocks.")
        return None

def backtest_orca(
    data_root,
    model_path='orca_model.pth',
    start_year=2021,
    end_year=2021,
    device=None
):
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
             device = 'mps'
        else:
            device = 'cpu'

    print(f"Running Backtest on {device}...")

    # 1. Load Data (Monthly) for Clustering
    # We still use monthly features to determine the clusters
    loader = ORCADataLoader(data_root)
    print(f"Loading monthly data for years {start_year-1} to {end_year}...")
    loader.load_data(start_year, end_year)
    df_monthly = loader.build_orca_features()

    # 2. Filter by Universe (if trained on subset)
    universe = load_universe(model_path)
    if universe is not None:
        print(f"Filtering Backtest Universe: {len(universe)} stocks")
        df_monthly = df_monthly[df_monthly['permno'].isin(universe)].copy()

    if df_monthly.empty:
        print("Error: No data for backtest.")
        return

    # 3. Load Model
    # Need to infer n_features from data cols
    exclude_cols = ['permno', 'date', 'gvkey', 'cusip', 'valid_from',
                    'datadate', 'cusip6', 'ncusip6', 'namedt', 'nameendt']
    feature_cols = [c for c in df_monthly.columns if c not in exclude_cols]

    print(f"Features: {len(feature_cols)}")

    model = ORCAModel(n_features=len(feature_cols), n_clusters=30).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        return

    model.eval()

    # 4. Predict Clusters (Monthly)
    print("Predicting Monthly Clusters...")
    # Preprocess same as train
    if 'date' in df_monthly.columns:
        df_monthly = df_monthly.sort_values(['date', 'permno'])

    features = df_monthly[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    # Standardize & Winsorize (Using statistics from BACKTEST data? Or should use Train stats?
    # Ideally Train stats. For simplicity/baseline, we re-standardize per batch or globally.
    # Here implementing same logic as new train.py: Global standardization of loaded data + Clip.
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features = (features - mean) / std
    features = np.clip(features, -5, 5) # Winsorize

    X = torch.tensor(features).to(device)

    batch_size = 4096
    clusters = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size]
            h = model(x_batch) # Forward returns h (Backbone features)
            c_logits = model.get_cluster_prob(h) # Returns probabilities/logits
            c_ids = torch.argmax(c_logits, dim=1).cpu().numpy()
            clusters.append(c_ids)

    df_monthly['cluster_label'] = np.concatenate(clusters)

    # Create Mapping: (Date, Permno) -> Cluster
    # Date in monthly df is Month-End.
    # We will forward-fill this to daily data.
    # Ensure date is datetime
    df_monthly['date'] = pd.to_datetime(df_monthly['date'])

    # Keep only relevant columns for merging
    cluster_map = df_monthly[['date', 'permno', 'cluster_label']].copy()

    # 5. Load Daily Data for Trading
    # Since we don't have perfect daily data loader integrated yet, we construct it from
    # whatever 'dsf' files are available or fallback to monthly 'msf' if daily undefined.
    # User Requirement: "Read daily data... use BaselineStrategy"

    # Try loading Daily DSF
    # Assuming standard parquet structure `crsp_dsf/dsf_YYYY.parquet`
    # Load Daily Data with Buffer
    print("Loading Daily Price Data with Buffer...")
    daily_dfs = []
    dsf_path = os.path.join(data_root, 'crsp_dsf')
    msf_fallback = False

    # Load prev year for buffer
    years_to_load = [start_year - 1, start_year] if start_year == end_year else range(start_year - 1, end_year + 1)

    if os.path.exists(dsf_path):
        for y in years_to_load:
            fpath = os.path.join(dsf_path, f"dsf_{y}.parquet")
            if os.path.exists(fpath):
                daily_dfs.append(pd.read_parquet(fpath))

    if not daily_dfs:
        print("Warning: No Daily Data (crsp_dsf) found. Falling back to Monthly (MSF) as daily proxy.")
        # This basically runs the monthly strategy but passed through the daily engine logic
        # Re-use df_monthly but ensure returns are correct
        daily_df = df_monthly[['date', 'permno', 'ret']].copy()
        # Rename 'date' -> 'date' is fine
        msf_fallback = True
    else:
        daily_df = pd.concat(daily_dfs, ignore_index=True)

    # Ensure Filter by Universe
    if universe is not None:
        daily_df = daily_df[daily_df['permno'].isin(universe)].copy()

    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values(['permno', 'date'])

    # Determine Test Start Date for Slicing later
    test_start_date = pd.Timestamp(f"{start_year}-01-01")

    # 6. Merge Clusters into Daily Data
    # Merge AsOf or Forward Fill?
    # Cluster is determined at Month T-1 (or T). Strategy trades in Month T+1?
    # Usually: Features from Month T (End) -> Cluster for Month T+1.
    # Let's align:
    # cluster_map dates are Month Ends.
    # If using 'resampled' daily, simply FFILL.

    # Pivot Daily to have a continuous date index per permno
    # Actually `pd.merge_asof` is best for "Last known cluster".

    daily_df = daily_df.sort_values('date')
    cluster_map = cluster_map.sort_values('date')

    merged = pd.merge_asof(
        daily_df,
        cluster_map,
        on='date',
        by='permno',
        direction='backward' # daily date >= cluster date. Cluster determined at prev month end.
    )

    # Drop rows where cluster is NaN (before first month)
    merged = merged.dropna(subset=['cluster_label'])

    # Rename for Strategy
    # Strategy needs: date, permno, ret (daily return)
    # Check if 'ret' in daily
    if 'ret' not in merged.columns and 'retx' in merged.columns:
         merged['ret'] = merged['retx'] # Fallback

    # 7. Run Baseline Strategy
    print("Running Baseline Strategy (Daily)...")
    strategy = BaselineStrategy(
        entry_z=2.0,
        exit_z=0.0,
        window=20, # Daily Window for Idio Return? Z-score window.
        signal_method='threshold',
        stop_entry_last_n=5 # Use class logic
    )

    res = strategy.generate_signals(merged)

    # Extract Daily Returns
    daily_ret = res['daily_ret'] # Series

    # Slice to Test Period ONLY (Remove Buffer)
    final_port_ret = daily_ret[daily_ret.index >= test_start_date]
    if end_year:
        test_end_date = pd.Timestamp(f"{end_year}-12-31")
        final_port_ret = final_port_ret[final_port_ret.index <= test_end_date]

    print(f"Backtest Period: {final_port_ret.index.min()} to {final_port_ret.index.max()}")

    # 8. Metrics & Plotting

    # 8. Metrics & Plotting
    metrics = strategy.get_summary_metrics(final_port_ret)

    print("\n" + "="*40)
    print("       BACKTEST PERFORMANCE SUMMARY       ")
    print("="*40)
    print(f"Total Return:      {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Sharpe Ratio:      {metrics['sharpe']:.4f}")
    print(f"Max Drawdown:      {metrics['max_drawdown']*100:.2f}%")
    print("="*40)

    # Save Results
    df_res = pd.DataFrame({
        'date': final_port_ret.index,
        'ret': final_port_ret.values
    })
    df_res.to_csv("backtest_results.csv", index=False)
    print("Detailed results saved to backtest_results.csv")

    # Plot
    plt.figure(figsize=(10, 6))
    (1 + final_port_ret).cumprod().plot()
    plt.title(f"Cumulative Return (Sharpe: {metrics['sharpe']:.2f})")
    plt.grid(True, alpha=0.3)
    plt.savefig("backtest_plots.png")
    print("Plots saved to backtest_plots.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/raw_ghz")
    parser.add_argument("--model_path", type=str, default="orca_model.pth")
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2021)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    backtest_orca(args.data_root, args.model_path, start_year=args.start_year, end_year=args.end_year, device=args.device)
