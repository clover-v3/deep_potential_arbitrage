
"""
Baseline Replication Pipeline (Han et al. 2021) - Rolling Window
End-to-end workflow:
1. Load Data (Merged Factors)
2. Iterative Walk-Forward:
   - For each Test Month M:
     - Train on [M-Lookback, M-1]
     - Fit PCA/Clusterer on Snapshot (Last Day of M-1)
     - Assign Clusters (Static for Month M)
     - Trade & Record PnL
3. Evaluate Performance
"""

import pandas as pd
import numpy as np
import os
import argparse
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.baseline.han_clustering import HanClusteringPipeline
from src.baseline.strategy import BaselineStrategy

def load_data(start_date, end_date, data_dir="data/processed/merged_factors"):
    if not os.path.exists(data_dir):
        # Graceful fallback for testing enviros or alternate paths
        if os.path.exists("src/data/processed/merged_factors"):
             data_dir = "src/data/processed/merged_factors"
        else:
             print(f"Warning: {data_dir} not found. Checking if files are local...")

    # Identify required months
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # We load slightly wider range to ensure we cover everything, or strict?
    # Strict matching of files is better.
    # But files are YYYY_MM.
    target_months = pd.period_range(start=start_ts, end=end_ts, freq='M')

    dfs = []
    found_any = False

    for period in target_months:
        y = period.year
        m = period.month
        filename = f"merged_factors_{y}_{m:02d}.parquet"
        path = os.path.join(data_dir, filename)

        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
            found_any = True

    if not found_any:
        # Check if legacy file exists
        files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and 'merged_factors_' in f] if os.path.exists(data_dir) else []
        if files:
            print("Warning: Partitioned monthly files not found. Attempting legacy load...")
            dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in files]
        else:
            print(f"No data found in {data_dir}. Returning empty DF.")
            return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])

    mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
    df = df[mask].sort_values(['date', 'permno'])

    return df

def get_feature_cols(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Metadata and Targets to exclude
    keys = ['permno', 'date', 'year', 'month', 'gvkey', 'cik', 'sic', 'naics', 'fyear', 'datadate']
    targets = ['ret', 'ret_x', 'ret_y', 'retx', 'prc', 'openprc', 'askhi', 'bidlo',
               'vol', 'vol_x', 'vol_y', 'shrout', 'shrout_x', 'shrout_y', 'cfacpr', 'cfacshr']

    feature_cols = []
    for c in numeric_cols:
        if c in keys or c in targets:
            continue
        if c.endswith('_x') or c.endswith('_y'):
            continue
        feature_cols.append(c)

    return feature_cols

def clean_features(df_train, feature_cols):
    """
    Robust cleaning for PCA input.
    Returns cleaned X_train (DataFrame) and list of used features.
    """
    if df_train.empty: return pd.DataFrame(), []

    # 0. Convert Infs
    X = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 1. Drop Sparse (>30% missing)
    nan_counts = X.isna().mean()
    valid_cols = nan_counts[nan_counts < 0.3].index.tolist()

    X = X[valid_cols]

    # 2. Impute with 0 (Standardized assumption)
    X = X.fillna(0)

    # 3. Drop Constant
    stds = X.std()
    non_const = stds[stds > 1e-6].index.tolist()

    X = X[non_const]

    return X, non_const

def run_rolling_pipeline(start_date: str, end_date: str, lookback_months: int = 12,
                         method: str = 'kmeans', n_clusters: int = 10, dist_quantile: float = None,
                         outlier_percentile: float = 95.0,
                         feature_set: str = 'all',  # 'all' or 'price_only'
                         universe_top_n: float | int | None = 2000,
                         strategy_kwargs: dict = None,
                         data_dir: str = "data/processed/merged_factors",
                         output_subdir: str = "default",
                         return_metrics: bool = False,
                         preloaded_df: pd.DataFrame = None,
                         preprocessed_data: bool = False):
    """
    Run the rolling window pipeline.

    feature_set: 'all' (default), 'price_only' (drop GHZ)
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}

    print(f"--- Running Rolling Pipeline ({method}) ---")
    print(f"Range: {start_date} to {end_date}")
    print(f"Features: {feature_set}")
    print(f"Strategy Kwargs: {strategy_kwargs}")

    # Strategy rolling window (for idiosyncratic z-score) – used to decide how
    # many days of pre-history we need to attach to each test month so that
    # trading can start from the first test day.
    # If user passed a custom 'window' into BaselineStrategy via strategy_kwargs,
    # we honor it; otherwise default to 20.
    signal_window = int(strategy_kwargs.get('window', 20))

    # ... setup paths ...
    base_res_dir = f"results/baseline/{output_subdir}"
    os.makedirs(base_res_dir, exist_ok=True)

    # 1. Load Data with Buffer
    test_start_dt = pd.to_datetime(start_date)
    # Approximate lookback buffer (Lookback + 1 month buffer for safety)
    if preloaded_df is not None:
        if preprocessed_data:
            # Zero-copy optimization: Assume passed DF is already filtered, sorted, and has 'year_month'
            print("Using Pre-Processed Data (Zero Copy)...")
            full_df = preloaded_df
        else:
            print("Using Preloaded Data...")
            # Approximate lookback buffer (Lookback + 1 month buffer for safety)
            buffer_days = (lookback_months + 1) * 32
            load_start = test_start_dt - pd.Timedelta(days=buffer_days)

            # Optimize: Filter FIRST, then copy. Avoids deep copy of unrelated data.
            mask = (preloaded_df['date'] >= load_start) & (preloaded_df['date'] <= pd.to_datetime(end_date))
            full_df = preloaded_df.loc[mask].copy()
            full_df.sort_values(['date', 'permno'], inplace=True)
    else:
        # Approximate lookback buffer (Lookback + 1 month buffer for safety)
        buffer_days = (lookback_months + 1) * 32
        load_start = test_start_dt - pd.Timedelta(days=buffer_days)
        print(f"Loading data from {load_start.date()} (buffer for training)")
        full_df = load_data(load_start, end_date, data_dir=data_dir)


    if full_df.empty:
        print("Error: No data loaded.")
        return None if return_metrics else None

    # Feature Filtering logic
    if feature_set == 'price_only' and not preprocessed_data:
        # Identify columns to DROP. GHZ factors usually don't have standard prefixes in this merged file?
        # We need to examine `daily_factors.py` vs `ghz_factors.py`.
        # Microstructure factors: 'ami_illig', 'parkinson_vol', 'turnover_var', 'downside_vol', 'overnight_gap', 'ret_skew', 'clv', 'ret_range'
        # GHZ factors: 'bm', 'ep', 'roe', etc.
        # It's safer to KEEP known microstructure columns + identifiers.

        identifiers = ['date', 'ticker', 'permno', 'shrout', 'prc', 'vol', 'siccd', 'industry', 'ret']
        micro_cols = ['ami_illig', 'turnover_var', 'parkinson_vol', 'downside_vol',
                      'overnight_gap', 'ret_skew', 'clv', 'ret_range'] # From daily_factors.py check

        # Check what's actually in df
        keep_cols = [c for c in full_df.columns if c in identifiers or c in micro_cols]
        print(f"Filtering features: Keeping {len(keep_cols)} columns (Price/Microstructure only).")
        full_df = full_df[keep_cols]

    if 'year_month' not in full_df.columns:
        full_df['year_month'] = full_df['date'].dt.to_period('M')

    # Identify Test Months
    test_months = full_df[full_df['date'] >= test_start_dt]['year_month'].unique()
    test_months = sorted(test_months)
    print(f"Test Months: {len(test_months)}")

    all_signals = []

    for m in test_months:
        print(f"\nProcessing Month: {m}")

        # Test (Trading) Data for this month
        mask_test = (full_df['year_month'] == m)
        df_test = full_df[mask_test].copy()
        if df_test.empty:
            continue

        # Train Window
        train_end_month = m - 1
        train_start_month = m - lookback_months

        mask_train = (full_df['year_month'] >= train_start_month) & (full_df['year_month'] <= train_end_month)
        df_train = full_df[mask_train].copy()

        if df_train.empty:
            print(f"Warning: No training data for {m}. Skipping.")
            continue

        print(f"Train Window: {train_start_month} to {train_end_month} ({len(df_train)} rows)")

        # ------------------------------------------------------------------
        # Universe Selection (Top-N by Market Cap at last training day)
        # ------------------------------------------------------------------
        # 1) Determine last training day snapshot
        last_day_train = df_train['date'].max()
        df_snapshot = df_train[df_train['date'] == last_day_train].copy()
        df_snapshot = df_snapshot.drop_duplicates(subset=['permno'])

        # 2) If requested, restrict to a Top-N universe based on market cap proxy.
        #    - Prefer prc * shrout if both available.
        #    - Fallback to prc * vol if shrout missing.
        #    - If N is None / NaN / >= total, keep full universe.
        if universe_top_n is not None:
            try:
                # Handle np.nan gracefully
                if isinstance(universe_top_n, float) and np.isnan(universe_top_n):
                    effective_top_n = None
                else:
                    effective_top_n = int(universe_top_n)
            except (TypeError, ValueError):
                effective_top_n = None

            if effective_top_n is not None and effective_top_n > 0:
                # Build market-cap-like metric
                mkt_cap = None
                if {'prc', 'shrout'}.issubset(df_snapshot.columns):
                    mkt_cap = (df_snapshot['prc'].astype(float) *
                               df_snapshot['shrout'].astype(float))
                elif {'prc', 'vol'}.issubset(df_snapshot.columns):
                    mkt_cap = (df_snapshot['prc'].astype(float) *
                               df_snapshot['vol'].astype(float))

                if mkt_cap is not None:
                    df_snapshot = df_snapshot.assign(_universe_mktcap=mkt_cap)
                    df_snapshot = df_snapshot.dropna(subset=['_universe_mktcap'])

                    if not df_snapshot.empty:
                        df_snapshot_sorted = df_snapshot.sort_values(
                            '_universe_mktcap', ascending=False
                        )
                        all_permnos = df_snapshot_sorted['permno'].unique()

                        # If N >= universe size, keep all; else take Top-N
                        if effective_top_n >= len(all_permnos):
                            selected_permnos = all_permnos
                        else:
                            selected_permnos = all_permnos[:effective_top_n]

                        selected_permnos = set(selected_permnos)

                        # Apply universe filter to train / test data for this month
                        df_train = df_train[df_train['permno'].isin(selected_permnos)].copy()
                        df_test = df_test[df_test['permno'].isin(selected_permnos)].copy()

                        # Rebuild snapshot after filtering
                        df_snapshot = df_train[df_train['date'] == last_day_train].copy()
                        df_snapshot = df_snapshot.drop_duplicates(subset=['permno'])

                        print(f"Universe filter applied: Top-{effective_top_n} names "
                              f"(actual {len(selected_permnos)} permnos) on {last_day_train.date()}")

        if df_snapshot.empty or len(df_snapshot) < 50:
            print("Snapshot insufficient.")
            continue

        print(f"Snapshot Data ({last_day_train.date()}): {len(df_snapshot)} tickers")

        # Clean Features
        feature_cols_snap = get_feature_cols(df_snapshot)
        X_snap, valid_feats = clean_features(df_snapshot, feature_cols_snap)

        if X_snap.empty:
             print("Features empty.")
             continue

        # Fit Pipeline
        # 2. Clustering
        # Pass kwargs for dynamic handling
        pipeline = HanClusteringPipeline(method=method, n_clusters=n_clusters,
                                        dist_quantile=dist_quantile,
                                        outlier_percentile=outlier_percentile)

        # Fit
        try:
             pipeline.fit(X_snap) # Fit on the cleaned snapshot features
        except ValueError as e:
             print(f"Fit failed for {m}: {e}")
             continue

        # Predict Labels for Test Selection
        # Wait, Han et al: "Form clusters on t-1, trade on t".
        # We need to assign test stocks to clusters.
        # But if clusters are "sub-universes", do we re-assign?
        # Han's method: Cluster formation period (Training).
        # Trading period: Objects are fixed?
        # Actually in Rolling Window: We cluster stocks using Training Data.
        # The result is: "Stock A belongs to Cluster 1".
        # We apply this mapping to Test Data.

        # Get labels for the LAST observation of each ticker in df_train to define its cluster for the test month
        # df_train sorted by date?
        # df_train_last = df_train.sort_values('date').groupby('ticker').last()

        # Re-predict? pipeline.predict(df_test)?
        # If dataset is characteristics, we should predict based on current characteristics?
        # Strategy: "At the beginning of month t..." -> Use characteristics at t-1.
        # So we predict cluster labels using Data at (Test Start - 1 Day).
        # Or just use the labels from training fit if it used the last snapshot.

        # HanClusteringPipeline.fit uses the WHOLE df_train provided?
        # Let's check fit implementation. It likely uses all rows.
        # Usually valid pair trading clusters are formed on a "Formation Period".
        # We assume the cluster membership holds for the "Trading Period".

        # Simplest consistent approach:
        # Predict clusters for df_test using the fitted model?
        # No, that implies clusters change every day in test.
        # We want STABLE clusters for the month.
        # Solution: Predict using the LAST available fundamental data before Test Month.
        # which is exactly what df_train[-1] is structurally.

        # Let's use predict on the last day of training data to establish "Month's Clusters"
        # The model was fitted on X_snap (cleaned features from df_snapshot).
        # We need to get the labels for the *original* df_snapshot based on the fit.

        labels_snapshot = pipeline.labels_ # These are the labels for X_snap

        # Map ticker -> label
        # labels_snapshot indices are integers from df.
        # Need to join back.
        df_snapshot_with_labels = df_snapshot.loc[X_snap.index].copy() # Ensure alignment with X_snap
        df_snapshot_with_labels['cluster'] = labels_snapshot
        ticker_cluster_map = df_snapshot_with_labels.set_index('permno')['cluster'].to_dict()

        # ------------------------------------------------------------------
        # Build signal DataFrame with extra pre-history for rolling window
        # ------------------------------------------------------------------
        # Trading period for this month:
        trade_start = df_test['date'].min()
        trade_end = df_test['date'].max()

        # To allow the strategy's rolling window to be "warm" from the first
        # trading day, we attach ~signal_window worth of history before the
        # month starts. Use 2x for calendar-day safety (holidays, weekends).
        hist_buffer_days = max(signal_window * 2, signal_window)
        hist_start = trade_start - pd.Timedelta(days=hist_buffer_days)

        # Signal universe is exactly the set of permnos present in the snapshot
        # / cluster map (after any Top-N filtering).
        signal_permnos = set(ticker_cluster_map.keys())

        mask_signals = (
            (full_df['date'] >= hist_start)
            & (full_df['date'] <= trade_end)
            & (full_df['permno'].isin(signal_permnos))
        )
        df_signals = full_df[mask_signals].copy()

        # Attach cluster labels to the extended history
        df_signals['cluster_label'] = df_signals['permno'].map(ticker_cluster_map)

        # Drop unclustered (NaN or -1)
        df_signals = df_signals.dropna(subset=['cluster_label'])
        df_signals = df_signals[df_signals['cluster_label'] != -1]

        if df_signals.empty:
            print("No valid clusters for trading after attaching history.")
            continue

        # 3. Strategy
        strategy = BaselineStrategy(**strategy_kwargs)

        # Strategy.generate_signals expects Long format with 'cluster_label'
        month_res = strategy.generate_signals(df_signals)

        # Collect 'daily_ret' but only over the actual trading month
        if month_res['daily_ret'] is not None and not month_res['daily_ret'].empty:
            daily_ret = month_res['daily_ret']
            # Restrict to trading-period dates for this month
            mask_month = (daily_ret.index >= trade_start) & (daily_ret.index <= trade_end)
            daily_ret_month = daily_ret.loc[mask_month]
            if not daily_ret_month.empty:
                all_signals.append(daily_ret_month)

    if not all_signals:
        print("No signals generated.")
        full_pnl = pd.DataFrame()
    else:
        full_pnl = pd.concat(all_signals)
        full_pnl.sort_index(inplace=True)

    base_res_dir = f"results/baseline/{output_subdir}"
    os.makedirs(base_res_dir, exist_ok=True)
    full_pnl.to_csv(os.path.join(base_res_dir, f"rolling_pnl_{start_date}_{end_date}.csv"))
    print(f"Saved results to {base_res_dir}")

    # Calculate overall metrics if requested
    metrics = None
    if not full_pnl.empty:
        # Assuming full_pnl is a Series of returns? Or DataFrame?
        # strategy returns 'daily_ret' which is Series (or DF depending on implementation)
        # It's likely Series.
        strat = BaselineStrategy() # Dummy for static method if needed, but get_summary_metrics is instance method
        metrics = strat.get_summary_metrics(full_pnl)

    if return_metrics:
        return metrics, full_pnl

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', default='2024-05-01')
    parser.add_argument('--end_date', default='2024-05-31')
    parser.add_argument('--lookback', type=int, default=1, help="Months of training data")
    parser.add_argument('--method', default='kmeans', choices=['kmeans', 'dbscan', 'optics', 'agglomerative'])
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--outlier_percentile', type=float, default=95.0, help='Percentile for density outlier removal (0-100)')
    parser.add_argument('--eps', type=float, default=0.5, help='EPS for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=None, help='Min Samples for DBSCAN/OPTICS (None = log(N))')
    parser.add_argument('--distance_threshold', type=float, default=None, help='Distance Threshold for Agglomerative')
    parser.add_argument('--dist_quantile', type=float, default=None, help='Quantile (0-1) for dynamic threshold')

    # Ablation Args
    parser.add_argument('--feature_set', default='all', choices=['all', 'price_only'], help='Feature set to use')
    parser.add_argument('--cost_bps', type=float, default=0.0, help='Transaction cost in basis points per side')
    parser.add_argument('--signal_method', default='threshold', choices=['threshold', 'rank'], help='Signal generation method')
    parser.add_argument('--top_k_percent', type=int, default=20, help='Top K Percent for rank-based signal (e.g. 20 for 20%)')

    parser.add_argument('--output_subdir', type=str, default=None, help='Subdirectory for results')
    parser.add_argument('--data_dir', type=str, default="data/processed/merged_factors", help="Input Directory for Merged Factors")

    args = parser.parse_args()

    # Construct strategy_kwargs
    strat_kwargs = {
        'cost_bps': args.cost_bps,
        'signal_method': args.signal_method,
        'top_k_percent': args.top_k_percent
    }

    run_rolling_pipeline(
        args.start_date,
        args.end_date,
        args.lookback,
        args.method,
        args.n_clusters,
        args.dist_quantile,
        args.outlier_percentile,
        feature_set=args.feature_set,
        strategy_kwargs=strat_kwargs,
        data_dir=args.data_dir,
        output_subdir=args.output_subdir,
        return_metrics=False,
    )
