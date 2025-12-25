"""
Auto-Tuning Grid Search & Evaluation
Logic:
1. Split Data into Validation (In-Sample) and Test (Out-of-Sample) periods based on split_ratio.
2. Run Grid Search on Validation Period.
3. Select Best Hyperparameters (Max Sharpe).
4. Run Backtest on Test Period using Best Parameters.
"""
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import argparse
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.baseline.run_pipeline import run_rolling_pipeline, load_data

def run_single_experiment(params, start_date, end_date, lookback, preloaded_df):
    """
    Wrapper for parallel execution.
    """
    method = params['method']
    subdir = params['subdir']

    # print(f"-> Starting {subdir}...") # Reduce spam

    try:
        metrics = None
        if method == 'kmeans':
            metrics, _ = run_rolling_pipeline(
                start_date, end_date, lookback,
                method='kmeans', n_clusters=params['value'],
                outlier_percentile=95.0,
                output_subdir=subdir, return_metrics=True,
                preloaded_df=preloaded_df,
                preprocessed_data=True
            )
            res = {'Method': 'K-Means', 'Param': 'k', 'Value': params['value']}

        elif method == 'agglomerative':
            metrics, _ = run_rolling_pipeline(
                start_date, end_date, lookback,
                method='agglomerative',
                dist_quantile=params['value'],
                outlier_percentile=95.0,
                output_subdir=subdir, return_metrics=True,
                preloaded_df=preloaded_df,
                preprocessed_data=True
            )
            res = {'Method': 'Agglomerative', 'Param': 'quantile', 'Value': params['value']}

        elif method == 'dbscan':
            metrics, _ = run_rolling_pipeline(
                start_date, end_date, lookback,
                method='dbscan',
                dist_quantile=params['value'],
                outlier_percentile=95.0,
                output_subdir=subdir, return_metrics=True,
                preloaded_df=preloaded_df,
                preprocessed_data=True
            )
            res = {'Method': 'DBSCAN', 'Param': 'quantile', 'Value': params['value']}

        if metrics:
            res.update(metrics)
            return res

    except Exception as e:
        print(f"Failed {subdir}: {e}")
        return None

    return None

def run_auto_tune(start_date, end_date, lookback, data_dir, output_dir, split_ratio=0.8):

    # 1. Determine Splits
    full_range = pd.to_datetime([start_date, end_date])
    total_days = (full_range[1] - full_range[0]).days
    split_days = int(total_days * split_ratio)

    split_date_dt = full_range[0] + pd.Timedelta(days=split_days)
    # Align to month end usually, but simplistic date is fine for now
    split_date = split_date_dt.strftime('%Y-%m-%d')

    # Validation: Start -> Split
    # Test: Split -> End
    val_start = start_date
    val_end = split_date

    # Buffer for Test: Needs 'lookback' months before test_start
    # We simply set test_start = split_date + 1 day
    test_start = (split_date_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    test_end = end_date

    print("="*60)
    print(f"AUTO-TUNING CONFIGURATION")
    print(f"Full Range:      {start_date} to {end_date}")
    print(f"Validation (IS): {val_start} to {val_end} ({split_ratio*100:.0f}%)")
    print(f"Test (OOS):      {test_start} to {test_end}")
    print("="*60)

    # 0. Pre-load Data ONCE (Use Full Range to cover everything)
    print("Pre-loading Data for Efficiency...")
    # Calculate buffer for validation start
    val_start_dt = pd.to_datetime(val_start)
    buffer_days = (lookback + 1) * 32
    load_start = val_start_dt - pd.Timedelta(days=buffer_days)

    # Load everything needed
    full_df = load_data(load_start, end_date, data_dir=data_dir)

    if full_df.empty:
        print("Error: No data loaded. Aborting.")
        return

    print(f"Data Loaded: {len(full_df)} rows.")

    # --- PHASE 1: GRID SEARCH (VALIDATION) ---
    print(f"\n>>> Starting PHASE 1: Grid Search on Validation Set ({val_start} - {val_end})")

    # Optimize: Prepare Validation Data ONCE
    # 1. Slice
    val_start_dt = pd.to_datetime(val_start)
    buffer_days = (lookback + 1) * 32
    load_start = val_start_dt - pd.Timedelta(days=buffer_days)
    val_end_dt = pd.to_datetime(val_end)

    mask_val = (full_df['date'] >= load_start) & (full_df['date'] <= val_end_dt)
    val_df = full_df.loc[mask_val].copy()

    # 2. Sort & Pre-calc
    val_df.sort_values(['date', 'permno'], inplace=True)
    val_df['year_month'] = val_df['date'].dt.to_period('M')

    print(f"Validation Data Prepared: {len(val_df)} rows (Reduced from {len(full_df)})")

    # Grids
    k_grid = [10, 50, 100, 200, 300, 500, 1000]
    q_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]

    tasks = []
    for k in k_grid:
        tasks.append({'method': 'kmeans', 'value': k, 'subdir': f"val_kmeans_k_{k}"})
    for q in q_grid:
        tasks.append({'method': 'agglomerative', 'value': q, 'subdir': f"val_agglo_q_{q}"})
    for q in q_grid:
        tasks.append({'method': 'dbscan', 'value': q, 'subdir': f"val_dbscan_q_{q}"})

    n_jobs = max(20, multiprocessing.cpu_count() - 1)
    print(f"Running {len(tasks)} experiments with {n_jobs} cores...")

    # Pass 'val_df' instead of 'full_df'
    # And use wrapper that sets preprocessed_data=True
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_experiment)(task, val_start, val_end, lookback, val_df)
        for task in tasks
    )

    results = [r for r in results if r is not None]

    if not results:
        print("Grid Search Failed. No results.")
        return

    df_res = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    df_res.to_csv(os.path.join(output_dir, "validation_grid_search.csv"), index=False)

    print("\n>>> Validation Results:")
    cols = ['Method', 'Param', 'Value', 'sharpe', 'total_return', 'max_drawdown']
    print(df_res[[c for c in cols if c in df_res.columns]].to_markdown(index=False))

    # Select Best PER METHOD
    methods = df_res['Method'].unique()

    print(f"\n>>> Starting PHASE 2: Run Best Model PER METHOD on Test Set ({test_start} - {test_end})")

    for m in methods:
        df_m = df_res[df_res['Method'] == m]
        if df_m.empty: continue

        # Best params for this method
        best_row = df_m.loc[df_m['sharpe'].idxmax()]
        best_val = best_row['Value']
        best_sharpe = best_row['sharpe']

        print(f"\n[Method: {m}] Best Valid Sharpe: {best_sharpe:.4f} (Param={best_val})")

        # Map back to pipeline args
        method_arg = m.lower()
        if 'k-means' in method_arg: method_arg = 'kmeans'
        elif 'agglomerative' in method_arg: method_arg = 'agglomerative'
        elif 'dbscan' in method_arg: method_arg = 'dbscan'

        # Args
        n_clusters = 10
        dist_quantile = None
        if method_arg == 'kmeans':
            n_clusters = int(best_val)
        else:
            dist_quantile = float(best_val)

        final_subdir = f"FINAL_TEST_BEST_{method_arg}_{best_val}"

        run_rolling_pipeline(
            test_start, test_end, lookback,
            method=method_arg,
            n_clusters=n_clusters,
            dist_quantile=dist_quantile,
            outlier_percentile=95.0,
            output_subdir=final_subdir,
            preloaded_df=full_df, # Use cached data
            data_dir=data_dir
        )
        print(f"-> Saved to results/baseline/{final_subdir}")

    print("\nDone. All Best Models executed on Test Set.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default='1980-01-01')
    parser.add_argument('--end_date', type=str, default='2024-12-31')
    parser.add_argument('--lookback', type=int, default=12)
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Ratio for Validation Split (In-Sample)")
    parser.add_argument('--data_dir', type=str, default="data/processed/merged_factors")
    parser.add_argument('--output_dir', type=str, default="results/baseline/autotune")

    args = parser.parse_args()

    run_auto_tune(args.start_date, args.end_date, args.lookback, args.data_dir, args.output_dir, args.split_ratio)
