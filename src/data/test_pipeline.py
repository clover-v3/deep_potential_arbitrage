"""
Test Pipeline Script
"""

import os
import shutil
import pandas as pd
import numpy as np
from src.data.create_dummy_data import create_dummy_parquet
from src.data.loader import DataLoader
from src.data.features import FeatureEngine

def test_pipeline():
    root_dir = "test_data_root"

    # 1. Generate Dummy Data
    print("Generating dummy data...")
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    create_dummy_parquet(root_dir)

    # 2. Initialize Pipeline with Universe
    univ_path = os.path.join(root_dir, 'universe.parquet')
    loader = DataLoader(root_dir, universe_path=univ_path)
    # Check if universe loaded
    if loader.universe is not None:
        print(f"Universe loaded successfully: {loader.universe.shape}")
    else:
        print("Universe load failed (check if create_dummy_data created it)")

    engine = FeatureEngine(loader)

    # 3. Build Dataset
    start_date = "2023-01-01"
    end_date = "2023-04-10"
    print(f"Building dataset from {start_date} to {end_date}...")

    try:
        df = engine.build_dataset(start_date, end_date, use_1min_vol=True)
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        return

    # 4. Verification
    print("\nDataset Info:")
    print(df.info())
    print("\nSample Data:")
    print(df.head())

    # Checks
    print("\nVerification Checks:")

    # Check 1: MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        print("✅ Index is MultiIndex (Date, Ticker)")
    else:
        print("❌ Index is NOT MultiIndex")

    # Check 2: Technicals present
    if 'vol_20d' in df.columns or 'rsi_14' in df.columns: # Might be NaN due to short window
        print("✅ Technical columns created")
    else:
        print("❌ Technical columns missing (Check window sizes vs data length)")

    # Check 3: 1-min aggregated feature (Advanced)
    if 'realized_skew' in df.columns and 'amihud_illiquidity' in df.columns:
        print("✅ Advanced Intraday Features (Skew, Amihud, etc.) created")
    else:
        print("❌ Advanced Intraday Features missing")
        print(f"   Columns found: {df.columns.tolist()}")

    # Check 4: Fundamentals present
    if 'pe_ttm' in df.columns and 'market_cap' in df.columns:
        print("✅ Fundamentals present")
    else:
        print("❌ Fundamentals missing")

    # Check 5: Distribution (Z-score should have mean ~0)
    # Note: Short dummy data might not be perfect 0, but check logic run
    print(f"Mean of PE (Z-scored): {df['pe_ttm'].mean():.4f}")

    # 5. Clustering Verification
    print("\n-----------------------------------")
    print("Testing Clustering Engine...")
    from src.baseline.han_clustering import HanClusteringPipeline

    # Init pipeline
    cluster_pipe = HanClusteringPipeline(n_components=5, n_clusters=3) # 3 dummy industries

    # Run
    results = cluster_pipe.train_clustering(df)

    if 'error' in results:
        print(f"❌ Clustering Failed: {results['error']}")
    else:
        metrics = results['metrics']
        print(f"✅ Clustering Complete")
        print(f"   Explained Variance (PCA): {metrics['explained_variance']:.4f}")
        print(f"   Inertia (KMeans): {metrics['inertia']:.4f}")
        if 'nmi' in metrics:
            print(f"   NMI (vs Ground Truth): {metrics['nmi']:.4f}")
            print(f"   ARI (vs Ground Truth): {metrics['ari']:.4f}")

        # Check clusters
        sample_clusters = results['cluster_labels'].head(10)
        # print(f"\nSample Clusters:\n{sample_clusters}")

    # 6. Strategy Verification
    print("\n-----------------------------------")
    print("Testing Trading Strategy...")
    from src.baseline.strategy import BaselineStrategy

    # Needs prices (use 'close' feature if available in raw dummy data loading)
    # The 'df' from engine has features. We need raw prices.
    # We can fetch 'close' from df if we included it in build_dataset,
    # OR reload using loader.
    # In features.py we don't necessarily keep raw close in final output unless requested.
    # But we can reload for testing.

    prices = loader.load_daily_data('close', start_date, end_date)
    # Apply universe same way
    prices = loader._apply_universe_mask(prices)

    if prices.empty:
        print("❌ No price data for strategy test")
    else:
        # Use clusters from step 5
        # clusters is a Series (one snapshot). Strategy handles broadcasting? Yes.
        clusters = results['cluster_labels']

        strat = BaselineStrategy(entry_z=1.0, window=5) # Short window for test
        strat_res = strat.generate_signals(prices, clusters)

        print("✅ Strategy Signal Generation Complete")
        print(f"   Total PnL (Dummy): {strat_res['total_pnl'].sum():.4f}")
        print(f"   Active Positions: {np.abs(strat_res['positions']).sum().sum()}")

        # Test Metrics
        metrics = strat.get_summary_metrics(strat_res['total_pnl'])
        print(f"   Sharpe Ratio: {metrics['sharpe']:.4f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.4f}")

    # 7. Open Universe Verification (No Universe Path)
    print("\n-----------------------------------")
    print("Testing Open Universe Loading (universe_path=None)...")
    loader_open = DataLoader(root_dir, universe_path=None)
    # Load 'close' for full period.
    # Expectation: TIC_008, TIC_009 are NaN in Jan/Feb (Month 1/2) but present in Mar/Apr.
    open_df = loader_open.load_daily_data('close', "2023-01-01", "2023-04-10")

    if open_df.empty:
        print("❌ Open Load Failed")
    else:
        print(f"✅ Open Load Shape: {open_df.shape}")
        # Check specific NaN condition
        # Jan 2023:
        jan_data = open_df.loc["2023-01-01":"2023-01-31"]
        if 'TIC_008' in jan_data.columns:
            nan_count = jan_data['TIC_008'].isna().sum()
            total_count = len(jan_data)
            print(f"   Jan TIC_008 NaNs: {nan_count}/{total_count}")
            if nan_count == total_count:
                print("✅ Correctly filled missing columns with NaN")
            else:
                 print(f"❌ TIC_008 should be all NaN in Jan, found {total_count - nan_count} valid")
        else:
             print("❌ TIC_008 column missing completely (should be present but NaN)")

    print("\nTest Complete.")

if __name__ == "__main__":
    test_pipeline()
