import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.contrastive_bl.data_loader import ORCADataLoader

def test_pipeline(data_root, start_year, end_year):
    print("Testing ORCA Data Pipeline...")

    # Check if data exists
    if not os.path.exists(data_root):
        print(f"FAILED: Data root {data_root} does not exist.")
        return

    loader = ORCADataLoader(data_root)

    # Load 2020-2023 (Based on what we saw in the dir)
    print(f"1. Loading Data ({start_year}-{end_year})...")
    loader.load_data(start_year, end_year)

    if loader.builder.funda.empty:
        print("WARNING: Funda is empty.")
    else:
        print(f"   Funda: {loader.builder.funda.shape}")

    if loader.builder.msf.empty:
        print("WARNING: MSF is empty.")
    else:
        print(f"   MSF: {loader.builder.msf.shape}")

    print("2. Building ORCA Features...")
    try:
        df = loader.build_orca_features()

        if df.empty:
            print("FAILED: Resulting DataFrame is empty.")
        else:
            print(f"SUCCESS: Generated Features with shape {df.shape}")
            print("\nColumns:")
            print(df.columns.tolist())
            print("\nSample Data:")
            print(df.head())

            # Check for critical columns
            required = ['mom_1', 'mom_12', 'atq', 'niq']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"WARNING: Critical columns missing: {missing}")
            else:
                print("Verified: Critical columns present.")

    except Exception as e:
        print(f"FAILED with Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/raw_ghz")
    parser.add_argument("--start_year", type=int, default=2020)
    parser.add_argument("--end_year", type=int, default=2023)
    args = parser.parse_args()

    test_pipeline(args.data_root, args.start_year, args.end_year)
