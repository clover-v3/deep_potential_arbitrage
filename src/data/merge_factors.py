
"""
Merge Daily Factors (High Freq) and GHZ Factors (Low Freq - Monthly/Annual).
Aligns low-frequency data to high-frequency timestamps using forward-filling logic (asof merge).
"""

import pandas as pd
import numpy as np
import os
import argparse

def load_daily_factors(start_date, end_date, data_root):
    """
    Load partitioned daily factors within range.
    Expects files like daily_factors_YYYY_MM.parquet
    """
    print(f"Loading Daily Factors from {data_root} ({start_date} to {end_date})...")
    if not os.path.exists(data_root):
        print(f"Error: {data_root} does not exist.")
        return pd.DataFrame()

    files = sorted([f for f in os.listdir(data_root) if f.endswith('.parquet')])
    if not files:
        print("No daily factor files found.")
        return pd.DataFrame()

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    dfs = []
    for f in files:
        # Extract YYYY_MM from filename "daily_factors_2024_01.parquet"
        try:
            parts = f.replace('.parquet', '').split('_')
            year = int(parts[-2])
            month = int(parts[-1])
            file_date = pd.Timestamp(year=year, month=month, day=1)

            # Simple check: if file month is within range (roughly)
            file_end = file_date + pd.DateOffset(months=1)
            if file_end < start_ts or file_date > end_ts:
                continue

            path = os.path.join(data_root, f)
            df_chunk = pd.read_parquet(path)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)

    # Filter strict
    full_df['date'] = pd.to_datetime(full_df['date'])
    mask = (full_df['date'] >= start_ts) & (full_df['date'] <= end_ts)
    return full_df[mask].sort_values(['permno', 'date'])

def load_ghz_factors(start_year, end_year, data_root):
    """
    Load partitioned GHZ factors for a range of years.
    Expects ghz_factors_YYYY.parquet
    """
    dfs = []
    # Load extra prior year for asof merge safety
    load_start = start_year - 1

    for y in range(load_start, end_year + 1):
        path = os.path.join(data_root, f"ghz_factors_{y}.parquet")
        if os.path.exists(path):
            print(f"Loading GHZ {y} from {path}...")
            dfs.append(pd.read_parquet(path))
        else:
            print(f"Warning: GHZ file {path} not found.")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True).sort_values('date')

def merge_factors(start_date, end_date, daily_dir, ghz_dir, out_dir):
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # 1. Load Daily (High Freq Backbone)
    df_daily = load_daily_factors(start_date, end_date, data_root=daily_dir)
    if df_daily.empty:
        print("No Daily Data found.")
        return

    print(f"Daily Data Loaded: {df_daily.shape}. Range: {df_daily['date'].min()} - {df_daily['date'].max()}")

    # 2. Load GHZ Partitioned
    s_year = start_ts.year
    e_year = end_ts.year
    df_ghz = load_ghz_factors(s_year, e_year, data_root=ghz_dir)

    if df_ghz.empty:
        print(f"GHZ Factors file not found in {ghz_dir}. Run ghz_factors.py first.")
        return

    df_ghz['date'] = pd.to_datetime(df_ghz['date'])
    df_ghz = df_ghz.sort_values('date')

    # 3. Merge Logic
    df_daily = df_daily.sort_values('date')

    # Identify overlaps
    daily_cols = set(df_daily.columns)
    ghz_cols = set(df_ghz.columns)
    overlaps = daily_cols.intersection(ghz_cols)
    overlaps.discard('date')
    overlaps.discard('permno')

    if overlaps:
        print(f"Dropping overlapping columns from GHZ: {overlaps}")
        df_ghz = df_ghz.drop(columns=list(overlaps))

    print("Merging...")
    merged = pd.merge_asof(
        df_daily,
        df_ghz,
        on='date',
        by='permno',
        direction='backward',
        tolerance=pd.Timedelta(days=45)
    )

    print(f"Merged Total Shape: {merged.shape}")

    # 4. Save Partitioned (by Month)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output Directory: {out_dir}")

    merged['year'] = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month

    # Iterate through unique Year-Months in the merged data
    groups = merged.groupby(['year', 'month'])

    for (year, month), group in groups:
        out_file = os.path.join(out_dir, f"merged_factors_{year}_{month:02d}.parquet")
        group.drop(columns=['year', 'month']).to_parquet(out_file)
        print(f"Saved {year}-{month:02d} to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', default='2024-01-01')
    parser.add_argument('--end_date', default='2024-12-31')

    # Path Arguments
    parser.add_argument("--daily_dir", type=str, default="./data/processed/daily_factors", help="Input directory for Daily Factors")
    parser.add_argument("--ghz_dir", type=str, default="./data/processed/ghz_factors", help="Input directory for GHZ Factors")
    parser.add_argument("--out_dir", type=str, default="./data/processed/merged_factors", help="Output directory for Merged Factors")

    args = parser.parse_args()

    merge_factors(args.start_date, args.end_date, args.daily_dir, args.ghz_dir, args.out_dir)
