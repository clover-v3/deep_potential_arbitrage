"""
Earnings Announcement (EA) Factor Builder

Constructs EA-related factors using:
- Compustat Quarterly (fundq) with RDQ
- CRSP Stocknames (for GVKEY <-> PERMNO link)
- Daily returns from already-processed daily_factors (permno, date, ret)

Outputs event-level factors and attaches them to the daily backbone via merge_asof.
"""

import os
import pandas as pd
import numpy as np


def _load_fundq_year(raw_dir: str, year: int) -> pd.DataFrame:
    """Load a single year's Compustat quarterly data fundq_YYYY.parquet."""
    path = os.path.join(raw_dir, "comp_fundq", f"fundq_{year}.parquet")
    if not os.path.exists(path):
        print(f"Warning: {path} not found for year {year}.")
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"Error loading fundq for year {year} from {path}: {e}")
        return pd.DataFrame()


def _load_stocknames(raw_dir: str) -> pd.DataFrame:
    """Load CRSP stocknames table from GHZ raw directory."""
    path = os.path.join(raw_dir, 'crsp_stocknames.parquet')
    if not os.path.exists(path):
        print(f"Warning: {path} not found. EA factors will be skipped.")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _build_ea_events_for_year(raw_dir: str, stock: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Build EA event table for a single calendar year based on RDQ.
    Columns: permno, rdq.
    """
    fundq = _load_fundq_year(raw_dir, year)
    if fundq.empty:
        return pd.DataFrame()

    if 'rdq' not in fundq.columns or 'cusip' not in fundq.columns or 'gvkey' not in fundq.columns:
        print(f"fundq {year} missing required columns (rdq/cusip/gvkey). EA factors disabled for this year.")
        return pd.DataFrame()

    fundq = fundq.copy()
    fundq['rdq'] = pd.to_datetime(fundq['rdq'], errors='coerce')
    fundq = fundq[fundq['rdq'].notna()]
    if fundq.empty:
        print(f"No valid RDQ dates in fundq for year {year}.")
        return pd.DataFrame()

    # Prepare keys
    stock_local = stock.copy()
    stock_local['namedt'] = pd.to_datetime(stock_local['namedt'])
    stock_local['nameenddt'] = pd.to_datetime(stock_local['nameenddt']).fillna(pd.Timestamp.max)
    stock_local['ncusip_6'] = stock_local['ncusip'].astype(str).str[:6]

    fundq['cusip_6'] = fundq['cusip'].astype(str).str[:6]

    # Merge on 6-digit CUSIP, then filter by rdq within name range
    merged = pd.merge(
        fundq[['gvkey', 'cusip_6', 'rdq']],
        stock_local[['permno', 'ncusip_6', 'namedt', 'nameenddt']],
        left_on='cusip_6',
        right_on='ncusip_6',
        how='inner',
    )
    mask = (merged['rdq'] >= merged['namedt']) & (merged['rdq'] <= merged['nameenddt'])
    events = merged[mask].copy()

    if events.empty:
        print(f"No EA events after linking fundq and stocknames for year {year}.")
        return pd.DataFrame()

    # Deduplicate (permno, rdq) pairs if multiple gvkeys map to same security/date
    events = events[['permno', 'rdq']].drop_duplicates()
    return events


def _load_ea_events_for_range(
    events_dir: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load EA events partitioned by year from a directory.
    Expects files like ea_events_YYYY.parquet partitioned by RDQ year.
    """
    if not os.path.exists(events_dir):
        print(f"EA events directory {events_dir} does not exist.")
        return pd.DataFrame()

    files = sorted(f for f in os.listdir(events_dir) if f.endswith(".parquet"))
    if not files:
        print(f"No EA events parquet files found in {events_dir}.")
        return pd.DataFrame()

    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year

    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            parts = f.replace(".parquet", "").split("_")
            year = int(parts[-1])
        except Exception:
            # Skip files that do not match pattern
            continue

        # Simple heuristic: include events whose RDQ year is in [start_year-1, end_year+1]
        if year < start_year - 1 or year > end_year + 1:
            continue

        path = os.path.join(events_dir, f)
        try:
            dfs.append(pd.read_parquet(path))
        except Exception as e:
            print(f"Skipping EA events file {path}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def attach_ea_factors_to_daily(
    df_daily: pd.DataFrame,
    raw_dir: str,
    events_dir: str | None = None,
    pre_days: int = 1,
    post_days: int = 3,
) -> pd.DataFrame:
    """
    Attach EA-related factors to an existing daily backbone.

    Parameters
    ----------
    df_daily : DataFrame
        Must contain at least ['permno', 'date', 'ret'].
    raw_dir : str
        GHZ raw directory containing comp_fundq and crsp_stocknames.parquet. Only
        used if events_dir is None (fallback to on-the-fly construction).
    events_dir : str | None
        Optional directory containing precomputed EA events parquet files
        partitioned by year (ea_events_YYYY.parquet). If provided and non-empty,
        only those files are used and raw_dir is ignored for events.
    events_path : str | None
        Optional path to a precomputed EA events parquet file. If provided and exists,
        it will be loaded instead of rebuilding events from raw_dir.
    pre_days : int
        Number of calendar days before RDQ to include in the EA window (default 1 for -1).
    post_days : int
        Number of calendar days after RDQ to include in the EA window (default 3 for +3).

    Returns
    -------
    DataFrame
        df_daily with two additional columns:
        - ear_ea: cumulative return over [rdq-pre_days, rdq+post_days] (default [-1,+3])
        - aeavol_ea: standard deviation of daily returns over the same window
        These factors are made available from rdq+post_days+1 day forward (no look-ahead).
    """
    required_cols = {'permno', 'date', 'ret'}
    if not required_cols.issubset(df_daily.columns):
        print("Daily data missing required columns for EA factors; skipping EA.")
        return df_daily

    # Prefer precomputed EA event table if available
    # Decide how to obtain EA events
    df_daily_local = df_daily[['permno', 'date', 'ret']].copy()
    df_daily_local['date'] = pd.to_datetime(df_daily_local['date'])

    if events_dir is not None:
        print(f"Loading EA events from directory {events_dir} for observed date range...")
        events = _load_ea_events_for_range(
            events_dir,
            start_date=df_daily_local['date'].min(),
            end_date=df_daily_local['date'].max(),
        )
    else:
        print(f"No EA events directory provided. Building EA events from raw_dir={raw_dir} for observed date range...")
        start_year = df_daily_local['date'].min().year
        end_year = df_daily_local['date'].max().year
        stock = _load_stocknames(raw_dir)
        if stock.empty:
            return df_daily
        dfs_events: list[pd.DataFrame] = []
        for y in range(start_year - 1, end_year + 2):
            ev_y = _build_ea_events_for_year(raw_dir, stock, y)
            if not ev_y.empty:
                dfs_events.append(ev_y)
        events = pd.concat(dfs_events, ignore_index=True) if dfs_events else pd.DataFrame()
    if events.empty:
        return df_daily

    # Cartesian merge on permno, then filter to window around RDQ
    joined = pd.merge(
        df_daily_local,
        events,
        on='permno',
        how='inner'
    )
    joined['rdq'] = pd.to_datetime(joined['rdq'])

    joined['delta_days'] = (joined['date'] - joined['rdq']).dt.days
    window_mask = (joined['delta_days'] >= -pre_days) & (joined['delta_days'] <= post_days)
    joined = joined[window_mask]

    if joined.empty:
        print("No overlapping daily returns around RDQ; EA factors will be empty.")
        return df_daily

    # Aggregate per (permno, rdq) event
    grouped = joined.groupby(['permno', 'rdq'])
    ea_events = grouped['ret'].agg(
        ear_ea=lambda x: (1 + x).prod() - 1,
        aeavol_ea=lambda x: x.std(ddof=0)
    ).reset_index()

    # Availability: from rdq + post_days + 1 day forward (window fully observed)
    ea_events['valid_from_ea'] = ea_events['rdq'] + pd.Timedelta(days=post_days + 1)

    # Ensure left keys sorted by (date, permno)
    df_daily_sorted = (
        df_daily.copy()
        .assign(date=pd.to_datetime(df_daily['date']))
        .sort_values(['date', 'permno'])
        .reset_index(drop=True)
    )

    # Ensure right keys sorted by (permno, valid_from_ea)
    ea_events = ea_events.copy()
    ea_events['valid_from_ea'] = pd.to_datetime(ea_events['valid_from_ea'])
    ea_events_sorted = (
        ea_events[['permno', 'valid_from_ea', 'ear_ea', 'aeavol_ea']]
        .sort_values(['valid_from_ea', 'permno'])
        .reset_index(drop=True)
    )

    # Attach EA factors via as-of merge (carry forward until next event)
    merged = pd.merge_asof(
        df_daily_sorted,
        ea_events_sorted,
        left_on='date',
        right_on='valid_from_ea',
        by='permno',
        direction='backward',
        tolerance=pd.Timedelta(days=365),
    )

    # ------------------------------------------------------------------
    # Construct simple, more continuous EA features for downstream models
    # ------------------------------------------------------------------
    # 1) Days since last EA window became fully observable.
    #
    # valid_from_ea = rdq + (post_days + 1)
    # => days_since_ea ≈ (date - valid_from_ea).days + (post_days + 1)
    #    = (date - rdq).days
    #
    # For dates without any prior EA event (NaN valid_from_ea), we set a large
    # sentinel value so that the feature remains numeric and "continuous".
    merged['ea_days_since'] = (
        (merged['date'] - merged['valid_from_ea'])
        .dt.days
        .astype('Float64')
    )
    # Add back post_days + 1 to approximate days since RDQ itself.
    merged['ea_days_since'] = merged['ea_days_since'] + (post_days + 1)
    # For names with no EA history or outside tolerance window, fill with
    # a capped large value to avoid NaNs while keeping the scale reasonable.
    merged['ea_days_since'] = (
        merged['ea_days_since']
        .fillna(9999.0)
        .clip(lower=0, upper=3650)  # cap at ~10 years
    )

    # 2) A smoothed EA effect factor derived from cumulative EA return.
    #    - Fill missing ear_ea with 0 so that the feature is dense.
    #    - Optionally clip to reduce extreme tails; this keeps it simple
    #      and robust for PCA / clustering.
    merged['ea_effect'] = merged['ear_ea'].astype('Float64').fillna(0.0)
    merged['ea_effect'] = merged['ea_effect'].clip(lower=-1.0, upper=1.0)

    # 3) Drop original sparse EA factors from the final output to keep the
    #    feature space compact and avoid highly sparse columns downstream.
    drop_cols = [c for c in ['ear_ea', 'aeavol_ea', 'valid_from_ea'] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    return merged


def build_and_save_ea_events(raw_dir: str, out_dir: str, start_year: int, end_year: int) -> None:
    """
    Utility to precompute EA events table and save to parquet.
    This can be called offline so that attach_ea_factors_to_daily can
    simply load a single compact parquet instead of scanning all fundq files.
    """
    os.makedirs(out_dir, exist_ok=True)
    stock = _load_stocknames(raw_dir)
    if stock.empty:
        print("Stocknames missing; cannot build EA events.")
        return

    # Mirror GHZ load_data behaviour: include [start_year-1, end_year+1] as safety buffer
    for year in range(start_year - 1, end_year + 2):
        print(f"Building EA events for year {year}...")
        events_y = _build_ea_events_for_year(raw_dir, stock, year)
        if events_y.empty:
            continue
        out_path = os.path.join(out_dir, f"ea_events_{int(year)}.parquet")
        events_y.to_parquet(out_path)
        print(f"Saved EA events for {year} to {out_path} ({events_y.shape[0]} rows)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./data/raw_ghz",
        help="Raw GHZ directory containing comp_fundq and crsp_stocknames.parquet",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/processed/ea_events",
        help="Output directory for precomputed EA events parquet files (partitioned by year)",
    )
    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        help="Start calendar year for which to build EA events (inclusive).",
    )
    parser.add_argument(
        "--end_year",
        type=int,
        required=True,
        help="End calendar year for which to build EA events (inclusive).",
    )
    args = parser.parse_args()

    build_and_save_ea_events(args.raw_dir, args.out_dir, args.start_year, args.end_year)
