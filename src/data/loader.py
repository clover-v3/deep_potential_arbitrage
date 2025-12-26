"""
Data Loader Module for Baseline Replication

Handles loading of Daily and 1-Minute Parquet files.
Structure:
- Daily: {variable}/{year_month}.parquet (Row: Date, Col: Tickers)
- 1-Min: {variable}/{date}.parquet (Row: Time, Col: Tickers)
"""

import pandas as pd
import numpy as np
import os
from glob import glob
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, data_root: str, universe_path: Optional[str] = None):
        """
        Args:
            data_root: Root directory containing 'daily' and '1min' subdirectories.
            universe_path: Path to universe parquet file (Row: Date, Col: Tickers, Val: 1/0).
                           If None, the universe is automatically constructed by concatenating
                           all available data files (union of tickers). Missing columns are filled with NaN.
        """
        self.data_root = data_root
        self.daily_path = os.path.join(data_root, 'daily')
        self.min_path = os.path.join(data_root, '1min')

        self.universe = None
        if universe_path and os.path.exists(universe_path):
            try:
                self.universe = pd.read_parquet(universe_path)
                # Ensure index is datetime
                if not isinstance(self.universe.index, pd.DatetimeIndex):
                    self.universe.index = pd.to_datetime(self.universe.index)
                print(f"Loaded universe from {universe_path}: shape={self.universe.shape}")
            except Exception as e:
                print(f"Failed to load universe: {e}")

    def list_variables(self, freq: str = 'daily') -> List[str]:
        """List available variables for a given frequency."""
        path = self.daily_path if freq == 'daily' else self.min_path
        if not os.path.exists(path):
            return []
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    def _apply_universe_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame by universe.
        Keep data where universe == 1. Set others to NaN.
        """
        if self.universe is None:
            return df

        # 1. Align Dates (Intersection)
        common_dates = df.index.intersection(self.universe.index)
        if common_dates.empty:
            return df # Or empty? If disjoint, mask logic implies filtering everything.

        # 2. Reindex universe to match df's full shape (subset of dates, subset of columns)
        # We process efficienty:
        # Get universe slice
        univ_slice = self.universe.loc[common_dates]

        # Align columns
        univ_slice = univ_slice.reindex(columns=df.columns).fillna(0)

        # Align df index (in case df has dates not in universe - though loop above handles it)
        # Actually load_daily_data filters by start/end.

        # Align indices exactly
        aligned_univ = univ_slice.reindex(df.index).reindex(columns=df.columns).fillna(0)

        # Apply mask
        # 1 = valid, 0 = invalid.
        # We want to keep valid.
        return df.where(aligned_univ == 1)

    def load_daily_data(
        self,
        variable: str,
        start_date: str,
        end_date: str,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load daily data for a variable across a date range.
        Reads monthly parquet files: {year_month}.parquet
        """
        var_path = os.path.join(self.daily_path, variable)
        if not os.path.exists(var_path):
            # print(f"Variable path not found: {var_path}")
            return pd.DataFrame()

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate list of year_months to load
        current = start_dt.replace(day=1)
        months_to_load = []
        while current <= end_dt:
            months_to_load.append(current.strftime("%Y_%m"))
            next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            current = next_month

        dfs = []
        for ym in months_to_load:
            file_path = os.path.join(var_path, f"{ym}.parquet")
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    # Filter by date range
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        full_df = pd.concat(dfs, axis=0).sort_index()

        # Filter columns if tickers provided
        if tickers:
            available_cols = [c for c in tickers if c in full_df.columns]
            full_df = full_df[available_cols]

        # Apply Universe Mask
        full_df = self._apply_universe_mask(full_df)

        return full_df

    def load_1min_data(
        self,
        variable: str,
        date: str,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load 1-minute data for a specific date.
        File: {variable}/{date}.parquet

        Note: Universe filtering for 1-min data is tricky if universe is Daily.
        We check if the DATE and Ticker is valid in universe.
        """
        var_path = os.path.join(self.min_path, variable)
        # Try different date formats if needed, assuming Standard ISO YYYY-MM-DD or YYYYMMDD
        # User example: {date}.parquet. Let's try standard formats.

        candidates = [
            os.path.join(var_path, f"{date}.parquet"),
            os.path.join(var_path, f"{date.replace('-', '')}.parquet")
        ]

        file_path = None
        for p in candidates:
            if os.path.exists(p):
                file_path = p
                break

        if not file_path:
            return pd.DataFrame()

        try:
            df = pd.read_parquet(file_path)
            if tickers:
                available_cols = [c for c in tickers if c in df.columns]
                df = df[available_cols]

            # Apply Universe Mask (Daily Granularity)
            if self.universe is not None:
                dt = pd.to_datetime(date)
                if dt in self.universe.index:
                    # Get valid tickers for this day
                    univ_row = self.universe.loc[dt]
                    valid_tickers = univ_row[univ_row == 1].index
                    # Intersect with filtered columns
                    cols_to_keep = df.columns.intersection(valid_tickers)
                    df = df[cols_to_keep]
                else:
                     # Date not in universe? Keep all or drop all?
                     # Let's assume keep allowed if universe missing for date?
                     # Or drop if universe is strict whitelist.
                     # Safe: Return empty if universe exists but date not in it.
                     pass

            return df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def aggregate_1min_to_daily(
        self,
        variable: str,
        start_date: str,
        end_date: str,
        agg_func: str = 'sum', # 'sum' (vol), 'std' (volatility), 'last' (close)
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Iterate over 1-min files and aggregate to daily values.
        Useful for calculating realize volatility or total turnover from 1-min data.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days = pd.date_range(start_dt, end_dt)

        results = {}

        for day in days:
            day_str = day.strftime("%Y-%m-%d")
            # 1min loader handles universe filtering (columns dropping)
            df_min = self.load_1min_data(variable, day_str, tickers)

            if df_min.empty:
                continue

            if agg_func == 'std':
                # For Realized Vol, usually we want std of returns, not prices
                # Assuming input is PRICE, calculate returns first
                if variable in ['close', 'open', 'high', 'low']:
                    returns = df_min.pct_change().dropna()
                    daily_val = returns.std()
                else:
                    daily_val = df_min.std()
            elif agg_func == 'sum':
                daily_val = df_min.sum()
            elif agg_func == 'mean':
                daily_val = df_min.mean()
            elif agg_func == 'last':
                 if not df_min.empty:
                    daily_val = df_min.iloc[-1]
            elif agg_func == 'rv': # Standard Realized Volatility
                 returns = df_min.pct_change().dropna()
                 # Annualized Vol (assuming 240 mins) -> but output is daily scalar
                 # Just return daily stddev for now
                 daily_val = returns.std()
            else:
                raise ValueError(f"Unknown aggregation function: {agg_func}")

            results[day] = daily_val

        if not results:
            return pd.DataFrame()

        final_df = pd.DataFrame(results).T # Index: Date, Col: Tickers
        # Note: 1min loaded filtered cols, so different days might have different cols
        # pandas T will join them, producing NaNs for missing cols/days
        return final_df
