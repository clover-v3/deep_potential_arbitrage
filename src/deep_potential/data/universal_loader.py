
import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
import glob

class DataSource(ABC):
    """
    Abstract Base Class for Data Sources.
    """
    def __init__(self, name: str, data_root: str):
        self.name = name
        self.data_root = data_root

    @abstractmethod
    def load(self, start_date: pd.Timestamp, end_date: pd.Timestamp, universe: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data within range.
        Returns DataFrame with index [Date, Ticker] and columns [Features].
        """
        pass

class DailyDataSource(DataSource):
    """
    Loads partitioned daily data (e.g., Price, Volatility).
    Expects parquet files: data_root/prefix_YYYY_MM.parquet
    """
    def __init__(self, name: str, data_root: str, file_prefix: str = "daily_factors"):
        super().__init__(name, data_root)
        self.file_prefix = file_prefix

    def load(self, start_date: pd.Timestamp, end_date: pd.Timestamp, universe: Optional[List[str]] = None) -> pd.DataFrame:
        print(f"[{self.name}] Loading Daily Data from {self.data_root}...")
        if not os.path.exists(self.data_root):
            print(f"Warning: {self.data_root} does not exist.")
            return pd.DataFrame()

        files = sorted(glob.glob(os.path.join(self.data_root, f"{self.file_prefix}_*.parquet")))
        dfs = []

        # Optimize: Filter files by name if YYYY_MM is in filename
        # Assumes format ends with YYYY_MM.parquet
        s_ym = start_date.year * 100 + start_date.month
        e_ym = end_date.year * 100 + end_date.month

        for f in files:
            try:
                # Extract YYYY_MM
                parts = os.path.basename(f).replace('.parquet', '').split('_')
                year = int(parts[-2])
                month = int(parts[-1])
                ym = year * 100 + month

                if ym < s_ym or ym > e_ym:
                    continue

                df_chunk = pd.read_parquet(f)

                # Filter Date
                df_chunk['date'] = pd.to_datetime(df_chunk['date'])
                mask = (df_chunk['date'] >= start_date) & (df_chunk['date'] <= end_date)
                df_chunk = df_chunk[mask]

                if universe:
                     # Assumes 'permno' or 'ticker' column. Let's assume 'permno' for now based on project.
                     # TODO: Standardize ID
                     if 'permno' in df_chunk.columns:
                         df_chunk = df_chunk[df_chunk['permno'].isin(universe)]

                dfs.append(df_chunk)

            except Exception as e:
                print(f"Skipping {f}: {e}")

        if not dfs:
            return pd.DataFrame()

        full_df = pd.concat(dfs, ignore_index=True)
        # Set Index: Date, ID
        if 'permno' in full_df.columns:
             full_df = full_df.rename(columns={'permno': 'asset_id'})

        full_df = full_df.set_index(['date', 'asset_id']).sort_index()
        return full_df

class LowFreqDataSource(DataSource):
    """
    Loads Annual/Quarterly data (e.g., Fundamentals).
    Expects parquet files: data_root/prefix_YYYY.parquet
    """
    def __init__(self, name: str, data_root: str, file_prefix: str = "ghz_factors"):
        super().__init__(name, data_root)
        self.file_prefix = file_prefix

    def load(self, start_date: pd.Timestamp, end_date: pd.Timestamp, universe: Optional[List[str]] = None) -> pd.DataFrame:
        print(f"[{self.name}] Loading Low-Freq Data from {self.data_root}...")

        # Load extra history for forward filling
        load_start_year = start_date.year - 2
        load_end_year = end_date.year

        dfs = []
        for y in range(load_start_year, load_end_year + 1):
            f = os.path.join(self.data_root, f"{self.file_prefix}_{y}.parquet")
            if os.path.exists(f):
                dfs.append(pd.read_parquet(f))

        if not dfs:
            return pd.DataFrame()

        full_df = pd.concat(dfs, ignore_index=True)

        # Standardize Columns
        if 'permno' in full_df.columns:
             full_df = full_df.rename(columns={'permno': 'asset_id'})

        # Filter mostly by date if 'date' exists (e.g. public release date) to reduce size
        # But for low freq, we keep all and let the AsOf merge handle logic
        full_df['date'] = pd.to_datetime(full_df['date'])
        full_df = full_df.sort_values('date')

        if universe and 'asset_id' in full_df.columns:
             full_df = full_df[full_df['asset_id'].isin(universe)]

        # We generally return raw here, the Loader handles alignment
        return full_df

class UniversalDataLoader:
    """
    Master Loader that aligns multiple sources to a target trading calendar.
    """
    def __init__(self, freq: str = '1D'):
        self.sources: List[DataSource] = []
        self.freq = freq

    def add_source(self, source: DataSource):
        self.sources.append(source)

    def load_aligned(
        self,
        start_date: str,
        end_date: str,
        universe: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Loads all sources and aligns them to the daily master clock (Back-fill / Forward-fill).
        """
        s_dt = pd.to_datetime(start_date)
        e_dt = pd.to_datetime(end_date)

        # 1. Load Main Backbone (First Source is assumed to be High Freq / Skeleton)
        if not self.sources:
            return pd.DataFrame()

        # Prioritize Daily Price source as the 'Backbone'
        backbone_df = self.sources[0].load(s_dt, e_dt, universe)

        if backbone_df.empty:
            print("Backbone source returned empty.")
            return pd.DataFrame()

        aligned_df = backbone_df.copy()

        # 2. Merge other sources
        for source in self.sources[1:]:
            df_new = source.load(s_dt, e_dt, universe)

            if df_new.empty:
                continue

            # Check if index is set
            if 'asset_id' not in df_new.columns and 'asset_id' not in df_new.index.names:
                 # Check if permno exists if asset_id is missing
                 if 'permno' in df_new.columns:
                     df_new = df_new.rename(columns={'permno': 'asset_id'})

            # Reset index for merge
            base_reset = aligned_df.reset_index()
            new_reset = df_new.reset_index()

            if 'date' not in new_reset.columns or 'asset_id' not in new_reset.columns:
                print(f"Skipping {source.name}: Missing date/asset_id columns.")
                continue

            new_reset['date'] = pd.to_datetime(new_reset['date'])
            base_reset = base_reset.sort_values('date')
            new_reset = new_reset.sort_values('date')

            # Identify overlaps to avoid duplicates
            cols_to_use = new_reset.columns.difference(base_reset.columns)
            # But we need join keys
            cols_to_merge = list(cols_to_use) + ['date', 'asset_id']
            new_reset_slim = new_reset[cols_to_merge]

            print(f"Merging {source.name}...")
            # AsOf Merge: aligns Low Freq latest available date <= current date
            merged = pd.merge_asof(
                base_reset,
                new_reset_slim,
                on='date',
                by='asset_id',
                direction='backward',
                tolerance=pd.Timedelta(days=365) # Max lag 1 year for annual
            )

            # Restore Index
            if 'date' in merged.columns and 'asset_id' in merged.columns:
                aligned_df = merged.set_index(['date', 'asset_id']).sort_index()
            else:
                aligned_df = merged

        return aligned_df
