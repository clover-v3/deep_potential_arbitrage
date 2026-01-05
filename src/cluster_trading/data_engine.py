import os
import pandas as pd
import numpy as np
import torch
from typing import Optional, List, Union, Dict
from pathlib import Path

class UniverseManager:
    """
    Manages the universe of valid tickers.
    Ensures we only trade liquid/active assets.
    """
    def __init__(self, data_df: pd.DataFrame):
        """
        Args:
            data_df: DataFrame with 'date', 'ticker' columns.
        """
        self.universe_mask = self._build_universe_mask(data_df)

    def _build_universe_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        # Assumes df has unique (date, ticker) entries
        # Returns a boolean mask (Date x Ticker)
        # For now, we assume if data exists, it's in universe.
        # In real logic, filter by Volume > Threshold, MarketCap > Threshold etc.
        pivot = df.pivot(index='date', columns='ticker', values='close')
        return ~pivot.isna()

    def get_valid_tickers(self, date) -> List[str]:
        if date not in self.universe_mask.index:
            return []
        row = self.universe_mask.loc[date]
        return row[row].index.tolist()

class DataEngine:
    """
    Handles data loading, caching, and preprocessing.
    Designed for parallelism and vectorization.
    """
    def __init__(self, data_path: str, cache_dir: str = ".cache"):
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.raw_data: Optional[pd.DataFrame] = None
        self.price_tensor: Optional[torch.Tensor] = None # (T, N)
        self.dates: Optional[List] = None
        self.tickers: Optional[List] = None

    def load_data(self, start_date: str, end_date: str):
        """
        Loads data from parquet/csv, filters by date range.
        Checks cache first.
        """
        # 1. Try Cache
        cache_key = f"data_{start_date}_{end_date}.pt"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            print(f"Loading data from cache: {cache_path}")
            data_dict = torch.load(cache_path)
            self.price_tensor = data_dict['prices']
            self.dates = data_dict['dates']
            self.tickers = data_dict['tickers']
            return

        # 2. Load Raw (Simulation)
        # In real usage, this might read multiple parquet files
        # For this prototype, we expect a cleaned long-format DF
        # with columns: date, ticker, close
        print(f"Loading raw data from {self.data_path}...")
        if str(self.data_path).endswith('.parquet'):
            df = pd.read_parquet(self.data_path)
        else:
            # Fallback or error
            raise ValueError(f"Unsupported file format: {self.data_path}")

        # Filter Date
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df = df[mask]

        # 3. Vectorize (Pivot)
        print("Vectorizing data...")
        pivot_df = df.pivot(index='date', columns='ticker', values='close')
        pivot_df = pivot_df.sort_index()

        # Fill NA (Forward Fill then Backward Fill)
        pivot_df = pivot_df.ffill().bfill()

        self.dates = pivot_df.index.tolist()
        self.tickers = pivot_df.columns.tolist()

        # Convert to Tensor
        self.price_tensor = torch.tensor(pivot_df.values, dtype=torch.float32)

        # 4. Save Cache
        print(f"Saving cache to {cache_path}")
        torch.save({
            'prices': self.price_tensor,
            'dates': self.dates,
            'tickers': self.tickers
        }, cache_path)

    def load_wide_data(self, df: pd.DataFrame):
        """
        Loads data from a Wide DataFrame (Index=Date, Columns=Tickers).
        Aligns directly with user request.
        """
        print("Loading wide data directly...")
        # Validate index is Datetime
        if not np.issubdtype(df.index.dtype, np.datetime64):
             df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Handle Universe masking (NaNs)
        # We fill NaNs for tensor creation, but we should track valid mask?
        # For prototype, we fillna(method='ffill')
        if df.isna().any().any():
            print("Warning: Input contains NaNs. Filling...")
            df = df.ffill().bfill()

        self.dates = df.index.tolist()
        self.tickers = df.columns.tolist()

        # (T, N)
        self.price_tensor = torch.tensor(df.values, dtype=torch.float32)
        print(f"Data Loaded: {self.price_tensor.shape} (Time, Tickers)")

    def get_batch(self, t_start: int, t_end: int) -> torch.Tensor:
        """
        Returns price slice (t_start:t_end, N_tickers)
        """
        if self.price_tensor is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.price_tensor[t_start:t_end, :]

    def to_device(self, device):
        if self.price_tensor is not None:
            self.price_tensor = self.price_tensor.to(device)
