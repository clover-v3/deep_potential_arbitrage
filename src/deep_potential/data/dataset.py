
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

class DeepPotentialDataset(Dataset):
    """
    PyTorch Dataset for Deep Potential Model.

    Structure:
    - Sliding Window over Time.
    - Returns FULL Cross-Section (all stocks) at each step t.
    - Yields: (X_batch, y_batch, mask_batch, meta_batch)
      - X_batch: (1, N, T, D) -> We remove batch dim in collate if needed, but here we yield one "Time Step" which contains N stocks.

    Wait, standard logical batching:
    Usually we want batches of size B time-steps?
    Or batches of size B stocks?

    For Graph Learning, we need ALL stocks at time t to form the graph.
    So a "Sample" is one time-window for ALL stocks.
    (N, T, D)

    If we batch these, we get (B, N, T, D).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        window_size: int = 60,
        asset_col: str = 'asset_id',
        date_col: str = 'date'
    ):
        """
        Args:
            df: Aligned DataFrame with (Date, Asset) index.
            feature_cols: List of features to use.
            target_col: Prediction target (e.g. Fwd Return).
            window_size: Lookback window T.
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.target_col = target_col

        # 1. Pivot to 3D Tensor Structure (Time, Asset, Features)
        # Check index
        if asset_col not in df.index.names or date_col not in df.index.names:
            df = df.reset_index().set_index([date_col, asset_col]).sort_index()

        # Get Unique Dates and Assets
        self.dates = df.index.get_level_values(date_col).unique().sort_values()
        self.assets = df.index.get_level_values(asset_col).unique().sort_values()

        self.T_total = len(self.dates)
        self.N = len(self.assets)
        self.D = len(feature_cols)

        print(f"Dataset Init: {self.N} Assets, {self.T_total} Timesteps, {self.D} Features.")

        # We need a dense tensor for efficiency, even if some data is missing (masked)
        # Create a full index product
        full_idx = pd.MultiIndex.from_product([self.dates, self.assets], names=[date_col, asset_col])

        # Reindex to full (Time * N)
        df_full = df.reindex(full_idx)

        # Convert to Tensor
        # Features
        self.X_full = torch.tensor(
            df_full[feature_cols].values.reshape(self.T_total, self.N, self.D),
            dtype=torch.float32
        )

        # Targets
        if target_col in df_full.columns:
            self.y_full = torch.tensor(
                df_full[target_col].values.reshape(self.T_total, self.N),
                dtype=torch.float32
            )
        else:
            self.y_full = torch.zeros((self.T_total, self.N))

        # Mask (1 if data present, 0 if NaN/filled)
        # We assume if all features are NaN, it's missing
        # But we previously filled NaNs.
        # Ideally we track valid mask before fill.
        # For now, let's assume if it was in original index it's valid?
        # Reindexing introduced NaNs.

        # Let's verify NaNs in X_full.
        self.mask = ~torch.isnan(self.X_full[:, :, 0]) # Check first feature

        # Now fill NaNs in X for computation
        self.X_full = torch.nan_to_num(self.X_full, nan=0.0)
        self.y_full = torch.nan_to_num(self.y_full, nan=0.0)

    def __len__(self):
        # Number of potentially valid windows
        return max(0, self.T_total - self.window_size + 1)

    def __getitem__(self, idx):
        # Window [t, t + window_size]
        # x shape: (window_size, N, D) -> Permute to (N, T, D) usually expected by Conv1d?
        # Conv1D expects (Batch, Channels, Length).
        # Our "Batch" here is N stocks? Or are we learning ONE graph for N stocks?
        # The Model expects (Batch=1, N, T, D) or similar.

        # Let's standardize on (N, T, D) for the item.
        # Batch loader will stack to (B, N, T, D).

        # Slicing
        # X: [idx : idx+window] (T, N, D)
        X_window = self.X_full[idx : idx + self.window_size]

        # Transpose to (N, T, D)
        X_out = X_window.permute(1, 0, 2)

        # Target: Return at T+1 (Next step after window)
        # idx + window_size is the index of T+1
        if idx + self.window_size < self.T_total:
            y_out = self.y_full[idx + self.window_size] # (N,)
            mask_out = self.mask[idx + self.window_size] # (N,)
        else:
            # End of test
            y_out = torch.zeros(self.N)
            mask_out = torch.zeros(self.N, dtype=torch.bool)

        return dict(
            features=X_out, # (N, T, D)
            target=y_out,   # (N,)
            mask=mask_out,  # (N,)
            timestamp_idx=idx + self.window_size
        )
