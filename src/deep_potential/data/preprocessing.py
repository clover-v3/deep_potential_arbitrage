
import pandas as pd
import numpy as np

class Preprocessor:
    """
    Handles Cross-Sectional and Time-Series Normalization.
    """

    @staticmethod
    def cross_sectional_zscore(df: pd.DataFrame, group_col: str = 'date') -> pd.DataFrame:
        """
        Computes Z-Score cross-sectionally (per date).
        Standardizes features to have mean 0, std 1 across the universe at each time t.
        """
        # Avoid grouping on robust data if already indexed
        if group_col in df.index.names:
            levels = df.index.names
            df_reset = df.reset_index()
        else:
            levels = None
            df_reset = df.copy()

        numeric_cols = df_reset.select_dtypes(include=[np.number]).columns
        # Exclude ID columns if they were numeric? Usually handled by groupby exclusion if not in list
        # But 'asset_id' might be numeric. Be careful.

        # Safe approach: standardization function
        def zscore(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        # Apply per group
        transformed = df_reset.groupby(group_col)[numeric_cols].transform(zscore)

        # Update original
        df_reset[numeric_cols] = transformed

        if levels:
            df_reset = df_reset.set_index(levels)

        return df_reset

    @staticmethod
    def rolling_zscore(df: pd.DataFrame, window: int = 60, min_periods: int = 20) -> pd.DataFrame:
        """
        Computes Rolling Z-Score time-serially (per asset).
        """
        # Assumes MultiIndex (Date, AssetID) or sorting by asset
        # If MultiIndex with date/asset, usually we want to group by asset

        # Identifies the asset level
        asset_level = 'asset_id' if 'asset_id' in df.index.names else None

        if not asset_level and 'asset_id' in df.columns:
            grouper = 'asset_id'
        elif asset_level:
            grouper = asset_level
        else:
            raise ValueError("Data must have 'asset_id' in index or columns")

        rolling = df.groupby(grouper).rolling(window=window, min_periods=min_periods)
        mean = rolling.mean()
        std = rolling.std()

        # Reset index of result to match df?
        # Rolling on groupby usually returns MultiIndex (Asset, Date)

        # Align
        if asset_level:
            # Result indices should match original if sorted
            # But let's be safe.
            # Z = (X - Mean) / Std
            # Need to match shapes carefully
            pass

        # Simplified:
        # Just use transform with lambda? Slow.
        # Direct vectorized:
        z = (df - mean.reset_index(level=0, drop=True)) / (std.reset_index(level=0, drop=True) + 1e-8)

        return z

    @staticmethod
    def fill_na(df: pd.DataFrame, method: str = 'zero') -> pd.DataFrame:
        if method == 'zero':
            return df.fillna(0.0)
        elif method == 'check':
            # Check for infinites
            return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df
