import pandas as pd
import numpy as np

def safe_bool_to_int(series: pd.Series) -> pd.Series:
    """
    Robustly convert boolean series (possibly with NA) to int.
    NA -> False -> 0
    True -> 1
    False -> 0
    """
    return series.fillna(False).astype(int)

def safe_log(series: pd.Series) -> pd.Series:
    """
    Robust Log that handles:
    1. Nullable Types (Float64) by casting to numpy float64 (avoiding masked array warnings)
    2. Non-positive values (0, negative) by masking them as NaN
    """
    s = series.astype('float64', copy=True)
    mask = s <= 0
    s[mask] = np.nan
    return np.log(s)

def clean_infs(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Inf/-Inf with NaN to prevent rolling/reduce warnings"""
    return df.replace([np.inf, -np.inf], np.nan)

def rolling_std(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Helper for rolling std on grouped series"""
    return series.rolling(window=window, min_periods=min_periods).std().reset_index(level=0, drop=True)

def rolling_mean(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Helper for rolling mean on grouped series"""
    return series.rolling(window=window, min_periods=min_periods).mean().reset_index(level=0, drop=True)

def rolling_sum(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Helper for rolling sum on grouped series"""
    return series.rolling(window=window, min_periods=min_periods).sum().reset_index(level=0, drop=True)
