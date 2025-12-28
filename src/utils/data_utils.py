import os
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
    return pd.Series(np.log(s), index=series.index)

def clean_infs(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Inf/-Inf with NaN to prevent rolling/reduce warnings"""
    return df.replace([np.inf, -np.inf], np.nan)

def rolling_std(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Helper for rolling std on grouped series (pass a Series or Grouped Series)"""
    return series.rolling(window=window, min_periods=min_periods).std().reset_index(level=0, drop=True)

def rolling_mean(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Helper for rolling mean on grouped series (pass a Series or Grouped Series)"""
    return series.rolling(window=window, min_periods=min_periods).mean().reset_index(level=0, drop=True)

def rolling_sum(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Helper for rolling sum on grouped series (pass a Series or Grouped Series)"""
    return series.rolling(window=window, min_periods=min_periods).sum().reset_index(level=0, drop=True)


# --- Grouped rolling helpers that preserve index shape via transform (for daily_factors, etc.) ---

def group_rolling_apply(g, series_name: str, func, window: int = 21, min_periods: int = 15) -> pd.Series:
    """
    Generic grouped rolling apply.
    g: DataFrameGroupBy (e.g. df.groupby('permno'))
    series_name: column name within df
    func: function passed to rolling.apply(raw=True)
    """
    return g[series_name].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).apply(func, raw=True)
        if len(x) >= min_periods else pd.Series(np.nan, index=x.index)
    )


def group_rolling_mean(g, series_name: str, window: int = 21, min_periods: int = 15) -> pd.Series:
    """Grouped rolling mean using transform to preserve index alignment."""
    return g[series_name].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).mean()
    )


def group_rolling_std(g, series_name: str, window: int = 21, min_periods: int = 15) -> pd.Series:
    """Grouped rolling std using transform to preserve index alignment."""
    return g[series_name].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).std()
    )


def group_rolling_max(g, series_name: str, window: int = 21, min_periods: int = 15) -> pd.Series:
    """Grouped rolling max using transform to preserve index alignment."""
    return g[series_name].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).max()
    )


def group_rolling_skew(g, series_name: str, window: int = 21, min_periods: int = 15) -> pd.Series:
    """Grouped rolling skew using transform to preserve index alignment."""
    return g[series_name].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).skew()
    )


def group_rolling_sum(g, series_name: str, window: int = 21, min_periods: int = 15) -> pd.Series:
    """Grouped rolling sum using transform to preserve index alignment."""
    return g[series_name].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).sum()
    )


def preprocess_crsp_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize CRSP daily data before factor construction.

    - Ensures numeric dtype for key columns.
    - Cleans Inf/-Inf via clean_infs.
    - Constructs split-adjusted price columns using CFACPR if present:
        adj_prc, adj_openprc, adj_askhi, adj_bidlo
    - Optionally constructs split-adjusted volume using CFACSHR if present:
        adj_vol
    """
    # Ensure numeric types where expected
    numeric_cols = [
        'prc', 'vol', 'shrout', 'openprc',
        'askhi', 'bidlo', 'ret', 'retx',
        'cfacpr', 'cfacshr', 'numtrd'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Clean infinite values once up front
    df = clean_infs(df)

    # Price adjustments using CFACPR (if available)
    if 'cfacpr' in df.columns:
        cfacpr_safe = df['cfacpr'].replace(0, np.nan)
        if 'prc' in df.columns:
            df['adj_prc'] = df['prc'] / cfacpr_safe
        if 'openprc' in df.columns:
            df['adj_openprc'] = df['openprc'] / cfacpr_safe
        if 'askhi' in df.columns:
            df['adj_askhi'] = df['askhi'] / cfacpr_safe
        if 'bidlo' in df.columns:
            df['adj_bidlo'] = df['bidlo'] / cfacpr_safe

    # Volume adjustment using CFACSHR (if available)
    # CRSP convention: adjusted shares ~ SHR * CFACSHR, so we treat volume similarly.
    if 'cfacshr' in df.columns and 'vol' in df.columns:
        df['adj_vol'] = df['vol'] * df['cfacshr'].astype(float)

    return df


from typing import Optional

def load_trading_days(path: str) -> Optional[pd.DatetimeIndex]:
    """
    Load trading day calendar from a parquet or csv file with a 'date' column.

    Recommended upstream: trading_days = db.get_table(library='crsp', table='dsi', columns=['date'])
    and save to parquet at `path`.
    """
    if not os.path.exists(path):
        print(f"Trading days file {path} does not exist. Falling back to calendar days.")
        return None

    try:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to load trading days from {path}: {e}")
        return None

    if 'date' not in df.columns:
        print(f"Trading days file {path} missing 'date' column.")
        return None

    dates = pd.to_datetime(df['date']).dropna().sort_values().unique()
    return pd.DatetimeIndex(dates)


def shift_to_next_trading_day(dates: pd.Series, trading_days: Optional[pd.DatetimeIndex]) -> pd.Series:
    """
    Map each calendar date to the next trading day using a trading day index.
    If trading_days is None, falls back to date + 1 calendar day.
    """
    dates = pd.to_datetime(dates)
    if trading_days is None or trading_days.empty:
        # Fallback: simple calendar-day shift
        return dates + pd.Timedelta(days=1)

    # Ensure trading_days sorted
    td_values = trading_days.sort_values().values
    date_values = dates.values

    idx = np.searchsorted(td_values, date_values, side="right")
    result = pd.Series(pd.NaT, index=dates.index)

    in_range = idx < len(td_values)
    result.iloc[in_range] = td_values[idx[in_range]]

    # For dates beyond last trading day, leave as NaT; caller can drop if needed.
    return result