"""
Feature Engineering Module for Han et al. Replication

Handles:
1. Technical Indicator Calculation (RSI, Volatility, Momentum)
2. Fundamental Data Processing (Winzorization, Standardization)
3. Dataset Construction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from .loader import DataLoader

class FeatureEngine:
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def _winsorize_series(self, series: pd.Series, limits=(0.01, 0.01)) -> pd.Series:
        """Winsorize a series at given quantiles."""
        return series.clip(lower=series.quantile(limits[0]), upper=series.quantile(1 - limits[1]))

    def _zscore_series(self, series: pd.Series) -> pd.Series:
        """Calculate Z-Score (standardization)."""
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            return series * 0
        return (series - mean) / std

    def preprocess_cross_section(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional preprocessing (per day):
        1. Winsorize (1% - 99%)
        2. Z-Score Standardization

        Args:
            df: DataFrame (Index: Date, Columns: Tickers)

        Returns:
            processed_df: Standardized DataFrame
        """
        # Apply row-wise (axis=1) for cross-sectional
        # Pandas apply(axis=1) is slow.
        # Better: Iterate rows or use vectorized operations if possible.
        # But quantiles are row-specific.

        def process_row(row):
            # 1. Winsorize
            w_row = self._winsorize_series(row)
            # 2. Z-Score
            z_row = self._zscore_series(w_row)
            return z_row

        return df.apply(process_row, axis=1)

    def compute_technical_features(self, close_df: pd.DataFrame, volume_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Compute standard technical features.

        Args:
            close_df: Daily close prices
            volume_df: Daily volume

        Returns:
            Dict of feature DataFrames
        """
        features = {}

        # 1. Returns
        returns = close_df.pct_change()

        # 2. Volatility (20D)
        features['vol_20d'] = returns.rolling(window=20).std()
        features['vol_60d'] = returns.rolling(window=60).std()

        # 3. Momentum
        # RSI 14
        delta = close_df.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # ROC (Rate of Change) / Momentum
        features['mom_20d'] = close_df.pct_change(periods=20)
        features['mom_60d'] = close_df.pct_change(periods=60)

        # 4. Turnover (if volume available)
        # Note: True turnover requires shares outstanding (market_cap / price).
        # If 'turnover' is already a daily variable in data, load it directly.
        # Here assuming volume is raw volume.

        return features

    def build_dataset(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[List[str]] = None,
        use_1min_vol: bool = False
    ) -> pd.DataFrame:
        """
        Build complete feature dataset for clustering.

        Returns:
            MultiIndex DataFrame (Date, Ticker) -> Features
        """
        # 1. Load Raw Fundamental Data
        # Variables: market_cap, pe_ttm, pb_lf, roe_ttm, operating_revenue_growth
        fund_vars = ['market_cap', 'pe_ttm', 'pb_lf', 'roe_ttm', 'operating_revenue_growth']
        raw_funds = {}

        for var in fund_vars:
            df = self.loader.load_daily_data(var, start_date, end_date, tickers)
            if not df.empty:
                # Log transform Market Cap
                if var == 'market_cap':
                    df = np.log(df.replace(0, np.nan))

                # Invert PE and PB to get Earnings Yield / Book-to-Market (better for linear models)
                # Handle zeros and negatives carefully
                if var in ['pe_ttm', 'pb_lf']:
                    # Simple inverse, keep sign? Or just process raw.
                    # Han et al often use raw characteristics normalized.
                    # Let's stick to raw for now, Z-score will handle scale.
                    pass

                raw_funds[var] = df

        # 2. Load Price Data for Technicals
        close_df = self.loader.load_daily_data('close', start_date, end_date, tickers)

        if close_df.empty:
            raise ValueError("No close price data found!")

        # 3. Compute Technicals
        tech_feats = self.compute_technical_features(close_df)

        if use_1min_vol:
            # New Advanced Intraday Features
            print("Computing Advanced Intraday Features (20+ metrics)...")
            from .intraday_features import IntradayFeatureEngine

            # We need to iterate days and load 1min data
            # Structure: 1min/{var}/{date}.parquet
            # We need OHLCV for each stock-day.

            # Efficient approach:
            # Iterate dates in range
            # For each date: load open, high, low, close, volume (if avail)
            # Combine into 1 DataFrame per Ticker? Or just pass aligned DFs

            # loader.load_1min_data(var, date) -> DataFrame(Index=Time, Col=Tickers)

            dates = pd.date_range(start_date, end_date)
            daily_feature_list = []

            for d in dates:
                d_str = d.strftime("%Y-%m-%d")
                # Load all required variables
                try:
                    df_close = self.loader.load_1min_data('close', d_str, tickers)
                    if df_close.empty:
                        continue

                    # Assume other vars exist if close exists (same universe)
                    df_open = self.loader.load_1min_data('open', d_str, tickers)
                    df_high = self.loader.load_1min_data('high', d_str, tickers)
                    df_low = self.loader.load_1min_data('low', d_str, tickers)
                    df_vol = self.loader.load_1min_data('volume', d_str, tickers)

                    # If variables missing, use close for OHL (fallback)
                    if df_open.empty: df_open = df_close
                    if df_high.empty: df_high = df_close
                    if df_low.empty: df_low = df_close
                    if df_vol.empty: df_vol = pd.DataFrame(1, index=df_close.index, columns=df_close.columns) # Dummy vol

                    # Compute per ticker
                    # This double loop (Days * Tickers) might be slow for large universe.
                    # For baseline replication (500 stocks), it's acceptable.

                    for ticker in df_close.columns:
                         # Construct 1-min DF for this ticker
                         # Ensure alignment
                         df_t = pd.DataFrame({
                             'open': df_open[ticker],
                             'high': df_high[ticker],
                             'low': df_low[ticker],
                             'close': df_close[ticker],
                             'volume': df_vol[ticker]
                         })

                         feats = IntradayFeatureEngine.compute_all(df_t)
                         if not feats.empty:
                             feats['date'] = d
                             feats['ticker'] = ticker
                             daily_feature_list.append(feats)

                except Exception as e:
                    print(f"Error processing {d_str}: {e}")
                    continue

            if daily_feature_list:
                intraday_df = pd.DataFrame(daily_feature_list)
                # Pivot to (Date, Ticker) -> Features
                # Current: rows are date-ticker-features.
                # Target: we want dict of Panels (Date x Ticker) for each feature?
                # Or just keep it as Long format and merge later?
                # Method below expects Dict[feature_name -> DataFrame(Date x Ticker)]

                # Verify unique keys
                intraday_df = intraday_df.set_index(['date', 'ticker'])

                # Split back into dict of DataFrames
                for col in intraday_df.columns:
                    tech_feats[col] = intraday_df[col].unstack()
            else:
                print("Warning: No intraday features computed.")

        # 4. Combine and Preprocess
        # We need to stack everything into a long format or dict of panels
        # Target format for PCA: (N_samples, N_features) where N_samples = Days * Tickers
        # But we need to normalize Cross-Sectionally FIRST.

        all_features = {}

        # Add Fundamentals
        for name, df in raw_funds.items():
            all_features[name] = self.preprocess_cross_section(df)

        # Add Technicals
        for name, df in tech_feats.items():
            all_features[name] = self.preprocess_cross_section(df)

        # 5. Load Ground Truth (Industry) - No processing needed, categorical
        industry_df = self.loader.load_daily_data('industry_citic', start_date, end_date, tickers)

        # 6. Align and Stack
        # Use pandas.concat on the collected DataFrames
        # Keys to columns

        # Stack to (Date, Ticker) MultiIndex
        stacked_dfs = []
        for feat_name, df in all_features.items():
            s = df.stack()
            s.name = feat_name
            stacked_dfs.append(s)

        if not stacked_dfs:
            return pd.DataFrame()

        full_df = pd.concat(stacked_dfs, axis=1)

        # Add industry if available
        if not industry_df.empty:
            ind_s = industry_df.stack()
            ind_s.name = 'industry'
            full_df = full_df.join(ind_s, how='left')

        return full_df.dropna()
