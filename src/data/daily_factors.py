
"""
Daily Factor Builder
Strictly uses CRSP Daily Stock File (DSF) to construct 10+ advanced microstructure and statistical factors.
Avoids simple momentum. Focuses on volatility, liquidity, and distribution characteristics.
"""

import pandas as pd
import numpy as np
import os

from src.utils.data_utils import (
    preprocess_crsp_daily,
    group_rolling_mean,
    group_rolling_std,
    group_rolling_max,
    group_rolling_skew,
    group_rolling_sum,
    load_trading_days,
    shift_to_next_trading_day,
)

class DailyFactorsBuilder:
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.dsf = pd.DataFrame()

    def load_data(self):
        """Load Parquet files from crsp_dsf directory"""
        print("Loading daily data...")
        path = os.path.join(self.data_root, 'crsp_dsf')
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist.")
            return

        files = [f for f in os.listdir(path) if f.endswith('.parquet')]
        dfs = []
        for f in sorted(files):
            dfs.append(pd.read_parquet(os.path.join(path, f)))

        if not dfs:
            print("No daily data found.")
            return

        self.dsf = pd.concat(dfs, ignore_index=True)
        print(f"Loaded DSF: {self.dsf.shape}")

    def process_daily_factors(self, start_date=None, end_date=None) -> pd.DataFrame:
        if self.dsf.empty:
            return pd.DataFrame()

        # Sort strictly to ensure rolling window consistency
        df = self.dsf.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Filter by date if provided (Optimization: do this early)
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        if df.empty:
            print("No data in specified date range.")
            return pd.DataFrame()

        df = df.sort_values(['permno', 'date'])

        # Standard CRSP daily preprocessing (adjust prices/volumes, clean infs, dtypes)
        df = preprocess_crsp_daily(df)

        # Shift observation date to next trading day (availability date) using trading calendar
        trading_days_path = os.path.join(self.data_root, 'crsp_trading_days.parquet')
        trading_days = load_trading_days(trading_days_path)
        df['date'] = shift_to_next_trading_day(df['date'], trading_days)
        # Drop rows where next trading day is unavailable (e.g., last date in sample)
        df = df[df['date'].notna()]

        # Define GroupBy object once
        g = df.groupby('permno')

        # Use split-adjusted columns when available; otherwise fall back to raw
        price = df['adj_prc'] if 'adj_prc' in df.columns else df['prc']
        high = df['adj_askhi'] if 'adj_askhi' in df.columns else df['askhi']
        low = df['adj_bidlo'] if 'adj_bidlo' in df.columns else df['bidlo']
        open_price = df['adj_openprc'] if 'adj_openprc' in df.columns else df.get('openprc')
        vol = df['adj_vol'] if 'adj_vol' in df.columns else df['vol']

        print("Calculating Liquidity Factors...")
        # 1. Amihud Illiquidity: Mean(|Ret| / (Price * Vol))
        df['dollar_vol'] = price * vol
        # Avoid division by zero
        df['illiq_daily'] = df['ret'].abs() / df['dollar_vol'].replace(0, np.nan)
        df['amihud_1m'] = group_rolling_mean(g, 'illiq_daily')

        # 2. Turnover Instability: Std(Vol / Shrout)
        df['turnover'] = df['vol'] / df['shrout']
        df['turnover_var'] = group_rolling_std(g, 'turnover')

        # 3. Dollar Volume Coeff of Variation (Liquidity Reliability)
        dvol_std = group_rolling_std(g, 'dollar_vol')
        dvol_mean = group_rolling_mean(g, 'dollar_vol')
        df['dvol_cv_1m'] = dvol_std / dvol_mean.replace(0, np.nan)

        print("Calculating Volatility Factors...")
        # 4. Parkinson Volatility (High-Low Range)
        # 1 / (4 * ln(2)) * (ln(High / Low))^2
        hl_ratio = high / low
        # Filter bad data (Ask < Bid or Bid=0)
        valid_hl = (df['bidlo'] > 0) & (df['askhi'] >= df['bidlo'])
        hl_ratio = hl_ratio.where(valid_hl)
        df['parkinson_daily'] = (np.log(hl_ratio) ** 2) / (4 * np.log(2))
        # Volatility is usually RMSE of this variance estimator, or just mean of variance proxy?
        # Parkinson estimator for sigma^2 is the mean of the daily estimators.
        # So sigma = sqrt(mean(daily_est))
        df['parkinson_vol_1m'] = np.sqrt(group_rolling_mean(g, 'parkinson_daily'))

        # 5. Downside Volatility (Semivariance)
        # Sqrt(Mean(Ret^2)) where Ret < 0
        df['neg_ret_sq'] = np.where(df['ret'] < 0, df['ret']**2, 0)
        df['downside_vol_1m'] = np.sqrt(group_rolling_mean(g, 'neg_ret_sq'))

        # 6. Upside Variability
        df['pos_ret_sq'] = np.where(df['ret'] > 0, df['ret']**2, 0)
        df['upside_vol_1m'] = np.sqrt(group_rolling_mean(g, 'pos_ret_sq'))

        print("Calculating Distribution Factors...")
        # 7. Maximum Daily Return (Lottery Demand)
        df['max_ret_1m'] = group_rolling_max(g, 'ret')

        # 8. Realized Skewness
        df['skew_1m'] = group_rolling_skew(g, 'ret')

        print("Calculating Microstructure Factors...")
        # 9. Overnight Return Gap Volatility (Split-Adjusted)
        # Gap = Adj_Open / Adj_Prev_Close - 1
        if open_price is not None:
            # Basic Shift
            df['prev_prc'] = price.groupby(df['permno']).shift(1)

            # Prefer split-adjusted prices when available
            if 'adj_openprc' in df.columns and 'adj_prc' in df.columns:
                df['adj_open'] = df['adj_openprc']
                df['adj_close'] = df['adj_prc']
                df['prev_adj_close'] = df.groupby('permno')['adj_close'].shift(1)

                valid_gap = (df['adj_open'] > 0) & (df['prev_adj_close'] > 0)
                df['ret_gap_daily'] = np.nan
                df.loc[valid_gap, 'ret_gap_daily'] = (
                    df.loc[valid_gap, 'adj_open'] / df.loc[valid_gap, 'prev_adj_close']
                ) - 1
            else:
                # Fallback to raw prices if adjusted not available
                valid_open = (open_price > 0) & (df['prev_prc'] > 0)
                df['ret_gap_daily'] = np.nan
                df.loc[valid_open, 'ret_gap_daily'] = (
                    df.loc[valid_open, 'openprc'] / df.loc[valid_open, 'prev_prc']
                ) - 1

            # Rolling Std of Gap
            df['ret_gap_vol_1m'] = group_rolling_std(g, 'ret_gap_daily')
        else:
            # print("Warning: 'openprc' column missing. Skipping ret_gap_vol_1m.")
            df['ret_gap_vol_1m'] = np.nan

        # 10. Close Location Value (CLV) Mean
        # (Close - Low) / (High - Low)
        # Proxy for accumulation (close near high) vs distribution (close near low)
        # Check if askhi/bidlo available
        if 'askhi' in df.columns and 'bidlo' in df.columns:
            range_daily = high - low
            # Avoid div zero
            df['clv_daily'] = (price - low) / range_daily.replace(0, np.nan)
            df.loc[range_daily <= 0, 'clv_daily'] = 0.5 # Neutral if no range
            # Clip
            df['clv_daily'] = df['clv_daily'].clip(0, 1)
            df['clv_mean_1m'] = group_rolling_mean(g, 'clv_daily')
        else:
            df['clv_mean_1m'] = np.nan

        # 11. Zero Return Days (Illiquidity)
        df['is_zero_ret'] = (df['ret'] == 0).astype(float)
        df['zero_ret_pct_1m'] = group_rolling_mean(g, 'is_zero_ret')

        print("Calculating Extended Factors (Trade/Split Based)...")
        # 12. Average Trade Size (Institutional Presence)
        # Vol / NumTrd. Higher = More Institutional.
        if 'numtrd' in df.columns:
            # Avoid div zero
            df['trade_size'] = df['vol'] / df['numtrd'].replace(0, np.nan)
            df['avg_trade_size_1m'] = group_rolling_mean(g, 'trade_size')

            # 13. Trade-Based Illiquidity (Kyle's Lambda Proxy)
            # |Ret| / NumTrd
            df['illiq_numtrd_daily'] = df['ret'].abs() / df['numtrd'].replace(0, np.nan)
            df['illiq_numtrd_1m'] = group_rolling_mean(g, 'illiq_numtrd_daily')
        else:
            df['avg_trade_size_1m'] = np.nan
            df['illiq_numtrd_1m'] = np.nan

        # 14. Payout Yield (High Frequency)
        # Sum(Ret - Retx). Captures dividends paid in the month.
        if 'retx' in df.columns:
            df['div_yield_daily'] = df['ret'] - df['retx']
            # Sum over month
            df['payout_yield_1m'] = group_rolling_sum(g, 'div_yield_daily')
        else:
            df['payout_yield_1m'] = np.nan

        # 15 & 16. Intraday Return & Volatility (Split Adjusted)
        # Intraday Ret = (Close / Open) - 1.
        if open_price is not None:
            # Prefer adjusted prices where available
            if 'adj_openprc' in df.columns and 'adj_prc' in df.columns:
                open_for_intra = df['adj_openprc']
                close_for_intra = df['adj_prc']
            else:
                open_for_intra = df['openprc']
                close_for_intra = price

            valid_intra = (open_for_intra > 0)
            df['intraday_ret_daily'] = np.nan
            df.loc[valid_intra, 'intraday_ret_daily'] = (
                close_for_intra[valid_intra] / open_for_intra[valid_intra]
            ) - 1

            # Factor 15: Mean Intraday Return (Daytime Momentum)
            df['intraday_ret_1m'] = group_rolling_mean(g, 'intraday_ret_daily')

            # Factor 16: Intraday Volatility (Daytime Risk vs Gap Risk)
            df['intraday_vol_1m'] = group_rolling_std(g, 'intraday_ret_daily')
        else:
            df['intraday_ret_1m'] = np.nan
            df['intraday_vol_1m'] = np.nan

        # Output Preparation
        out_cols = [
            'permno', 'date',
            'prc', 'ret', 'retx', 'vol', 'shrout', # Include Raw for Backtest
            'amihud_1m', 'turnover_var', 'dvol_cv_1m',
            'parkinson_vol_1m', 'downside_vol_1m', 'upside_vol_1m',
            'max_ret_1m', 'skew_1m',
            'ret_gap_vol_1m', 'clv_mean_1m', 'zero_ret_pct_1m',
            'avg_trade_size_1m', 'illiq_numtrd_1m', 'payout_yield_1m',
            'intraday_ret_1m', 'intraday_vol_1m'
        ]

        # Describe data frequency
        # "Factors are as-of Close. Used to predict T+1."

        # Filter columns to only those that exist
        final_cols = [c for c in out_cols if c in df.columns]
        return df[final_cols]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', default='2020-01-01')
    parser.add_argument('--end_date', default='2024-12-31')
    parser.add_argument("--raw_dir", type=str, default="./data/raw_ghz", help="Directory containing raw parquet files (crsp_dsf)")
    parser.add_argument("--out_dir", type=str, default="./data/processed/daily_factors", help="Output directory for calculated factors")
    args = parser.parse_args()

    print(f"Running Daily Factor Builder from: {args.raw_dir}")
    print(f"Output Directory: {args.out_dir}")

    builder = DailyFactorsBuilder(data_root=args.raw_dir)
    builder.load_data()
    factors = builder.process_daily_factors(start_date=args.start_date, end_date=args.end_date)

    if not factors.empty:
        print(f"Factors Generated: {factors.shape}")

        # SAVE MONTHLY
        print("Saving Monthly Files...")
        factors['year'] = factors['date'].dt.year
        factors['month'] = factors['date'].dt.month

        os.makedirs(args.out_dir, exist_ok=True)

        for (year, month), group in factors.groupby(['year', 'month']):
            fname = f"daily_factors_{year}_{month:02d}.parquet"
            fpath = os.path.join(args.out_dir, fname)
            # Drop partition cols
            group.drop(columns=['year', 'month']).to_parquet(fpath)
            print(f"Saved {fname} ({group.shape[0]} rows)")

    else:
        print("No factors generated.")
