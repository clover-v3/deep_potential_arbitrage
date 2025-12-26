"""
Intraday Feature Engineering Module

Computes 20+ distinct daily features from 1-minute OHLCV data.
Categories:
1. Volatility (Realized Vol, Skew, Kurt)
2. Extremes (Drawdown, Parkinson, Tails)
3. Liquidity (Amihud, Vol Variance)
4. Microstructure (Hurst, Autocorr, Path)
"""

import pandas as pd
import numpy as np
from typing import Dict

class IntradayFeatureEngine:
    @staticmethod
    def compute_all(df_1min: pd.DataFrame) -> pd.Series:
        """
        Compute all intraday features for a single stock-day.
        Args:
            df_1min: DataFrame with index (Datetime) and cols: open, high, low, close, volume
                     Must be sorted by time.
        Returns:
            pd.Series: Aggregated daily features
        """
        if df_1min.empty or len(df_1min) < 10:
            return pd.Series()

        # Pre-calculation
        returns = df_1min['close'].pct_change().dropna()
        log_ret = np.log(df_1min['close'] / df_1min['close'].shift(1)).dropna()
        high = df_1min['high']
        low = df_1min['low']
        open_p = df_1min['open']
        close = df_1min['close']
        volume = df_1min['volume']
        # Avoid division by zero
        vwap_denom = (volume * close).replace(0, np.nan)

        features = {}

        # --- A. Volatility & Distribution ---
        features['realized_vol'] = returns.std(ddof=1)
        features['realized_skew'] = returns.skew()
        features['realized_kurtosis'] = returns.kurtosis()

        down_rets = returns[returns < 0]
        up_rets = returns[returns > 0]
        features['downside_vol'] = down_rets.std(ddof=1) if len(down_rets) > 0 else 0
        features['upside_vol'] = up_rets.std(ddof=1) if len(up_rets) > 0 else 0

        # --- B. Extremes & Range ---
        # Intraday Max Drawdown
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        features['intraday_max_drawdown'] = dd.min()

        # Parkinson Vol (High-Low)
        # 1 / (4 * ln(2)) * mean(ln(H/L)^2)
        hl_ratio_sq = np.log(high / low) ** 2
        features['parkinson_vol'] = np.sqrt(1.0 / (4.0 * np.log(2.0)) * hl_ratio_sq.mean())

        # Rogers-Satchell (Drift independent)
        # mean( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
        term1 = np.log(high / close) * np.log(high / open_p)
        term2 = np.log(low / close) * np.log(low / open_p)
        features['rogers_satchell_vol'] = np.sqrt((term1 + term2).mean())

        # Normalized Range
        features['high_low_ratio'] = ((high.max() - low.min()) / open_p.iloc[0])

        # Tail Risk (95% - 5%) - actually looking for magnitude of tails
        features['tail_risk_ratio'] = np.abs(returns.quantile(0.95)) + np.abs(returns.quantile(0.05))

        # --- C. Liquidity & Volume ---
        # Amihud (Intraday proxy): mean( |ret| / (Price * Vol) )
        # Scale can be very small, maybe log transform later.
        illiq = returns.abs() / (close.iloc[1:] * volume.iloc[1:])
        features['amihud_illiquidity'] = illiq.replace([np.inf, -np.inf], np.nan).mean()

        features['volume_volatility'] = volume.std() / (volume.mean() + 1e-6) # CV of volume

        # Price-Vol Corr
        if len(returns) == len(volume.iloc[1:]):
             features['price_vol_corr'] = returns.corr(volume.iloc[1:])
        else:
             features['price_vol_corr'] = np.nan

        features['volume_skew'] = volume.skew()

        # Signed Volume Imbalance (Aggr Buy - Aggr Sell)
        # Proxy: Close > Open => Buy, Close < Open => Sell
        # Or Tick rule on 1-min close?
        # Let's use Ret > 0 as Buy
        buy_vol = volume.iloc[1:][returns > 0].sum()
        sell_vol = volume.iloc[1:][returns < 0].sum()
        features['signed_volume_imbalance'] = (buy_vol - sell_vol) / (volume.sum() + 1e-6)

        # --- D. Microstructure & Efficiency ---
        # Autocorrelation (Lag 1)
        features['autocorr_1min'] = returns.autocorr(lag=1)

        # Path Length (Sum of abs changes)
        features['price_path_length'] = returns.abs().sum()

        # Jump Count (3 std events)
        std_ret = features['realized_vol']
        if std_ret > 0:
            features['jump_count'] = (returns.abs() > 3 * std_ret).sum()
        else:
            features['jump_count'] = 0

        # Reversal Intensity (Count sign flips)
        # (r_t * r_{t-1}) < 0
        sign_flip = (returns * returns.shift(1)) < 0
        features['reversal_intensity'] = sign_flip.sum() / len(returns)

        # Hurst Exponent (Simple R/S implementation)
        # R/S analysis on the whole day
        # Range of cumulative dev / Std dev
        # X = cumsum(ret - mean)
        # R = max(X) - min(X)
        # S = std(ret)
        # H = log(R/S) / log(N) (Approximation)
        ret_mean = returns.mean()
        cum_dev = (returns - ret_mean).cumsum()
        R = cum_dev.max() - cum_dev.min()
        S = std_ret
        if S > 0 and R > 0:
            features['hurst_exponent'] = np.log(R/S) / np.log(len(returns))
        else:
            features['hurst_exponent'] = 0.5 # Random walk

        # --- E. Intraday Momentum ---
        features['intraday_mom'] = (close.iloc[-1] - open_p.iloc[0]) / open_p.iloc[0]
        # Open Gap (needs prev close, passed in or assumed from first open?
        # Actually gap is Open_t - Close_{t-1}.
        # Without previous day, we can't calculate GAP precisely here from single file.
        # We can calculate "Overnight" if we had prev day.
        # For now, let's omit Gap or use Open - Low?
        # Let's use Close-High (Distance from High)
        features['close_dist_high'] = (high.max() - close.iloc[-1]) / high.max()

        return pd.Series(features)
