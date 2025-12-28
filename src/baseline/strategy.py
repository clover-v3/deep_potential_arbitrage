"""
Han et al. (2021) Baseline Trading Strategy

Logic:
1. Calculate Cluster Equal-Weighted Index (or Market-Cap Weighted)
2. Calculate StockIdiosyncratic Return = R_i - R_cluster
3. Signal: Z-Score of Cumulative Idiosyncratic Return
4. Trade: Long if Z < -Threshold, Short if Z > Threshold
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class BaselineStrategy:
    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.0, window: int = 20,
                 cost_bps: float = 0.0, signal_method: str = 'threshold', top_k_percent: int = 20,
                 stop_entry_last_n: int = 3):
        """
        Args:
            entry_z: 开仓/反向开仓阈值（绝对值）。
            exit_z: 目前未显式使用，预留给将来的显式平仓逻辑。
            window: 计算累积 idio 收益和波动的滚动窗口长度（交易日）。
            cost_bps: 单边交易成本（bp）。
            signal_method: 'threshold' 或 'rank'。
            top_k_percent: rank 模式下簇内多空比例。
            stop_entry_last_n: 交易期最后 N 天禁止开新仓（仅允许平仓），N<=0 表示关闭该功能。
        """
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.window = window
        self.cost_bps = cost_bps
        self.signal_method = signal_method
        self.top_k_percent = top_k_percent
        self.stop_entry_last_n = stop_entry_last_n


    def generate_signals(self, merged_df: pd.DataFrame, return_metrics=True):
        """
        Generate signals and calculate returns.
        merged_df: Columns [date, ticker, price, cluster_label, ...]
        """
        # print('merged_df.columns: ', merged_df.columns)
        merged = merged_df.copy()
        merged['date'] = pd.to_datetime(merged['date'])

        # 1. Calculate Returns
        # We need pivoted returns for weights multiplication later
        # But here we work with Long format for signal generation
        # Ensure sorted
        merged = merged.sort_values(['permno', 'date'])

        # Simple returns
        # Check if 'ret' exists
        if 'ret' not in merged.columns:
            # Calculate from close if available, else need 'ret'
            if 'prc' in merged.columns:
                merged['ret'] = merged.groupby('permno')['prc'].pct_change().fillna(0)
            else:
                raise ValueError("DataFrame must contain 'ret' or 'prc'")

        # 2. Cluster Returns (Market/Cluster Mode)
        # We need the mean return of the cluster for each day
        # Group by [date, cluster]
        cluster_rets = merged.groupby(['date', 'cluster_label'])['ret'].transform('mean')
        merged['cluster_ret'] = cluster_rets

        # 3. Idiosyncratic Return
        merged['idio_ret'] = merged['ret'] - merged['cluster_ret']

        # 4. Fill NaNs (for initial periods)
        merged['idio_ret'] = merged['idio_ret'].fillna(0)

        # 5. Cumulative Idiosyncratic (Price Deviation)
        merged['cum_idio'] = merged.groupby('permno')['idio_ret'].transform(
            lambda x: x.rolling(self.window).sum()
        )

        # 6. Z-Score
        merged['std_idio'] = merged.groupby('permno')['idio_ret'].transform(
            lambda x: x.rolling(self.window).std()
        )
        merged['z_score'] = merged['cum_idio'] / (merged['std_idio'] + 1e-6)

        # 7. Signal Generation
        merged['pos'] = 0

        if self.signal_method == 'threshold':
            # Mean Reversion: High Z -> Short, Low Z -> Long
            merged.loc[merged['z_score'] > self.entry_z, 'pos'] = -1
            merged.loc[merged['z_score'] < -self.entry_z, 'pos'] = 1

        elif self.signal_method == 'rank':
            # INTRA-CLUSTER Ranking (Top/Bottom K%)
            # Sort by Z-score within each Date-Cluster group
            # Long Bottom K% (Undervalued), Short Top K% (Overvalued)

            # GroupBy is slow if many groups. Transform is faster.
            # We rank Z-score ascending: Low Z (Rank 1) -> Long
            g = merged.groupby(['date', 'cluster_label'])['z_score']

            merged['n_cluster'] = g.transform('count')
            merged['rank_in_cluster'] = g.rank(method='first', ascending=True)

            # Calculate Stock Count threshold (Ceiling)
            # e.g. 10 stocks, 20% -> 2 stocks.
            # 4 stocks, 20% -> 0.8 -> 1 stock.
            k_count = np.ceil(merged['n_cluster'] * (self.top_k_percent / 100.0))

            # Signal Logic
            # Bottom K% -> Long
            merged.loc[merged['rank_in_cluster'] <= k_count, 'pos'] = 1

            # Top K% -> Short
            # i.e. Rank > N - K
            merged.loc[merged['rank_in_cluster'] > (merged['n_cluster'] - k_count), 'pos'] = -1

        # Format back to Wide for Portfolio Calc
        pos_df = merged.pivot(index='date', columns='permno', values='pos').fillna(0)
        returns = merged.pivot(index='date', columns='permno', values='ret').fillna(0)

        # ------------------------------------------------------------------
        # 额外约束 1：交易期最后 N 天停止“开新仓”，仅允许平仓
        # ------------------------------------------------------------------
        if self.stop_entry_last_n is not None and self.stop_entry_last_n > 0 and not pos_df.empty:
            # 以当前传入的 merged_df 为“一个交易期”，例如单个测试月
            all_dates = pos_df.index.sort_values()
            n = min(int(self.stop_entry_last_n), len(all_dates))
            last_dates = list(all_dates[-n:])

            # 逐日在宽表上施加“只许平仓，不许新开/反向”的限制
            for i, d in enumerate(last_dates):
                # 找到 d 的前一交易日（如果在当前期内不存在，则视为全平仓起点）
                idx = all_dates.get_loc(d)
                if idx == 0:
                    prev_pos_row = pos_df.iloc[0] * 0  # 第一日当作之前全空仓
                else:
                    prev_pos_row = pos_df.loc[all_dates[idx - 1]]

                raw_pos_row = pos_df.loc[d]
                new_row = prev_pos_row.copy()

                # 1）如果之前已经有仓位，只允许“平仓”（从非 0 -> 0），不允许反向开新仓
                mask_prev_nonzero = prev_pos_row != 0
                close_mask = mask_prev_nonzero & (raw_pos_row == 0)
                new_row[close_mask] = 0

                # 2）如果之前是空仓，则保持空仓（禁止从 0 -> ±1）
                #    其余情况均保持 prev_pos_row
                pos_df.loc[d] = new_row

            # ------------------------------------------------------------------
            # 额外约束 2：交易期最后一天强制全平仓
            # ------------------------------------------------------------------
            last_day = all_dates[-1]
            pos_df.loc[last_day] = 0

        # Align columns
        pos_df = pos_df.reindex(columns=returns.columns, fill_value=0)

        # PnL (Lagged Position * Return)
        pos_lag = pos_df.shift(1).fillna(0)

        # Weights: Equal Weight across ALL active positions
        # Gross Exposure = Sum(|pos|)
        gross_exposure = pos_lag.abs().sum(axis=1)

        # Avoid division by zero
        weights = pos_lag.div(gross_exposure.replace(0, 1.0), axis=0)

        # Portfolio Gross Return
        port_ret = (weights * returns).sum(axis=1)

        # Transaction Costs
        if self.cost_bps > 0:
            # Turnover computation
            # weights_t - weights_{t-1}
            # Note: Weights change due to price moves too (Drift), but here we assume Daily Rebalance to Target Weights
            # So the trade is exactly the change in target weights.
            # Ideally: Trade = Weight_Target - Weight_Drifted.
            # Simplified: Trade = Weight_t - Weight_{t-1}
            turnover = weights.diff().abs().sum(axis=1).fillna(0)
            cost = turnover * (self.cost_bps / 10000.0)
            port_ret = port_ret - cost

        return {
            'z_score': merged.pivot(index='date', columns='permno', values='z_score'),
            'positions': pos_df,
            'weights': weights,
            'daily_ret': port_ret,
            'daily_pnl': weights * returns, # Attribution (Pre-cost)
            'total_pnl': port_ret # Series (Post-cost)
        }

    def get_summary_metrics(self, daily_ret: pd.Series) -> Dict[str, float]:
        """
        Calculate Sharpe, Max Drawdown, Cumulative Return.
        """
        if daily_ret.empty or daily_ret.std() == 0:
            return {'sharpe': 0.0, 'max_drawdown': 0.0, 'total_return': 0.0, 'annualized_return': 0.0}

        # Annualized Sharpe (assuming daily data)
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)

        # Annualized Return (Geometric)
        cum_ret_series = (1 + daily_ret).cumprod()
        total_ret = cum_ret_series.iloc[-1] - 1

        n_days = len(daily_ret)
        if n_days > 0:
             ann_ret = (1 + total_ret) ** (252 / n_days) - 1
        else:
             ann_ret = 0.0

        # Max Drawdown
        peak = cum_ret_series.cummax()
        drawdown = (cum_ret_series - peak) / peak
        max_dd = drawdown.min()

        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': total_ret,
            'annualized_return': ann_ret
        }
