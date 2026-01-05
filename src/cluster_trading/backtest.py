import torch
import pandas as pd

class BacktestEngine:
    """
    Vectorized Backtester for Differentiable Trading.
    Handles PnL, Turnover, and Costs.
    """
    def __init__(self, cost_bps: float = 10.0):
        self.cost_rate = cost_bps / 10000.0

    def compute_stats(self, weights: torch.Tensor, prices: torch.Tensor) -> dict:
        """
        Args:
            weights: (Batch, N, T) - Target weights at time t
            prices: (Batch, N, T) - Prices at time t

        Returns dict with:
            'ret_pre_cost': (Batch, T)
            'ret_post_cost': (Batch, T)
            'turnover': (Batch, T)
        """
        # 1. Calculate Returns of potential assets
        # r_{t+1} = (p_{t+1} - p_t) / p_t
        # Note: weights[t] decides exposure to r_{t+1} (or r_t depending on alignment)
        # Typically: Signals generated at Close[t], executed at Close[t] (or Open[t+1]).
        # Here we assume execution at Close[t], capturing return from t to t+1.

        # prices: (B, N, T)
        future_prices = prices.roll(-1, dims=-1) # Shift left (t+1 moves to t)
        # Last element is invalid
        asset_returns = (future_prices - prices) / (prices + 1e-8)

        # Zero out last return (unknown)
        asset_returns[..., -1] = 0.0

        # 2. Portfolio Return (Pre-Cost)
        # R_p = sum(w_i * r_i)
        port_ret = (weights * asset_returns).sum(dim=1) # (B, T)

        # 3. Turnover
        # Delta W = |w_t - w_{t-1} * (1 + r_{t-1})|  <-- Self-financing drift?
        # Simplified: |w_t - w_{t-1}|
        # Shift weights right to get w_{t-1}
        prev_weights = weights.roll(1, dims=-1)
        prev_weights[..., 0] = 0.0

        turnover = (weights - prev_weights).abs().sum(dim=1) # (B, T)

        # 4. Costs
        costs = turnover * self.cost_rate

        net_ret = port_ret - costs

        # Cumulative PnL
        cum_ret = torch.cumsum(net_ret, dim=-1)

        return {
            'daily_ret': net_ret,
            'turnover': turnover,
            'cum_ret': cum_ret,
            'costs': costs
        }
