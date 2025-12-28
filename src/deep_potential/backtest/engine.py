import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .cost_model import TransactionCostModel, LinearCostModel

class BacktestEngine:
    """
    Vectorized Backtest Engine.
    Simulates portfolio evolution over time.
    """
    def __init__(
        self,
        initial_capital: float = 1e6,
        cost_model: TransactionCostModel = None
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or LinearCostModel(bps=10.0)

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        returns: pd.DataFrame = None
    ) -> Dict[str, pd.Series]:
        """
        Run simulation.
        signals: (T, N) Target Weights.
        prices: (T, N) Execution Prices (Open/Close).
        returns: (T, N) Period Returns.
        """
        # Ensure alignment
        common_idx = signals.index.intersection(prices.index)
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]

        if returns is None:
            returns = prices.pct_change().fillna(0.0)
        else:
            returns = returns.loc[common_idx]

        # Simulation Arrays
        n_steps, n_assets = signals.shape
        portfolio_value = np.zeros(n_steps)
        positions = np.zeros((n_steps, n_assets)) # In Dollars? Or Shares?
        # Target Weights strategy usually implies rebalancing to target % of Equity.

        cash = self.initial_capital
        equity = self.initial_capital

        # Helper: Current Holdings
        # We can implement a fully vectorized approximation or loop.
        # Loop is safer for costs and compounding.

        # State
        current_holdings = np.zeros(n_assets) # In Dollars

        pnl_series = []
        turnover_series = []
        cost_series = []

        # Iterate
        # signals[t] determines positions held from t to t+1 ?
        # Or signals[t] uses data up to t, executes at t+1 Open?
        # Let's assume Signal T uses Data <= T. Execution at T+1 Close (or T Close if robust).
        # Standard: Calculate Signal at Close T, Trade at Close T (MOC) or Open T+1.
        # We'll assume signals align with 'returns' such that signal[t] captures return[t+1].
        # wait. signal[t] * return[t+1] = PnL.

        signals_vals = signals.values
        ret_vals = returns.values

        # Pre-shift signals??
        # Usually signals generated at T affect returns at T+1.
        # If input 'signals' is aligned such that signals.iloc[i] corresponds to returns.iloc[i],
        # then we just multiply.
        # But usually Backtester takes 'Target Weights' at time T.
        # And we earn return T+1.

        # Let's shift returns in loop logic.

        equity_curve = [self.initial_capital]

        # positions[0] = 0
        current_weights = np.zeros(n_assets)

        for t in range(n_steps - 1):
            # 1. Start of Period t (which is actually T for signal, T+1 for return)
            # Let's say t is the time we set the portfolio.

            target_w = signals_vals[t]

            # Rebalance
            # Target Dollars = Target W * Current Equity
            target_pos = target_w * equity

            # Turnover
            trade_val = target_pos - current_holdings
            turnover = np.sum(np.abs(trade_val))

            # Cost
            cost = self.cost_model.compute_cost(turnover)

            # Update Equity (deduct cost)
            equity -= cost

            # Update Holdings
            current_holdings = target_pos

            # 2. End of Period (Apply Return)
            # Returns from t to t+1
            period_ret = ret_vals[t+1] # Lookahead?
            # returns usually: price[t] / price[t-1] - 1.
            # So ret[t+1] is return from t to t+1. Correct.

            pnl = np.sum(current_holdings * period_ret)
            equity += pnl
            current_holdings *= (1 + period_ret) # drift

            equity_curve.append(equity)
            turnover_series.append(turnover)
            cost_series.append(cost)

        # Finalize
        equity_series = pd.Series(equity_curve, index=signals.index[:len(equity_curve)])

        stats = {
            'equity': equity_series,
            'turnover': np.mean(turnover_series),
            'total_cost': np.sum(cost_series),
            'sharpe': self._sharpe(equity_series),
            'max_dd': self._max_dd(equity_series)
        }

        return stats

    def _sharpe(self, equity: pd.Series) -> float:
        ret = equity.pct_change().dropna()
        if ret.std() == 0: return 0.0
        return ret.mean() / ret.std() * np.sqrt(252) # Ann

    def _max_dd(self, equity: pd.Series) -> float:
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return dd.min()
