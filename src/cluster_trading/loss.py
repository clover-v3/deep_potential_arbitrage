import torch
import torch.nn as nn

class DifferentiableTradingLoss(nn.Module):
    """
    Differentiable Loss Function for CoopTradingSystem.
    Components:
    1. Returns (Maximize)
    2. Sharpe Ratio (Maximize risk-adjusted returns)
    3. Turnover Probability/Cost (Minimize)
    """
    def __init__(self,
                 turnover_penalty: float = 0.0,
                 leverage_penalty: float = 0.0,
                 max_leverage: float = 1.0,
                 use_sharpe: bool = True,
                 cost_bps: float = 10.0):
        super().__init__()
        self.turnover_penalty = turnover_penalty
        self.leverage_penalty = leverage_penalty
        self.max_leverage = max_leverage
        self.use_sharpe = use_sharpe
        self.cost_rate = cost_bps / 10000.0

    def forward(self, weights: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: (Batch, N, T_out) - Weights generated from windowed features.
            prices: (Batch, N, T_in) - Raw prices. T_out = T_in - Window + 1.

        Returns:
            loss: Scalar tensor
        """
        # 1. Alignment
        # Feature Window W.
        # weights[0] corresponds to window ending at time W-1 (index W-1 of prices).
        # We want to trade from Closed Price[W-1] to Price[W].
        # So we align weights[t] with Return(Price[t+W-1] -> Price[t+W]).

        B, N, T_in = prices.shape
        _, _, T_out = weights.shape

        # Window size inferred
        W = T_in - T_out + 1

        # Calculate Returns for whole price series first
        # r_t = (p_{t+1} - p_t) / p_t
        # This return belongs to the holding period STARTING at t.
        future_prices = prices.roll(-1, dims=-1)
        all_returns = (future_prices - prices) / (prices + 1e-8)
        # Last return is invalid (wrap around), mask it later or ignore

        # We need returns starting from index W-1 up to T_in - 2
        # weights[0] -> trade at W-1 -> return[W-1] (which is (p_W - p_{W-1})/p_{W-1})
        # weights[T_out-1] -> trade at T_in-1 (Last Step) -> return[T_in-1] (Invalid/Future)

        # Wait, if weights[last] corresponds to time T_in-1 (last available price),
        # we cannot trade into the future. So the last weight is useless for training PnL (no label).
        # We truncate the last weight.

        # Returns of interest:
        # Start index: W-1
        # End index: T_in - 2 (inclusive) -> length T_out - 1

        valid_returns = all_returns[..., W-1 : -1] # Shape (B, N, T_out - 1)
        valid_weights = weights[..., :-1]          # Shape (B, N, T_out - 1)

        if valid_returns.shape[-1] == 0:
             # Handle minimal case
             return torch.tensor(0.0, device=weights.device, requires_grad=True)

        # Portfolio Returns
        # (B, T_valid)
        port_ret = (valid_weights * valid_returns).sum(dim=1)

        # 2. Compute Turnover
        # Sum(|w_t - w_{t-1}|)
        prev_weights = valid_weights.roll(1, dims=-1)
        prev_weights[..., 0] = 0.0 # First step turnover assumed from 0
        turnover = (valid_weights - prev_weights).abs().sum(dim=1)

        # 3. Transaction Costs impact on Returns
        net_ret = port_ret - (turnover * self.cost_rate)

        # 4. Loss Components
        if self.use_sharpe:
            mean_ret = net_ret.mean(dim=-1)
            std_ret = net_ret.std(dim=-1) + 1e-6
            performance_metric = mean_ret / std_ret
        else:
            performance_metric = net_ret.mean(dim=-1)

        # B. Turnover Penalty
        avg_turnover = turnover.mean(dim=-1)

        # C. Leverage Penalty (Soft Constraint)
        # Global Leverage at each step = sum(|w|)
        current_leverage = valid_weights.abs().sum(dim=1) # (B, T_valid)
        # Penalty = ReLU(Leverage - Max)^2
        leverage_excess = torch.relu(current_leverage - self.max_leverage)
        leverage_cost = (leverage_excess ** 2).mean(dim=-1) # Mean over time

        # Total Loss
        loss = -performance_metric + \
               (self.turnover_penalty * avg_turnover) + \
               (self.leverage_penalty * leverage_cost)

        return loss.mean()
