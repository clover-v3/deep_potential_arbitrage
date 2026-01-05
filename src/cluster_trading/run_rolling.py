import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

from src.cluster_trading.data_engine import DataEngine
from src.cluster_trading.system import CoopTradingSystem
from src.cluster_trading.loss import DifferentiableTradingLoss
from src.cluster_trading.backtest import BacktestEngine

class RollingTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_model(self, prices_train: torch.Tensor):
        """
        Trains a fresh model on the given training data.
        Returns the trained model.
        """
        # Init System
        system = CoopTradingSystem(
            n_clusters=self.args.n_clusters,
            feature_window=self.args.window,
            entry_threshold=self.args.threshold,
            stop_threshold=self.args.stop_threshold,
            similarity_threshold=self.args.similarity_threshold,
            temp=self.args.temp,
            scaling_factor=self.args.scaling_factor
        ).to(self.device)

        # Init Centroids (helps convergence)
        system.initialize_clusters(prices_train)

        # Loss & Optimizer
        criterion = DifferentiableTradingLoss(
            turnover_penalty=self.args.turnover_penalty,
            leverage_penalty=self.args.leverage_penalty,
            max_leverage=self.args.max_leverage,
            cost_bps=self.args.cost_bps,
            use_sharpe=True
        )
        optimizer = optim.Adam(system.parameters(), lr=self.args.lr)

        # Training Loop
        prices_train = prices_train.to(self.device)

        for _ in range(self.args.epochs):
            optimizer.zero_grad()
            out = system(prices_train, hard=False)
            weights = system.get_portfolio_weights(out['positions'], method='independent')
            loss = criterion(weights, prices_train)
            loss.backward()
            optimizer.step()

        return system

    def run_rolling(self):
        print(f"=== Starting Rolling Training ===")
        print(f"Train Period: {self.args.train_days} days")
        print(f"Test Period: {self.args.test_days} days")

        # 1. Load All Data
        engine = DataEngine(data_path=self.args.data_path)
        if self.args.data_path == 'virtual':
             # Virtual Data
             print("Generating Virtual Data...")
             B, N, T = 1, 50, 2000
             returns = torch.randn(N, T) * 0.001
             prices = 100 * torch.exp(torch.cumsum(returns, dim=1))
             engine.price_tensor = prices.transpose(0, 1)
             engine.dates = pd.date_range(start='2020-01-01', periods=T, freq='D')
             engine.tickers = [f"T{i}" for i in range(N)]
        else:
             fpath = Path(self.args.data_path)
             if fpath.is_file() and fpath.suffix == '.parquet':
                 df = pd.read_parquet(fpath)
                 engine.load_wide_data(df)
             else:
                 engine.load_data(self.args.start_date, self.args.end_date)

        full_prices = engine.price_tensor.T.unsqueeze(0).to(self.device) # (1, N, T_total)
        dates = engine.dates
        total_steps = full_prices.shape[-1]

        train_steps = self.args.train_days
        test_steps = self.args.test_days

        current_idx = 0

        # We will collect ALL test weights into a single timeline
        # Initialize with zeros.
        # Note: The first 'train_steps' will have 0 weights (start of simulation).
        # We only start trading after the first training period.
        global_test_weights = torch.zeros_like(full_prices) # (1, N, T)

        test_periods_coverage = [] # To track where we actually have predictions

        while current_idx + train_steps + test_steps <= total_steps:
             # Indices
            train_start = current_idx
            train_end = current_idx + train_steps
            test_start = train_end
            test_end = train_end + test_steps

            # Slices
            prices_train = full_prices[..., train_start:train_end]
            prices_test = full_prices[..., test_start:test_end]

            period_str = f"Train[{dates[train_start].date()}:{dates[train_end-1].date()}] -> Test[{dates[test_start].date()}:{dates[test_end-1].date()}]"
            print(f"\nProcessing: {period_str}")

            # A. Train
            model = self.train_model(prices_train)

            # B. Test (Inference)
            # Input Context: Train Window Tail + Test Data
            window = self.args.window
            context_prices = torch.cat([prices_train[..., -window:], prices_test], dim=-1)

            with torch.no_grad():
                out = model(context_prices, hard=True)

            # Extract Test Positions
            # context valid length is len(context) - window + 1 = test_steps + 1
            # We want positions corresponding to the test steps.
            # positions shape: (1, N, test_steps + 1)
            # The last element corresponds to signal at T_end.
            # We align such that positions[t] is the target holding for t to t+1.
            # So we take [..., :-1] if size is test_steps + 1?
            # Let's count precise steps.
            # Context size C = W + M.
            # Out size O = M + 1.
            # Output indices 0..M.
            # Index 0 corresponds to window ending at start of test. -> Signal used for first test step return.
            # Index M corresponds to window ending at end of test.

            # We need positions for range [test_start, test_end). M steps.
            # So we take out['positions'][..., :test_steps]

            positions_test = out['positions'][..., :test_steps] # (1, N, M)
            weights_test = model.get_portfolio_weights(positions_test, method='independent')

            # Stitch into Global Weights
            global_test_weights[..., test_start:test_end] = weights_test

            test_periods_coverage.append((test_start, test_end))

            # Slide
            current_idx += test_steps

        # 3. Global Backtest
        print("\n=== Running Global Backtest (Stitched Signals) ===")
        backtester = BacktestEngine(cost_bps=self.args.cost_bps)

        # We only care about stats during the testing period (after first train end)
        if not test_periods_coverage:
            print("No testing periods completed.")
            return -999.0, -999.0

        first_test_idx = test_periods_coverage[0][0]
        last_test_idx = test_periods_coverage[-1][1]

        # Valid range for analysis
        analysis_weights = global_test_weights[..., first_test_idx:last_test_idx]
        analysis_prices = full_prices[..., first_test_idx:last_test_idx]

        stats = backtester.compute_stats(analysis_weights, analysis_prices)

        daily_ret = stats['daily_ret'].squeeze(0).cpu().numpy()
        sharpe = np.mean(daily_ret) / (np.std(daily_ret) + 1e-9) * np.sqrt(252)
        total_ret = np.sum(daily_ret)

        print(f"Global Test Sharpe: {sharpe:.4f}")
        print(f"Global Total Return: {total_ret:.4f}")

        return sharpe, total_ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_path', type=str, default='virtual')
    parser.add_argument('--start_date', type=str, default='2020-01-01')
    parser.add_argument('--end_date', type=str, default='2023-01-01')

    # Rolling Config (Days)
    parser.add_argument('--train_days', type=int, default=252) # ~1 year
    parser.add_argument('--test_days', type=int, default=63)   # ~3 months

    # Model Config (Same as train.py)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=2.0)
    parser.add_argument('--stop_threshold', type=float, default=4.0)
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--scaling_factor', type=float, default=5.0)

    # Training Config
    parser.add_argument('--epochs', type=int, default=50) # Faster for rolling
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--turnover_penalty', type=float, default=0.5)
    parser.add_argument('--leverage_penalty', type=float, default=1.0)
    parser.add_argument('--max_leverage', type=float, default=5.0)
    parser.add_argument('--cost_bps', type=float, default=10.0)

    args = parser.parse_args()

    trainer = RollingTrainer(args)
    trainer.run_rolling()
