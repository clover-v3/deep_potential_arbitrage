import argparse
import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.cluster_trading.data_engine import DataEngine
from src.cluster_trading.system import CoopTradingSystem
from src.cluster_trading.loss import DifferentiableTradingLoss

def train(args):
    print(f"=== Training Cluster Trading System ===")
    print(f"Config: {args}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Data Loading
    engine = DataEngine(data_path=args.data_path)

    if args.data_path == 'virtual':
        # Generate Virtual Data for testing if path is 'virtual'
        print("Generating Virtual Data...")
        # 100 tickers, 2000 steps
        B, N, T = 1, 100, 2000
        # Random Walk + some structure
        returns = torch.randn(N, T) * 0.001
        prices = 100 * torch.exp(torch.cumsum(returns, dim=1))
        engine.price_tensor = prices.transpose(0, 1) # (N, T)
        engine.dates = pd.date_range(start='2020-01-01', periods=T, freq='Min')
        engine.tickers = [f"T{i}" for i in range(N)]
    else:
        # Load actual data
        # Assuming parquet file for now or directory
        # For simplicity, we assume data_path points to a parquet file
        # that load_wide_data or load_data can handle.
        # If it's a file, read it.
        fpath = Path(args.data_path)
        if fpath.is_file() and fpath.suffix == '.parquet':
            df = pd.read_parquet(fpath)
            engine.load_wide_data(df)
        else:
            # Fallback to engine's default load method (requires date range)
            engine.load_data(args.start_date, args.end_date)

    # Prepare Tensor: (Batch, N, T)
    # We treat the whole history as one batch for this prototype,
    # or chunk it. Here we use full batch.
    prices = engine.price_tensor.T.unsqueeze(0).to(device) # (1, N, T)

    # 2. Model Init
    system = CoopTradingSystem(
        n_clusters=args.n_clusters,
        feature_window=args.window,
        entry_threshold=args.threshold,
        stop_threshold=args.stop_threshold,
        similarity_threshold=args.similarity_threshold,
        temp=args.temp,
        scaling_factor=args.scaling_factor
    ).to(device)

    # Print Learnable Parameters
    print("\n=== Model Parameters ===")
    for name, param in system.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    print("========================\n")

    # Initialize Centroids Data-Driven
    print("Initializing Centroids...")
    system.initialize_clusters(prices)

    # 3. Loss & Optimizer
    criterion = DifferentiableTradingLoss(
        turnover_penalty=args.turnover_penalty,
        leverage_penalty=args.leverage_penalty,
        max_leverage=args.max_leverage,
        cost_bps=args.cost_bps,
        use_sharpe=True
    )

    # We optimize:
    # 1. Cluster Centroids (system.cluster_layer.centroids)
    # 2. (Optional) Feature Extractor parameters if any (currently deterministic)
    # 3. (Optional) Signal Engine params if made learnable (scaling factor etc)
    optimizer = optim.Adam(system.parameters(), lr=args.lr)

    # 4. Training Loop
    best_loss = float('inf')

    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        # Forward Pass (Soft Mode for Gradients)
        out = system(prices, hard=False)

        # Get Weights
        positions = out['positions']
        # Use INDEPENDENT method for E2E optimization
        weights = system.get_portfolio_weights(positions, method='independent')

        # Compute Loss
        loss = criterion(weights, prices)

        # Backward
        loss.backward()
        optimizer.step()

        # Logging
        current_loss = loss.item()
        pbar.set_description(f"Loss: {current_loss:.4f}")

        if current_loss < best_loss:
            best_loss = current_loss
            # Save Checkpoint
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                torch.save(system.state_dict(), f"{args.save_dir}/best_model.pth")

    print(f"Training Complete. Best Loss: {best_loss:.4f}")
    if args.save_dir:
        print(f"Model saved to {args.save_dir}/best_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='virtual', help='Path to data parquet or "virtual"')
    parser.add_argument('--start_date', type=str, default='2020-01-01')
    parser.add_argument('--end_date', type=str, default='2020-12-31')

    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=2.0)
    parser.add_argument('--stop_threshold', type=float, default=4.0)
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--scaling_factor', type=float, default=5.0)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--turnover_penalty', type=float, default=0.5)
    parser.add_argument('--leverage_penalty', type=float, default=1.0)
    parser.add_argument('--max_leverage', type=float, default=5.0)
    parser.add_argument('--cost_bps', type=float, default=10.0)

    parser.add_argument('--save_dir', type=str, default='checkpoints')

    args = parser.parse_args()
    train(args)
