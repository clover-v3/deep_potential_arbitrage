
import sys
import os
import torch
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from deep_potential.data.dataset import DeepPotentialDataset
from deep_potential.model.model import DeepPotentialModel
from deep_potential.strategy.controller import PhaseSpaceController
from deep_potential.backtest.engine import BacktestEngine

def test_full_pipeline():
    print("=== Starting Integration Test ===")

    # 1. Create Dummy Data
    print("[1] Generating Synthetic Data...")
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    assets = [f'Asset_{i}' for i in range(10)]

    # Create multi-index DF
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset_id'])
    n_rows = len(idx)

    # Random Prices and Returns
    df = pd.DataFrame(index=idx)
    df['close'] = np.random.randn(n_rows).cumsum() + 100
    df['return'] = df.groupby('asset_id')['close'].pct_change().fillna(0)
    df['feature_1'] = np.random.randn(n_rows)

    # 2. Dataset
    print("[2] Initializing Dataset...")
    dataset = DeepPotentialDataset(
        df=df,
        feature_cols=['return', 'feature_1'],
        target_col='return',
        window_size=20
    )

    print(f"    Dataset Size: {len(dataset)}")

    # 3. Model
    print("[3] Initializing Dual-Tower Model...")
    model = DeepPotentialModel(
        input_dim=2, # return, feature_1
        hidden_dim=16,
        n_heads=2,
        top_k=5
    )

    # 4. Forward Pass (Single Batch)
    print("[4] Running Forward Pass...")
    # Get a sample
    sample = dataset[0] # Returns dict containing (N, T, D) tensors

    # Batch dimension needed for model (B, N, T, D)
    x_dyn = sample['features'].unsqueeze(0) # (1, N, T, D)

    with torch.no_grad():
        out = model(x_dyn)

    print("    Model Output Keys:", out.keys())
    print("    Force Shape:", out['force'].shape)

    # 5. Controller
    print("[5] Generating Signals...")
    controller = PhaseSpaceController()

    # Convert tensors to numpy for controller
    force_np = out['force'].squeeze(0).numpy() # (N, D)
    stiffness_np = out['stiffness'].squeeze(0).numpy() # (N, 1)

    # Fake velocity
    velocity_np = np.random.randn(*force_np.shape)

    signals = controller.compute_signal(force_np, stiffness_np, velocity_np)
    print("    Signals Shape:", signals.shape)

    # 6. Backtest
    print("[6] Running Backtest Engine...")
    engine = BacktestEngine(initial_capital=10000)

    # Create Mock Signal DataFrame (Time, Asset)
    # We need signal for every timestep.
    # Simple Loop:
    signal_list = []

    # Generate random signals for validation
    signal_df = pd.DataFrame(
        np.random.randn(len(dates), len(assets)),
        index=dates,
        columns=assets
    )

    # Price DF
    price_df = df['close'].unstack()

    stats = engine.run(signal_df, price_df)

    print("    Backtest Stats:")
    print(f"    Sharpe: {stats['sharpe']:.4f}")
    print(f"    Total Cost: {stats['total_cost']:.2f}")

    print("=== Integration Test Passed ===")

if __name__ == "__main__":
    test_full_pipeline()
