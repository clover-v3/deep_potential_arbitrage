import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from src.contrastive_bl.data_loader import ORCADataLoader
from src.contrastive_bl.orca import ORCAModel

def augment(x, mask_prob=0.1, noise_std=0.1):
    """
    Data Augmentation for Contrastive Learning on Tabular Data.
    x: (Batch, Features)
    """
    # 1. Random Masking
    mask = torch.rand_like(x) < mask_prob
    x_aug = x.clone()
    x_aug[mask] = 0.0

    # 2. Gaussian Noise
    noise = torch.randn_like(x) * noise_std
    x_aug = x_aug + noise

    return x_aug

def train_orca(
    data_root,
    epochs=100,
    batch_size=1024,
    lr=0.002,
    alpha=1.0,
    beta=1.0,
    save_path='orca_model.pth',
    device = None,
    start_year=2000,
    end_year=2023,
    smoke_test=False
):
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
             device = 'mps'
        else:
            device = 'cpu'

    print(f"Training ORCA on {device}...")

    # 1. Load Data
    loader = ORCADataLoader(data_root)
    # Load recent data for training (e.g., 2000-2023)
    print(f"Loading and building features ({start_year}-{end_year})...")
    loader.load_data(start_year, end_year)
    df = loader.build_orca_features()

    if df.empty:
        print("Error: No data loaded. Cannot train.")
        return

    print(f"Data Loaded: {df.shape}")

    # 2. Preprocess
    # Features: All columns except keys
    # Keys: permno, date (index or columns)
    # Check columns.
    # Exclude IDs and metadata
    exclude_cols = ['permno', 'date', 'gvkey', 'cusip', 'valid_from',
                    'datadate', 'cusip6', 'ncusip6', 'namedt', 'nameendt']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Sort by Date for Time-Series Split
    if 'date' in df.columns:
        df = df.sort_values(['date', 'permno'])

    # Robust conversion: Handle pd.NA / Object types / Coercion
    features = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)

    # Normalize (StandardScaler) - Critical for PLE
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features = (features - mean) / std

    # Returns for PINN
    # We need r_t and r_{t-1}.
    # Let's add a raw 'ret' column to use for PINN if available in DF.
    if 'ret' in df.columns:
        ret_series = pd.to_numeric(df['ret'], errors='coerce').fillna(0).values.astype(np.float32)
    else:
        # Fallback to mom_1
        print("Warning: 'ret' column missing. Using mom_1 as proxy.")
        idx_mom1 = feature_cols.index('mom_1') if 'mom_1' in feature_cols else 0
        ret_series = features[:, idx_mom1]

    # Create (r_t, r_{t-1})
    r_prev = np.roll(ret_series, 1)
    r_prev[0] = 0

    # Stack returns for loader: (N, 2)
    returns_tensor = np.stack([ret_series, r_prev], axis=1)

    # 3. Validation Split (Time Series)
    N = len(features)
    split_idx = int(0.8 * N)
    print(f"Time-Series Split: Train {split_idx}, Val {N-split_idx}")

    train_idx = np.arange(0, split_idx)
    val_idx = np.arange(split_idx, N)

    # Create Datasets
    X_train = torch.tensor(features[train_idx])
    R_train = torch.tensor(returns_tensor[train_idx])

    X_val = torch.tensor(features[val_idx])
    R_val = torch.tensor(returns_tensor[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, R_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, R_val), batch_size=batch_size, shuffle=False)

    # 4. Model & Optimizer
    model = ORCAModel(n_features=len(feature_cols), n_clusters=30).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 5. Training Loop
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_ins = 0
        total_clu = 0
        total_pinn = 0

        for i, (x, r) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{epochs}", total=len(train_loader)):
            x, r = x.to(device), r.to(device)

            # Augment
            x_a = augment(x)
            x_b = augment(x)

            optimizer.zero_grad()

            l_ins, l_clu, l_pinn = model.compute_loss(x_a, x_b, r)

            loss = l_ins + alpha * l_clu + beta * l_pinn

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ins += l_ins.item()
            total_clu += l_clu.item()
            total_pinn += l_pinn.item()

            if smoke_test and i >= 5:
                # print("Smoke test limit reached.")
                break

        avg_loss = total_loss / (i + 1)
        print(f"Train Loss: {avg_loss:.4f} (Ins: {total_ins/(i+1):.4f}, Clu: {total_clu/(i+1):.4f}, PINN: {total_pinn/(i+1):.4f})")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i_v, (x, r) in enumerate(val_loader):
                if smoke_test and i_v >= 2: break

                x, r = x.to(device), r.to(device)
                x_a = augment(x)
                x_b = augment(x)
                l_ins, l_clu, l_pinn = model.compute_loss(x_a, x_b, r)

                loss = l_ins + alpha * l_clu + beta * l_pinn
                val_loss += loss.item()

        avg_val_loss = val_loss / (i_v + 1)
        print(f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss or smoke_test:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

        if smoke_test: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/raw_ghz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--smoke_test", action="store_true", help="Run fast smoke test")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu)")
    args = parser.parse_args()

    train_orca(args.data_root, args.epochs, args.batch_size,
               start_year=args.start_year, end_year=args.end_year,
               smoke_test=args.smoke_test, device=args.device)
