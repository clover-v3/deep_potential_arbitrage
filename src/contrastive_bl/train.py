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

from torch.utils.data import Sampler

class MonthlyBatchSampler(Sampler):
    """
    Samples batches ensuring all samples in a batch belong to the same month (or date).
    Shuffles:
    - Order of months
    - Tickers within each month
    """
    def __init__(self, dates, batch_size):
        self.dates = dates
        self.batch_size = batch_size

        # Group indices by date
        # dates is expected to be a numpy array or list aligned with dataset
        self.date_to_indices = {}
        for idx, date in enumerate(dates):
            if date not in self.date_to_indices:
                self.date_to_indices[date] = []
            self.date_to_indices[date].append(idx)

        self.unique_dates = list(self.date_to_indices.keys())

    def __iter__(self):
        # 1. Shuffle months
        np.random.shuffle(self.unique_dates)

        for date in self.unique_dates:
            indices = np.array(self.date_to_indices[date])
            # 2. Shuffle tickers within month
            np.random.shuffle(indices)

            # 3. Yield batches
            # If month has > batch_size elements, chunk it
            # If < batch_size, yield one smaller batch (or drop? standard is yield)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i+self.batch_size]

    def __len__(self):
        # Approximation: sum of batches per month
        count = 0
        for indices in self.date_to_indices.values():
            count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count

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
    smoke_test=False,
    batch_mode='global',
    d_model=128,
    n_bins=64,
    dropout=0.1
):
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
             device = 'mps'
        else:
            device = 'cpu'

    print(f"Training ORCA on {device} (Batch Mode: {batch_mode}, d_model={d_model})...")

    # 1. Load Data
    loader = ORCADataLoader(data_root)
    # Load recent data for training
    print(f"Loading and building features ({start_year}-{end_year})...")
    loader.load_data(start_year, end_year)
    df = loader.build_orca_features()

    if df.empty:
        print("Error: No data loaded. Cannot train.")
        return

    print(f"Data Loaded: {df.shape}")

    # 2. Preprocess
    # Exclude IDs and metadata
    exclude_cols = ['permno', 'date', 'gvkey', 'cusip', 'valid_from',
                    'datadate', 'cusip6', 'ncusip6', 'namedt', 'nameendt']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # Universe Selection (Top 2000 by Market Cap at Last Date)
    if 'mve_m' in df.columns and 'date' in df.columns:
        last_date = df['date'].max()
        print(f"Selecting Universe based on Market Cap at {last_date}...")

        # Get data for last month
        last_month_df = df[df['date'] == last_date]

        # Sort by MVE and take top 2000
        # Check if mve_m is valid
        if not last_month_df.empty:
            top_2000 = last_month_df.sort_values('mve_m', ascending=False).head(2000)
            universe_permnos = top_2000['permno'].unique()
            print(f"Universe selected: {len(universe_permnos)} stocks.")

            # Filter Training Data to this Universe
            df = df[df['permno'].isin(universe_permnos)].copy()

            # Save Universe
            uni_save_path = save_path.replace('.pth', '_universe.npy')
            np.save(uni_save_path, universe_permnos)
            print(f"Saved universe to {uni_save_path}")
        else:
             print("Warning: Last month data empty, skipping universe selection.")
    else:
        print("Warning: 'mve_m' or 'date' missing. Skipping universe selection.")

    # Sort by Date for Time-Series Split (and Monthly Sampler)
    if 'date' in df.columns:
        df = df.sort_values(['date', 'permno'])

    # Store Dates for Sampler
    dates_array = df['date'].values if 'date' in df.columns else np.zeros(len(df))

    # Robust conversion
    features = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)

    # Normalize (StandardScaler) + Winsorize (Clip at 5 std)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features = (features - mean) / std

    # Winsorize: Clip at +/- 5
    features = np.clip(features, -5, 5)

    # Returns for PINN
    if 'ret' in df.columns:
        ret_series = pd.to_numeric(df['ret'], errors='coerce').fillna(0).values.astype(np.float32)
    else:
        print("Warning: 'ret' column missing. Using mom_1 as proxy.")
        idx_mom1 = feature_cols.index('mom_1') if 'mom_1' in feature_cols else 0
        ret_series = features[:, idx_mom1]

    # Create (r_t, r_{t-1})
    r_prev = np.roll(ret_series, 1)
    r_prev[0] = 0

    # Stack returns for loader
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

    # Sampler Logic
    if batch_mode == 'monthly':
        # Pass dates corresponding to TRAIN indices
        train_dates = dates_array[train_idx]
        train_sampler = MonthlyBatchSampler(train_dates, batch_size=batch_size)
        train_loader = DataLoader(TensorDataset(X_train, R_train), batch_sampler=train_sampler)
    else:
        # Global Shuffle
        train_loader = DataLoader(TensorDataset(X_train, R_train), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(X_val, R_val), batch_size=batch_size, shuffle=False)

    # 4. Model & Optimizer
    model = ORCAModel(
        n_features=len(feature_cols),
        n_clusters=30,
        d_model=d_model,
        n_bins=n_bins,
        dropout=dropout
    ).to(device)
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
    parser.add_argument("--batch_mode", type=str, default="global", help="global/monthly")
    parser.add_argument("--save_path", type=str, default="orca_model.pth", help="Path to save model")

    # Hyperparams
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_bins", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    train_orca(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=0.002,
        save_path=args.save_path,
        device=args.device,
        start_year=args.start_year,
        end_year=args.end_year,
        smoke_test=args.smoke_test,
        batch_mode=args.batch_mode,
        d_model=args.d_model,
        n_bins=args.n_bins,
        dropout=args.dropout
    )
