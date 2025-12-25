"""
GNN Training Logic with Robust Loss and Correlation Regularization

Implements the training loop and loss functions for learning graph structures
based on dynamics consistency and correlation guidance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from gnn_models import SimpleGNN, compute_laplacian_torch

def hybrid_loss(
    A_pred: torch.Tensor,
    f_batch: torch.Tensor,
    corr_matrix: torch.Tensor,
    dt: float = 0.01,
    delta: float = 1.0,
    alpha_corr: float = 1.0,  # Weight for correlation loss
    alpha_dyn: float = 1.0    # Weight for dynamics loss
) -> Dict[str, torch.Tensor]:
    """
    Hybrid Loss: Robust Dynamics + Correlation Guidance

    Args:
        A_pred: Predicted Adjacency (Batch, N, N)
        f_batch: State series (Batch, T, N)
        corr_matrix: Correlation matrix (Batch, N, N) or (N, N)
        dt: Time step
        delta: Huber delta
        alpha_corr: Weight for correlation term
        alpha_dyn: Weight for dynamics term

    Returns:
        losses: Dict containing total and component losses
    """
    # === 1. Dynamics Loss (Huber) ===
    # df/dt
    df_dt = (f_batch[:, 1:, :] - f_batch[:, :-1, :]) / dt

    # Force = -L * f
    L_pred = compute_laplacian_torch(A_pred)
    f_curr = f_batch[:, :-1, :].transpose(1, 2)
    force_pred = -torch.matmul(L_pred, f_curr).transpose(1, 2)

    # Residual
    residual = df_dt - force_pred

    # Huber Loss
    loss_dyn = nn.HuberLoss(reduction='mean', delta=delta)(residual, torch.zeros_like(residual))

    # === 2. Correlation Loss ===
    # We want A_ij to be related to Corr_ij
    # Since A is sparse and non-negative, and Corr is dense and [-1, 1],
    # we might encourage A to align with |Corr|

    # Expand corr if needed
    if corr_matrix.dim() == 2:
        corr_target = corr_matrix.unsqueeze(0).expand(A_pred.size(0), -1, -1)
    else:
        corr_target = corr_matrix

    # Use absolute correlation as a "soft target" or "prior"
    # We don't force A to equal Corr, but penalize large deviations
    # specially where Corr is high.
    # Actually, simpler: MSE(A, |Corr|) might be too strong because A should be sparse.
    # Better: Contrastive?
    # Let's simple MSE for now as requested: "Can we add corr to loss?"

    loss_corr = nn.MSELoss()(A_pred, torch.abs(corr_target))

    # === 3. Regularization ===
    l1_reg = torch.mean(torch.abs(A_pred))

    # Total Loss
    total_loss = alpha_dyn * loss_dyn + alpha_corr * loss_corr + 0.01 * l1_reg

    return {
        'total': total_loss,
        'dyn': loss_dyn,
        'corr': loss_corr,
        'reg': l1_reg
    }

def train_gnn_model(
    f_series: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 32,
    window_size: int = 50,
    learning_rate: float = 0.001,
    dt: float = 0.01,
    alpha_corr: float = 0.5,
    device: str = 'cpu',
    verbose: bool = True
) -> SimpleGNN:
    """
    训练GNN模型 (Updated with Correlation Prior)
    """
    total_steps, n_stocks = f_series.shape

    # Calculate global correlation matrix as prior
    # Ideally this should be rolling, but for now global is fine as a prior
    corr_matrix_np = np.corrcoef(f_series.T)
    corr_matrix = torch.FloatTensor(corr_matrix_np).to(device)

    # Prepare Data
    n_samples = total_steps - window_size + 1
    X = []
    stride = 5
    for i in range(0, n_samples, stride):
        window = f_series[i : i + window_size]
        X.append(window)
    X = np.array(X)
    dataset = torch.FloatTensor(X).to(device)

    # Initialize Model
    model = SimpleGNN(n_stocks=n_stocks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    n_batches = len(dataset) // batch_size

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        indices = torch.randperm(len(dataset))

        for i in range(n_batches):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size]
            batch_data = dataset[batch_idx]

            optimizer.zero_grad()
            A_pred = model(batch_data)

            # Use Hybrid Loss
            loss_dict = hybrid_loss(
                A_pred, batch_data, corr_matrix,
                dt=dt,
                alpha_corr=alpha_corr
            )

            loss = loss_dict['total']
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    return model, losses

def predict_graph(model, f_series, window_size=50, device='cpu'):
    model.eval()
    indices = np.linspace(0, len(f_series) - window_size, 10, dtype=int)
    samples = []
    for idx in indices:
        samples.append(f_series[idx : idx + window_size])
    inputs = torch.FloatTensor(np.array(samples)).to(device)
    with torch.no_grad():
        A_preds = model(inputs)
    return A_preds.mean(dim=0).cpu().numpy()
