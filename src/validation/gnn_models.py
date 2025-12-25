"""
GNN Models for Graph Structure Learning (Optimized)

Implements GNN architectures to learn adjacency matrices from time series data.
Specific optimizations for Pairwise Similarity (Dot Product).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class SimpleGNN(nn.Module):
    """
    Simple GNN for pairwise relationship learning.

    Structure:
    1. Temporal Encoder: Conv1d to extract features from time series
    2. Node Embedding: Linear layer to project features
    3. Edge Predictor: FORCED DOT PRODUCT (h_i @ h_j.T)
    """

    def __init__(
        self,
        input_dim: int = 1,      # Number of features per stock (usually 1: price/state)
        hidden_dim: int = 32,    # Hidden dimension
        n_stocks: int = 50,      # Number of stocks
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_stocks = n_stocks

        # 1. Temporal Encoder
        # Input: (Batch, N, T, input_dim) -> (Batch*N, input_dim, T)
        # We use a simple 1D convolution to capture local temporal patterns
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Pool over time: (Batch*N, hidden_dim, 1)
        )

        # 2. Node Embedding - project to latent space for dot product
        self.node_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            # nn.ReLU(), # Remove ReLU to allow negative embeddings (anti-correlation) if needed
            # But for Laplacian A_ij > 0, we might want similarity > 0
            # Let's keep it simple: Linear -> Norm
        )

        # Scale factor for dot product (like Attention)
        self.scale = nn.Parameter(torch.tensor(1.0 / np.sqrt(hidden_dim)))

        # Bias for sparsity control
        self.edge_bias = nn.Parameter(torch.tensor(-1.0)) # Start with negative bias -> sparse graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict adjacency matrix.

        Args:
            x: Input tensor of shape (Batch, T, N) or (Batch, N, T) or (Batch, T, N, F)
               Our convention: (Batch, T, N) where F=1

        Returns:
            A: Predicted adjacency matrix (Batch, N, N)
        """
        # Standarize input to (Batch, N, T, F)
        if x.dim() == 3:
            # Assume (Batch, T, N) -> permute to (Batch, N, T)
            # Add feature dimension: (Batch, N, T, 1)
            x = x.permute(0, 2, 1).unsqueeze(-1)

        batch_size, n_stocks, n_timesteps, n_features = x.shape

        # Reshape for temporal encoder: (Batch*N, F, T)
        x_flat = x.permute(0, 1, 3, 2).reshape(batch_size * n_stocks, n_features, n_timesteps)

        # 1. Encode temporal features
        h = self.temporal_encoder(x_flat)  # (Batch*N, hidden_dim, 1)
        h = h.squeeze(-1)                  # (Batch*N, hidden_dim)

        # 2. Project node embeddings
        h = self.node_projector(h)         # (Batch*N, hidden_dim)

        # Reshape back to (Batch, N, hidden_dim)
        h = h.reshape(batch_size, n_stocks, self.hidden_dim)

        # 3. Predict edges: FORCED DOT PRODUCT
        # Score(i, j) = scale * (h_i . h_j) + bias
        # This forces the model to encode "similarity" in the embedding space

        # h: (Batch, N, H)
        # scores: (Batch, N, N)
        scores = torch.matmul(h, h.transpose(1, 2)) * self.scale

        # Add bias
        scores = scores + self.edge_bias

        # Sigmoid to get probability/weight between 0 and 1
        A = torch.sigmoid(scores)

        # Remove self-loops (diagonal = 0)
        mask = torch.eye(n_stocks, device=x.device).unsqueeze(0).bool()
        A = A.masked_fill(mask, 0)

        return A

def compute_laplacian_torch(A: torch.Tensor) -> torch.Tensor:
    """
    Compute Laplacian matrix L = D - A in PyTorch.

    Args:
        A: Adjacency matrix (Batch, N, N)

    Returns:
        L: Laplacian matrix (Batch, N, N)
    """
    # Degree: sum over columns (Batch, N)
    degree = A.sum(dim=2)

    # Create degree matrix: (Batch, N, N)
    D = torch.diag_embed(degree)

    L = D - A

    return L
