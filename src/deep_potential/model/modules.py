
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalEncoder(nn.Module):
    """
    1D-Conv Encoder to extract dynamic features from time-series.
    Input: (B, N, T, D) -> Permutes to (B*N, D, T) -> Conv -> (B*N, H)
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Pool over time -> (B*N, H, 1)
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N, T, D)
        B, N, T, D = x.shape

        # Flatten Batch and N for shared processing
        x_flat = x.view(B * N, T, D).permute(0, 2, 1) # (B*N, D, T)

        h = self.conv_net(x_flat) # (B*N, H, 1)

        # Reshape back to (B, N, H)
        return h.view(B, N, self.hidden_dim)

class FeatureEncoder(nn.Module):
    """
    MLP Encoder for static or low-freq features.
    Input: (B, N, D_stat) -> (B, N, H)
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GraphAttention(nn.Module):
    """
    Multi-Head Self-Attention to learn adjacency A.
    Returns sparse or soft adjacency matrix.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_k = hidden_dim // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, H)
        B, N, H = h.shape

        Q = self.W_q(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2) # (B, Heads, N, d_k)
        K = self.W_k(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Attention Scores: (Q @ K.T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale # (B, Heads, N, N)

        # Softmax over neighbors
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Average over heads to get single A
        A = attn.mean(dim=1) # (B, N, N)

        # Symmetrize (Optional, but Laplacian usually requires separation of undirected vs directed)
        # For Force dynamic F = -Lf, L usually symmetric for conservation?
        # Not rigorously required, but usually L = D - A symmetric allows spectral theory.
        # Let's force symmetry for stability.
        A = (A + A.transpose(1, 2)) / 2.0

        return A
