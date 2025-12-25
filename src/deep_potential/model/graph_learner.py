
import torch
import torch.nn as nn
from typing import Tuple
from .modules import TemporalEncoder, FeatureEncoder, GraphAttention

class GraphLearner(nn.Module):
    """
    Branch I: Hybrid Graph Structure Learner.
    Learns Adjacency A and computes Laplacian L.
    """
    def __init__(
        self,
        input_dim_dyn: int,
        input_dim_stat: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        top_k: int = 10,
        alpha_prior: float = 0.0
    ):
        """
        Args:
            top_k: Number of neighbors to keep (sparsity).
            alpha_prior: Weight for static prior graph (0 = purely learned).
        """
        super().__init__()

        # Encoders
        self.enc_dyn = TemporalEncoder(input_dim_dyn, hidden_dim)
        if input_dim_stat > 0:
            self.enc_stat = FeatureEncoder(input_dim_stat, hidden_dim)
            self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.enc_stat = None

        # Graph Learning
        self.attention = GraphAttention(hidden_dim, n_heads)

        self.top_k = top_k
        self.alpha = alpha_prior

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor = None, A_prior: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            L: Laplacian (B, N, N)
            A: Adjacency (B, N, N)
        """
        # 1. Encode Features
        h = self.enc_dyn(x_dyn) # (B, N, H)

        if self.enc_stat and x_stat is not None:
            h_s = self.enc_stat(x_stat)
            h = torch.cat([h, h_s], dim=-1)
            h = self.fusion(h)

        # 2. Learn Adjacency
        A_learn = self.attention(h)

        # 3. Sparsify (Top-K)
        B, N, _ = A_learn.shape
        if self.top_k < N:
            # Keep top k values per row
            topk_vals, topk_inds = torch.topk(A_learn, k=self.top_k, dim=-1)

            # Create sparse mask
            mask = torch.zeros_like(A_learn)
            mask.scatter_(-1, topk_inds, 1.0)

            A_sparse = A_learn * mask

            # Re-symmetrize generally beneficial for L, though scatter might break symmetry
            # Apply OR logic: if i->j OR j->i, keep.
            A_sparse = (A_sparse + A_sparse.transpose(1, 2)) / 2.0
            A_learn = A_sparse

        # 4. Fusion with Prior
        if self.alpha > 0 and A_prior is not None:
            # Ensure A_prior shape (B, N, N)
            if A_prior.dim() == 2:
                A_prior = A_prior.unsqueeze(0).expand(B, -1, -1)
            A_final = self.alpha * A_prior + (1 - self.alpha) * A_learn
        else:
            A_final = A_learn

        # 5. Compute Laplacian: L = D - A
        # Degree D_ii = sum_j A_ij
        degree = A_final.sum(dim=-1) # (B, N)
        D = torch.diag_embed(degree) # (B, N, N)

        L = D - A_final

        return L, A_final
