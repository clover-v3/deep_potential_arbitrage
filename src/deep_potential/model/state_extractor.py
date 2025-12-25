
import torch
import torch.nn as nn
from typing import Optional

class StateExtractor(nn.Module):
    """
    Branch II: Latent State Extractor.
    Extracts the 'Pure' state f_t that should revert.

    Modes:
    1. 'residual': f_t = raw_price - Market/Sector Factor (Pre-calculated or learned linear).
    2. 'autoencoder': f_t = raw - reconstructed (Idiosyncratic).
    """
    def __init__(self, input_dim: int, method: str = 'residual', hidden_dim: int = 32):
        super().__init__()
        self.method = method
        self.input_dim = input_dim

        if method == 'residual':
            # Linear projection to 'Factor Space' to remove beta
            # f = x - Beta * F_mkt
            # Or simplified: f = x - Linear(x) ?
            # No, standard residual is: x - Beta * Market.
            # If input x includes Market Index, we can learn Beta.

            # Simple Learnable Demeaning:
            # f = x - mean(x) (Cross-sectional)
            # This is robust.
            pass

        elif method == 'autoencoder':
            # AE Reconstructs 'Systematic' component
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4) # Bottleneck (Low rank factors)
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input: x (B, N, D) - usually D=1 (Cumulative Return / Price)
        Output: f (B, N, D)
        """
        if self.method == 'residual':
            # 1. Cross-Sectional Neutralization (remove Market Mode)
            # Mean over N (masked)
            if mask is not None:
                # Weighted mean
                # mask: (B, N, 1) or (B, N)
                if mask.dim() == 2: mask = mask.unsqueeze(-1)

                sum_x = (x * mask).sum(dim=1, keepdim=True)
                count = mask.sum(dim=1, keepdim=True) + 1e-8
                mean_x = sum_x / count

                # Global Mean removal
                f = x - mean_x

                # Apply mask again to zero out invalid
                f = f * mask
            else:
                mean_x = x.mean(dim=1, keepdim=True)
                f = x - mean_x

            return f

        elif self.method == 'autoencoder':
            # 2. Non-linear Factor Removal
            # Flatten to batch independent
            # Actually, factors are usually Cross-Sectional (PCA).
            # So applying Linear on (N, D) mixes features, not stocks?
            # PCA on returns: X ~ F * B.
            # We want Residual E = X - FB.

            # Implementation:
            # Simple AE on per-stock basis? No, that learns stock features.
            # We want Global Factor Model.

            # Allow fallback to residual for now as per plan
            return x - x.mean(dim=1, keepdim=True)

        return x
