
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .graph_learner import GraphLearner
from .state_extractor import StateExtractor

class DeepPotentialModel(nn.Module):
    """
    Dual-Tower Coupling Model.

    Branch I:  X -> L (Laplacian)
    Branch II: X -> f (State)
    Coupling:  g = 2 L f (Force)
    """
    def __init__(
        self,
        input_dim: int = 1, # Usually price/return
        n_static: int = 0,  # Fundamental features
        hidden_dim: int = 64,
        n_heads: int = 4,
        top_k: int = 20,
        alpha_prior: float = 0.0,
        state_method: str = 'residual'
    ):
        super().__init__()

        # Branch I: Structure
        self.learner = GraphLearner(
            input_dim_dyn=input_dim,
            input_dim_stat=n_static,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            top_k=top_k,
            alpha_prior=alpha_prior
        )

        # Branch II: State
        self.extractor = StateExtractor(
            input_dim=input_dim,
            method=state_method,
            hidden_dim=hidden_dim
        )

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_stat: torch.Tensor = None,
        mask: torch.Tensor = None,
        A_prior: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns Dictionary of physical quantities:
        - 'L': Laplacian
        - 'f': State
        - 'force': Restoring Force
        - 'stiffness': Confidence (2*Degree or similar)
        """
        # 1. Branch I: Get Structure
        # x_dyn: (B, N, T, D)
        # We need to reshape slightly if using (B, N, T, D) convention, Model expects it.
        # But GraphLearner was designed for that inputs.

        L, A = self.learner(x_dyn, x_stat, A_prior)

        # 2. Branch II: Get State
        # We extract state from the *latest* timestep or learning valid sequence?
        # Usually dynamics are instantaneous: F_t = -L_t * f_t
        # So f_t is the state at time T.
        # x_dyn_last = x_dyn[:, :, -1, :] # (B, N, D)

        # Actually StateExtractor might operate on window (e.g. removing trend).
        # But for 'residual' it's usually just cross-sectional at T.
        x_last = x_dyn[:, :, -1, :]
        f = self.extractor(x_last, mask)

        # 3. Coupling (Physics)
        # F = 2 * L * f
        # L: (B, N, N)
        # f: (B, N, D)
        # Need to handle D (features). Force per feature.

        # L @ f -> (B, N, N) @ (B, N, D) -> (B, N, D)
        force = 2.0 * torch.matmul(L, f)

        # Stiffness (Confidence)
        # k = diag(2L) = 2 * Degree (if L = D - A)
        # Degree D_ii = L_ii + A_ii? No L_ii = D_ii - A_ii (=-A_ii).
        # Actually L = D - A. Diagonal of L is D (since A diag is 0).
        # So diag(L) is Degree.
        # k = 2 * diag(L)

        deg = torch.diagonal(L, dim1=1, dim2=2) # (B, N)
        stiffness = 2.0 * deg.unsqueeze(-1) # (B, N, 1) to broadcast to D features

        return {
            'L': L,
            'A': A,
            'f': f,
            'force': force,
            'stiffness': stiffness
        }
