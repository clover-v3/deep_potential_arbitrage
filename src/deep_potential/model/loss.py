
import torch
import torch.nn as nn
from typing import Dict

class DeepPotentialLoss(nn.Module):
    """
    Robust Loss Function for Deep Potential Model.

    L_total = L_dyn + lambda_1 * L_sparse + lambda_2 * L_degree

    1. L_dyn (Dynamics Consistency):
       Checks if Next Return can be explained by Current Force.
       Use Robust Huber Loss to ignore outliers.

       Target: r_{t+1}
       Pred:   alpha * Force_t + beta * v_t (Taylor Expansion?)

       Ideally, we train Branch I (L) such that Force = 2Lf is predictive.
       Since we don't know the scalar scaling factor 'eta', we can learn it or use Cosine Similarity.

       Option A: Cosine Sim(Force, -Return) -> Maximize alignment.
       Option B: Regression MSE(r_{t+1}, -eta * Force).

    2. L_reg (Regularization):
       - Sparsity (L1 on A).
       - Connectivity (Degree > 0).
    """
    def __init__(self, delta: float = 1.0, alpha_reg: float = 0.01):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.alpha_reg = alpha_reg
        self.eta = nn.Parameter(torch.tensor(0.01)) # Learnable step size

    def forward(self, force: torch.Tensor, next_return: torch.Tensor, A: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            force: (B, N, D) calculated from 2 L f
            next_return: (B, N, D)
            A: Adjacency (B, N, N)
        """
        # Dynamics Loss
        # Hypothesis: r_{t+1} ~ - eta * Force
        # Note: Force = 2Lf points towards mean. So Price moves in direction of -Force?
        # Wait.
        # Potential E = f^T L f.
        # Force g = dE/df = 2Lf.
        # Gradient Descent: df/dt = - g = -2Lf.
        # So change in f (return) should be proportional to -Force.

        pred_change = -self.eta * force

        # Robust Regression
        loss_dyn = self.huber(next_return, pred_change)

        # Regularization
        # 1. Sparsity on A
        loss_sparse = torch.mean(torch.abs(A))

        # 2. No isolated nodes (Degree penalty) -> Optional
        # degree = A.sum(dim=-1)
        # loss_iso = torch.mean(torch.relu(1.0 - degree)) # Penalty if degree < 1

        total = loss_dyn + self.alpha_reg * loss_sparse

        return {
            'total': total,
            'dyn': loss_dyn,
            'reg': loss_sparse
        }
