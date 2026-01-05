import torch
import torch.nn as nn

class SignalEngine(nn.Module):
    """
    Computes trading signals based on Z-Score deviation from Cluster Consensus.
    """
    def __init__(self, entry_threshold: float = 2.0, stop_threshold: float = 4.0, scaling_factor: float = 5.0):
        super().__init__()
        self.entry_threshold = entry_threshold
        self.stop_threshold = stop_threshold
        self.scaling_factor = scaling_factor

    def forward(self, z_scores: torch.Tensor, assignments: torch.Tensor, outlier_gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_scores: (Batch, N, T, D) - Raw Z-scores
            assignments: (Batch, N, T, K) - Soft assignment weights
            outlier_gate: (Batch, N, T, 1) - 1.0 if core, 0.0 if outlier

        Returns:
            positions: (Batch, N, T)
        """
        # 1. Clean Consensus Calculation (Filter out outliers)
        # Apply gate to assignments.
        # If a point is an outlier, its weight in ALL clusters becomes 0.
        # gated_assignments: (B, N, T, K)
        gated_assignments = assignments * outlier_gate

        # Numerator: sum_i (g_w_ik * z_i) -> (B, T, K, D)
        w_perm = gated_assignments.permute(0, 2, 3, 1) # (B, T, K, N)
        z_perm = z_scores.permute(0, 2, 1, 3)     # (B, T, N, D)

        cluster_sum = torch.einsum('btkn, btnd -> btkd', w_perm, z_perm)
        w_sum = w_perm.sum(dim=-1, keepdim=True) + 1e-6
        cluster_consensus = cluster_sum / w_sum # (B, T, K, D)

        # 2. Expected Z for each stock i
        # We use original assignments here? Or gated?
        # If I am an outlier, my expected Z should probably be "my cluster's Z"
        # even if I didn't contribute to it. So use original assignments to project.
        expected_z = torch.einsum('bntk, btkd -> bntd', assignments, cluster_consensus)

        # 3. Residual
        residuals = z_scores - expected_z # (B, N, T, D)

        # 4. Signal = Residual at CURRENT time step
        signal_val = residuals[..., -1] # (Batch, N, T)

        # 5. Advanced Activation with Soft Stop-Loss
        # Logic: Signal is active if Entry < |Residual| < Stop
        # Implementation: Difference of Sigmoids

        # Magnitude checks
        abs_signal = signal_val.abs()

        # Soft Step Functions
        # is_above_entry = 1 if |Z| > Entry
        is_above_entry = torch.sigmoid(self.scaling_factor * (abs_signal - self.entry_threshold))

        # is_below_stop = 1 if |Z| < Stop (equivalent to 1 - is_above_stop)
        is_below_stop = 1.0 - torch.sigmoid(self.scaling_factor * (abs_signal - self.stop_threshold))

        # Band Activation: Both must be true
        activation_magnitude = is_above_entry * is_below_stop

        # Direction: Mean Reversion
        # If Z > 0 (Overpriced), Short (-1)
        # If Z < 0 (Underpriced), Long (+1)
        # soft_sign = -tanh(k * Z)
        direction = -torch.tanh(self.scaling_factor * signal_val)

        pos = direction * activation_magnitude

        # 6. Forced Exit for Outliers
        # If I am an outlier (gate -> 0), my position must be closed.
        # Note: z_scores has T window, signal_val is at current time.
        # outlier_gate shape is (B, N, T, 1). We need gate at current T (last element).
        # Actually z_scores is (B, N, T_valid, W). Wait.
        # Let's check input shapes from system.py call.

        # Re-verify shapes:
        # features: (B, N, T, W) -> No
        # FeatureExtractor returns normalized_features (B, N, T', W) where T' is valid steps.
        # ClusterLayer forward(features) -> returns assignments (B, N, T', W, K)?
        # No, ClusterLayer forward expects (..., D).
        # So assignments is (B, N, T', K)? No.

        # features from FeatureExtractor:
        #   log_prices.unfold -> (B, N, T', W)
        #   normalized_features -> (B, N, T', W)
        # ClusterLayer(features):
        #   Input: (B, N, T', W)
        #   Centroids: (K, W)
        #   Einsum: ...d, kd -> ...k
        #   Output: (B, N, T', K).
        # Ah, so assignments is (B, N, T', K).

        # outlier_gate is also (B, N, T', 1).

        # So signal_val (residual) is derived from the FULL vector similarity.
        # We simply use outlier_gate for the current step?
        # NO. We are trading based on the similarity of the *entire window* (current state).
        # So outlier_gate corresponds to the current state of the stock being an outlier or not.

        # pos shape: (B, N, T')  (One signal per valid time step)
        # outlier_gate shape: (B, N, T', 1)

        pos = pos * outlier_gate.squeeze(-1)

        return pos
