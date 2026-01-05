import torch
import torch.nn as nn

class SphereNorm(nn.Module):
    """
    Projects vectors onto a unit sphere.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Dim) or (Batch, T, Dim)
        Returns:
            Normalized tensor where vectors have norm 1.
        """
        # Norm across the last dimension (features)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + self.eps)

class FeatureExtractor(nn.Module):
    """
    Transforms raw price series into features suitable for clustering.
    Pipeline:
    1. Log Prices
    2. Rolling Standardization (Mean/Std)
    3. Spherical Projection
    """
    def __init__(self, window_size: int = 60, eps: float = 1e-8):
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        self.sphere_norm = SphereNorm(eps)

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prices: (Batch, N_tickers, Time) or (N_tickers, Time)
        Returns:
            features: (Batch, N_tickers, Time, FeatureDim)
            Here FeatureDim is effectively 1 if we just use price,
            but usually we might want a window history as the feature vector.

            Simplified "Hyperplane" Logic:
            We treat the standardized price series ITSELF as the stream of vectors.
            At each time t, the feature vector for stock i is simply its scalar standardized price?
            No, clustering usually needs a vector.

            Re-reading User Request: "normalize tickers ... to points on unit sphere"
            If we treat the TIME SERIES as the vector (e.g. daily returns over last N days), then N-dim vector.

            HOWEVER, the user says "minute level price data... standardized... cluster these points".
            If we cluster instantaneously at time t, the "point" must be a vector of characteristics.

            Interpretation A: Feature vector is the path of last N minutes.
            Interpretation B (Simplest): We cluster stocks based on their correlation.
            The "Point" on the sphere is the normalized return history vector of length N_window.

            Let's implement Interpretation A:
            Feature(i, t) = [NormReturn(i, t), NormReturn(i, t-1), ..., NormReturn(i, t-W+1)]
            Then we normalize this W-dim vector to unit length.
        """
        # Ensure input is (Batch, N, T)
        if prices.dim() == 2:
            prices = prices.unsqueeze(0) # (1, N, T)

        B, N, T = prices.shape

        # 1. Log Returns approximation or Log Prices
        # User said: "normalize (-mean/std)" -> Z-Score of price or return? usually Price for cointegration/pairs.
        # Let's use Log Prices.
        log_prices = torch.log(prices + self.eps)

        # 2. Rolling Window Unfold
        # We want at time t, a window of size W.
        # Output shape: (B, N, T', W)
        # We pad beginning
        # pad_size = self.window_size - 1
        # padded = torch.nn.functional.pad(log_prices, (pad_size, 0), mode='replicate') # simple padding

        # Use unfold to get sliding windows
        # dimension 2 is Time
        # unfold(dimension, size, step)
        # We want to maintain time dimension T.
        # If we unfold, we get (B, N, NumberOfWindows, W)

        # Implementation Trick: standardized locally
        # mu = mean(window), std = std(window)
        # This is expensive to do with unfold if W is large.

        # Optimized: 1D Convolution for Mean/Std?
        # But let's stick to unfold for clarity first as W is small (e.g. 60 mins).

        # To vectorizingly get rolling windows:
        # Input: (B, N, T)
        # Output: (B, N, T, W)

        # Using strides (as_strided) or unfold.
        # Let's use unfold.
        features_unfolded = log_prices.unfold(dimension=-1, size=self.window_size, step=1)
        # shape: (B, N, T - W + 1, W)

        # We need to pad output to length T? Or just return valid?
        # Typically backtest aligns date. Let's return valid.

        # 3. Micro-Batch Normalization (Z-Score within window)
        # x_norm = (x - mean) / std
        m = features_unfolded.mean(dim=-1, keepdim=True)
        s = features_unfolded.std(dim=-1, keepdim=True)
        z_scores = (features_unfolded - m) / (s + self.eps)

        # 4. Sphere Projection
        # The vector v_i is the z-score sequence.
        # Project to unit sphere: v_i / ||v_i||
        normalized_features = self.sphere_norm(z_scores)

        return normalized_features, z_scores # (B, N, T_valid, W)
