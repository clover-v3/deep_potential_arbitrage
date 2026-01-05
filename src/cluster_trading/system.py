import torch
import torch.nn as nn
from typing import Dict, Optional

from .features import FeatureExtractor
from .clustering import ClusterLayer
from .signal import SignalEngine

class CoopTradingSystem(nn.Module):
    """
    End-to-End Differentiable Pair Trading System.
    Structure: Features -> Cluster -> Signal -> Position
    """
    def __init__(self,
                 n_clusters: int = 10,
                 feature_window: int = 60,
                 entry_threshold: float = 2.0,
                 stop_threshold: float = 4.0,
                 similarity_threshold: float = 0.5,
                 temp: float = 1.0,
                 scaling_factor: float = 5.0):
        super().__init__()

        self.feature_extractor = FeatureExtractor(window_size=feature_window)
        self.cluster_layer = ClusterLayer(n_clusters=n_clusters,
                                          feature_dim=feature_window,
                                          temp=temp,
                                          similarity_threshold=similarity_threshold)
        self.signal_engine = SignalEngine(entry_threshold=entry_threshold,
                                          stop_threshold=stop_threshold,
                                          scaling_factor=scaling_factor)

    def forward(self, prices: torch.Tensor, hard: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            prices: (Batch, N, T)
            hard: If True, uses hard assignments (Inference/Backtest).
        """
        # 1. Extract Features
        # features: Normalized (Sphere), z_scores: Raw Magnitude
        features, z_scores = self.feature_extractor(prices)

        # 2. Clustering assignments (Using Normalized Features)
        if hard:
            indices = self.cluster_layer.hard_assignment(features) # (B, N, T)
            assignments = torch.zeros(*indices.shape, self.cluster_layer.n_clusters, device=prices.device)
            assignments.scatter_(-1, indices.unsqueeze(-1), 1.0)
            # In hard mode, we should perhaps also compute the gate?
            # hard_assignment only returns indices.
            # For simplicity in 'hard' mode (inference debug), let's just forward normally to get gate
            # Or assume gate=1. But let's verify.
            # Ideally hard mode should also check similarity threshold.
            # Let's just run soft forward for gate, or re-implement hard check.
            # Re-running soft forward partially:
            _, gate = self.cluster_layer(features)
        else:
            assignments, gate = self.cluster_layer(features) # (B, N, T, K)

        # 3. Signal (Using Raw Z-Scores and Assignments)
        # Note: We no longer pass centroids to signal engine, it computes consensus dynamically
        positions = self.signal_engine(z_scores, assignments, gate)

        return {
            'features': features,
            'z_scores': z_scores,
            'assignments': assignments,
            'gate': gate,
            'positions': positions
        }

    def initialize_clusters(self, prices: torch.Tensor):
        print("Initializing centroids with K-Means++...")
        with torch.no_grad():
            features, _ = self.feature_extractor(prices)
            flat_features = features.reshape(-1, features.shape[-1])
            if flat_features.shape[0] > 10000:
                indices = torch.randperm(flat_features.shape[0])[:10000]
                sample = flat_features[indices]
            else:
                sample = flat_features
            self.cluster_layer.init_centroids_kmeanspp(sample)

    def get_portfolio_weights(self, positions: torch.Tensor, method: str = 'long_short_neutral') -> torch.Tensor:
        """
        Convert raw signals [-1, 1] into Portfolio Weights.
        Args:
            positions: (Batch, N, T)
            method:
                'long_short_neutral': Scale by Gross Exposure (Dollar Neutral).
                'equal_weight': Fixed weight per active signal.
        """
        # 1. Zero out weak signals
        active_pos = positions.clone()
        # active_pos[active_pos.abs() < 0.1] = 0.0 # already handled by tanh soft thresholding usually

        if method == 'long_short_neutral':
            # Gross Exposure = Sum(|pos|)
            gross = active_pos.abs().sum(dim=1, keepdim=True) # (Batch, 1, T)
            weights = active_pos / (gross + 1e-6) # Normalized to Sum(|w|) = 1

        elif method == 'equal_weight_cluster':
            # Complex: Equal weight per cluster?
            # Simplified: Just normalize by count of active signals
            count = (active_pos.abs() > 1e-3).float().sum(dim=1, keepdim=True)
            weights = active_pos / (count + 1e-6)

        elif method == 'independent':
            # Raw positions are directly the weights (e.g. 50% leverage per stock)
            # Usually we scale it down by some default factor to prevent initial explosion
            # e.g. weight = 0.1 * position
            weights = active_pos * 0.1

        else:
            weights = active_pos

        return weights
