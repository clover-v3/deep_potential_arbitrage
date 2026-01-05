import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ClusterLayer(nn.Module):
    """
    Differentiable Clustering Layer (Soft K-Means variant).
    Maintains centroids on the unit sphere.
    """
    def __init__(self, n_clusters: int, feature_dim: int, temp: float = 1.0, similarity_threshold: float = 0.5):
        super().__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.temp = temp
        self.similarity_threshold = similarity_threshold

        # Learnable Centroids: (K, D)
        # We initialize them randomly, they should be normalized during forward pass?
        # Or we project them.
        self.centroids = nn.Parameter(torch.randn(n_clusters, feature_dim))

    def _get_normalized_centroids(self):
        # Always project centroids to unit sphere to match feature space
        return F.normalize(self.centroids, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Soft Assignments (Attention Weights) AND Outlier Gate.
        Args:
            x: Features (..., D)
        Returns:
            assignments: (..., K) - Softmax weights
            outlier_gate: (..., 1) - 1.0 if core member, 0.0 if outlier
        """
        # x: (Batch, N, T, D) or similar
        # W: (K, D)
        W = self._get_normalized_centroids()

        # Similarity: Cosine Similarity (since both x and W are normalized)
        # Dot product
        # sim = x @ W.T
        sim = torch.einsum('...d, kd -> ...k', x, W)

        # Outlier Detection
        # Find similarity to the CLOSEST centroid (max sim)
        max_sim, _ = sim.max(dim=-1, keepdim=True)

        # Soft Gate: Sigmoid((Sim - Threshold) * Scaling)
        # Scaling=20.0 gives a reasonable transition width around threshold
        outlier_gate = torch.sigmoid(20.0 * (max_sim - self.similarity_threshold))

        # Softmax
        # assignment_{ik} = exp(beta * sim) / sum indices
        assignments = F.softmax(sim / self.temp, dim=-1)

        return assignments, outlier_gate

    def hard_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns hard cluster indices.
        Args:
            x: Features (..., D)
        Returns:
            indices: (..., ) LongTensor
        """
        W = self._get_normalized_centroids()
        sim = torch.einsum('...d, kd -> ...k', x, W)
        return torch.argmax(sim, dim=-1)

    def init_centroids_kmeanspp(self, x: torch.Tensor):
        """
        Initialize centroids using K-Means++ logic on a batch of data x.
        x: (M, D) flattened batch of features
        """
        with torch.no_grad():
            M, D = x.shape
            # 1. Randomly pick first centroid
            idx = torch.randint(0, M, (1,))
            self.centroids.data[0] = x[idx]

            # 2. Pick remaining
            # Only implementing simple random choice for now for speed/simplicity
            # checks if K < M
            if M > self.n_clusters:
                indices = torch.randperm(M)[:self.n_clusters]
                self.centroids.data = x[indices].clone()
            else:
                # Should not happen in real training
                pass

            # Normalize
            self.centroids.data = F.normalize(self.centroids.data, p=2, dim=-1)
