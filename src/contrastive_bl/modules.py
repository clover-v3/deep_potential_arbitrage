import torch
import torch.nn as nn
import numpy as np

class PLEPhase1(nn.Module):
    """
    Piece-wise Linear Encoding (Phase 1): Binning
    Transforms scalar feature x into T-dimensional vector.
    """
    def __init__(self, n_bins: int = 64):
        super().__init__()
        self.n_bins = n_bins
        # Bin boundaries will be registered as buffer after data analysis
        # For now, we assume standard normal inputs (approx -3 to 3) or use a fixed range.
        # Paper says: "We normalize all features using a standard scaler."
        # Then defines bins b_0 ... b_T.
        # We allow learnable boundaries or fixed.
        # Fixed quantiles of Normal distribution is a good default for StandardScaled data.

        # Initialize boundaries (k+1 boundaries for k bins)
        # We use linspace for simplicity on Standard Scaled data (-4 to 4)
        boundaries = torch.linspace(-4, 4, n_bins + 1)
        self.register_buffer('boundaries', boundaries)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N_features)
        # We apply PLE to each feature independently.
        # Output: (Batch, N_features, n_bins)

        # x: (B, F) -> (B, F, 1)
        val = x.unsqueeze(-1)

        # boundaries: (T+1) -> (1, 1, T+1)
        b = self.boundaries.view(1, 1, -1)

        # b_0 ... b_T
        # We need pairs (b_{t-1}, b_t)
        b_lower = b[..., :-1] # (..., T)
        b_upper = b[..., 1:]  # (..., T)

        # Case 1: x < b_{t-1} -> 0
        # Case 2: x >= b_t -> 1
        # Case 3: b_{t-1} <= x < b_t -> (x - b_{t-1}) / (b_t - b_{t-1})

        # Vectorized logic:
        # Relu((x - lower) / (upper - lower)) clipped at 1?
        # Let's verify formula:
        # if x < lower: (neg)/diff -> neg. Relu -> 0. Correct.
        # if x > upper: (big)/diff -> >1. Clip(1) -> 1. Correct.
        # if in between: fraction. Correct.

        width = b_upper - b_lower
        encoded = torch.clamp( (val - b_lower) / width, min=0.0, max=1.0)

        # Paper defines first bin (-inf, b1) and last (bT-1, inf).
        # Our linspace -4 to 4 roughly covers it.
        # For strict paper adherence:
        # Bin 1: ( -inf, b_1 ). if x < b_1, value is 1?
        # Paper Eq(1):
        # if x < b_{t-1}: 0 (for t>1). Bin 1 lower bound is -inf.
        # Bin 1 (t=1): b_0 = -inf. Formula: x - (-inf) ??
        # The paper says: "The first bin B1=(-inf, b2)..."
        # Actually Eq(1) says: if t>1 and x < b_{t-1}, e_t=0.
        # This implies e_1 is special?
        # Usually PLE encodes position in CDF.
        # Let's stick to the clamp method which works for bounded bins.
        # For the tails, if x < -4, all 0? No, should be 0.
        # If x > 4, all 1.

        return encoded

class FeatureEncoder(nn.Module):
    """
    Full Feature Encoder: PLE -> Linear -> Embedding
    """
    def __init__(self, n_features: int, n_bins: int = 64, d_model: int = 128):
        super().__init__()
        self.n_features = n_features
        self.ple = PLEPhase1(n_bins)

        # Each feature maps from T bins to d_model
        # We use a shared linear layer or separate?
        # Paper: "passed through a trainable linear layer" (singular?)
        # "enc_k(xk)... for each feature k". Implies separate weights per feature?
        # "Finally, this T-dimensional vector is passed through a trainable linear layer"
        # Usually in Tabular, it's separate embeddings.

        self.feature_embeddings = nn.ModuleList([
            nn.Linear(n_bins, d_model) for _ in range(n_features)
        ])

        # Output will be (Batch, N_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N_features)

        # 1. Binning
        # encoded: (Batch, N_features, n_bins)
        p = self.ple(x)

        # 2. Linear Projection per feature
        embeddings = []
        for k in range(self.n_features):
            # p[:, k, :] -> (Batch, n_bins)
            emb = self.feature_embeddings[k](p[:, k, :]) # (Batch, d_model)
            embeddings.append(emb.unsqueeze(1))

        # Stack: (Batch, N_features, d_model)
        return torch.cat(embeddings, dim=1)

class ORCABackbone(nn.Module):
    """
    Transformer Backbone
    """
    def __init__(self, n_features: int, n_bins: int = 64, d_model: int = 128, n_layers: int = 2, n_heads: int = 8):
        super().__init__()

        self.encoder = FeatureEncoder(n_features, n_bins, d_model)

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N_features)

        # 1. Embed features
        x_emb = self.encoder(x) # (B, F, D)

        # 2. Prepend CLS
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1) # (B, F+1, D)

        # 3. Transformer interactions
        out = self.transformer(x_seq)

        # 4. Return CLS output
        return out[:, 0, :] # (B, D)
