# Cluster-Based Differentiable Pairs Trading System - Design Doc

## 1. System Overview
A fully differentiable, end-to-end trainable pairs trading system that learns to cluster stocks and generate trading signals. The system optimizes for risk-adjusted returns (Sharpe) while managing turnover and leverage through soft constraints.

## 2. Core Modules

### 2.1 Feature Extraction (`features.py`)
-   **Input**: Raw Price Tensor $(B, N, T)$.
-   **Process**:
    1.  **Log Prices**: $p = \log(P)$.
    2.  **Rolling Z-Score**: $Z_t = (p_t - \mu_{t-W:t}) / \sigma_{t-W:t}$.
    3.  **Spherical Projection**: Normalizes feature vectors to the unit hypersphere to focus on shape/correlation rather than magnitude.
-   **Output**:
    -   `features`: Normalized vectors for clustering.
    -   `z_scores`: Raw Z-scores for signal generation (preserving magnitude).

### 2.2 Differentiable Clustering (`clustering.py`)
-   **Algorithm**: **Gated Soft K-Means (Outlier Rejection)**.
-   **Mechanism**:
    -   **Similarity**: Cosine Similarity $S_{ik}$ between feature $x_i$ and centroid $\mu_k$.
    -   **Outlier Gate**: Detects if a point is far from *all* centroids.
        -   $S_{max, i} = \max_k(S_{ik})$.
        -   $\text{Gate}_i = \text{Sigmoid}(\gamma \cdot (S_{max, i} - S_{threshold}))$.
    -   **Soft Assignments**: Standard Softmax assignments $q_{ik}$.
-   **Output**: Returns both assignments and the gate.

### 2.3 Signal Engine (`signal.py`) [UPDATED]
-   **Clean Consensus**: Uses **Gated Assignments** to compute centroid price.
    -   $q'_{ik} = q_{ik} \cdot \text{Gate}_i$.
    -   Outliers (Gate $\approx 0$) do not pollute the cluster consensus.
-   **Signal Generation**:
    -   Residual: $r_t = Z_{i,t} - \hat{Z}_{i,t}$.
    -   **Activation**: Differentiable Band-Pass Filter ($Entry < |r| < Stop$).
-   **Forced Exit**:
    -   Final Signal = Raw Signal $\times$ **Outlier Gate**.
    -   If a stock drifts too far from its cluster (becoming an outlier), its position is smoothly forced to 0.

### 2.4 Portfolio Construction (`system.py`)
-   **Method**: **Independent Position Sizing**.
    -   $w_i = 0.1 \times \text{Signal}_i$.
    -   No forced normalization to avoid dilution.
-   **Leverage Control**: Soft Leverage Constraint in Loss.

## 3. Loss Function (`loss.py`)
$$ L = - \text{Sharpe}(R_p) + \lambda_1 \cdot \text{Turnover} + \lambda_2 \cdot \text{LeveragePenalty} $$

## 4. Differentiable Time-To-Live (TTL) Layer (Future/Proposed)

A recurrent gate to enforce "Time-Based Exit" (Max Hold Duration) while maintaining differentiability.

### Logic
The goal is to force a position to close if it has been held for longer than $T_{max}$ steps, regardless of profitability.

### Recursive Formulation
We define a state variable $Age_t$ for each asset.
1.  **Continuity Check**:
    -   Has position? $Active_t = \tanh(|w_t|) > \epsilon$.
    -   Same direction? $Dir_t = \text{ReLU}(\text{sgn}(w_t) \cdot \text{sgn}(w_{t-1}))$.
    -   $Keep_t = Active_t \cdot Dir_t$.
2.  **Age Update**:
    -   $Age_t = (Age_{t-1} \cdot Keep_t) + 1$.
3.  **Gate generation (Soft)**:
    -   $Mask_t = \sigma(\beta \cdot (T_{max} - Age_t))$.
    -   As Age approaches $T_{max}$, Mask drops from 1 to 0.
4.  **Application**:
    -   $w_{final, t} = w_{raw, t} \cdot Mask_t$.

### JIT Implementation
Since $Age_t$ depends on $Age_{t-1}$, this cannot be parallelized efficiently in pure Python. We use `torch.jit.script`.

```python
import torch

@torch.jit.script
def age_gating(weights: torch.Tensor, max_hold: int = 100, beta: float = 5.0):
    # weights: (Batch, N, T)
    B, N, T = weights.shape
    age = torch.zeros((B, N), device=weights.device)
    mask_list = []

    # Store previous sign
    prev_sign = torch.zeros((B, N), device=weights.device)

    for t in range(T):
        w_t = weights[:, :, t]
        curr_sign = torch.sign(w_t)

        # Check if active and same direction
        is_active = (w_t.abs() > 1e-3).float()
        is_same_dir = (curr_sign * prev_sign > 0).float()

        # If position kept, age increases. Else resets to 1 (new pos) or 0 (no pos)
        # Simplified logic: If active and same dir, Age++, else Age=1

        keep = is_active * is_same_dir
        age = age * keep + 1.0

        # Compute Gate
        # If age > max_hold, gate -> 0
        gate = torch.sigmoid(beta * (max_hold - age))
        mask_list.append(gate)

        prev_sign = curr_sign

    mask_seq = torch.stack(mask_list, dim=-1) # (B, N, T)
    return mask_seq
```
*Note: In the differentiable training version, strict `torch.sign` would kill gradients. We would use `torch.tanh` and soft comparisons.*
