# ORCA Implementation Documentation

## Overview
This directory (`src/contrastive_bl/`) contains the implementation of the paper **"Deep Mean-Reversion: A Physics-Informed Contrastive Approach to Pairs Trading"** (Kim et al., 2025). The model, named **ORCA**, integrates Contrastive Learning with Physics-Informed Neural Networks (PINNs) to find tradable, mean-reverting asset clusters.

## Directory Structure

```
src/
├── contrastive_bl/
│   ├── data_loader.py   # ORCADataLoader: Features (Momentum + Fundamentals)
│   ├── modules.py       # Neural Modules: PLE Encoder, Transformer Backbone
│   ├── orca.py          # ORCAModel: Combined Architecture + Unified Loss
│   ├── train.py         # Training Loop with Augmentation
│   ├── backtest.py      # Momentum Spread Strategy Backtest
├── utils/
│   ├── data_utils.py    # Shared data helpers (safe_log, clean_infs, etc.)
```

## Data Specification

The ORCA model uses a strictly defined set of **36 features** for each asset at each time step $t$.

### 1. Momentum Features (24)
**Source**: Calculated in `data_loader.py` from raw CRSP `msf` returns.
*   `mom_1`: Return in month $t-1$.
*   `mom_2` ... `mom_24`: Cumulative return from month $t-i$ to $t-1$.
    *   *Calculation*: Rolling sum of log-returns (re-implemented to match paper exactness, not using GHZ pre-calc).

### 2. Fundamental Features (12)
**Source**: Raw Compustat `fundq` columns. Proxies calculated in `data_loader.py` if specific columns (e.g., `niq`) are missing in raw feed.
*   **Alignment**: Uses `valid_from` logic from `GHZFactorBuilder` (RDQ or Date+45d) to prevent look-ahead bias.
*   **Balance Sheet**:
    *   `atq`: Total Assets
    *   `ltq`: Total Liabilities
    *   `dlcq`: Debt in Current Liabilities
    *   `dlttq`: Long-Term Debt (Proxy: `ltq - lctq` if missing)
    *   `seqq`: Shareholders' Equity
    *   `cheq`: Cash and Equivalents
*   **Income Statement**:
    *   `saleq`: Sales/Turnover
    *   `niq`: Net Income (Proxy: `ibq`)
    *   `oiadpq`: Operating Income After Depreciation
    *   `piq`: Pretax Income
    *   `dpq`: Depreciation and Amortization
    *   `epspxq`: Earnings Per Share (Basic) - Used to derive Price-to-Earnings ratios implicitly.

## Detailed Component Analysis


### 1. Data Loader (`data_loader.py`)
*   **Goal**: Generate the exact 36 input features used in the paper.
*   **Inheritance**: Reuses `src.data.ghz_factors.GHZFactorBuilder` to handle raw WRDS data loading and cleaning.
*   **Features with Special Handling**:
    *   **Momentum (24 features)**: `mom_1` is simple $t-1$ return. `mom_i` ($i>1$) is the cumulative return over previous $i$ months (excluding current). Implemented via `rolling(window=i).sum()` on log-returns.
    *   **Fundamentals**: Extracts quarterly items (e.g., `atq`, `ltq`, `niq`) and aligns them to monthly data using `valid_from` (RDQ or Date+45d) via `merge_asof`.
*   **Key Detail**: The `_merge_custom` method implements a precise `merge_asof` logic to align lower-frequency quarterly data with monthly price data, respecting "knowledge dates" to prevent look-ahead bias.

### 2. Modules (`modules.py`)
*   **PLE Phase 1 (Binning)**:
    *   Implements **Piece-wise Linear Encoding**.
    *   Use $T=64$ bins (configurable).
    *   Encodes scalar features into a dense vector representing its position in the distribution.
    *   **Note**: Uses `torch.linspace(-4, 4)` for efficient binning on StandardScaled data.
*   **Backbone**:
    *   A **Bidirectional Transformer** (2 layers, 8 heads).
    *   Input: `[CLS]` token + 36 feature embeddings.
    *   Output: The embedding of the `[CLS]` token serves as the asset representation $h$.

### 3. ORCA Model & Unified Loss (`orca.py`)
*   **Architecture**:
    *   **Backbone**: `ORCABackbone` (as above).
    *   **Instance Head**: MLP projecting $h$ to $z$ for instance contrastive learning.
    *   **Cluster Head**: MLP projecting $h$ to $y$ (Softmax over 30 clusters).
    *   **OU Head**: MLP projecting Cluster Centroid $\bar{h}_k$ to OU parameters $(\theta, \mu, \sigma)$.
*   **Losses**:
    *   **$L_{ins}$ (Instance)**: NT-Xent (SimCLR) loss between augmented views of the same asset. Maximizes similarity $z_a \cdot z_b$.
    *   **$L_{clu}$ (Cluster)**: Contrastive loss on cluster assignments $y_a, y_b$. Plus Entropy Maximization to prevent cluster collapse.
    *   **$L_{PINN}$ (Physics-Informed)**: The core innovation.
        *   Calculates weighted average centroid $\bar{h}_k$ for each cluster.
        *   Predicts $(\theta_k, \mu_k, \sigma_k)$ for that centroid.
        *   Validates if asset returns in that cluster follow $dr_t = \theta(\mu - r_t)dt + \sigma dW_t$.
        *   Minimizes negative log-likelihood of the residual.

### 4. Training (`train.py`)
*   **Augmentation**:
    *   **Masking**: Randomly zeros 10% of features.
    *   **Gaussian Noise**: Adds $\mathcal{N}(0, 0.1)$.
*   **Data Feeding**:
    *   Feeds pairs of returns $(r_t, r_{t-1})$ to the model to compute the PINN loss residuals.

### 5. Backtest (`backtest.py`)
*   **Strategy**: Cluster-based Mean Reversion.
*   **Logic**:
    *   For each Cluster:
        *   Rank assets by `mom_1` (Prior 1-month return).
        *   Calculate Z-Score of `mom_1`.
        *   **Long**: Z-Score < $-\gamma$ (Losers).
        *   **Short**: Z-Score > $+\gamma$ (Winners).
    *   Equal weighted portfolio.
    *   Monthly rebalancing.

## Refactoring Notes
*   Common utility functions (`safe_bool_to_int`, `safe_log`, `clean_infs`, `rolling_*`) were extracted from `src/data/ghz_factors.py` to `src/utils/data_utils.py`.
*   This ensures that the new `ORCADataLoader` and the existing `GHZFactorBuilder` share the exact same robust numerical handling logic without code duplication.
