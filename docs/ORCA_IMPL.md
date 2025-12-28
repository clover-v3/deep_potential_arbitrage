# ORCA Implementation Documentation

## Overview
This directory (`src/contrastive_bl/`) contains the implementation of the paper **"Deep Mean-Reversion: A Physics-Informed Contrastive Approach to Pairs Trading"** (Kim et al., 2025). The model, named **ORCA**, integrates Contrastive Learning with Physics-Informed Neural Networks (PINNs) to find tradable, mean-reverting asset clusters.

## Directory Structure

```
src/
├── contrastive_bl/
│   ├── data_loader.py       # ORCADataLoader: Features (Momentum + Fundamentals)
│   ├── modules.py           # Neural Modules: PLE Encoder, Transformer Backbone
│   ├── orca.py              # ORCAModel: Combined Architecture + Unified Loss
│   ├── train.py             # Training Loop (Augmentation, Universe Selection)
│   ├── backtest.py          # Daily Trading Backtest (BaselineStrategy)
│   ├── run_rolling.py       # Rolling Window Experiment Orchestrator
│   ├── run_autotune_orca.py # Hyperparameter Grid Search Script
├── utils/
│   ├── data_utils.py        # Shared data helpers (safe_log, clean_infs, etc.)
```

## 1. Data Pipeline & Processing

### Features (36 Total)
*   **Momentum (24)**: `mom_1` (Last month return) + `mom_i` (Cumulative return $t-i \dots t-1$).
*   **Fundamentals (12)**: Quarterly metrics (Asset, Debt, Income, etc.) aligned to monthly price data.

### Preprocessing Logic (`train.py`)
1.  **Universe Selection**:
    *   **Logic**: Top 2000 stocks by Market Capitalization (`mve_m`).
    *   **Timing**: Selection is based on the **last month** of the training period.
    *   **Constraint**: Both training and subsequent backtesting are strictly filtered to this universe.
    *   **Artifact**: The list of 2000 `pernmo`s is saved as `*_universe.npy`.
2.  **Normalization**:
    *   z-score normalization: $\frac{x - \mu}{\sigma}$.
3.  **Winsorization**:
    *   **Clipping**: Values are clamped to the range $[-5, +5]$ standard deviations (`np.clip`).
    *   **Purpose**: Outlier suppression (fat tails).

### Augmentation (for Contrastive Learning)
Used in `train.py` to generate views $x_a, x_b$:
*   **mask_prob**: 0.1 (10% of features set to 0).
*   **noise_std**: 0.1 (Gaussian noise added).

## 2. Model Architecture (`orca.py`, `modules.py`)

*   **PLE Encoder**: Piece-wise Linear Encoding ($T$ bins, $D$ embedding dim).
    *   Defaults: `n_bins=64`, `d_model=128`.
*   **Transformer Backbone**: 2 Layers, 8 Heads.
*   **Heads**:
    *   Instance Head -> $z$ (Contrastive)
    *   Cluster Head -> $y$ (Softmax over 30 clusters)
    *   PINN Head -> OU Params $(\theta, \mu, \sigma)$

## 3. Training Strategy (`train.py`)

*   **Loss Function**: $L_{total} = L_{ins} + \alpha L_{clu} + \beta L_{PINN}$
*   **Batching**:
    *   **Global Shuffle** (Default): Random batches.
    *   **Monthly Batching** (`--batch_mode monthly`): Batches contain assets from the *same month* only, to remove temporal noise and focus on cross-sectional alpha.
*   **Hyperparameters**: Exposed via CLI (`--d_model`, `--dropout`, etc.).

## 4. Backtesting Strategy (`backtest.py`)

### Setup
*   **Data**: Daily Price Data (`crsp_dsf`).
    *   *Fallback*: If daily data is missing, uses Monthly data (`crsp_msf`) as a proxy.
*   **History Buffer**: Loads `start_year - 1` to ensure the strategy's rolling window is warm-started at Day 0.
*   **Clusters**: Predictions from the trained ORCA model (monthly freq) are forward-filled to daily.

### Trading Logic (`BaselineStrategy`)
*   **Signal**: Z-Score of cumulative idiosyncratic returns ($R_{i} - R_{cluster}$).
*   **Entry**: |Z| > 2.0 (Mean Reversion).
*   **Exit Rules**:
    1.  **Last 5 Days**: No new positions allowed.
    2.  **Last Day**: Force Close all positions.

## 5. Advanced Workflows

### Rolling Window (`run_rolling.py`)
Automates the lifecycle:
1.  **Train**: Years $T \dots T+K$
2.  **Test**: Year $T+K+1$
3.  **Slide**: $T \leftarrow T+1$, repeat.

### Auto-Tuning (`run_autotune_orca.py`)
Grid search optimization validation Sharpe Ratio.
*   Grid: `d_model` [64, 128], `n_bins` [32, 64], `dropout` [0.1, 0.3].
