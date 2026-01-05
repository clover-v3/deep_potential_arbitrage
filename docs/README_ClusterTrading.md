# Deep Potential Arbitrage: Cluster-Based Pairs Trading

## Overview
This project implements an **End-to-End Differentiable Pairs Trading System**. Unlike traditional methods that separate clustering (unsupervised) and trading (rule-based), this system integrates both into a single differentiable pipeline. It allows the model to learn cluster centroids and signal parameters that directly optimize **Risk-Adjusted Returns (Sharpe Ratio)** while adhering to soft leverage and turnover constraints.

## Key Features
-   **Differentiable Clustering**: Soft K-Means on the Hypersphere allows gradients to flow back to feature extraction.
-   **Outlier Rejection (Gated Softmax)**: Automatically detects points far from any centroid and "gates" them out. This purifies the cluster consensus and forces the exit of drifting positions.
-   **Independent Position Sizing**: Discards "Pie-Chart" allocation in favor of "Checkbook" sizing to prevent opportunity dilution.
-   **Soft Stop-Loss**: Differentiable Band-Pass activation function ($Entry < |Z| < Stop$) allowing the model to learn structural breaks.
-   **Smart Regularization**: Custom Loss function with penalties for Turnover and Excess Leverage.

## Architecture
1.  **Input**: Raw Price Tensor $(T, N)$.
2.  **Features**: Log-prices $\to$ Rolling Z-Score $\to$ Spherical Normalization.
3.  **Clustering**:
    -   Compute Cosine Similarity.
    -   **Gate**: $\text{Sigmoid}(\text{MaxSim} - \text{Threshold})$. Points with low similarity are marked as outliers.
    -   **Soft Assignments**: Standard Softmax.
4.  **Signal**:
    -   **Clean Consensus**: Weighted Average using Gated Assignments.
    -   **Residual**: $Z_{stock} - Z_{consensus}$.
    -   **Activation**: Band-Pass Filter.
    -   **Forced Exit**: Final Signal is multiplied by the Gate.
5.  **Portfolio**: Independent weights scaled by signal strength.

## Installation
Ensure you have `torch` and `pandas` installed.
```bash
pip install torch pandas numpy tqdm
```

## Quick Start (Training)
The main entry point is `src/cluster_trading/train.py`.

### 1. Training on Virtual Data (Demo)
Run a quick training session on synthetic random walk data to verify the pipeline.
```bash
python3 -m src.cluster_trading.train \
    --data_path virtual \
    --n_clusters 10 \
    --window 60 \
    --epochs 50
```

### 2. Training on Real Data
Load your Wide-Format Parquet file (Index=Datetime, Columns=Tickers).
```bash
python3 -m src.cluster_trading.train \
    --data_path /path/to/your/prices.parquet \
    --n_clusters 20 \
    --window 60 \
    --threshold 2.0 \
    --stop_threshold 4.0 \
    --similarity_threshold 0.5 \
    --leverage_penalty 1.0 \
    --max_leverage 5.0 \
    --epochs 200 \
    --lr 0.005 \
    --save_dir checkpoints
```

### 3. Rolling / Walk-Forward Training
For rigorous backtesting, use the rolling trainer which automatically splits data into Train/Test segments and slides forward.
```bash
python3 -m src.cluster_trading.run_rolling \
    --data_path /path/to/data.parquet \
    --train_days 252 \
    --test_days 63 \
    --epochs 50
```

### 4. Automated Grid Search
To optimize hyperparameters (`n_clusters`, `window`, `similarity_threshold`) using the rolling framework:
```bash
python3 -m src.cluster_trading.run_grid_search \
    --data_path /path/to/data.parquet \
    --train_days 252 \
    --test_days 63
```
Results are saved to `grid_search_results.csv`.

## CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_path` | `virtual` | Path to parquet file or 'virtual' for synthetic data. |
| `--train_days`| `252` | Length of In-Sample training period in days. |
| `--test_days` | `63` | Length of Out-of-Sample test period in days. |
| `--n_clusters` | `10` | Number of clusters (centroids). |
| `--window` | `60` | Lookback window size for Z-Score features. |
| `--threshold` | `2.0` | Entry Z-Score threshold. |
| `--stop_threshold` | `4.0` | Stop-Loss Z-Score threshold (Signal decays after this). |
| `--similarity_threshold` | `0.5` | Threshold for Outlier Rejection (Gate < 0.5). |
| `--temp` | `1.0` | Temperature for Softmax clustering (lower = harder assignments). |
| `--scaling_factor` | `5.0` | Sharpness of the Tanh/Sigmoid activation functions. |
| `--leverage_penalty`| `1.0` | Weight of the Soft Leverage Constraint in Loss. |
| `--max_leverage` | `5.0` | Target Maximum Gross Leverage (Sum of Abs Weights). |

| `--temp` | `1.0` | Temperature for Softmax clustering (lower = harder assignments). |
| `--scaling_factor` | `5.0` | Sharpness of the Tanh/Sigmoid activation functions. |
| `--leverage_penalty`| `1.0` | Weight of the Soft Leverage Constraint in Loss. |
| `--max_leverage` | `5.0` | Target Maximum Gross Leverage (Sum of Abs Weights). |
| `--turnover_penalty`| `0.5` | Penalty weight for daily turnover. |
| `--cost_bps` | `10.0` | Transaction cost in basis points (used for Net Return calc). |

## Advanced Concepts
-   **Independent Sizing**: Weights are calculated as $w_i = 0.1 \times \text{Signal}_i$. There is no normalization across stocks. This ensures that a new opportunity does not force the sale of existing profitable positions.
-   **Leverage Penalty**: Since weights are independent, total leverage can fluctuate. The loss function adds a penalty $\text{ReLU}(\text{TotalLev} - \text{MaxLev})^2$ to keep risk in check.
