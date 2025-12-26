# Walkthrough: Han et al. (2021) Baseline Replication

> [!NOTE]
> This document details the implementation of the "Fundamental Decomposition of Stock Returns" baseline. This system serves as a pure performance benchmark for our Deep Potential Controller.

## 1. System Architecture

The baseline replication is built in three modular phases:

1.  **Data & Features**: Ingestion, Universe Filtering, Feature Engineering.
2.  **Clustering Engine**: PCA (Risk Factors) -> K-Means (Segments).
3.  **Trading Strategy**: "Cluster-Relative Reversion" logic.

### Directory Structure
```
src/
├── data/
│   ├── loader.py        # Parquet Loader with Universe Filtering
│   ├── features.py      # Technical & Fundamental Features (Z-Score, Winsorize)
│   └── test_pipeline.py # End-to-End Verification Script
├── baseline/
│   ├── han_clustering.py # PCA + K-Means Pipeline
│   └── strategy.py       # Trading Logic (Signals, PnL, Metrics)
```

## 2. Implementation Details

### Phase 1: Data Loader & Universe
We implemented a robust `DataLoader` that handles:
- **Daily & 1-Min Data**: Reads partitioned parquet files.
- **Universe Masking**: Filters valid tickers daily using a user-provided `universe.parquet` (0/1 mask).

```python
# src/data/loader.py
def _apply_universe_mask(self, df: pd.DataFrame) -> pd.DataFrame:
    """Filters data intersections between available prices and valid universe."""
    # ... (Aligns dates and columns, sets invalid entries to NaN)
```
> [!NOTE]
> **Open Universe Loading**: If `universe_path` is `None`, the loader automatically constructs the universe by concatenating all available parquet files, ensuring no data is filtered and missing columns are filled with `NaN`.

### Phase 2: Clustering Engine
Implements the exact methodology from Han et al.:
1.  **Pivot**: Convert `(Date, Ticker) -> Features` to cross-sectional matrices.
2.  **PCA**: Extract latent risk factors.
3.  **K-Means**: Cluster stocks based on factor loadings.

### Phase 3: Trading Strategy
The **Cluster-Relative Reversion** strategy:
1.  **Cluster Index**: Calculates equal-weighted return of each cluster.
2.  **Idiosyncratic Return**: $R_{idiosyncratic} = R_{stock} - R_{cluster}$
3.  **Signal**: Cumulative sum of idiosyncratic returns, standardized (Z-Score).
    - *Short* if Price deviates positively from Cluster (Overvalued).
    - *Long* if Price deviates negatively from Cluster (Undervalued).

```python
# src/baseline/strategy.py
merged['z_score'] = merged['cum_idio'] / (merged['std_idio'] + 1e-6)
merged.loc[merged['z_score'] > 2.0, 'pos'] = -1
merged.loc[merged['z_score'] < -2.0, 'pos'] = 1
```

## 3. Verification Results

We verified the pipeline using `src/data/test_pipeline.py` with synthetic dummy data.

| Component | Status | Result |
| :--- | :--- | :--- |
| **Data Ingestion** | ✅ Pass | 100+ days loaded, Universe Mask applied correctly. |
| **Feature Engine** | ✅ Pass | Features (PE, Vol, Mom) computed without NaN errors. |
| **Clustering** | ✅ Pass | PCA Variance ~60%, Clusters assigned for all stocks. |
| **Strategy** | ✅ Pass | Signals generated, Active Positions > 0, PnL calculated. |

> [!IMPORTANT]
> The infrastructure is code-complete and verified. The next step is to execute this on **Real Data** to establish the numerical benchmark (Sharpe/Drawdown) for the project.
