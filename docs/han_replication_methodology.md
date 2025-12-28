# Han et al. (2021) "Pairs Trading via Unsupervised Learning" Replication Methodology

This document details the implementation of the Unsupervised Learning Pipeline for Pairs Trading, strictly following the methodology of *Han et al. (2021)* and its advanced extensions.

## 1. Data Processing Methodology

The pipeline processes raw equity data to construct a robust feature set for clustering.

### 1.1 Data Universe & Frequency
- **Source**: WRDS / CRSP / Compustat.
- **Frequency**: Monthly Rolling Window.
- **Training Data**: A "Snapshot" of the market is taken on the last trading day of month $T-1$.
- **Testing**: The derived clusters are used statically for trading throughout month $T$.

### 1.2 Feature Engineering (175+ Factors)
We implement the full suite of **GHZ (Green, Hand, Zhang 2017)** characteristics:
- **Fundamental Factors**: Valuation (EP, BP), Profitability (ROE, ROA), Investment (Asset Growth).
- **Price Factors**: Momentum (12m, 1m), Reversal, Volatility (Idiosyncratic Vol), Beta.
- **Liquidity Factors**: Turnover, Illiquidity, Bid-Ask Spread.

### 1.3 Data Cleaning Pipeline
Before clustering, the feature matrix $X_{T-1}$ (Dimensions $N \times F$) undergoes rigorous cleaning:
1.  **Inf Handling**: Infinite values are replaced with `NaN`.
2.  **Sparcity Filter**: Features with **>30% Missing Values** are dropped to ensure data quality (`nan_counts < 0.3`).
3.  **Imputation**: Remaining missing values are filled with `0`. Since features are later standardized, this effectively imputes with the cross-sectional mean (neutral value).
4.  **Constant Filter**: Features with zero variance (std < 1e-6) are removed to prevent singular matrices in PCA.

---

## 2. Clustering Methodology

The core engine groups stocks into "Pairs" (or Clusters) based on the similarity of their risk factor exposures.

### 2.1 Dimensionality Reduction (PCA)
- **Algorithm**: Principal Component Analysis (PCA).
- **Purpose**: Extract latent "Risk Factors" from the 100+ raw characteristics.
- **Components ($K_{pca}$)**: Default is **5** (configurable).
- **Scaling**: Features are Standard Scaled ($z = \frac{x - \mu}{\sigma}$) prior to PCA.

### 2.2 Dynamic Clustering Parameters
We implement the **adaptive parameter design** implied by density-based clustering literature and Han et al.'s advanced settings. This removes the need for arbitrary hyperparameter guessing.

#### A. Distance Metric: L1 Norm (Manhattan)
- **Requirement**: "Similarly to DBSCAN, l1 norm is used as the distance metric."
- **Implementation**: All distance calculations (KNN graph, Clustering, Density Filter) leverage `metric='manhattan'` (Sum of absolute differences).
  $$d(x, y) = \sum_{i} |x_i - y_i|$$

#### B. Dynamic MinPts (Minimum Points)
- **Rule**: $MinPts = \ln(N)$, where $N$ is the number of stocks in the cross-section.
- **Implementation**: The pipeline automatically calculates this per month.
  - *Example (May 2024)*: $N=9432 \rightarrow MinPts = \lfloor \ln(9432) \rfloor = 9$.

#### C. Dynamic Threshold ($\epsilon$)
- **Method**: **K-Distance Graph** (Ester et al. 1996).
- **Logic**:
    1.  For every stock, calculate the L1 distance to its $k$-th nearest neighbor (where $k=MinPts$).
    2.  Sort these distances to form a distribution.
    3.  Set $\epsilon$ (or `distance_threshold`) to the **$q$-th percentile** of these distribution (`dist_quantile`).
- **Benefit**: Adapts the strictness of clustering to the market's current volatility regime.

---

## 3. Models Implemented

The architecture supports multiple unsupervised learners, swappable via CLI arguments.

### 3.1 Agglomerative Clustering (Primary)
- **Type**: Hierarchical, Bottom-up.
- **Linkage**: `average` (Robust to noise, supports L1 metric).
- **Configuration**: Uses the dynamic `distance_threshold` calculated above.
- **Behavior**: Merges clusters until the inter-cluster L1 distance exceeds dynamic $\epsilon$.

### 3.2 DBSCAN
- **Type**: Density-Based Spatial Clustering.
- **Metric**: Manhattan (L1).
- **Parameters**:
    - `eps`: Dynamic (from K-Dist Graph).
    - `min_samples`: Dynamic $\ln(N)$.
- **Behavior**: Finds high-density regions; classifies low-density points as outliers (-1).

### 3.3 OPTICS
- **Type**: Generalized DBSCAN (Ordering Points To Identify the Clustering Structure).
- **Metric**: Manhattan (L1).
- **Benefit**: Less sensitive to specific $\epsilon$, better dealing with varying density clusters.

### 3.4 K-Means (Baseline)
- **Type**: Centroid-based.
- **Metric**: Euclidean (L2) - *Note: Standard K-Means is L2-only. Used for baseline comparison.*
- **Parameters**: Fixed $K$ (e.g., 5, 10, 50).

---

## 4. Generic Density Filtering (Post-Processing)
Regardless of the algorithm used (even K-Means), we apply a **Density Filter** to remove "loose" stocks that don't fit well.

1.  **Empirical Centroid**: Calculate the center (mean) of the assigned cluster.
2.  **Distance Calculation**: Compute **L1 Distance** of each stock to its cluster centroid.
3.  **Outlier Cutoff**: Discard stocks whose distance is in the top $5^{th}$ percentile (`outlier_percentile=95.0`).
4.  **Result**: Only "tight" pairs remain for trading.

---

## 5. Execution & Reproducibility

### CLI Command Example
To replicate the full dynamic L1-norm pipeline:

```bash
python3 src/baseline/run_pipeline.py \
    --start_date 2024-05-01 \
    --end_date 2024-05-31 \
    --lookback 1 \
    --method agglomerative \
    --dist_quantile 0.1 \
    --outlier_percentile 95.0
```

### Artifacts (Source Code)
- **Pipeline Logic**: `src/baseline/run_pipeline.py`
- **Clustering Engine**: `src/baseline/han_clustering.py`
- **Feature Source**: `src/data/ghz_factors.py`
