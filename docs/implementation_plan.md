# Implementation Plan: Core Assumption Validation (Updated)

**Date:** 2025-12-10
**Goal:** Validate the two fundamental assumptions of the Deep Potential Arbitrage framework

---

## 🎯 核心逻辑链条

### 我们的方法论

```
股票市场 → 存在均值回归 (OU process) → 可用拉普拉斯动力学建模 → GNN学习聚类结构 → Controller利用势能梯度交易
```

### 关键假设

1. **假设1：GNN能捕捉拉普拉斯动力学形成的聚类关系**
   - 拉普拉斯动力学会自然形成同步簇
   - GNN能够从数据中学习到这些簇
   - 学到的簇与真实的动力学簇一致

2. **假设2：股票市场存在拉普拉斯动力学**
   - 股票价格遵循类似OU过程的均值回归
   - 这种回归可以用拉普拉斯动力学建模
   - 恢复力能预测未来价格变化

---

## ⚠️ 重要澄清

### ❌ 我们不做什么

**不是"动力学反推图"：**
- 我们不会用 `dynamics_based_graph()` 作为主要方法
- 那只是一个baseline，用来验证概念
- 实际系统中，我们用**GNN学习图**

### ✅ 我们做什么

**GNN学习聚类：**
- 用GNN从时间序列数据中学习图结构
- 验证GNN学到的簇与拉普拉斯动力学形成的簇一致
- 证明GNN能捕捉到动力学的本质

---

## 验证策略

### 阶段1：合成数据验证（Controlled Experiment）
- 生成已知拉普拉斯动力学的数据
- 用GNN学习图结构
- 对比GNN学到的簇 vs 真实动力学簇

### 阶段2：真实数据验证（Real-World Test）
- 使用真实股票数据
- 测试是否存在拉普拉斯动力学
- 验证GNN能否发现这种结构

---

## Part 1: 假设1验证 - GNN能学习拉普拉斯动力学聚类

### 1.1 合成数据生成（保持不变）

使用 `synthetic_data.py` 生成符合拉普拉斯动力学的数据：
```python
dataset = generate_complete_dataset(
    n_stocks=50,
    n_clusters=5,
    n_timesteps=1000
)
# 包含：
# - A_true: 真实图结构
# - cluster_labels: 真实簇标签
# - f_series: 状态演化（符合 df/dt = -Lf）
```

---

### 1.2 GNN图学习（核心方法）

**目标：** 用GNN从时间序列中学习图结构

#### 方法A：简单GNN（先实现这个）

```python
class SimpleGNN(nn.Module):
    """
    简单的GNN图学习器

    输入：时间序列 X (N, T, d)
    输出：邻接矩阵 A (N, N)
    """
    def __init__(self, d_input, d_hidden, n_stocks):
        super().__init__()
        # 时序编码器
        self.temporal_encoder = nn.Conv1d(d_input, d_hidden, kernel_size=5)

        # 节点嵌入
        self.node_embedding = nn.Linear(d_hidden, d_hidden)

        # 边预测器（pairwise）
        self.edge_predictor = nn.Bilinear(d_hidden, d_hidden, 1)

    def forward(self, X):
        # X: (N, T, d)
        # 1. 时序编码
        h = self.temporal_encoder(X.transpose(1, 2))  # (N, d_hidden, T')
        h = h.mean(dim=2)  # (N, d_hidden) - 池化

        # 2. 节点嵌入
        h = self.node_embedding(h)  # (N, d_hidden)

        # 3. 边预测（所有配对）
        N = len(h)
        A = torch.zeros(N, N)
        for i in range(N):
            for j in range(i+1, N):
                score = self.edge_predictor(h[i], h[j])
                A[i, j] = A[j, i] = torch.sigmoid(score)

        return A
```

#### 方法B：图结构学习（可选，更高级）

使用可微分的图学习方法（如LDS, NRI等）

---

### 1.3 训练目标

**核心问题：** 用什么loss训练GNN？

#### 选项1：监督学习（如果有真实图）

```python
# 在合成数据上，我们知道真实图A_true
loss = binary_cross_entropy(A_pred, A_true)
```

**优点：** 直接，容易收敛
**缺点：** 真实数据没有ground truth

#### 选项2：动力学一致性（推荐 - 鲁棒升级版）

为了与 Controller v3.1 的鲁棒设计保持一致，模型训练也必须**鲁棒化**。我们不能让 GNN 为了拟合一个巨大的黑天鹅（Outlier）而破坏了整个图结构。

```python
# 学到的图应该能解释动力学，但要忽略黑天鹅
# df/dt ≈ -L * f

def robust_dynamics_loss(A_pred, f_series, delta=1.0):
    L_pred = compute_laplacian(A_pred)

    # 计算 df/dt
    df_dt = (f_series[1:] - f_series[:-1]) / dt

    # 预测的力：F_pred = -L * f
    force_pred = -L_pred @ f_series[:-1].T

    # 预测误差：Residual = df/dt - F_pred
    residual = df_dt - force_pred.T

    # 使用 Huber Loss 代替 MSE
    # 当误差大时（黑天鹅），只产生线性的梯度，而不是平方梯度
    loss = torch.nn.HuberLoss(delta=delta)(residual, torch.zeros_like(residual))

    return loss
```

**对齐 Controller：**
- **Training:** Huber Loss 忽略训练数据中的 Outliers。
- **Trading:** Huber Potential 忽略实时行情中的 Outliers。
- **Result:** Training Objective 和 Trading Objective 完美统一（Robust Estimation & Control）。

#### 选项3：对比学习（最灵活）

```python
# 同一簇的股票应该相似，不同簇的应该不同
def contrastive_loss(A_pred, f_series):
    # 用A_pred定义相似度
    # 最大化同簇股票的相似度
    # 最小化不同簇股票的相似度
    pass
```

---

### 1.4 评估指标（更新）

**核心指标：簇一致性**

```python
def evaluate_gnn_clustering(A_gnn, A_true, cluster_labels_true):
    """
    评估GNN学到的聚类质量

    重点：
    - NMI (Normalized Mutual Information)
    - ARI (Adjusted Rand Index)
    - 簇纯度

    次要：
    - 边的F1（不是最重要的）
    """
    # 1. 用GNN学到的图进行谱聚类
    n_clusters = len(np.unique(cluster_labels_true))
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed'
    )
    cluster_labels_gnn = clustering.fit_predict(A_gnn)

    # 2. 计算簇一致性
    nmi = normalized_mutual_info_score(cluster_labels_true, cluster_labels_gnn)
    ari = adjusted_rand_score(cluster_labels_true, cluster_labels_gnn)

    # 3. 簇纯度
    purity = compute_purity(cluster_labels_true, cluster_labels_gnn)

    return {
        'nmi': nmi,  # 最重要
        'ari': ari,  # 最重要
        'purity': purity,
        # 边的F1作为参考
        'edge_f1': compute_edge_f1(A_gnn, A_true)
    }
```

**通过标准（更新）：**
- ✅ **NMI > 0.7**（簇识别准确）
- ✅ **ARI > 0.6**（簇分配一致）
- ✅ **Purity > 0.8**（簇纯度高）

边的F1不是主要指标，因为：
- 簇内的具体连接方式不重要
- 重要的是能否识别出哪些股票属于同一簇

---

### 1.5 对比方法（更新）

```python
methods = {
    'GNN': SimpleGNN(...),  # 我们的方法
    'Han2021': han_et_al_clustering,  # Priority Baseline (Paper Replication)
    'Correlation': correlation_graph,  # Baseline 1
    'Dynamics': dynamics_based_graph,  # Baseline 2（仅用于验证概念）
    'Spectral': spectral_clustering_on_correlation,  # Baseline 3
}
```

---

## Part 3: Baseline复现 - Han et al. (2021)

**目标：** 复现 Reference 中的论文方法，作为核心Baseline
**方法特点：** 结合基本面数据和量价数据进行聚类

### 3.1 特征工程
需要构建两类特征：

1.  **量价特征 (Price-Volume)**
    - 波动率 (Volatility)
    - 流动性 (Liquidity/Turnover)
    - 动量 (Momentum)
    - 贝塔 (Beta)

2.  **基本面特征 (Fundamental)**
    - 估值: P/E, P/B
    - 规模: Market Cap
    - 盈利能力: ROE
    - 成长性: Revenue Growth

**Goal:** 复现 "Fundamental Decomposition of Stock Returns" 论文方法，构建基于基本面的聚类基线。即使是简单的 K-Means，有了正确的基本面特征，也能产生极佳的板块结构。

#### 3.1 Data Pipeline & Utils
**Corrected Data Format (All files: Row=Time, Col=Tickers):**
*   **Daily Data:** `path/to/daily/{variable}/{year_month}.parquet` (e.g., `close`, `market_cap`).
*   **1-Min Data:** `path/to/1min/{variable}/{date}.parquet`.

**Required Raw Variables (for Han et al. Features):**
1.  **Price/Volume:** `close`, `open`, `high`, `low`, `volume`, `amount`.
    *   For: Volatility, Momentum (RSI), Turnover, Liquidity.
2.  **Fundamental:**
    *   `market_cap` (Size factor)
    *   `pe_ttm` (Valuation)
    *   `pb_lf` (Valuation)
    *   `roe_ttm` (Profitability)
    *   `operating_revenue_growth` (Growth)
    *   `industry_citic` (Ground Truth for validation)
    *   `co_filedate` (Precise Annual Availability)

### 3.1.2 Data Processing Improvements (User Requested)
1.  **Lazy Loading**: Load only relevant parquet files based on `start_date` and `end_date` (with buffer).
2.  **Precise Availability (`valid_from`)**:
    *   **Annual**: Use `comp.co_filedate`. Fallback: `datadate + 3 months`.
    *   **Quarterly**: Use `rdq`. Fallback: `datadate + 45 days`.
    *   **Monthly**: `date + 1 day`.
3.  **Data Cleaning**:
    *   Annual: Drop rows with `sales` is NaN.
    *   Quarterly: Drop rows with `salesq` is NaN.
    *   Monthly: Drop rows with `prc` is NaN.

### 3.1.2 Manual CUSIP Linking (Fallback Strategy)
Since `ccmxpf_linktable` is unavailable, we implement manual linking:
1.  **Compustat Data**: Ensure `comp_funda` includes `cusip` (from `comp.company` header or `funda`).
    *   *Note*: Header CUSIP (`comp.company`) is static (current), introducing some survivorship/look-ahead bias if ticker changed, but acceptable as fallback.
2.  **CRSP Data**: Use `crsp_stocknames` (downloaded in Step 5 of pull script).
    *   Columns: `permno`, `ncusip` (Historical CUSIP), `namedt`, `nameenddt`.
3.  **Merging Logic**:
    *   Match `funda.cusip` (first 6 digits) == `stocknames.ncusip` (first 6 digits).
    *   Filter: `funda.datadate` within `[stocknames.namedt, stocknames.nameenddt]`.
    *   Assign `permno` to Compustat rows.

#### 3.2 Methodology: Clustering Algorithms (The Baselines)
**Goal:** Evaluate standard unsupervised learning methods.
1.  **K-Means:** The standard. Rigid, spherical clusters.
2.  **DBSCAN:** Density-based. Can handle noise and non-spherical shapes. Important for filtering "outlier" stocks.
3.  **Agglomerative:** Hierarchical. useful for nested sector structures.

#### [Baseline] Han et al. (2021) Methodology Refined

> [!IMPORANT]
> **Revised Objective**: The goal is NOT to reproduce Industry Classification, but to find latent structural similarities that traditional sectors miss. We evaluate cluster validity using statistical metrics (Silhouette, Stability) and Trading Performance (Sharpe), rather than NMI against Industry.

#### 1. Experimental Modes
We will test three clustering configurations:
1.  **Fundamental Only**: Low-frequency (Quarterly/Monthly) metrics. Captures "Value/Growth" structure.
2.  **Price/Volume Only**: High-frequency aggregated metrics. Captures "Behavioral/Microstructure" structure.
3.  **Combined**: The fusion of both.

#### 2. Advanced Feature Engineering (Daily Microstructure)
We implement **16+ Distinct Features** in `src/data/daily_factors.py` using daily OHLCV and Trade Count data:

**A. Volatility & Distribution**
1.  `parkinson_vol`: High-Low range-based volatility estimator.
2.  `downside_vol`: Std of negative daily returns.
3.  `upside_vol`: Std of positive daily returns.
4.  `max_ret`: Maximum daily return in month (Lottery).
5.  `skew`: Skewness of daily returns.

**B. Liquidity & Volume**
6.  `amihud_illiquidity`: Mean(|Ret| / (Price * Volume)).
7.  `turnover_var`: Std of daily turnover.
8.  `dvol_cv`: Coeff of Variation of Dollar Volume (Liquidity Stability).
9.  `avg_trade_size`: Vol / NumTrades (Institutional Presence).
10. `illiq_numtrd`: Trade-based Illiquidity (Kyle's Lambda proxy).

**C. Microstructure & Efficiency**
11. `ret_gap_vol`: Volatility of overnight gaps (Split-adjusted).
12. `clv_mean`: Close Location Value (Accumulation/Distribution proxy).
13. `zero_ret_pct`: Percentage of days with zero return.
14. `payout_yield`: Sum(Return - Retx) (Realized Dividend Yield).
15. `intraday_ret`: Mean((Close/Open) - 1).
16. `intraday_vol`: Std((Close/Open) - 1).

#### 3.3 Comparison Metrics
*   **Cluster Quality:** NMI against true Sector Labels.
*   **Trading Performance:** Use this $A$ in our *Deep Potential Controller*.
    *   Does Fundamental $A$ generate higher IC than Price Correlation $A$?
    *   Hypothesis: Yes, especially in "Diversified" regimes.
### 3.2 聚类算法
通常流程：
1.  特征标准化 (Z-score)
2.  降维 (PCA)
3.  聚类 (DBSCAN/OPTICS/K-Means)

### 3.3 实施计划
- [ ] **Data Loader**: 扩充数据加载器以支持基本面数据
- [ ] **Feature Extractor**: 实现常用的因子计算
- [ ] **Clustering Model**: 实现论文中的聚类流程
- [ ] **Comparison**: 在Part 1和Part 2中加入此方法的对比

---

## Part 4: 假设2验证 - 股票市场存在拉普拉斯动力学

### 2.1 核心测试（保持不变）

**测试：** 恢复力 $F = -2Lf$ 是否预测未来收益

```python
def test_laplacian_dynamics_in_market(f_series, L):
    """
    测试市场是否遵循拉普拉斯动力学

    核心：F_t 与 Δf_{t+1} 的相关性
    """
    ic_results = test_force_return_correlation(f_series, L, lag=1)

    # 通过标准
    return {
        'ic_mean': ic_results['ic_mean'],  # > 0.05
        'p_value': ic_results['p_value'],  # < 0.05
        'passed': (ic_results['ic_mean'] > 0.05 and
                  ic_results['p_value'] < 0.05)
    }
```

### 2.2 不同时间尺度测试

```python
# 测试不同频率的数据
time_scales = {
    '1min': load_1min_data(),
    '5min': load_5min_data(),
    '1hour': load_1hour_data(),
    '1day': load_1day_data(),
}

for scale, data in time_scales.items():
    result = test_laplacian_dynamics_in_market(data, L)
    print(f"{scale}: IC = {result['ic_mean']:.4f}")
```

**目标：** 找到拉普拉斯动力学最显著的时间尺度

---

## 实施文件结构（更新）

```
src/validation/
├── __init__.py
├── synthetic_data.py          # ✅ 已实现
├── gnn_models.py              # 🆕 GNN模型定义
│   ├── SimpleGNN
│   └── AdvancedGNN (可选)
│
├── gnn_training.py            # 🆕 GNN训练逻辑
│   ├── train_gnn_supervised()
│   ├── train_gnn_dynamics()
│   └── train_gnn_contrastive()
│
├── graph_learning.py          # ✅ 已实现（作为baseline）
│   ├── correlation_graph()
│   ├── dynamics_based_graph()  # 仅用于概念验证
│   └── spectral_clustering()
│
├── dynamics_test.py           # ✅ 已实现
├── metrics.py                 # 🔄 需更新（强调簇指标）
└── run_validation.py          # 🔄 需更新（GNN为主）

### Design Decisions: Baseline Replication (Updated)

### Addressing Data Frequency Imbalance (Monthly vs Daily)
**User Question**: "How to mix frequencies without one dominating?"
**Solution**: **PCA (Principal Component Analysis)**.
*   **Mechanism**: We do NOT cluster on the 116 raw features (100 Monthly + 16 Daily). Instead, we cluster on the Top $K$ Principal Components.
*   **Why it works**:
    *   The 100+ Monthly Fundamental factors are highly collinear (Value, Growth, Profitability groups). PCA compresses this redundancy into a few "Fundamental Components".
    *   The 16 Daily Microstructure factors are distinct. They form "Microstructure Components".
    *   **Result**: The clustering inputs are orthogonal Risk Factors, balanced by *Information Content (Variance)*, not by the raw count of columns.

### Clustering Models (To Be Implemented)
1.  **K-Means**: Standard baseline. Simple, rigid clusters.
2.  **DBSCAN**: Density-based. Crucial for filtering noise stocks that don't follow any group.
3.  **OPTICS**: Advanced density clustering, better for varying density.
4.  **Agglomerative**: Hierarchical clustering.

### Baseline Pipeline Workflow (Rolling Walk-Forward)
Strictly implementing Han et al.'s rolling methodology:
1.  **Iterative Loop**: For each Month $M$ in Test Range:
    *   **Train Data**: Rolling window $[M - \text{Lookback}, M - 1]$.
    *   **Preprocessing**:
        *   Clean NaNs (Drop sparse, Impute 0).
        *   Remove Constant Features.
        *   **StandardScaler** (Fit on Train).
        *   **PCA** (Fit on Train) -> Extract Risk Factors.
    *   **Clustering**: Fit Model (K-Means/DBSCAN/etc) on PCA Components.
    *   **Prediction**: Apply fitted Scaler/PCA/Clusterer to Month $M$ data.
    *   **Trading**: Generate signals based on Mean Reversion ($R_i - R_{cluster}$).
    *   **Store**: Save Month $M$ PnL and Signals.
2.  **Evaluation**: Concatenate all monthly results and compute global metrics.
```

---

## 验证计划（更新）

### Week 1: GNN实现和合成数据测试

**Day 1-2: 实现GNN模型**
- [ ] `gnn_models.py` - SimpleGNN
- [ ] `gnn_training.py` - 动力学一致性loss
- [ ] 测试：能否在合成数据上收敛

**Day 3-4: 训练和评估**
- [ ] 训练GNN on合成数据
- [ ] 评估簇识别质量（NMI, ARI）
- [ ] 对比baseline方法

**Day 5-7: 调优和验证**
- [ ] 调整超参数
- [ ] 测试不同噪声水平
- [ ] 生成报告：NMI > 0.7?

### Week 2: 真实数据验证

**Day 1-2: 数据准备**
- [ ] 下载股票数据（XLK等）
- [ ] 预处理：去趋势、标准化

**Day 3-5: 动力学测试**
- [ ] 测试不同时间尺度
- [ ] IC测试：是否存在拉普拉斯动力学
- [ ] 找到最优时间尺度

**Day 6-7: GNN应用**
- [ ] 用GNN学习真实数据的图
- [ ] 验证学到的簇是否稳定
- [ ] 最终决策：Go or No-Go

---

## 成功标准（更新）

### 假设1（GNN聚类）通过标准：
- ✅ 合成数据：**NMI > 0.7, ARI > 0.6**
- ✅ GNN优于correlation baseline（NMI提升 > 0.1）
- ✅ 簇稳定性：滚动窗口相似度 > 0.6

### 假设2（动力学）通过标准：
- ✅ 合成数据：IC > 0.3 (p < 0.01)
- ✅ 真实数据：**IC > 0.05 (p < 0.05)**
- ✅ 至少一个时间尺度显著

### 综合决策：
- **绿灯（继续）：** 两个假设都通过
- **黄灯（修正）：** 一个假设通过
- **红灯（停止）：** 两个假设都不通过

---

## 关键区别总结

| 方面 | 旧理解 | 新理解（正确） |
|-----|--------|--------------|
| 主要方法 | dynamics_based_graph | **GNN学习图** |
| 验证重点 | 边的F1 | **簇的NMI/ARI** |
| dynamics_based_graph的作用 | 主要方法 | Baseline/概念验证 |
| 核心假设 | 能从动力学反推图 | **GNN能学到动力学簇** |

---

## 下一步

1. **实现GNN模型**（`gnn_models.py`）
2. **实现训练逻辑**（`gnn_training.py`）
3. **更新评估指标**（强调NMI/ARI）
4. **更新验证脚本**（GNN为主，dynamics_based_graph为baseline）

这个逻辑链条更清晰：
```
拉普拉斯动力学 → 形成簇 → GNN学习簇 → Controller利用势能交易
```
