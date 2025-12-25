# Project Specification: Deep Potential Arbitrage (v2.2)

## Title: Learning to Synchronize — Deep Graph Laplacian Dynamics for Robust Statistical Arbitrage

### 1. 核心假设与公理 (Axioms & Hypothesis)

本项目建立在以下第一性原理之上：

1. **结构存在性：** 市场存在隐式的同步簇，其拓扑结构可用图拉普拉斯矩阵 $L$ 描述。
2. **拉普拉斯动力学：** 系统倾向于能量最小化。当潜在状态 $f$ 违背图结构 $L$ 时，产生恢复力 $g = 2Lf$，预示均值回归。

### 1.1 理论基础：拉普拉斯动力学即广义均值回归 (Theoretical Proof)

用户提出的深刻洞察："拉普拉斯动力学是 Mean Reversion 在图空间的表示"，在数学上是严格成立的。

**定义 1 (标量均值回归 - OU 过程):**
经典的 Ornstein-Uhlenbeck 过程描述如下：
$$ d x_t = -\theta (x_t - \mu) dt + \sigma dW_t $$
* $x_t$: 当前价格
* $\mu$: 全局均衡均值
* $-\theta(x_t - \mu)$: 恢复力，方向指向 $\mu$。

**定义 2 (图拉普拉斯算子):**
对于节点 $i$，拉普拉斯运算 $(Lf)_i$ 定义为：
$$ (Lf)_i = \sum_{j} A_{ij} (f_i - f_j) = d_i f_i - \sum_{j} A_{ij} f_j $$
$$ = d_i (f_i - \frac{\sum A_{ij} f_j}{d_i}) $$

此处，$\frac{\sum A_{ij} f_j}{d_i}$ 正是节点 $i$ 的邻居（相关资产）的加权平均值，记为 $\bar{f}_{neighbors}$。

**推论 (图均值回归):**
当我们将动力学定义为 $\frac{df}{dt} = -Lf$ 时，对节点 $i$ 而言：
$$ \frac{df_i}{dt} = - d_i (f_i - \bar{f}_{neighbors}) $$

**结论:**
这与 OU 过程 $dx = -\theta(x - \mu)$ 形式完全一致！
* **刚度 (Stiffness) $\theta$**: 对应节点的度 $d_i$ (连接强度)。
* **均衡点 (Equilibrium) $\mu$**: 对应**邻居的加权均值** $\bar{f}_{neighbors}$ (局部共识)。

因此，**寻找拉普拉斯动力学 $\iff$ 寻找能够正确定义局部均衡均值的图结构 $L$**。
验证了 $L$ 的存在，即验证了高维空间中 Pair Trading (配对交易) 的广义形式。

---

### 2. Tier 0: 第一性原理验证 (The Go/No-Go Decision)

在开发复杂模型前，必须先完成：

1. **验证 A (表达能力):** 禁用先验，验证 GNN 能否从合成数据中还原 Block-Diagonal 结构。
2. **验证 B (物理有效性):** 使用 XLK ETF 数据，验证 $g=2Lf$（使用全连接图）与未来收益 $r_{t+1}$ 的负相关性。

---

### 3. 模型整体架构 (System Architecture)

这是本次更新的核心部分。模型采用 **"双塔耦合架构" (Dual-Tower Coupling Architecture)**。

#### 3.1 架构图解 (Architecture Diagram)

数据流向呈 "Y" 字形，两个分支并行处理，最终汇聚。

```mermaid
graph TD
    Input((原始输入 Data)) --> |(N, T, d)| Branch1(Branch I: 结构学习器)
    Input --> |(N, T, d)| Branch2(Branch II: 状态提取器)

    subgraph Branch1 [构建地图: 谁和谁是一伙?]
        B1_Enc[时序编码 1D-Conv] --> B1_Emb[特征 H]
        B1_Emb --> B1_Score[两两打分 Pairwise Scorer]
        B1_Score --> B1_Fusion[先验融合 Fusion]
        B1_Fusion --> A[邻接矩阵 A]
        A --> L[拉普拉斯矩阵 L = D - A]
    end

    subgraph Branch2 [确定坐标: 现在位置在哪?]
        B2_Trans[去趋势 / 神经网络映射] --> f[潜在状态向量 f]
    end

    L --> Interaction{耦合交互 Coupling}
    f --> Interaction

    Interaction --> Energy[能量 E = f^T L f]
    Interaction --> Force[恢复力 g = 2 L f]

    Force --> Controller[相空间控制器] --> Signal(交易信号)
```

#### 3.2 模块间交互机制 (Interaction Mechanism)

模型由两个平行的分支组成，它们在计算动力学时发生**耦合**：

* **Branch I (The Map):** 负责输出 $L_t$ (形状 $N \times N$)。它描述了市场的**"约束规则"**。
* **Branch II (The State):** 负责输出 $f_t$ (形状 $N \times 1$)。它描述了资产的**"当前位置"**。
* **Coupling (The Physics):** 两者通过矩阵乘法 $g = 2 L_t f_t$ 结合。只有当"位置"违背"规则"时，信号才会产生。

---

### 4. 详细模块规格 (Module Specifications)

#### 4.0 输入数据定义 (Input Definition)

* **张量形状:** $X \in \mathbb{R}^{B \times N \times T \times d}$
    * $N$: 股票数量 (Universe Size, e.g., 50)
    * $T$: 回望窗口 (Window Size, e.g., 60 minutes)
    * $d$: 特征通道 (Log-returns, Volume, Spread, Volatility)
* **辅助输入:** $Z \in \mathbb{R}^{N \times K}$ (静态属性：行业 One-hot，市值)

#### 4.1 Branch I: 混合图学习器 (Hybrid Graph Learner)

**目标:** 输出反映市场结构的拉普拉斯矩阵 $L_t$。

1. **先验构建 (Prior Path):**
    * 输入: $Z$
    * 计算: $A_0$ (静态稀疏矩阵，行业内=1，否则=0)。
2. **数据驱动学习 (Learner Path):**
    * **Time Encoder:** $X \to \text{Conv1D} \to \text{Pool} \to H \in (N, d_h)$。
    * **Pairwise Scorer:** $S_{ij} = \text{MLP}(h_i, h_j)$。输出原始分数矩阵 $S$。
    * **Sparsification:** 对 $S$ 进行 Top-K 截断和对称化，得到 $A_{learn}$。
3. **融合 (Fusion):**
    * $A_{final} = \alpha A_0 + (1-\alpha) A_{learn}$。
4. **拉普拉斯构建 (Laplacian Construction) [关键]:**
    * 计算度矩阵: $D_{ii} = \sum_{j=1}^N (A_{final})_{ij}$ (即行求和)。
    * 输出: $L_t = D_t - A_{final}$。
    * *作用:* $D$ 的引入使得 $L$ 具备了"差分算子"的物理意义，即 $Lf$ 代表"我 - 邻居均值"。

#### 4.2 Branch II: 潜在状态提取器 (Latent State Extractor)

**目标:** 输出适合进行均值回归的纯净状态 $f_t$。

* **输入:** $X$ (原始价格序列)。
* **Track 1 (Baseline - 必须先做):**
    * 逻辑: 统计残差。
    * 计算: $f_t = P_t - \beta \cdot P_{Index}$ (去大盘趋势)。
    * *作用:* 剥离 Beta，确保 $f$ 是平稳的，防止全市场暴涨导致虚假信号。
* **Track 2 (Advanced):**
    * 逻辑: 神经网络提取非线性 Alpha。
    * 计算: $f_t = \text{MLP}(H)$ (复用 Branch I 的特征 $H$ 或独立编码)。

#### 4.3 动力学引擎 (Dynamics Engine)

**位置:** 双塔汇聚点。

* **输入:** $L_t$ (来自 Branch I), $f_t$ (来自 Branch II)。
* **计算物理量:**
  1. **恢复力 (Force/Gradient):** $g_t = 2 L_t f_t$
      * *(解释: 向量，每个元素代表该股票受到的回归拉力)*
  2. **刚度 (Stiffness):** $k_t = \text{diag}(2 L_t)$
      * *(解释: 向量，每个元素代表该股票与 Cluster 连接的紧密程度/置信度)*
  3. **速度 (Velocity):** $v_t = f_t - f_{t-1}$
      * *(解释: 向量，当前运动方向)*

---

### 5. 执行策略：相空间控制器 (Phase Space Controller)

利用动力学引擎输出的三个物理量进行决策。

**控制方程:**
$$\text{AlphaScore}_i = - \frac{g_{i,t}}{\sqrt{k_{i,t}} + \epsilon}$$

**决策逻辑:**

```python
def Controller(AlphaScore, Velocity, Force):
    # 1. 过滤假信号：如果刚度 k 太小，AlphaScore 会被分母放大？
    # 不，分母小会导致 Score 变大吗？
    # 修正逻辑：刚度 k 是置信度。我们应该 乘以 k 或者 设定 k 的阈值。
    # 新逻辑：Weighted Signal = - Force * sqrt(k) (力越大且连接越紧，信号越强)

    Signal = - Force * sqrt(Stiffness)

    # 2. 动能过滤
    # 正在回归 (Converging): 力与速度同向 -> 加强信号
    # 正在发散 (Diverging): 力与速度反向 -> 信号置 0 (不接飞刀)
    if (Force * Velocity) < 0:
        Signal = 0

    return Signal
```

---

### 6. 训练与损失函数 (Training Methodology)

由于模型有两个分支，训练需要协调。

#### Stage 1: 替代损失预热 (Surrogate Warm-up)

* **固定 Branch II:** $f_t$ 锁死为统计残差（不训练）。
* **训练 Branch I:** 让 GNN 学习 $L_t$。
* **Loss:** $\mathcal{L} = \| f_{t+1} - (f_t - \eta \cdot 2 L_t f_t) \|^2$。
    * *含义:* 寻找一个 $L_t$，使得计算出的恢复力能最好地预测下一刻的均值回归。

#### Stage 2: 全参数微调 (Joint Fine-tuning)

* **解锁 Branch II:** 允许微调 $f_t$。
* **Loss:** Sliced Score Matching (SSM)。
* **正则化:** 加入 $\lambda \|A\|_1$ (稀疏) 和 $\lambda (\lambda_2(L)-c)^2$ (连通性)。

---

### 7. 实施 Roadmap (Implementation Plan)

1. **Week 1 (Tier 0 验证):**
    * 编写"合成数据生成器"。
    * 编写"可视化脚本"。
    * 验证：不加先验，GNN 能否还原 Block-Diagonal 结构。
    * 验证：XLK 数据中，$2Lf$ 是否与未来收益负相关。

2. **Week 2 (原型机搭建):**
    * 实现 `1D-Conv` 编码器。
    * 实现 `Fusion` 模块（融合 $A_0$）。
    * 实现 `Controller` 逻辑。
    * **里程碑:** 跑通 pipeline，输出第一笔交易信号。

3. **Week 3 (策略优化):**
    * 引入 `Surrogate Loss` 进行训练。
    * 在光伏/半导体板块进行回测。
    * 对比 Baseline (普通 Pair Trading)。

---

### 8. 交付物清单 (Deliverables)

1. **`data_loader.py`**: 处理 $(N, T, d)$ 数据，包含去趋势预处理。
2. **`model.py`**: 定义 `Branch1` (GraphLearner), `Branch2` (StateExtractor), 和 `DynamicsEngine`。
3. **`train.py`**: 实现 Surrogate Loss 训练循环。
4. **`backtest.py`**: 实现相空间控制器逻辑，计算 Sharpe 和 Drawdown。
