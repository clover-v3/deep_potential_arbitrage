# Controller Design v3.1: Robust Vector Field (Huber Potential)

**Date:** 2025-12-10
**Version:** 3.1 - Robustness Upgrade (Huber/Tanh Saturation)
**Parent:** v3.0 Vector Field Approach

---

## 🛡️ 核心升级：鲁棒性 (Robustness)

### 问题分析：平方势能的危险性
v3.0 使用平方势能 $E(f) = f^T L f = \sum A_{ij}(f_i - f_j)^2$。
- **物理特性：** 线性弹簧 ($F = -kx$)
- **优点：** 均值回归从不停止，偏差越大，恢复力越大。
- **致命弱点：**
  - 假设数据是高斯分布 (Gaussian)。但金融数据是**肥尾 (Fat-tailed)** 分布。
  - 当发生**黑天鹅 (Black Swan)** 或**结构性崩塌 (Structural Break)** 时，价差可能拉大到 10 倍标准差。
  - 线性力会产生巨大的交易信号，导致在"接飞刀"时疯狂加仓，最终爆仓。

### 解决方案：Huber 势能 / 饱和力
引入**鲁棒统计**思想，将势能函数改为 Huber 函数或 Pseudo-Huber 函数。

**Huber 势能函数：**
$$E(f) = \sum_{(i,j) \in \mathcal{E}} A_{ij} \cdot \rho(f_i - f_j)$$

**对应的恢复力 (Robust Force)：**
$$F_i = -\sum_{j} A_{ij} \cdot \psi(f_i - f_j)$$
其中 $\psi(x) = \rho'(x)$ 是影响函数 (Influence Function)。

---

## 📉 数学原理与物理意义

### 1. 两种状态的平滑切换

我们使用 $\tanh(x)$ 或 Softsign 作为 $\psi(x)$ 的近似：

| 偏差状态 | 数学表现 | 物理模型 | 交易行为 |
|---------|---------|---------|---------|
| **正常震荡** (Small $\Delta$) | $\psi(x) \approx x$ | 线性弹簧 (Harmonic Oscillator) | **高精度均值回归**：偏差越大，力度越大，积极套利。 |
| **异常极端** (Large $\Delta$) | $\psi(x) \to \pm 1$ | 恒力 (Constant Force) | **风险饱和 (Saturation)**：虽然偏离极大，但模型认为"这可能是个错误/黑天鹅"，信号不再增加，限制最大仓位，防止爆仓。 |

### 2. 改进后的控制器逻辑

**原版 (v3.0):**
```python
force = -2 * L @ f  # 线性聚合
# 等价于 force_i = -2 * sum(A_ij * (f_i - f_j))
```

**鲁棒版 (v3.1):**
```python
# 推荐：Edge-wise Robustness (更细腻)
# 对每一对关系单独应用饱和，防止单个邻居的异常值污染整体
robust_force_i = sum( A_ij * tanh(gamma * (f_j - f_i)) )

# 简化版：Global Robustness (用户建议)
# 对总合力进行饱和
linear_force = -2 * L @ f
robust_force = tanh(gamma * linear_force)
```

**我们采用一种混合策略：**
1. **Edge-wise 饱和：** 计算 `message = tanh(f_j - f_i)`。这保证了一个坏掉的邻居不会把整条船拖翻（Robust Estimation）。
2. **Global 归一化：** 最终信号也进行范围限制。

---

## 💻 Pseudocode v3.1

```python
def RobustVectorController(L, f_current, f_previous, A,
                           gamma=1.0,          # 饱和系数，控制线性区域宽度
                           threshold_z=2.0,
                           min_confidence=0.1):
    """
    v3.1 鲁棒相空间控制器

    Upgrade:
    - 使用 Edge-wise Tanh 势能，防止单个异常值污染
    - 信号强度自动饱和，天然风险控制
    """
    N = len(f_current)

    # ========== Step 1: 计算鲁棒恢复力 (Edge-wise) ==========

    # 原始可加性力：F_i = sum_j A_ij * (f_j - f_i)
    # 鲁棒力：F_i = sum_j A_ij * tanh( gamma * (f_j - f_i) )

    robust_force = np.zeros(N)

    for i in range(N):
        # 找到邻居
        neighbors = np.where(A[i, :] > 0)[0]
        if len(neighbors) == 0:
            continue

        weights = A[i, neighbors]
        diffs = f_current[neighbors] - f_current[i]

        # 核心修改：非线性激活
        # 当 diffs 小时，近似 linear
        # 当 diffs 大时，饱和为 +/- 1
        nonlinear_diffs = np.tanh(gamma * diffs)

        # 加权求和
        robust_force[i] = np.sum(weights * nonlinear_diffs)

    # 既然 force 指向 f 增加的方向（上坡），我们需要反过指交易方向
    # 这里的 diff 是 (f_j - f_i)，如果 f_j > f_i，我被拉向高处
    # 所以 robust_force 指向"目标值"。
    # 交易方向 = robust_force (即：如果不满，它会推着我走)
    # Wait, check directions:
    # Potential E ~ (f_i - f_j)^2
    # Force on i = - dE/df_i = - 2(f_i - f_j) = 2(f_j - f_i)
    # Positive force means push f_i UP.
    # So Force IS the trading direction (Long).

    force_vector = robust_force

    # ========== Step 2: 速度与动量 ==========

    velocity = f_current - f_previous

    # ========== Step 3: 置信度与Z-score ==========

    # 计算有效邻居波动率 (Robust Scaling)
    # 使用 MAD (Median Absolute Deviation) 替代 Std 进一步增强鲁棒性
    neighbor_mad = np.zeros(N)
    confidence = np.zeros(N)

    for i in range(N):
        neighbors = A[i, :] > 0
        if not neighbors.any():
            neighbor_mad[i] = 1.0
            continue

        local_diffs = f_current[neighbors] - f_current[i]
        neighbor_mad[i] = np.median(np.abs(local_diffs)) + 1e-6

        confidence[i] = A[i, neighbors].sum() / neighbor_mad[i]

    # ========== Step 4: 交易逻辑 ==========

    # 信号强度 (已饱和)
    # tanh 已经把力限制在一定范围内，不需要再除以 std 了吗？
    # 不，除以 std 可以让 gamma 的 scale 适应波动率
    # 这里我们直接用 robust_force 作为主要信号

    # 过滤条件
    # 1. 回归中：Force * Velocity > 0
    is_converging = (force_vector * velocity) > 0

    # 2. 显著性：虽然力有 tanh 限制，但如果是微小噪音，tanh值也很小
    # 我们可以设定一个最小力阈值
    is_significant = np.abs(force_vector) > (threshold_z * 0.01) # 需要校准

    # 3. 置信度
    is_confident = confidence > min_confidence

    active_mask = is_converging & is_significant & is_confident

    # ========== Step 5: 输出 ==========

    weights = np.zeros(N)
    weights[active_mask] = force_vector[active_mask] * np.sqrt(confidence[active_mask])

    # 市场中性化 & 归一化
    weights -= weights.mean()
    weights /= (np.sum(np.abs(weights)) + 1e-8)

    return weights
```

## ⚖️ 评价结论

这个修正（鲁棒化）对 Deep Potential Strategy 具有**决定性意义**：

1.  **生存能力 (Survivability):** 它解决了基于均值回归策略最大的死穴 —— 肥尾风险。防止在资产发生结构性崩溃时把所有资金加进去。
2.  **物理一致性 (Consistency):** 在小偏差下退化为 Linear Laplacian，保留了 v3.0 的所有数学优美性；在大偏差下变为 L1 Laplacian，符合鲁棒统计原则。
3.  **实现简单:** 只需要在聚合前加一个 `tanh`，计算代价极低。

**建议：** 将此作为默认的生产环境 Controller 实现。
