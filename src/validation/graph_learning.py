"""
Graph Learning Methods

Implements various methods for learning graph structure from time series data.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, Ridge


def correlation_graph(
    f_series: np.ndarray,
    threshold: float = 0.5,
    method: str = 'pearson'
) -> np.ndarray:
    """
    基于相关系数构建图（Baseline方法）

    Args:
        f_series: (T, N) 时间序列
        threshold: 相关系数阈值
        method: 'pearson' or 'spearman'

    Returns:
        A: (N, N) 邻接矩阵
    """
    T, N = f_series.shape

    # 计算相关系数矩阵
    if method == 'pearson':
        corr_matrix = np.corrcoef(f_series.T)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr_matrix, _ = spearmanr(f_series, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 取绝对值
    A = np.abs(corr_matrix)

    # 阈值截断
    A[A < threshold] = 0

    # 去除自环
    np.fill_diagonal(A, 0)

    return A


def dynamics_based_graph(
    f_series: np.ndarray,
    method: str = 'regression',
    alpha: float = 0.01,
    dt: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从动力学方程反推图结构（我们的核心方法）

    已知：df/dt ≈ -L * f
    求解：L (进而得到 A)

    Args:
        f_series: (T, N) 状态时间序列
        method: 'regression' or 'least_squares'
        alpha: 正则化系数（用于Lasso）
        dt: 时间步长

    Returns:
        A: (N, N) 邻接矩阵
        L: (N, N) 拉普拉斯矩阵
    """
    T, N = f_series.shape

    # 1. 计算 df/dt
    df_dt = np.diff(f_series, axis=0) / dt  # (T-1, N)
    f_t = f_series[:-1]  # (T-1, N)

    # 2. 求解 df/dt = -L * f
    # 对每个股票i，求解：df_i/dt = -sum_j L_ij * f_j

    L = np.zeros((N, N))

    if method == 'regression':
        # 用Lasso回归（稀疏化）
        for i in range(N):
            # 目标：df_i/dt
            y = df_dt[:, i]

            # 特征：-f (所有股票的状态)
            X = -f_t

            # Lasso回归
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)
            model.fit(X, y)

            # L的第i行
            L[i, :] = model.coef_

    elif method == 'least_squares':
        # 最小二乘法（无正则化）
        # df/dt = -L * f
        # 展开：(T-1, N) = (T-1, N) @ (N, N)^T
        # 求解：L^T = (f^T f)^{-1} f^T (-df/dt)

        # 为了数值稳定，添加小的正则化
        reg = 1e-6 * np.eye(N)
        L = -np.linalg.solve(f_t.T @ f_t + reg, f_t.T @ df_dt).T

    else:
        raise ValueError(f"Unknown method: {method}")

    # 3. 从 L 恢复 A
    # L = D - A，其中 D_ii = sum_j A_ij
    # 假设：L_ii = D_ii（对角线）
    # 则：A_ij = -L_ij (i != j)

    A = -L.copy()
    np.fill_diagonal(A, 0)

    # 4. 对称化（取平均）
    A = (A + A.T) / 2

    # 5. 确保非负
    A = np.maximum(A, 0)

    # 6. 重新计算L（确保一致性）
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A

    return A, L


def granger_causality_graph(
    f_series: np.ndarray,
    max_lag: int = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """
    基于格兰杰因果性构建图

    测试：f_j 是否格兰杰因果于 f_i
    如果是，则 A_ij > 0

    Args:
        f_series: (T, N) 时间序列
        max_lag: 最大滞后阶数
        alpha: 显著性水平

    Returns:
        A: (N, N) 邻接矩阵（可能非对称）
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    T, N = f_series.shape
    A = np.zeros((N, N))

    # 对每对 (i, j) 测试
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            try:
                # 测试 j 是否格兰杰因果于 i
                # 数据格式：[y, x] 即 [f_i, f_j]
                data = np.column_stack([f_series[:, i], f_series[:, j]])

                # 运行测试
                result = grangercausalitytests(data, max_lag, verbose=False)

                # 提取p值（使用F检验）
                p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                min_p_value = min(p_values)

                # 如果显著，设置边
                if min_p_value < alpha:
                    A[i, j] = 1.0 - min_p_value  # 权重 = 1 - p值

            except:
                # 如果测试失败，跳过
                continue

    return A


def topk_sparsify(A: np.ndarray, k: int) -> np.ndarray:
    """
    Top-K稀疏化：每个节点只保留k个最强的连接

    Args:
        A: (N, N) 邻接矩阵
        k: 每个节点保留的边数

    Returns:
        A_sparse: (N, N) 稀疏化后的邻接矩阵
    """
    N = len(A)
    A_sparse = np.zeros_like(A)

    for i in range(N):
        # 找到第i行最大的k个值
        row = A[i, :]
        if k >= N - 1:
            A_sparse[i, :] = row
        else:
            # 获取top-k的索引（排除自己）
            indices = np.argsort(row)[::-1]
            indices = indices[indices != i][:k]
            A_sparse[i, indices] = row[indices]

    # 对称化
    A_sparse = (A_sparse + A_sparse.T) / 2

    return A_sparse


# ============================================================================
# 示例使用
# ============================================================================

if __name__ == "__main__":
    # 加载合成数据
    from synthetic_data import generate_complete_dataset

    print("生成合成数据...")
    dataset = generate_complete_dataset(
        n_stocks=30,  # 减少股票数量以加快测试
        n_clusters=3,
        n_timesteps=500,
        seed=42
    )

    f_series = dataset['f_series']
    A_true = dataset['A_true']

    print(f"\n真实图密度: {(A_true > 0).sum() / 30**2:.2%}")

    # 测试不同方法
    print("\n测试图学习方法...")

    # 方法1：相关性图
    print("\n1. 相关性图...")
    A_corr = correlation_graph(f_series, threshold=0.3)
    print(f"   密度: {(A_corr > 0).sum() / 30**2:.2%}")

    # 方法2：动力学图
    print("\n2. 动力学图...")
    A_dyn, L_dyn = dynamics_based_graph(f_series, method='regression', alpha=0.01)
    print(f"   密度: {(A_dyn > 0).sum() / 30**2:.2%}")

    # 方法3：Top-K稀疏化
    print("\n3. Top-K稀疏化（k=5）...")
    A_dyn_sparse = topk_sparsify(A_dyn, k=5)
    print(f"   密度: {(A_dyn_sparse > 0).sum() / 30**2:.2%}")

    print("\n完成！")
