"""
Synthetic Data Generator for Laplacian Dynamics Validation

This module generates synthetic stock price data that follows Laplacian dynamics,
with known ground-truth graph structure for validation purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering


def create_ground_truth_graph(
    n_stocks: int = 50,
    n_clusters: int = 5,
    intra_cluster_weight_range: Tuple[float, float] = (0.8, 1.0), # Increased coupling
    inter_cluster_weight_range: Tuple[float, float] = (0.0, 0.1),
    intra_cluster_density: float = 1.0, # Make clusters dense/obvious
    inter_cluster_density: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建块对角结构的真实图（ground truth）

    Args:
        n_stocks: 股票数量
        n_clusters: 簇数量
        intra_cluster_weight_range: 簇内边权重范围
        inter_cluster_weight_range: 簇间边权重范围
        intra_cluster_density: 簇内连接密度（0-1）
        inter_cluster_density: 簇间连接密度（0-1）
        seed: 随机种子

    Returns:
        A: (N, N) 邻接矩阵（对称，非负）
        cluster_labels: (N,) 簇标签
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. 分配股票到簇（尽量均匀）
    stocks_per_cluster = n_stocks // n_clusters
    cluster_labels = np.repeat(np.arange(n_clusters), stocks_per_cluster)

    # 处理余数
    remainder = n_stocks - len(cluster_labels)
    if remainder > 0:
        cluster_labels = np.concatenate([
            cluster_labels,
            np.arange(remainder)
        ])

    # 2. 初始化邻接矩阵
    A = np.zeros((n_stocks, n_stocks))

    # 3. 填充簇内连接
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        n_in_cluster = len(cluster_indices)

        # 生成簇内连接
        for i in range(n_in_cluster):
            for j in range(i + 1, n_in_cluster):
                # 以一定概率连接
                if np.random.rand() < intra_cluster_density:
                    # 权重从指定范围采样
                    weight = np.random.uniform(*intra_cluster_weight_range)
                    idx_i = cluster_indices[i]
                    idx_j = cluster_indices[j]
                    A[idx_i, idx_j] = weight
                    A[idx_j, idx_i] = weight  # 对称

    # 4. 填充簇间连接（稀疏）
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            # 如果不在同一簇
            if cluster_labels[i] != cluster_labels[j]:
                # 以较低概率连接
                if np.random.rand() < inter_cluster_density:
                    weight = np.random.uniform(*inter_cluster_weight_range)
                    A[i, j] = weight
                    A[j, i] = weight

    return A, cluster_labels


def compute_laplacian(A: np.ndarray) -> np.ndarray:
    """
    计算拉普拉斯矩阵

    L = D - A
    其中 D 是度矩阵（对角矩阵）

    Args:
        A: (N, N) 邻接矩阵

    Returns:
        L: (N, N) 拉普拉斯矩阵
    """
    # 计算度（每行的和）
    degree = A.sum(axis=1)

    # 度矩阵
    D = np.diag(degree)

    # 拉普拉斯矩阵
    L = D - A

    return L


def generate_laplacian_dynamics(
    A: np.ndarray,
    n_timesteps: int = 1000,
    dt: float = 0.01,
    noise_std: float = 0.02,
    noise_type: str = 'gaussian', # 'gaussian' or 'student-t'
    initial_state: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据拉普拉斯动力学生成时间序列

    动力学方程：
        df/dt = -L * f + noise

    离散化（欧拉法）：
        f_{t+1} = f_t - dt * L * f_t + sqrt(dt) * noise

    Args:
        A: (N, N) 邻接矩阵
        n_timesteps: 时间步数
        dt: 时间步长
        noise_std: 噪声标准差
        noise_type: 噪声分布类型 ('gaussian', 'student-t')
        initial_state: 初始状态
        seed: 随机种子

    Returns:
        f_series: (T, N) 状态时间序列
        L: (N, N) 拉普拉斯矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(A)

    # 计算拉普拉斯矩阵
    L = compute_laplacian(A)

    # 初始化状态
    if initial_state is None:
        f_0 = np.random.randn(N)
    else:
        f_0 = initial_state.copy()

    # 存储时间序列
    f_series = np.zeros((n_timesteps, N))
    f_series[0] = f_0

    # 欧拉法迭代
    for t in range(1, n_timesteps):
        # 确定性部分：-L * f
        deterministic = -L @ f_series[t-1]

        # 随机部分：噪声
        if noise_type == 'gaussian':
            raw_noise = np.random.randn(N)
        elif noise_type == 'student-t':
            # df=3 has finite variance (heavy tails)
            # Normalize to have approx unit variance for fair comparison
            # Var(t_v) = v / (v-2) for v > 2. For v=3, Var=3.
            # So divide by sqrt(3) to get std ~ 1
            raw_noise = np.random.standard_t(df=3, size=N) / np.sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        noise = raw_noise * noise_std * np.sqrt(dt)

        # 更新
        f_series[t] = f_series[t-1] + dt * deterministic + noise

    return f_series, L


def convert_to_prices(
    f_series: np.ndarray,
    add_market_trend: bool = True,
    trend_mu: float = 0.0001,
    trend_sigma: float = 0.01,
    initial_price: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将状态序列转换为价格序列

    f 是去趋势的log价格，我们需要：
    1. 添加市场趋势（可选）
    2. 转换为价格

    Args:
        f_series: (T, N) 状态时间序列
        add_market_trend: 是否添加市场趋势
        trend_mu: 市场趋势均值（每步）
        trend_sigma: 市场趋势波动
        initial_price: 初始价格

    Returns:
        prices: (T, N) 价格序列
        log_prices: (T, N) log价格序列
    """
    T, N = f_series.shape

    # 1. 如果需要，添加市场趋势
    if add_market_trend:
        # 生成市场趋势（随机游走）
        market_trend = np.cumsum(
            np.random.randn(T) * trend_sigma + trend_mu
        )
        # 广播到所有股票
        log_prices = f_series + market_trend[:, np.newaxis]
    else:
        log_prices = f_series.copy()

    # 2. 转换为价格
    # P_t = P_0 * exp(log_price_t)
    prices = initial_price * np.exp(log_prices)

    return prices, log_prices


def generate_complete_dataset(
    n_stocks: int = 50,
    n_clusters: int = 5,
    n_timesteps: int = 1000,
    dt: float = 0.01,
    noise_std: float = 0.01, # Reduced noise (was 0.02)
    noise_type: str = 'gaussian',
    seed: Optional[int] = None
) -> Dict:
    """
    生成完整的合成数据集

    包括：
    - 真实图结构
    - 状态演化
    - 价格序列

    Args:
        n_stocks: 股票数量
        n_clusters: 簇数量
        n_timesteps: 时间步数
        dt: 时间步长
        noise_std: 噪声标准差
        seed: 随机种子

    Returns:
        dataset: 包含所有数据的字典
    """
    A_true, cluster_labels = create_ground_truth_graph(
        n_stocks=n_stocks,
        n_clusters=n_clusters,
        seed=seed
    )

    # 2. 生成动力学演化
    f_series, L_true = generate_laplacian_dynamics(
        A=A_true,
        n_timesteps=n_timesteps,
        dt=dt,
        noise_std=noise_std,
        noise_type=noise_type,
        seed=seed
    )

    # 2.5 添加簇特定的趋势（模拟不同板块的分化）
    # 每个簇有一个自己的随机游走趋势
    if seed is not None:
        np.random.seed(seed + 1) # 使用不同的种子

    cluster_trend_std = 0.0005 # 簇趋势的波动率
    cluster_trends = np.zeros((n_timesteps, n_clusters))
    for k in range(n_clusters):
        cluster_trends[:, k] = np.cumsum(np.random.randn(n_timesteps) * cluster_trend_std)

    # 将簇趋势叠加到各自的股票上
    # f_series 是去趋势的状态，这里我们把板块趋势加进去，看看GNN能不能处理
    # 注意：理论上VectorController应该对这种共同趋势免疫（因为它看的是相对差异）
    # 但对于Correlation Graph，这会增强簇内相关性
    f_with_trends = f_series.copy()
    for k in range(n_clusters):
        mask = (cluster_labels == k)
        f_with_trends[:, mask] += cluster_trends[:, k][:, np.newaxis]

    # 3. 转换为价格
    # 使用带有簇趋势的 f 系列
    prices, log_prices = convert_to_prices(f_with_trends)

    # 4. 计算收益率
    returns = np.diff(log_prices, axis=0)

    return {
        'A_true': A_true,
        'L_true': L_true,
        'cluster_labels': cluster_labels,
        'f_series': f_series,
        'prices': prices,
        'log_prices': log_prices,
        'returns': returns,
        'n_stocks': n_stocks,
        'n_clusters': n_clusters,
        'n_timesteps': n_timesteps,
        'dt': dt,
        'noise_std': noise_std,
    }


def visualize_synthetic_data(dataset: Dict, save_path: Optional[str] = None):
    """
    可视化合成数据

    包括：
    1. 真实图的邻接矩阵
    2. 状态演化
    3. 价格演化
    4. 簇结构
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 邻接矩阵热图
    ax = axes[0, 0]
    im = ax.imshow(dataset['A_true'], cmap='viridis', aspect='auto')
    ax.set_title('Ground Truth Adjacency Matrix')
    ax.set_xlabel('Stock Index')
    ax.set_ylabel('Stock Index')
    plt.colorbar(im, ax=ax)

    # 2. 状态演化（前10只股票）
    ax = axes[0, 1]
    for i in range(min(10, dataset['n_stocks'])):
        ax.plot(dataset['f_series'][:, i], alpha=0.7, linewidth=0.8)
    ax.set_title('State Evolution (First 10 Stocks)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State f')
    ax.grid(True, alpha=0.3)

    # 3. 价格演化（前10只股票）
    ax = axes[1, 0]
    for i in range(min(10, dataset['n_stocks'])):
        ax.plot(dataset['prices'][:, i], alpha=0.7, linewidth=0.8)
    ax.set_title('Price Evolution (First 10 Stocks)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)

    # 4. 簇结构（按簇排序的邻接矩阵）
    ax = axes[1, 1]
    # 按簇标签排序
    sorted_indices = np.argsort(dataset['cluster_labels'])
    A_sorted = dataset['A_true'][sorted_indices][:, sorted_indices]
    im = ax.imshow(A_sorted, cmap='viridis', aspect='auto')
    ax.set_title('Adjacency Matrix (Sorted by Cluster)')
    ax.set_xlabel('Stock Index (Sorted)')
    ax.set_ylabel('Stock Index (Sorted)')
    plt.colorbar(im, ax=ax)

    # 添加簇边界线
    cluster_sizes = np.bincount(dataset['cluster_labels'])
    boundaries = np.cumsum(cluster_sizes)[:-1]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color='red', linestyle='--', linewidth=1)
        ax.axvline(boundary - 0.5, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    plt.show()


# ============================================================================
# 示例使用
# ============================================================================

if __name__ == "__main__":
    print("生成合成数据集...")

    # 生成数据
    dataset = generate_complete_dataset(
        n_stocks=50,
        n_clusters=5,
        n_timesteps=1000,
        dt=0.01,
        noise_std=0.02,
        seed=42
    )

    print(f"\n数据集信息:")
    print(f"  股票数量: {dataset['n_stocks']}")
    print(f"  簇数量: {dataset['n_clusters']}")
    print(f"  时间步数: {dataset['n_timesteps']}")
    print(f"  邻接矩阵密度: {(dataset['A_true'] > 0).sum() / dataset['n_stocks']**2:.2%}")

    # 分析簇结构
    print(f"\n簇结构:")
    for cluster_id in range(dataset['n_clusters']):
        n_in_cluster = (dataset['cluster_labels'] == cluster_id).sum()
        print(f"  簇 {cluster_id}: {n_in_cluster} 只股票")

    # 可视化
    print("\n生成可视化...")
    visualize_synthetic_data(dataset)

    # 保存数据（可选）
    # np.savez('synthetic_data.npz', **dataset)
    # print("\n数据已保存到: synthetic_data.npz")
