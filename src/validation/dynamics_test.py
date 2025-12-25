"""
Dynamics Testing Module

Tests whether Laplacian dynamics holds in the data:
- Force-return correlation
- Dynamics regression
- Prediction accuracy
- Graph stability
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


def test_force_return_correlation(
    f_series: np.ndarray,
    L: np.ndarray,
    lag: int = 1,
    min_samples: int = 50
) -> Dict:
    """
    测试恢复力与未来收益的相关性

    核心问题：F_t = -2Lf_t 是否预测 Δf_{t+lag}？

    Args:
        f_series: (T, N) 状态时间序列
        L: (N, N) 拉普拉斯矩阵
        lag: 预测步长
        min_samples: 最小样本数

    Returns:
        results: 包含相关性、显著性等统计量的字典
    """
    T, N = f_series.shape

    if T < min_samples + lag:
        raise ValueError(f"Time series too short: {T} < {min_samples + lag}")

    ic_series = []  # Information Coefficient时间序列

    # 对每个时间点计算截面IC
    for t in range(T - lag):
        # 当前恢复力
        f_t = f_series[t]
        F_t = -2 * L @ f_t

        # 未来收益
        delta_f = f_series[t + lag] - f_t

        # 计算截面相关性（IC）
        # 只有当两个序列都有变化时才计算
        if np.std(F_t) > 1e-8 and np.std(delta_f) > 1e-8:
            ic = np.corrcoef(F_t, delta_f)[0, 1]
            ic_series.append(ic)
        else:
            ic_series.append(0.0)

    ic_series = np.array(ic_series)

    # 统计检验：IC均值是否显著不为0
    t_stat, p_value = stats.ttest_1samp(ic_series, 0)

    # 计算信息比率（IR）
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    ic_ir = ic_mean / (ic_std + 1e-8)

    # 胜率（IC > 0的比例）
    win_rate = (ic_series > 0).mean()

    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        't_statistic': t_stat,
        'p_value': p_value,
        'win_rate': win_rate,
        'ic_series': ic_series,
        'n_samples': len(ic_series),
    }


def test_dynamics_regression(
    f_series: np.ndarray,
    L: np.ndarray,
    dt: float = 0.01
) -> Dict:
    """
    测试动力学方程的拟合优度

    模型：df/dt = -L * f + epsilon

    Args:
        f_series: (T, N) 状态时间序列
        L: (N, N) 拉普拉斯矩阵
        dt: 时间步长

    Returns:
        results: R²、系数等
    """
    T, N = f_series.shape

    # 1. 计算 df/dt
    df_dt = np.diff(f_series, axis=0) / dt  # (T-1, N)
    f_t = f_series[:-1]  # (T-1, N)

    # 2. 计算 -L * f
    Lf = -L @ f_t.T  # (N, T-1)
    Lf = Lf.T  # (T-1, N)

    # 3. 线性回归：df/dt = beta * Lf + epsilon
    # 展平为向量
    y = df_dt.flatten()
    X = Lf.flatten()

    # 计算回归系数（应该接近1）
    beta = np.sum(X * y) / (np.sum(X * X) + 1e-8)

    # 预测值
    y_pred = beta * X

    # 计算R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-8)

    # 计算每个股票的R²
    r_squared_per_stock = np.zeros(N)
    for i in range(N):
        y_i = df_dt[:, i]
        X_i = Lf[:, i]
        y_pred_i = beta * X_i
        ss_res_i = np.sum((y_i - y_pred_i) ** 2)
        ss_tot_i = np.sum((y_i - np.mean(y_i)) ** 2)
        r_squared_per_stock[i] = 1 - ss_res_i / (ss_tot_i + 1e-8)

    return {
        'beta': beta,
        'r_squared': r_squared,
        'r_squared_per_stock': r_squared_per_stock,
        'r_squared_mean': np.mean(r_squared_per_stock),
        'r_squared_std': np.std(r_squared_per_stock),
    }


def test_prediction_accuracy(
    f_series: np.ndarray,
    L: np.ndarray,
    prediction_horizon: int = 5,
    dt: float = 0.01
) -> Dict:
    """
    测试用动力学方程预测未来状态的准确性

    预测：f_{t+h} ≈ f_t - h * dt * L * f_t
    真实：f_{t+h} (observed)

    Args:
        f_series: (T, N) 状态时间序列
        L: (N, N) 拉普拉斯矩阵
        prediction_horizon: 预测步数
        dt: 时间步长

    Returns:
        results: MSE、方向准确率等
    """
    T, N = f_series.shape

    mse_list = []
    mae_list = []
    direction_accuracy_list = []

    for t in range(T - prediction_horizon):
        # 当前状态
        f_t = f_series[t]

        # 预测未来状态（一阶欧拉法）
        f_pred = f_t - prediction_horizon * dt * L @ f_t

        # 真实未来状态
        f_true = f_series[t + prediction_horizon]

        # 计算误差
        mse = np.mean((f_pred - f_true) ** 2)
        mae = np.mean(np.abs(f_pred - f_true))

        # 计算方向准确率
        delta_pred = f_pred - f_t
        delta_true = f_true - f_t
        direction_correct = (delta_pred * delta_true > 0).mean()

        mse_list.append(mse)
        mae_list.append(mae)
        direction_accuracy_list.append(direction_correct)

    return {
        'mse_mean': np.mean(mse_list),
        'mse_std': np.std(mse_list),
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'direction_accuracy': np.mean(direction_accuracy_list),
        'direction_accuracy_std': np.std(direction_accuracy_list),
    }


def test_graph_stability(
    f_series: np.ndarray,
    graph_learning_func,
    window_size: int = 100,
    step: int = 20,
    **kwargs
) -> Dict:
    """
    测试学到的图是否随时间稳定

    Args:
        f_series: (T, N) 状态时间序列
        graph_learning_func: 图学习函数
        window_size: 窗口大小
        step: 步长
        **kwargs: 传递给graph_learning_func的参数

    Returns:
        results: 稳定性指标
    """
    T, N = f_series.shape

    # 滚动窗口学习图
    graphs = []
    timestamps = []

    for t in range(0, T - window_size, step):
        # 提取窗口数据
        window_data = f_series[t:t+window_size]

        # 学习图
        if 'dynamics_based_graph' in str(graph_learning_func):
            A, _ = graph_learning_func(window_data, **kwargs)
        else:
            A = graph_learning_func(window_data, **kwargs)

        graphs.append(A)
        timestamps.append(t)

    # 计算相邻图的相似度
    similarities = []

    for i in range(len(graphs) - 1):
        sim = graph_similarity(graphs[i], graphs[i+1])
        similarities.append(sim)

    similarities = np.array(similarities)

    return {
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'similarity_series': similarities,
        'timestamps': timestamps,
        'n_windows': len(graphs),
    }


def graph_similarity(A1: np.ndarray, A2: np.ndarray) -> float:
    """
    计算两个图的相似度

    使用多种指标的平均值：
    1. Frobenius相关性
    2. 边重叠率
    3. 加权边重叠

    Args:
        A1, A2: (N, N) 邻接矩阵

    Returns:
        similarity: 0-1之间的相似度
    """
    # 1. Frobenius相关性
    A1_flat = A1.flatten()
    A2_flat = A2.flatten()

    if np.std(A1_flat) > 1e-8 and np.std(A2_flat) > 1e-8:
        frobenius_corr = np.corrcoef(A1_flat, A2_flat)[0, 1]
    else:
        frobenius_corr = 0.0

    # 2. 边重叠率（Jaccard系数）
    edges1 = (A1 > 0)
    edges2 = (A2 > 0)
    intersection = (edges1 & edges2).sum()
    union = (edges1 | edges2).sum()

    if union > 0:
        jaccard = intersection / union
    else:
        jaccard = 0.0

    # 3. 加权边重叠（余弦相似度）
    dot_product = np.sum(A1 * A2)
    norm1 = np.sqrt(np.sum(A1 ** 2))
    norm2 = np.sqrt(np.sum(A2 ** 2))

    if norm1 > 1e-8 and norm2 > 1e-8:
        cosine_sim = dot_product / (norm1 * norm2)
    else:
        cosine_sim = 0.0

    # 综合相似度（平均）
    similarity = (frobenius_corr + jaccard + cosine_sim) / 3

    return similarity


def visualize_dynamics_test_results(results: Dict, save_path: Optional[str] = None):
    """
    可视化动力学测试结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. IC时间序列
    if 'ic_series' in results:
        ax = axes[0, 0]
        ic_series = results['ic_series']
        ax.plot(ic_series, alpha=0.7, linewidth=1)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(results['ic_mean'], color='green', linestyle='--',
                   alpha=0.7, label=f"Mean: {results['ic_mean']:.4f}")
        ax.set_xlabel('Time')
        ax.set_ylabel('IC')
        ax.set_title(f"Information Coefficient (p={results['p_value']:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. IC分布
    if 'ic_series' in results:
        ax = axes[0, 1]
        ax.hist(ic_series, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(results['ic_mean'], color='green', linestyle='--', linewidth=2)
        ax.set_xlabel('IC')
        ax.set_ylabel('Frequency')
        ax.set_title('IC Distribution')
        ax.grid(True, alpha=0.3)

    # 3. 累计IC
    if 'ic_series' in results:
        ax = axes[1, 0]
        cumulative_ic = np.cumsum(ic_series)
        ax.plot(cumulative_ic, linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative IC')
        ax.set_title('Cumulative Information Coefficient')
        ax.grid(True, alpha=0.3)

    # 4. 统计摘要
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    Dynamics Test Summary
    ═══════════════════════════════

    IC Statistics:
      Mean IC:        {results.get('ic_mean', 0):.6f}
      Std IC:         {results.get('ic_std', 0):.6f}
      IR (IC/Std):    {results.get('ic_ir', 0):.4f}
      Win Rate:       {results.get('win_rate', 0):.2%}

    Significance:
      t-statistic:    {results.get('t_statistic', 0):.3f}
      p-value:        {results.get('p_value', 1):.6f}
      Significant:    {'✅ Yes' if results.get('p_value', 1) < 0.05 else '❌ No'}

    Regression (if available):
      Beta:           {results.get('beta', 0):.4f}
      R²:             {results.get('r_squared', 0):.4f}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    plt.show()


# ============================================================================
# 示例使用
# ============================================================================

if __name__ == "__main__":
    from synthetic_data import generate_complete_dataset

    print("生成合成数据...")
    dataset = generate_complete_dataset(
        n_stocks=30,
        n_clusters=3,
        n_timesteps=500,
        dt=0.01,
        noise_std=0.02,
        seed=42
    )

    f_series = dataset['f_series']
    L_true = dataset['L_true']

    print("\n测试动力学假设...")

    # Test 1: Force-return correlation
    print("\n1. 恢复力-收益相关性测试...")
    results = test_force_return_correlation(f_series, L_true, lag=1)
    print(f"   IC均值: {results['ic_mean']:.6f}")
    print(f"   p值: {results['p_value']:.6f}")
    print(f"   显著: {'✅' if results['p_value'] < 0.05 else '❌'}")

    # Test 2: Dynamics regression
    print("\n2. 动力学回归测试...")
    reg_results = test_dynamics_regression(f_series, L_true, dt=0.01)
    print(f"   Beta: {reg_results['beta']:.4f} (应该接近1)")
    print(f"   R²: {reg_results['r_squared']:.4f}")

    # Test 3: Prediction accuracy
    print("\n3. 预测准确性测试...")
    pred_results = test_prediction_accuracy(f_series, L_true, prediction_horizon=5)
    print(f"   方向准确率: {pred_results['direction_accuracy']:.2%}")

    # Visualize
    print("\n生成可视化...")
    visualize_dynamics_test_results(results)
