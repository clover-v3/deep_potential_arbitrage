"""
Evaluation Metrics for Graph Recovery

Implements various metrics to evaluate the quality of learned graph structures.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering


def evaluate_graph_recovery(
    A_pred: np.ndarray,
    A_true: np.ndarray,
    cluster_labels_true: np.ndarray,
    threshold: float = 0.1
) -> Dict:
    """
    评估图恢复质量

    **重点：簇识别质量（NMI, ARI, Purity）**
    边预测指标作为次要参考

    Args:
        A_pred: (N, N) 预测的邻接矩阵
        A_true: (N, N) 真实的邻接矩阵
        cluster_labels_true: (N,) 真实的簇标签
        threshold: 二值化阈值

    Returns:
        metrics: 包含各种评估指标的字典
    """
    N = len(A_true)

    # ========== 主要指标：簇质量 ==========

    n_clusters = len(np.unique(cluster_labels_true))

    try:
        # 用预测的图进行谱聚类
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        cluster_labels_pred = clustering.fit_predict(A_pred + 1e-6)

        # 计算NMI和ARI（最重要的指标）
        nmi = normalized_mutual_info_score(cluster_labels_true, cluster_labels_pred)
        ari = adjusted_rand_score(cluster_labels_true, cluster_labels_pred)

        # 计算簇纯度
        purity = compute_cluster_purity(cluster_labels_true, cluster_labels_pred)

    except:
        nmi = 0.0
        ari = 0.0
        purity = 0.0

    # ========== 次要指标：边预测 ==========

    # 1. 二值化预测图
    A_pred_binary = (A_pred > threshold).astype(float)
    A_true_binary = (A_true > 0).astype(float)

    # 去除对角线
    np.fill_diagonal(A_pred_binary, 0)
    np.fill_diagonal(A_true_binary, 0)

    # 2. 边级别的精度/召回率/F1
    # 只考虑上三角（因为是对称矩阵）
    mask = np.triu(np.ones((N, N)), k=1).astype(bool)

    pred_edges = A_pred_binary[mask]
    true_edges = A_true_binary[mask]

    # True Positives, False Positives, False Negatives
    TP = np.sum((pred_edges == 1) & (true_edges == 1))
    FP = np.sum((pred_edges == 1) & (true_edges == 0))
    FN = np.sum((pred_edges == 0) & (true_edges == 1))
    TN = np.sum((pred_edges == 0) & (true_edges == 0))

    # 计算指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # 3. 矩阵误差（Frobenius范数）
    frobenius_error = np.linalg.norm(A_pred - A_true, 'fro')
    frobenius_error_normalized = frobenius_error / np.linalg.norm(A_true, 'fro')

    # 4. 权重相关性（对于加权图）
    true_edge_mask = (A_true > 0)
    if true_edge_mask.sum() > 0:
        true_weights = A_true[true_edge_mask]
        pred_weights = A_pred[true_edge_mask]

        if np.std(true_weights) > 1e-8 and np.std(pred_weights) > 1e-8:
            weight_correlation = np.corrcoef(true_weights, pred_weights)[0, 1]
        else:
            weight_correlation = 0.0
    else:
        weight_correlation = 0.0

    return {
        # ========== 主要指标：簇质量 ==========
        'nmi': nmi,                    # 最重要
        'ari': ari,                    # 最重要
        'purity': purity,              # 重要

        # ========== 次要指标：边预测 ==========
        'edge_precision': precision,
        'edge_recall': recall,
        'edge_f1': f1,
        'edge_accuracy': accuracy,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),

        # ========== 其他指标 ==========
        'frobenius_error': frobenius_error,
        'frobenius_error_normalized': frobenius_error_normalized,
        'weight_correlation': weight_correlation,
    }


def compute_cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    计算簇纯度

    Purity = (1/N) * sum_k max_j |cluster_k ∩ class_j|

    Args:
        labels_true: 真实标签
        labels_pred: 预测标签

    Returns:
        purity: 0-1之间的纯度分数
    """
    from scipy.optimize import linear_sum_assignment

    # 构建混淆矩阵
    n_samples = len(labels_true)
    n_clusters = len(np.unique(labels_pred))
    n_classes = len(np.unique(labels_true))

    confusion_matrix = np.zeros((n_clusters, n_classes))

    for i in range(n_samples):
        confusion_matrix[labels_pred[i], labels_true[i]] += 1

    # 对每个簇，找到最多的类
    purity = np.sum(np.max(confusion_matrix, axis=1)) / n_samples

    return purity


def compute_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    axis: int = 1
) -> np.ndarray:
    """
    计算Information Coefficient（信息系数）

    IC是截面相关系数，衡量预测信号与实际收益的相关性

    Args:
        predictions: (T, N) 预测值
        targets: (T, N) 目标值
        axis: 计算相关性的轴（1=截面，0=时间序列）

    Returns:
        ic: (T,) 或 (N,) IC序列
    """
    if axis == 1:
        # 截面IC（每个时间点）
        T = len(predictions)
        ic = np.zeros(T)
        for t in range(T):
            if np.std(predictions[t]) > 1e-8 and np.std(targets[t]) > 1e-8:
                ic[t] = np.corrcoef(predictions[t], targets[t])[0, 1]
    else:
        # 时间序列IC（每只股票）
        N = predictions.shape[1]
        ic = np.zeros(N)
        for i in range(N):
            if np.std(predictions[:, i]) > 1e-8 and np.std(targets[:, i]) > 1e-8:
                ic[i] = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]

    return ic


def print_evaluation_report(metrics: Dict, title: str = "Evaluation Report"):
    """
    打印评估报告
    """
    print("=" * 60)
    print(title)
    print("=" * 60)

    print("\n【边预测性能】")
    print(f"  Precision:  {metrics['edge_precision']:.4f}")
    print(f"  Recall:     {metrics['edge_recall']:.4f}")
    print(f"  F1 Score:   {metrics['edge_f1']:.4f}")
    print(f"  Accuracy:   {metrics['edge_accuracy']:.4f}")
    print(f"  TP/FP/FN/TN: {metrics['TP']}/{metrics['FP']}/{metrics['FN']}/{metrics['TN']}")

    print("\n【簇质量】")
    print(f"  NMI:        {metrics['nmi']:.4f}")
    print(f"  ARI:        {metrics['ari']:.4f}")

    print("\n【矩阵误差】")
    print(f"  Frobenius Error:       {metrics['frobenius_error']:.4f}")
    print(f"  Normalized Error:      {metrics['frobenius_error_normalized']:.4f}")
    print(f"  Weight Correlation:    {metrics['weight_correlation']:.4f}")

    print("\n【通过标准】")
    passed = []
    failed = []

    if metrics['edge_f1'] > 0.8:
        passed.append("✅ F1 > 0.8")
    else:
        failed.append(f"❌ F1 = {metrics['edge_f1']:.4f} < 0.8")

    if metrics['nmi'] > 0.7:
        passed.append("✅ NMI > 0.7")
    else:
        failed.append(f"❌ NMI = {metrics['nmi']:.4f} < 0.7")

    for item in passed:
        print(f"  {item}")
    for item in failed:
        print(f"  {item}")

    print("\n" + "=" * 60)


# ============================================================================
# 示例使用
# ============================================================================

if __name__ == "__main__":
    from synthetic_data import generate_complete_dataset
    from graph_learning import correlation_graph, dynamics_based_graph

    print("生成合成数据...")
    dataset = generate_complete_dataset(
        n_stocks=30,
        n_clusters=3,
        n_timesteps=500,
        seed=42
    )

    f_series = dataset['f_series']
    A_true = dataset['A_true']
    cluster_labels = dataset['cluster_labels']

    print("\n测试图学习方法...")

    # 方法1：相关性图
    print("\n=== 方法1: 相关性图 ===")
    A_corr = correlation_graph(f_series, threshold=0.3)
    metrics_corr = evaluate_graph_recovery(A_corr, A_true, cluster_labels)
    print_evaluation_report(metrics_corr, "相关性图评估")

    # 方法2：动力学图
    print("\n=== 方法2: 动力学图 ===")
    A_dyn, _ = dynamics_based_graph(f_series, method='regression', alpha=0.01)
    metrics_dyn = evaluate_graph_recovery(A_dyn, A_true, cluster_labels)
    print_evaluation_report(metrics_dyn, "动力学图评估")

    # 对比
    print("\n=== 对比总结 ===")
    print(f"相关性图 F1: {metrics_corr['edge_f1']:.4f}")
    print(f"动力学图 F1: {metrics_dyn['edge_f1']:.4f}")
    print(f"改进: {(metrics_dyn['edge_f1'] - metrics_corr['edge_f1']):.4f}")
