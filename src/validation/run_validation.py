"""
Integrated Validation Script

Runs complete validation of both core assumptions:
1. Graph structure can capture Laplacian dynamics
2. Laplacian dynamics holds in financial markets
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_data import generate_complete_dataset, visualize_synthetic_data
from graph_learning import correlation_graph, dynamics_based_graph, topk_sparsify
from dynamics_test import (
    test_force_return_correlation,
    test_dynamics_regression,
    test_prediction_accuracy,
    test_graph_stability,
    visualize_dynamics_test_results
)
from metrics import evaluate_graph_recovery, print_evaluation_report


def run_assumption1_validation(
    n_stocks: int = 50,
    n_clusters: int = 5,
    n_timesteps: int = 1000,
    seed: int = 42,
    dt: float = 0.01,
    noise_std: float = 0.01,
    noise_type: str = 'gaussian'
) -> Dict:
    """
    验证假设1：图结构能捕捉拉普拉斯动力学

    测试：
    - 能否从合成数据中恢复真实图结构
    - 不同方法的对比

    Returns:
        results: 验证结果字典
    """
    print("\n" + "=" * 70)
    print("假设1验证：图结构学习能力")
    print("=" * 70)

    # 1. 生成合成数据
    print("\n[1/4] 生成合成数据...")
    dataset = generate_complete_dataset(
        n_stocks=n_stocks,
        n_clusters=n_clusters,
        n_timesteps=n_timesteps,
        seed=seed,
        dt=dt,
        noise_std=noise_std,
        noise_type=noise_type
    )

    print(f"  ✓ 生成 {n_stocks} 只股票，{n_clusters} 个簇，{n_timesteps} 个时间步")
    print(f"  ✓ 真实图密度: {(dataset['A_true'] > 0).sum() / n_stocks**2:.2%}")

    f_series = dataset['f_series']
    A_true = dataset['A_true']
    cluster_labels = dataset['cluster_labels']

    # 2. 测试不同图学习方法
    print("\n[2/4] 测试图学习方法...")

    methods = {}

    # 方法1：相关性图（Baseline）
    print("  - 相关性图...")
    A_corr = correlation_graph(f_series, threshold=0.3)
    methods['correlation'] = {
        'A': A_corr,
        'name': '相关性图',
    }

    # 方法2：动力学图（Baseline）
    print("  - 动力学图 (Baseline)...")
    A_dyn, L_dyn = dynamics_based_graph(f_series, method='regression', alpha=0.01)
    methods['dynamics'] = {
        'A': A_dyn,
        'L': L_dyn,
        'name': '动力学图(Baseline)',
    }

    # 方法3：GNN学习图（主要方法）
    print(f"  - GNN学习图 (Robust Training)...")
    from gnn_training import train_gnn_model, predict_graph

    # 训练GNN
    print("    开始训练GNN...")
    gnn_model, losses = train_gnn_model(
        f_series,
        n_epochs=50,  # 快速验证用50轮
        verbose=False
    )

    # 预测图
    A_gnn = predict_graph(gnn_model, f_series)

    # 简单的后处理（Top-K稀疏化，因为GNN输出全连接权重）
    A_gnn_sparse = topk_sparsify(A_gnn, k=n_stocks // n_clusters * 2)

    # 计算对应的L
    degree = A_gnn_sparse.sum(axis=1)
    L_gnn = np.diag(degree) - A_gnn_sparse

    methods['gnn'] = {
        'A': A_gnn_sparse,
        'L': L_gnn,
        'name': 'GNN (Ours)',
    }

    # 3. 评估每个方法
    print("\n[3/4] 评估图恢复质量...")

    results = {}
    for method_name, method_data in methods.items():
        print(f"\n  === {method_data['name']} ===")
        metrics = evaluate_graph_recovery(
            method_data['A'],
            A_true,
            cluster_labels,
            threshold=0.1
        )
        results[method_name] = metrics
        print_evaluation_report(metrics, method_data['name'])

    # 4. 对比总结
    print("\n[4/4] 对比总结")
    print("\n" + "-" * 80)
    print(f"{'方法':<20} {'F1':<10} {'NMI(簇)':<10} {'ARI(簇)':<10} {'Purity':<10}")
    print("-" * 80)

    for method_name, metrics in results.items():
        method_display = methods[method_name]['name']
        f1 = metrics['edge_f1']
        nmi = metrics['nmi']
        ari = metrics['ari']
        purity = metrics['purity']
        print(f"{method_display:<20} {f1:<10.4f} {nmi:<10.4f} {ari:<10.4f} {purity:<10.4f}")

    print("-" * 80)

    # 更新决策逻辑
    results['dataset'] = dataset
    results['methods'] = methods

    return results


def run_assumption2_validation(
    dataset: Dict,
    L: np.ndarray,
    method_name: str = "动力学图"
) -> Dict:
    """
    验证假设2：拉普拉斯动力学在数据中成立

    测试：
    - 恢复力是否预测未来收益
    - 动力学方程的拟合优度
    - 预测准确性

    Args:
        dataset: 数据集（来自assumption1）
        L: 拉普拉斯矩阵
        method_name: 方法名称

    Returns:
        results: 验证结果字典
    """
    print("\n" + "=" * 70)
    print(f"假设2验证：拉普拉斯动力学有效性（使用{method_name}）")
    print("=" * 70)

    f_series = dataset['f_series']

    results = {}

    # Test 1: Force-return correlation
    print("\n[1/4] 恢复力-收益相关性测试...")
    ic_results = test_force_return_correlation(f_series, L, lag=1)
    results['ic_test'] = ic_results

    print(f"  IC均值:     {ic_results['ic_mean']:.6f}")
    print(f"  IC标准差:   {ic_results['ic_std']:.6f}")
    print(f"  信息比率:   {ic_results['ic_ir']:.4f}")
    print(f"  t统计量:    {ic_results['t_statistic']:.3f}")
    print(f"  p值:        {ic_results['p_value']:.6f}")
    print(f"  胜率:       {ic_results['win_rate']:.2%}")

    if ic_results['p_value'] < 0.05:
        print(f"  ✅ 显著性: 通过 (p < 0.05)")
    else:
        print(f"  ❌ 显著性: 未通过 (p >= 0.05)")

    # Test 2: Dynamics regression
    print("\n[2/4] 动力学回归测试...")
    reg_results = test_dynamics_regression(f_series, L, dt=dataset['dt'])
    results['regression_test'] = reg_results

    print(f"  Beta系数:   {reg_results['beta']:.4f} (应接近1)")
    print(f"  R²:         {reg_results['r_squared']:.4f}")
    print(f"  平均R²:     {reg_results['r_squared_mean']:.4f}")

    if reg_results['r_squared'] > 0.5:
        print(f"  ✅ 拟合优度: 良好 (R² > 0.5)")
    else:
        print(f"  ⚠️  拟合优度: 一般 (R² < 0.5)")

    # Test 3: Prediction accuracy
    print("\n[3/4] 预测准确性测试...")
    pred_results = test_prediction_accuracy(f_series, L, prediction_horizon=5, dt=dataset['dt'])
    results['prediction_test'] = pred_results

    print(f"  MSE:        {pred_results['mse_mean']:.6f}")
    print(f"  MAE:        {pred_results['mae_mean']:.6f}")
    print(f"  方向准确率: {pred_results['direction_accuracy']:.2%}")

    if pred_results['direction_accuracy'] > 0.55:
        print(f"  ✅ 方向预测: 优于随机 (> 55%)")
    else:
        print(f"  ❌ 方向预测: 未优于随机 (< 55%)")

    # Test 4: Summary
    print("\n[4/4] 综合评估")
    print("\n" + "-" * 70)

    # 判断是否通过
    passed_tests = []
    failed_tests = []

    if ic_results['p_value'] < 0.05 and ic_results['ic_mean'] > 0.05:
        passed_tests.append("✅ IC显著且为正")
    else:
        failed_tests.append(f"❌ IC = {ic_results['ic_mean']:.4f}, p = {ic_results['p_value']:.4f}")

    if reg_results['r_squared'] > 0.5:
        passed_tests.append("✅ 动力学拟合良好")
    else:
        failed_tests.append(f"❌ R² = {reg_results['r_squared']:.4f} < 0.5")

    if pred_results['direction_accuracy'] > 0.55:
        passed_tests.append("✅ 方向预测优于随机")
    else:
        failed_tests.append(f"❌ 方向准确率 = {pred_results['direction_accuracy']:.2%}")

    print("通过的测试:")
    for test in passed_tests:
        print(f"  {test}")

    if failed_tests:
        print("\n未通过的测试:")
        for test in failed_tests:
            print(f"  {test}")

    print("-" * 70)

    # 可视化
    print("\n生成可视化...")
    visualize_dynamics_test_results(ic_results)

    return results


def run_complete_validation(
    n_stocks: int = 50,
    n_clusters: int = 5,
    n_timesteps: int = 1000,
    seed: int = 42,
    dt: float = 0.01,
    noise_std: float = 0.01,
    noise_type: str = 'gaussian'
) -> Dict:
    """
    运行完整的验证流程

    Returns:
        complete_results: 包含所有验证结果的字典
    """
    print("\n" + "=" * 70)
    print("深度势能套利 - 核心假设验证")
    print("=" * 70)
    print(f"\n参数设置:")
    print(f"  股票数量: {n_stocks}")
    print(f"  簇数量:   {n_clusters}")
    print(f"  时间步数: {n_timesteps}")
    print(f"  随机种子: {seed}")
    print(f"  噪声分布: {noise_type}")

    # 假设1验证
    assumption1_results = run_assumption1_validation(
        n_stocks=n_stocks,
        n_clusters=n_clusters,
        n_timesteps=n_timesteps,
        seed=seed,
        dt=dt,
        noise_std=noise_std,
        noise_type=noise_type
    )

    # 假设2验证（使用GNN学习的图 - 如果GNN学得好，动力学应该更显著）
    dataset = assumption1_results['dataset']

    # 优先使用GNN的结果，如果没有则回退到dynamics
    if 'gnn' in assumption1_results['methods']:
        L_best = assumption1_results['methods']['gnn']['L']
        method_name = "GNN学习图"
    else:
        L_best = assumption1_results['methods']['dynamics']['L']
        method_name = "动力学图(Baseline)"

    assumption2_results = run_assumption2_validation(
        dataset=dataset,
        L=L_best,
        method_name=method_name
    )

    # 最终决策
    print("\n" + "=" * 70)
    print("最终决策")
    print("=" * 70)

    # 假设1通过标准（更新）
    # 重点关注簇质量指标
    if 'gnn' in assumption1_results['methods']:
        gnn_metrics = assumption1_results['gnn']
        assumption1_passed = (gnn_metrics['nmi'] > 0.7 and gnn_metrics['ari'] > 0.6)

        print(f"\n假设1（GNN聚类能力）: {'✅ 通过' if assumption1_passed else '❌ 未通过'}")
        print(f"  - NMI = {gnn_metrics['nmi']:.4f} {'>' if gnn_metrics['nmi'] > 0.7 else '<'} 0.7")
        print(f"  - ARI = {gnn_metrics['ari']:.4f} {'>' if gnn_metrics['ari'] > 0.6 else '<'} 0.6")
        print(f"  - Purity = {gnn_metrics['purity']:.4f}")
    else:
        # Fallback logic
        assumption1_passed = False
        print("\n假设1: ❌ 未通过 (GNN未运行)")

    # 假设2通过标准
    ic_test = assumption2_results['ic_test']
    assumption2_passed = (
        ic_test['p_value'] < 0.05 and
        ic_test['ic_mean'] > 0.05
    )

    print(f"\n假设2（动力学存在性）: {'✅ 通过' if assumption2_passed else '❌ 未通过'}")
    print(f"  - IC = {ic_test['ic_mean']:.6f} {'>' if ic_test['ic_mean'] > 0.05 else '<'} 0.05")
    print(f"  - p值 = {ic_test['p_value']:.6f} {'<' if ic_test['p_value'] < 0.05 else '>'} 0.05")

    print("\n" + "-" * 70)

    if assumption1_passed and assumption2_passed:
        print("🟢 绿灯：两个假设都通过，建议继续开发完整系统")
    elif assumption1_passed or assumption2_passed:
        print("🟡 黄灯：一个假设通过，建议修正后重新验证")
    else:
        print("🔴 红灯：两个假设都未通过，建议重新思考核心方法")

    print("=" * 70)

    return {
        'assumption1': assumption1_results,
        'assumption2': assumption2_results,
        'decision': {
            'assumption1_passed': assumption1_passed,
            'assumption2_passed': assumption2_passed,
            'overall': 'green' if (assumption1_passed and assumption2_passed) else
                      'yellow' if (assumption1_passed or assumption2_passed) else
                      'red'
        }
    }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 运行完整验证
    results = run_complete_validation(
        n_stocks=50,
        n_clusters=5,
        n_timesteps=1000,
        seed=42,
        dt=0.01,
        noise_std=0.01,
        noise_type='student-t' # Run 4: Heavy Tails
    )

    # 保存结果（可选）
    # np.savez('validation_results.npz', **results)
    # print("\n结果已保存到: validation_results.npz")
