"""
Validation Module for Deep Potential Arbitrage

This module provides tools for validating the core assumptions:
1. Graph structure can capture Laplacian dynamics
2. Laplacian dynamics holds in financial markets
"""

from .synthetic_data import (
    create_ground_truth_graph,
    generate_laplacian_dynamics,
    generate_complete_dataset,
    visualize_synthetic_data
)

from .graph_learning import (
    correlation_graph,
    dynamics_based_graph,
    topk_sparsify
)

from .dynamics_test import (
    test_force_return_correlation,
    test_dynamics_regression,
    test_prediction_accuracy,
    test_graph_stability,
    graph_similarity,
    visualize_dynamics_test_results
)

from .metrics import (
    evaluate_graph_recovery,
    compute_ic,
    print_evaluation_report
)

__all__ = [
    # Synthetic data
    'create_ground_truth_graph',
    'generate_laplacian_dynamics',
    'generate_complete_dataset',
    'visualize_synthetic_data',

    # Graph learning
    'correlation_graph',
    'dynamics_based_graph',
    'topk_sparsify',

    # Dynamics testing
    'test_force_return_correlation',
    'test_dynamics_regression',
    'test_prediction_accuracy',
    'test_graph_stability',
    'graph_similarity',
    'visualize_dynamics_test_results',

    # Metrics
    'evaluate_graph_recovery',
    'compute_ic',
    'print_evaluation_report',
]
