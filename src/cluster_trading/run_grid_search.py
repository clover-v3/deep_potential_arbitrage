import argparse
import pandas as pd
import itertools
from tabulate import tabulate
import copy

from src.cluster_trading.run_rolling import RollingTrainer

def run_grid_search():
    parser = argparse.ArgumentParser()
    # Base arguments (fixed for all runs)
    parser.add_argument('--data_path', type=str, default='virtual')
    parser.add_argument('--start_date', type=str, default='2020-01-01')
    parser.add_argument('--end_date', type=str, default='2023-01-01')
    parser.add_argument('--train_days', type=int, default=252)
    parser.add_argument('--test_days', type=int, default=63)

    # Defaults for non-search params
    parser.add_argument('--threshold', type=float, default=2.0)
    parser.add_argument('--stop_threshold', type=float, default=4.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--scaling_factor', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--turnover_penalty', type=float, default=0.5)
    parser.add_argument('--leverage_penalty', type=float, default=1.0)
    parser.add_argument('--max_leverage', type=float, default=5.0)
    parser.add_argument('--cost_bps', type=float, default=10.0)

    # Search Space Override (Optional, usually hardcoded or config file)

    base_args = parser.parse_args()

    # === SEARCH SPACE ===
    # Focusing on the "Must Search" parameters identified in analysis
    search_space = {
        'n_clusters': [5, 10, 20],
        'window': [30, 60, 90],
        'similarity_threshold': [0.3, 0.5, 0.7]
    }

    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    print(f"=== Starting Grid Search ===")
    print(f"Total Combinations: {len(combinations)}")

    for i, params in enumerate(combinations):
        print(f"\n--- Run {i+1}/{len(combinations)}: {params} ---")

        # Create a copy of args and update with current params
        run_args = copy.deepcopy(base_args)
        for k, v in params.items():
            setattr(run_args, k, v)

        # Run Rolling Trainer
        try:
            trainer = RollingTrainer(run_args)
            sharpe, total_ret = trainer.run_rolling()

            results.append({
                **params,
                'sharpe': sharpe,
                'total_return': total_ret
            })
        except Exception as e:
            print(f"Run failed: {e}")
            results.append({
                **params,
                'sharpe': -999.0,
                'total_return': -999.0,
                'error': str(e)
            })

    # Summary
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='sharpe', ascending=False)

    print("\n=== Grid Search Results (Top 10) ===")
    print(tabulate(df_results.head(10), headers='keys', tablefmt='psql', floatfmt=".4f"))

    # Save
    df_results.to_csv("grid_search_results.csv", index=False)
    print("Results saved to grid_search_results.csv")

if __name__ == "__main__":
    run_grid_search()
