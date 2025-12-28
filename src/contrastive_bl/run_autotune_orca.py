
import itertools
import subprocess
import pandas as pd
import argparse
import sys
import os

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Auto-Tune ORCA Hyperparameters")
    parser.add_argument("--data_root", type=str, default="./data/raw_ghz")
    parser.add_argument("--device", type=str, default="cpu")
    # Define search/validation period
    parser.add_argument("--start_year", type=int, default=2019)
    parser.add_argument("--end_year", type=int, default=2021)
    parser.add_argument("--train_years", type=int, default=2)
    parser.add_argument("--test_years", type=int, default=1)

    args = parser.parse_args()

    # Define Grid
    # Focused grid for Top 2000 (Small vs Large)
    grid = {
        'd_model': [64, 128],
        'n_bins': [32, 64],
        'dropout': [0.1, 0.3]
    }

    keys = list(grid.keys())
    combinations = list(itertools.product(*[grid[k] for k in keys]))

    print(f"Starting Grid Search over {len(combinations)} combinations...")
    print(f"Grid: {grid}")

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n>>> Trial {i+1}/{len(combinations)}: {params}")

        # We run a short rolling window to evaluate this config
        # Use existing run_rolling.py
        # We need to parse the output CSV to get metrics.
        # run_rolling.py writes 'rolling_results_full.csv'

        # Clean previous results
        if os.path.exists("rolling_results_full.csv"):
            os.remove("rolling_results_full.csv")

        cmd = (
            f"python -m src.contrastive_bl.run_rolling "
            f"--data_root {args.data_root} "
            f"--start_year {args.start_year} "
            f"--end_year {args.end_year} "
            f"--train_years {args.train_years} "
            f"--test_years {args.test_years} "
            f"--epochs 5 " # Short epochs for tuning
            f"--device {args.device} "
            f"--batch_mode monthly " # Best practice
            f"--d_model {params['d_model']} "
            f"--n_bins {params['n_bins']} "
            f"--dropout {params['dropout']} "
        )

        success = run_command(cmd)

        if success and os.path.exists("rolling_results_full.csv"):
            df = pd.read_csv("rolling_results_full.csv")
            if not df.empty and 'ret' in df.columns:
                # Calculate Sharpe
                ann_ret = df['ret'].mean() * 12
                ann_vol = df['ret'].std() * (12**0.5) + 1e-9
                sharpe = ann_ret / ann_vol

                res_entry = params.copy()
                res_entry['sharpe'] = sharpe
                res_entry['ann_ret'] = ann_ret
                results.append(res_entry)
                print(f"Trial {i+1} Result: Sharpe={sharpe:.4f}, Ret={ann_ret:.2%}")
            else:
                print(f"Trial {i+1} Failed: Empty results.")
        else:
             print(f"Trial {i+1} Failed: Execution Error.")

    # Summary
    if results:
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values('sharpe', ascending=False)
        print("\n" + "="*40)
        print("       GRID SEARCH RESULTS       ")
        print("="*40)
        print(res_df)
        res_df.to_csv("autotune_results.csv", index=False)

        best = res_df.iloc[0]
        print("\nBest Configuration:")
        print(best)
    else:
        print("No successful trials.")

if __name__ == "__main__":
    main()
