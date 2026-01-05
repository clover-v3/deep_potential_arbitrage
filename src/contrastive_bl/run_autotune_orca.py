
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

    # Augmentation
    parser.add_argument("--aug_mask", type=float, default=0.1)
    parser.add_argument("--aug_noise", type=float, default=0.1)

    parser.add_argument("--cache_dir", type=str, default=None, help="Feature cache directory")
    parser.add_argument("--universe_dir", type=str, default=None, help="Shared universe directory (optional)")

    # Loss Params
    parser.add_argument("--manual_loss", action="store_true", help="Use manual weights")
    parser.add_argument("--lambda_ins", type=float, default=1.0)
    parser.add_argument("--lambda_clu", type=float, default=1.0)
    parser.add_argument("--lambda_pinn", type=float, default=1.0)

    args = parser.parse_args()

    # Define Grid
    # Define Grid
    # Focused grid for Top 2000 (Small vs Large)
    # Focused grid for Top 2000 (Small vs Large)
    grid = {
        'd_model': [64, 128],
        'n_bins': [32, 64],
        'dropout': [0.1, 0.3],
        'lr': [0.001, 0.002]
    }

    keys = list(grid.keys())
    combinations = list(itertools.product(*[grid[k] for k in keys]))

    print(f"Starting Grid Search over {len(combinations)} combinations...")

    # Base Output Dir
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = f"contrastive_bl_results/autotune/run_{timestamp}"
    os.makedirs(base_out_dir, exist_ok=True)
    print(f"Saving results to: {base_out_dir}")

    # Shared Universe Directory
    if args.universe_dir:
        shared_univ_dir = args.universe_dir
    else:
        shared_univ_dir = os.path.join(base_out_dir, "shared_universe")

    os.makedirs(shared_univ_dir, exist_ok=True)
    print(f"Using shared universe: {shared_univ_dir}")

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n>>> Trial {i+1}/{len(combinations)}: {params}")

        # Unique Subdir for this trial
        # Replace . with p for filename safety (0.3 -> 0p3)
        do_str = str(params['dropout']).replace('.', 'p')
        lr_str = str(params['lr']).replace('.', 'p')
        trial_name = f"trial_{i+1}_d{params['d_model']}_n{params['n_bins']}_do{do_str}_lr{lr_str}"
        trial_dir = os.path.join(base_out_dir, trial_name)
        os.makedirs(trial_dir, exist_ok=True)

        cmd = (
            f"python -m src.contrastive_bl.run_rolling "
            f"--data_root {args.data_root} "
            f"--start_year {args.start_year} "
            f"--end_year {args.end_year} "
            f"--train_years {args.train_years} "
            f"--test_years {args.test_years} "
            f"--epochs 5 " # Short epochs for tuning
            f"--device {args.device} "
            f"--batch_mode monthly "
            f"--d_model {params['d_model']} "
            f"--n_bins {params['n_bins']} "
            f"--dropout {params['dropout']} "
            f"--lr {params['lr']} "
            f"--aug_mask {args.aug_mask} "
            f"--aug_noise {args.aug_noise} "
            f"--output_dir {trial_dir} "
        )
        if args.cache_dir:
            cmd += f"--cache_dir {args.cache_dir} "

        cmd += f"--universe_dir {shared_univ_dir} "

        # Loss Params
        if args.manual_loss:
            cmd += f"--manual_loss --lambda_ins {args.lambda_ins} --lambda_clu {args.lambda_clu} --lambda_pinn {args.lambda_pinn} "

        success = run_command(cmd)

        res_csv = os.path.join(trial_dir, "rolling_results_full.csv")
        if success and os.path.exists(res_csv):
            df = pd.read_csv(res_csv)
            if not df.empty and 'ret' in df.columns:
                # Calculate Sharpe
                ann_ret = df['ret'].mean() * 252
                ann_vol = df['ret'].std() * (252**0.5) + 1e-9
                sharpe = ann_ret / ann_vol

                # Metrics
                max_dd = 0.0
                if 'cum_ret' not in df.columns:
                     df['cum_ret'] = (1 + df['ret']).cumprod()

                # Win Rate
                win_rate = (df['ret'] > 0).mean()

                res_entry = params.copy()
                res_entry['sharpe'] = sharpe
                res_entry['ann_ret'] = ann_ret
                res_entry['ann_vol'] = ann_vol
                res_entry['win_rate'] = win_rate
                res_entry['trial_dir'] = trial_dir

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

        # Save summary to base dir
        summary_path = os.path.join(base_out_dir, "autotune_results.csv")
        res_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")

        best = res_df.iloc[0]
        print("\nBest Configuration:")
        print(best)
    else:
        print("No successful trials.")

if __name__ == "__main__":
    main()
