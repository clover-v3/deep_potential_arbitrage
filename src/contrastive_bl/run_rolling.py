import argparse
import subprocess
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(1)

def calc_metrics(res_df):
    if res_df.empty: return {}

    # Check if 'ret' is in columns
    if 'ret' not in res_df.columns:
        return {}

    # Annualized Return & Vol
    # Assuming monthly rebalancing (12 periods/year)
    ann_ret = res_df['ret'].mean() * 12
    ann_vol = res_df['ret'].std() * np.sqrt(12) + 1e-9

    # Sharpe Ratio
    sharpe = ann_ret / ann_vol

    # Max Drawdown
    res_df['cum_ret'] = (1 + res_df['ret']).cumprod()
    cum_series = res_df['cum_ret']
    running_max = cum_series.cummax()
    drawdown = (cum_series - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # Win Rate
    win_months = (res_df['ret'] > 0).sum()
    total_months = len(res_df)
    win_rate = win_months / total_months if total_months > 0 else 0.0

    return {
        'Total Return': (cum_series.iloc[-1] - 1) * 100,
        'Annualized Return': ann_ret * 100,
        'Annualized Vol': ann_vol * 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd * 100,
        'Calmar Ratio': calmar,
        'Win Rate': win_rate * 100
    }

def main():
    parser = argparse.ArgumentParser(description="Run Rolling Window Train/Test for ORCA")
    parser.add_argument("--data_root", type=str, default="./data/raw_ghz")
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--train_years", type=int, default=10, help="Number of years for training window")
    parser.add_argument("--test_years", type=int, default=1, help="Number of years for testing/trading window")
    parser.add_argument("--device", type=str, default=None, help="cuda/mps/cpu")
    parser.add_argument("--batch_mode", type=str, default="global", help="global/monthly")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--smoke_test", action="store_true", help="Run fast smoke test")

    # Hyperparams
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_bins", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    # Validation
    if args.end_year - args.start_year < args.train_years:
        print("Error: Total period implies shorter than one training window.")
        return

    all_results = []

    # Rolling Loop
    # We slide the test window start year
    # Test Start can go from (Start + Train) to (End - Test + 1)?
    # Actually, iterate current_start for training

    # Example: Start 2000, End 2023, Train 10, Test 1.
    # 1. Train 2000-2009 (10y), Test 2010 (1y)
    # 2. Train 2001-2010, Test 2011
    # ...
    # Last Test Year is 2023.
    # So Test Start Max is 2023.
    # Train Start Max = 2023 - 1 (Test Len) - 10 (Train Len) + 1 ?
    # Let's iterate on Test Start Year

    first_test_year = args.start_year + args.train_years
    last_test_year_end = args.end_year

    # The loop should ensure the test window fits in [first_test_year, last_test_year_end]
    # Iterate test_start_year

    current_test_start = first_test_year

    print(f"Starting Rolling Window: Train {args.train_years}y, Test {args.test_years}y")
    print(f"Period: {args.start_year} to {args.end_year}")

    while current_test_start + args.test_years - 1 <= args.end_year:
        test_end = current_test_start + args.test_years - 1

        train_start = current_test_start - args.train_years
        train_end = current_test_start - 1

        print(f"\n>>> Cycle: Train {train_start}-{train_end} | Test {current_test_start}-{test_end}")

        model_name = f"orca_model_{current_test_start}.pth"

        # 1. Train
        # Note: We pass batch_mode (if implemented)
        # Note: We pass device
        # Note: We pass args.data_root

        # Construct command
        train_cmd = (
            f"python -m src.contrastive_bl.train "
            f"--data_root {args.data_root} "
            f"--start_year {train_start} --end_year {train_end} "
            f"--epochs {args.epochs} "
            f"--save_path {model_name} "
        )
        if args.device:
            train_cmd += f"--device {args.device} "

        # Add batch_mode if user specified
        if args.batch_mode:
            train_cmd += f"--batch_mode {args.batch_mode} "

        if args.smoke_test:
            train_cmd += "--smoke_test "

        # Hyperparams
        train_cmd += f"--d_model {args.d_model} --n_bins {args.n_bins} --dropout {args.dropout} "

        run_command(train_cmd)

        # 2. Backtest
        backtest_cmd = (
            f"python -m src.contrastive_bl.backtest "
            f"--data_root {args.data_root} "
            f"--model_path {model_name} "
            f"--start_year {current_test_start} --end_year {test_end} "
        )
        if args.device:
            backtest_cmd += f"--device {args.device} "

        run_command(backtest_cmd)

        # 3. Collect Results
        if os.path.exists("backtest_results.csv"):
            df = pd.read_csv("backtest_results.csv")
            # Tag with cycle info
            df['train_start'] = train_start
            df['train_end'] = train_end
            all_results.append(df)
        else:
            print("Warning: backtest_results.csv not found.")

        # Move forward
        # Slide by test_years to ensure contiguous non-overlapping test windows
        current_test_start += args.test_years

    if not all_results:
        print("No results generated.")
        return

    # 4. Consolidate
    master_df = pd.concat(all_results, ignore_index=True)
    master_df.to_csv("rolling_results_full.csv", index=False)

    # 5. Global Metrics
    print("\n" + "="*40)
    print("       ROLLING BACKTEST SUMMARY       ")
    print("="*40)

    # We need to ensure dates are sorted and unique (if overlaps exist, handle them?)
    # With slide = test_years, no overlaps.
    master_df['date'] = pd.to_datetime(master_df['date'])
    master_df = master_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')

    metrics = calc_metrics(master_df)
    for k, v in metrics.items():
        if "Ratio" in k:
             print(f"{k}: {v:.4f}")
        else:
             print(f"{k}: {v:.2f}%")

    print("="*40)

    # 6. Plot
    try:
        plt.figure(figsize=(12, 8))

        # Cumulative
        master_df['cum_ret'] = (1 + master_df['ret']).cumprod()

        plt.subplot(2, 1, 1)
        plt.plot(master_df['date'], master_df['cum_ret'], label='Rolling Portfolio')
        plt.title('Rolling Window Cumulative Return')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Drawdown
        running_max = master_df['cum_ret'].cummax()
        dd = (master_df['cum_ret'] - running_max) / running_max

        plt.subplot(2, 1, 2)
        plt.plot(master_df['date'], dd, label='Drawdown', color='red')
        plt.fill_between(master_df['date'], dd, 0, color='red', alpha=0.3)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("rolling_backtest_plots.png")
        print("Plots saved to rolling_backtest_plots.png")

    except Exception as e:
        print(f"Plotting error: {e}")

if __name__ == "__main__":
    main()
