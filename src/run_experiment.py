"""
run_experiment.py
End-to-end experiment runner: for a given FinBench task (e.g. 'cd2', 'ld2'),
loads data, trains baseline ML models, trains a FinPT-like model, and saves
a results table + comparison bar chart to results/<dataset_name>/.

Usage:
    python run_experiment.py --dataset cd2
    python run_experiment.py --dataset ld2
    python run_experiment.py --dataset cd2 ld2     # run several tasks back to back
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_models import train_and_evaluate_baselines
from data_utils import load_finbench_split
from finpt_model import train_and_evaluate_finpt_like


def run_for_dataset(dataset_name: str, output_root: str = "../results", random_state: int = 42):
    print(f"\n{'=' * 60}\nRunning experiment for FinBench task: {dataset_name}\n{'=' * 60}")

    out_dir = os.path.join(output_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    data = load_finbench_split(dataset_name, split="train")
    X_train, X_test, y_train, y_test = train_test_split(
        data.X_raw, data.y, test_size=0.2, random_state=random_state
    )

    # --- Baseline models (use raw frames; preprocessing pipeline scales/encodes internally) ---
    baseline_results = train_and_evaluate_baselines(
        X_train, y_train, X_test, y_test, data.numerical_cols, data.categorical_cols
    )

    # --- FinPT-like model (raw values converted to text, then embedded) ---
    finpt_metrics = train_and_evaluate_finpt_like(X_train, y_train, X_test, y_test)

    combined = baseline_results.copy()
    combined.loc["FinPT-like Model"] = finpt_metrics

    # --- Save results table ---
    csv_path = os.path.join(out_dir, "model_performance.csv")
    combined.to_csv(csv_path)
    print(f"\nSaved results table to {csv_path}")
    print(combined)

    # --- Save comparison chart ---
    plt.figure(figsize=(12, 6))
    combined[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", figsize=(12, 6), edgecolor="black"
    )
    plt.title(f"Model Performance Comparison — '{dataset_name}' dataset (Baseline vs. FinPT-like)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(title="Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "model_performance.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved chart to {chart_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Run FinBench baseline vs FinPT-like comparison.")
    parser.add_argument(
        "--dataset", nargs="+", default=["cd2", "ld2"],
        help="FinBench task code(s) to run, e.g. cd2 ld2 (default: cd2 ld2)",
    )
    parser.add_argument("--output-root", default="../results", help="Where to save results")
    args = parser.parse_args()

    all_results = {}
    for ds in args.dataset:
        all_results[ds] = run_for_dataset(ds, output_root=args.output_root)

    print("\nDone. Summary:")
    for ds, df in all_results.items():
        print(f"\n--- {ds} ---")
        print(df.round(3))


if __name__ == "__main__":
    main()
