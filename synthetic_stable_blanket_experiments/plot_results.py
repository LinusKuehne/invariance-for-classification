from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from synthetic_experiments.plotting import save_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate experiment plots from saved results CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing results_per_run.csv and where plots will be saved.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional path to results_per_run.csv. Defaults to output-dir/results_per_run.csv.",
    )
    parser.add_argument(
        "--attack-mode",
        choices=["bound", "cost"],
        default="bound",
        help="Which x-axis/sweep type to plot.",
    )
    parser.add_argument(
        "--max-sweep-value",
        type=float,
        default=None,
        help="Optional maximum x-axis sweep value to include, e.g. 1.0 for bounds.",
    )
    parser.add_argument(
        "--endpoint-train-sizes-only",
        action="store_true",
        help="For train-size sweeps, plot only the smallest and largest train sizes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    results_csv = args.results_csv or output_dir / "results_per_run.csv"

    results = pd.read_csv(results_csv)
    save_plots(
        results,
        output_dir,
        attack_mode=args.attack_mode,
        max_sweep_value=args.max_sweep_value,
        endpoint_train_sizes_only=args.endpoint_train_sizes_only,
    )
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
