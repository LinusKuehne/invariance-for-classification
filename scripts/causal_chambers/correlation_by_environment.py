from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def print_correlations(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    print(f"Correlations for {csv_path.name}:" + "=" * 24)
    grouped = df.groupby("E", dropna=False)
    for env, group in grouped:
        corr_y_ir3 = group["Y"].corr(group["ir_3"])
        corr_y_vis3 = group["Y"].corr(group["vis_3"])
        print(
            f"env={env} | corr(Y, ir_3)={corr_y_ir3: .3f} | "
            f"corr(Y, vis_3)={corr_y_vis3: .3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-environment Pearson correlations for Y with ir_3 and vis_3."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        default=[
            Path("scripts/causal_chambers/data/2_train.csv"),
            Path("scripts/causal_chambers/data/2_test.csv"),
        ],
        help="CSV files to process. Defaults to 2_train.csv and 2_test.csv.",
    )
    args = parser.parse_args()

    for csv_file in args.csv_files:
        print_correlations(csv_file)


if __name__ == "__main__":
    main()
