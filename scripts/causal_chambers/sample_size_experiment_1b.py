"""
Sample size dependence of stabilized classification (TramGCM-RF) on dataset 1b.

For each of N_REPS repetitions and each n_e in N_E_VALUES:
  1. Randomly sample n_e observations per training environment
  2. Fit SC with TramGCM(RF) invariance test, RF predictor
  3. Evaluate on the full test set ("ensemble" and "best")

Produces two plots:
  - n_e vs accuracy, with 95% t-test CI bands (ensemble / best)
  - n_e vs subset counts (n_invariant / n_predictive), with 95% CI bands

Usage:
    python sample_size_experiment_1b.py
    python sample_size_experiment_1b.py --n-reps 2 --n-e-values 100 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from invariance_for_classification import StabilizedClassificationClassifier

# =============================================================================
# configuration
# =============================================================================

SEED = 42
DATASET = "1b"
N_E_VALUES_DEFAULT = [50, 100, 200, 300, 500, 750, 1000]
NORMAL_COLS = ["Y", "red", "green", "blue", "ir_1", "vis_1", "ir_3", "vis_3", "E"]


# =============================================================================
# data helpers
# =============================================================================


def _load_train_test(data_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, f"{DATASET}_train.csv")
    test_path = os.path.join(data_dir, f"{DATASET}_test.csv")
    if not os.path.exists(train_path):
        sys.exit(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        sys.exit(f"Test file not found: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    cols = [c for c in NORMAL_COLS if c in df_train.columns]
    df_train = df_train[cols]
    df_test = df_test[cols]

    return df_train, df_test


def _subsample_train(
    df_train: pd.DataFrame, n_obs_per_env: int, rng: np.random.Generator
) -> pd.DataFrame:
    dfs = []
    for env in sorted(df_train["E"].unique()):
        env_df = df_train[df_train["E"] == env]
        if len(env_df) <= n_obs_per_env:
            dfs.append(env_df)
        else:
            idx = rng.choice(len(env_df), size=n_obs_per_env, replace=False)
            dfs.append(env_df.iloc[idx])
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# statistics helper
# =============================================================================


def _ci_half_width(values: np.ndarray, confidence: float = 0.95) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(t_crit * se)


# =============================================================================
# experiment runner
# =============================================================================


def run_experiment(
    n_reps: int = 25,
    n_e_values: list[int] | None = None,
    n_jobs: int = 10,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the experiment. Returns (df_accuracy, df_subsets)."""
    if n_e_values is None:
        n_e_values = N_E_VALUES_DEFAULT

    df_train_full, df_test = _load_train_test()
    features = [c for c in df_train_full.columns if c not in ("Y", "E")]

    X_test = df_test[features].to_numpy()
    y_test = df_test["Y"].to_numpy().astype(int)

    print("=" * 60)
    print(f"Dataset         : {DATASET}")
    print(f"Features ({len(features)})   : {features}")
    print(f"Full train size : {len(df_train_full)}")
    print(f"Test size       : {len(df_test)}")
    print(f"n_e values      : {n_e_values}")
    print(f"N_REPS          : {n_reps}")
    print(f"N_JOBS          : {n_jobs}")
    print("=" * 60)

    acc_records: list[dict] = []
    subset_records: list[dict] = []

    for rep in range(n_reps):
        print(f"\nRepetition {rep + 1}/{n_reps}")

        for n_e in n_e_values:
            # Independent seed per (rep, n_e) so results don't depend on
            # which other n_e values are present in the same run.
            pair_seed = SEED + rep * 10_000 + n_e
            rng = np.random.default_rng(pair_seed)

            t0 = time.time()
            print(f"  n_e={n_e:>5d} ...", end=" ", flush=True)

            df_train = _subsample_train(df_train_full, n_e, rng)
            X_train = df_train[features].to_numpy()
            y_train = df_train["Y"].to_numpy().astype(int)
            E_train = df_train["E"].to_numpy()

            clf = StabilizedClassificationClassifier(
                alpha_inv=0.05,
                alpha_pred=0.05,
                pred_classifier_type=["RF"],
                invariance_test="tram_gcm",
                test_classifier_type="RF",
                pred_scoring="mean",
                n_bootstrap=250,
                verbose=1 if verbose else 0,
                n_jobs=n_jobs,
                random_state=pair_seed,
            )
            clf.fit(X_train, y_train, environment=E_train)

            # subset counts
            n_inv = int(clf.n_invariant_subsets_)
            n_pred_raw = clf.n_predictive_subsets_
            n_pred = (
                n_pred_raw["RF"] if isinstance(n_pred_raw, dict) else int(n_pred_raw)
            )
            subset_records.append(
                {"rep": rep, "n_e": n_e, "n_invariant": n_inv, "n_predictive": n_pred}
            )

            for method_label in ["ensemble", "best"]:
                y_pred = clf.predict(
                    X_test, pred_classifier_type="RF", method=method_label
                )
                acc = float(accuracy_score(y_test, y_pred))
                acc_records.append(
                    {"rep": rep, "n_e": n_e, "method": method_label, "accuracy": acc}
                )

            print(f"done ({time.time() - t0:.1f}s)")

    return pd.DataFrame(acc_records), pd.DataFrame(subset_records)


# =============================================================================
# plotting helpers
# =============================================================================


def _add_legend(fig, series: dict[str, tuple[str, str]]) -> None:
    handles = [
        mpatches.Patch(color=color, label=label, alpha=0.8)
        for _, (color, label) in series.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(series),
        fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )


# =============================================================================
# plots
# =============================================================================


def make_plot(df: pd.DataFrame, output_path: str) -> None:
    n_e_values = sorted(df["n_e"].unique())
    series = {
        "ensemble": ("#1f77b4", "Ensemble"),
        "best": ("#ff7f0e", "Best subset"),
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))

    positions = list(range(len(n_e_values)))
    for method, (color, _) in series.items():
        mdf = df[df["method"] == method]
        means, ci_lows, ci_highs = [], [], []
        for n_e in n_e_values:
            vals = mdf[mdf["n_e"] == n_e]["accuracy"].to_numpy()
            m = float(np.mean(vals))
            ci = _ci_half_width(vals)
            means.append(m)
            ci_lows.append(m - ci)
            ci_highs.append(m + ci)
        ax.plot(positions, means, marker="o", color=color)
        ax.fill_between(positions, ci_lows, ci_highs, alpha=0.2, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in n_e_values])
    ax.set_xlabel("Observations per environment ($n_e$)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)

    _add_legend(fig, series)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def make_subset_plot(df: pd.DataFrame, output_path: str) -> None:
    n_e_values = sorted(df["n_e"].unique())
    series = {
        "n_invariant": (
            "#2ca02c",
            r"$|\hat{\mathcal{I}}_{\mathcal{E}_{\mathrm{tr}}}|$",
        ),
        "n_predictive": (
            "#9467bd",
            r"$|\hat{\mathcal{C}}_{\mathcal{E}_{\mathrm{tr}}}|$",
        ),
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))

    positions = list(range(len(n_e_values)))
    for col, (color, _) in series.items():
        means, ci_lows, ci_highs = [], [], []
        for n_e in n_e_values:
            vals = df[df["n_e"] == n_e][col].to_numpy().astype(float)
            m = float(np.mean(vals))
            ci = _ci_half_width(vals)
            means.append(m)
            ci_lows.append(m - ci)
            ci_highs.append(m + ci)
        ax.plot(positions, means, marker="o", color=color)
        ax.fill_between(positions, ci_lows, ci_highs, alpha=0.2, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in n_e_values])
    ax.set_xlabel("Observations per environment ($n_e$)", fontsize=11)
    ax.set_ylabel("Number of subsets", fontsize=11)
    ax.set_ylim(bottom=0)

    _add_legend(fig, series)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


# =============================================================================
# main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample size experiment for SC (TramGCM-RF) on dataset 1b."
    )
    parser.add_argument(
        "--n-reps", type=int, default=25, help="Number of repetitions (default: 25)."
    )
    parser.add_argument(
        "--n-e-values",
        type=int,
        nargs="+",
        default=N_E_VALUES_DEFAULT,
        help="List of n_e values to evaluate (default: 50 100 200 300 500 750 1000).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=10, help="Parallel workers (default: 10)."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    save_dir = os.path.join(repo_root, "results", DATASET)
    os.makedirs(save_dir, exist_ok=True)

    df_acc, df_subsets = run_experiment(
        n_reps=args.n_reps,
        n_e_values=args.n_e_values,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    acc_path = os.path.join(save_dir, "sample_size_1b.csv")
    subsets_path = os.path.join(save_dir, "sample_size_1b_subsets.csv")
    df_acc.to_csv(acc_path, index=False)
    df_subsets.to_csv(subsets_path, index=False)
    print(f"\nResults saved to {acc_path}")
    print(f"Subset counts saved to {subsets_path}")

    make_plot(df_acc, os.path.join(save_dir, f"sample_size_1b_{args.n_jobs}.pdf"))
    make_subset_plot(
        df_subsets, os.path.join(save_dir, f"sample_size_1b_subsets_{args.n_jobs}.pdf")
    )


if __name__ == "__main__":
    main()
