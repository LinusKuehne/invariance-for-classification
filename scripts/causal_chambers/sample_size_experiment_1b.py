"""
Sample size dependence of stabilized classification (TramGCM-RF) on dataset 1b.

For each of N_REPS repetitions and each n_e in N_E_VALUES:
  1. Randomly sample n_e observations per training environment
  2. Fit SC with TramGCM(RF) invariance test, RF predictor
  3. Evaluate on the full test set ("ensemble" and "best")

Produces three plots:
  - n_e vs min. accuracy (ensemble / best), 95% CI bands
  - TPR and FPR (at alpha=0.05) vs n_e, dual y-axes
  - Fraction of ensemble members containing ir_3 vs n_e

Usage:
    python sample_size_experiment_1b.py
    python sample_size_experiment_1b.py --n-reps 2 --n-e-values 100 200
"""

from __future__ import annotations

import argparse
import itertools
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


def _gt_invariant_subsets_1b(features: list[str]) -> set[frozenset[str]]:
    """Ground truth invariant subsets for dataset 1b (same logic as evaluate_all_tests.py)."""
    base = [
        frozenset(),
        frozenset({"red"}),
        frozenset({"green"}),
        frozenset({"blue"}),
        frozenset({"red", "green"}),
        frozenset({"red", "blue"}),
        frozenset({"green", "blue"}),
        frozenset({"red", "green", "blue"}),
        frozenset({"red", "green", "blue", "vis_3"}),
    ]
    optional = [
        f for f in features if f not in {"red", "green", "blue", "vis_3", "ir_3"}
    ]
    opt_subsets = [frozenset()]
    for r in range(1, len(optional) + 1):
        for combo in itertools.combinations(optional, r):
            opt_subsets.append(frozenset(combo))
    result: set[frozenset[str]] = set()
    for b in base:
        for o in opt_subsets:
            result.add(b | o)
    return result


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the experiment. Returns (df_accuracy, df_pvalues, df_ensemble)."""
    if n_e_values is None:
        n_e_values = N_E_VALUES_DEFAULT

    df_train_full, df_test = _load_train_test()
    features = [c for c in df_train_full.columns if c not in ("Y", "E")]

    X_test = df_test[features].to_numpy()
    y_test = df_test["Y"].to_numpy().astype(int)
    E_test = df_test["E"].to_numpy()

    gt_invariant_subsets = _gt_invariant_subsets_1b(features)

    print("=" * 60)
    print(f"Dataset         : {DATASET}")
    print(f"Features ({len(features)})   : {features}")
    print(f"Full train size : {len(df_train_full)}")
    print(f"Test size       : {len(df_test)}")
    print(f"n_e values      : {n_e_values}")
    print(f"N_REPS          : {n_reps}")
    print(f"N_JOBS          : {n_jobs}")
    print(f"GT invariant subsets: {len(gt_invariant_subsets)}")
    print("=" * 60)

    acc_records: list[dict] = []
    pvalue_records: list[dict] = []
    ensemble_records: list[dict] = []

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

            # p-values for all evaluated subsets (for TPR/FPR plot)
            if hasattr(clf, "all_results_"):
                for r in clf.all_results_:
                    subset_feats = frozenset(features[i] for i in r["subset"])
                    pval = r.get("p_value", r.get("inv_score", float("nan")))
                    pvalue_records.append(
                        {
                            "rep": rep,
                            "n_e": n_e,
                            "p_value": pval,
                            "is_gt_invariant": subset_feats in gt_invariant_subsets,
                        }
                    )

            # Track fraction of ensemble members containing ir_3
            _active = getattr(clf, "active_subsets_", None)
            if _active is not None:
                if isinstance(_active, dict):
                    _active = _active.get("RF", next(iter(_active.values()), []))
                n_active = len(_active)
                ir3_count = sum(
                    1 for r in _active if "ir_3" in {features[i] for i in r["subset"]}
                )
                ensemble_records.append(
                    {
                        "rep": rep,
                        "n_e": n_e,
                        "ir3_frac": ir3_count / n_active
                        if n_active > 0
                        else float("nan"),
                    }
                )

            for method_label in ["ensemble", "best"]:
                y_pred = clf.predict(
                    X_test, pred_classifier_type="RF", method=method_label
                )
                acc = float(
                    min(
                        accuracy_score(y_test[E_test == e], y_pred[E_test == e])
                        for e in np.unique(E_test)
                    )
                )
                acc_records.append(
                    {"rep": rep, "n_e": n_e, "method": method_label, "accuracy": acc}
                )

            print(f"done ({time.time() - t0:.1f}s)")

    return (
        pd.DataFrame(acc_records),
        pd.DataFrame(pvalue_records),
        pd.DataFrame(ensemble_records),
    )


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


def _mean_ci(vals: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, lower, upper)."""
    m = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    ci = _ci_half_width(vals) if len(vals) >= 2 else float("nan")
    return m, m - ci, m + ci


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
            m, lo, hi = _mean_ci(vals)
            means.append(m)
            ci_lows.append(lo)
            ci_highs.append(hi)
        ax.plot(positions, means, marker="o", color=color)
        ax.fill_between(positions, ci_lows, ci_highs, alpha=0.2, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in n_e_values])
    ax.set_xlabel("Observations per environment ($n_e$)", fontsize=11)
    ax.set_ylabel("Worst-case accuracy", fontsize=11)

    _add_legend(fig, series)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def make_tpr_fpr_plot(df_pv: pd.DataFrame, output_path: str) -> None:
    """TPR and FPR at alpha=0.05 vs n_e, with dual y-axes."""
    n_e_values = sorted(df_pv["n_e"].unique())
    positions = list(range(len(n_e_values)))
    fpr_color = "#c75b5b"
    tpr_color = "#2ca02c"

    fpr_means, fpr_lows, fpr_highs = [], [], []
    tpr_means, tpr_lows, tpr_highs = [], [], []

    for n_e in n_e_values:
        sub = df_pv[df_pv["n_e"] == n_e]

        fpr_per_rep = (
            sub[~sub["is_gt_invariant"]]
            .groupby("rep")
            .apply(lambda g: float((g["p_value"] > 0.05).mean()))
            .to_numpy()
        )
        tpr_per_rep = (
            sub[sub["is_gt_invariant"]]
            .groupby("rep")
            .apply(lambda g: float((g["p_value"] > 0.05).mean()))
            .to_numpy()
        )

        m, lo, hi = _mean_ci(fpr_per_rep)
        fpr_means.append(m)
        fpr_lows.append(lo)
        fpr_highs.append(hi)
        m, lo, hi = _mean_ci(tpr_per_rep)
        tpr_means.append(m)
        tpr_lows.append(lo)
        tpr_highs.append(hi)

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()

    ax1.plot(positions, fpr_means, marker="o", color=fpr_color)
    ax1.fill_between(positions, fpr_lows, fpr_highs, alpha=0.2, color=fpr_color)
    ax1.axhline(0.05, color=fpr_color, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("FPR", fontsize=11, color=fpr_color)
    ax1.tick_params(axis="y", labelcolor=fpr_color)
    ax1.set_ylim(bottom=0)

    ax2.plot(positions, tpr_means, marker="o", color=tpr_color)
    ax2.fill_between(positions, tpr_lows, tpr_highs, alpha=0.2, color=tpr_color)
    ax2.set_ylabel("TPR", fontsize=11, color=tpr_color)
    ax2.tick_params(axis="y", labelcolor=tpr_color)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(v) for v in n_e_values])
    ax1.set_xlabel("Observations per environment ($n_e$)", fontsize=11)

    handles = [
        mpatches.Patch(color=fpr_color, label="FPR", alpha=0.8),
        mpatches.Patch(color=tpr_color, label="TPR", alpha=0.8),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def make_ensemble_ir3_plot(df_ens: pd.DataFrame, output_path: str) -> None:
    """Fraction of ensemble members containing ir_3 vs n_e."""
    n_e_values = sorted(df_ens["n_e"].unique())
    positions = list(range(len(n_e_values)))
    color = "#9467bd"

    means, ci_lows, ci_highs = [], [], []
    for n_e in n_e_values:
        vals = df_ens[df_ens["n_e"] == n_e]["ir3_frac"].dropna().to_numpy()
        m, lo, hi = _mean_ci(vals)
        means.append(m)
        ci_lows.append(lo)
        ci_highs.append(hi)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(positions, means, marker="o", color=color)
    ax.fill_between(positions, ci_lows, ci_highs, alpha=0.2, color=color)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in n_e_values])
    ax.set_xlabel("Observations per environment ($n_e$)", fontsize=11)
    ax.set_ylabel(
        "Fraction of subsets in "
        + r"$\hat{\mathcal{C}}_{\mathcal{E}_{\mathrm{tr}}}$"
        + "\ncontaining ir$_3$",
        fontsize=11,
    )
    ax.set_ylim(bottom=0)

    plt.tight_layout()
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
    nreps = args.n_reps

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    save_dir = os.path.join(repo_root, "results", DATASET)
    os.makedirs(save_dir, exist_ok=True)

    df_acc, df_pvalues, df_ensemble = run_experiment(
        n_reps=args.n_reps,
        n_e_values=args.n_e_values,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    acc_path = os.path.join(save_dir, "sample_size_1b.csv")
    pvalues_path = os.path.join(save_dir, "sample_size_1b_pvalues.csv")
    ensemble_path = os.path.join(save_dir, "sample_size_1b_ensemble.csv")
    df_acc.to_csv(acc_path, index=False)
    df_pvalues.to_csv(pvalues_path, index=False)
    df_ensemble.to_csv(ensemble_path, index=False)
    print(f"\nResults saved to {acc_path}")
    print(f"P-values saved to {pvalues_path}")
    print(f"Ensemble composition saved to {ensemble_path}")

    make_plot(df_acc, os.path.join(save_dir, f"sample_size_1b_{nreps}.pdf"))

    if len(df_pvalues) > 0:
        make_tpr_fpr_plot(
            df_pvalues, os.path.join(save_dir, f"sample_size_1b_tpr_fpr_{nreps}.pdf")
        )
    else:
        print("WARNING: no p-value data collected (all_results_ not available).")

    if len(df_ensemble) > 0:
        make_ensemble_ir3_plot(
            df_ensemble,
            os.path.join(save_dir, f"sample_size_1b_ens_ir3_{nreps}.pdf"),
        )


if __name__ == "__main__":
    main()
