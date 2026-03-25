"""
Sample size dependence of stabilized classification (TramGCM-RF) on dataset 1b.

For each of N_REPS repetitions and each n_e in N_E_VALUES:
  1. Randomly sample n_e observations per training environment
  2. Fit SC with TramGCM(RF) invariance test, RF predictor
  3. Evaluate on the full test set ("ensemble" and "best")

Produces six plots:
  - n_e vs accuracy (ensemble / best), 95% CI bands
  - n_e vs subset counts (n_invariant / n_predictive), 95% CI bands
  - Acceptance rate of the stable blanket {red, green, blue, vis_3} vs n_e
  - Avg. size of accepted (invariant) subsets vs n_e
  - TPR and FPR (at alpha=0.05) vs n_e, dual y-axes
  - Pairwise AUC (invariant vs non-invariant p-values) vs n_e

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

# Stable blanket for dataset 1b (ground-truth oracle feature set)
STABLE_BLANKET_1B = frozenset({"red", "green", "blue", "vis_3"})


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


def _pairwise_auc(inv_vals: np.ndarray, noninv_vals: np.ndarray) -> float:
    """Fraction of (inv, noninv) pairs where inv p-value > noninv p-value."""
    if len(inv_vals) == 0 or len(noninv_vals) == 0:
        return float("nan")
    count = sum(
        1.0 if i > n else 0.5 if i == n else 0.0 for i in inv_vals for n in noninv_vals
    )
    return count / (len(inv_vals) * len(noninv_vals))


# =============================================================================
# experiment runner
# =============================================================================


def run_experiment(
    n_reps: int = 25,
    n_e_values: list[int] | None = None,
    n_jobs: int = 10,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the experiment. Returns (df_accuracy, df_subsets, df_pvalues)."""
    if n_e_values is None:
        n_e_values = N_E_VALUES_DEFAULT

    df_train_full, df_test = _load_train_test()
    features = [c for c in df_train_full.columns if c not in ("Y", "E")]

    X_test = df_test[features].to_numpy()
    y_test = df_test["Y"].to_numpy().astype(int)
    E_test = df_test["E"].to_numpy()

    gt_invariant_subsets = _gt_invariant_subsets_1b(features)
    sb_key = ",".join(sorted(STABLE_BLANKET_1B))  # canonical string for CSV lookup

    print("=" * 60)
    print(f"Dataset         : {DATASET}")
    print(f"Features ({len(features)})   : {features}")
    print(f"Full train size : {len(df_train_full)}")
    print(f"Test size       : {len(df_test)}")
    print(f"n_e values      : {n_e_values}")
    print(f"N_REPS          : {n_reps}")
    print(f"N_JOBS          : {n_jobs}")
    print(f"GT invariant subsets: {len(gt_invariant_subsets)}")
    print(f"Stable blanket  : {sb_key}")
    print("=" * 60)

    acc_records: list[dict] = []
    subset_records: list[dict] = []
    pvalue_records: list[dict] = []
    best_records: list[dict] = []
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

            # subset counts
            n_inv = int(clf.n_invariant_subsets_)
            n_pred_raw = clf.n_predictive_subsets_
            n_pred = (
                n_pred_raw["RF"] if isinstance(n_pred_raw, dict) else int(n_pred_raw)
            )
            subset_records.append(
                {"rep": rep, "n_e": n_e, "n_invariant": n_inv, "n_predictive": n_pred}
            )

            # p-values for all evaluated subsets (for diagnostic plots)
            if hasattr(clf, "all_results_"):
                for r in clf.all_results_:
                    subset_feats = frozenset(features[i] for i in r["subset"])
                    subset_str = ",".join(sorted(subset_feats))
                    pval = r.get("p_value", r.get("inv_score", float("nan")))
                    pvalue_records.append(
                        {
                            "rep": rep,
                            "n_e": n_e,
                            "subset": subset_str,
                            "p_value": pval,
                            "is_gt_invariant": subset_feats in gt_invariant_subsets,
                            "subset_size": len(subset_feats),
                        }
                    )

            # Track best subset and ensemble composition
            _inv_fitted = getattr(clf, "_all_invariant_fitted_", None)
            if _inv_fitted is not None:
                if isinstance(_inv_fitted, dict):
                    _inv_fitted = _inv_fitted.get(
                        "RF", next(iter(_inv_fitted.values()), [])
                    )
                if _inv_fitted:
                    best_r = max(_inv_fitted, key=lambda x: x["score"])
                    best_feats = frozenset(features[i] for i in best_r["subset"])
                    best_records.append(
                        {
                            "rep": rep,
                            "n_e": n_e,
                            "subset": ",".join(sorted(best_feats)),
                            "is_gt_invariant": best_feats in gt_invariant_subsets,
                        }
                    )

            _active = getattr(clf, "active_subsets_", None)
            if _active is not None:
                if isinstance(_active, dict):
                    _active = _active.get("RF", next(iter(_active.values()), []))
                n_active = len(_active)
                ir3_count = sum(
                    1 for r in _active if "ir_3" in {features[i] for i in r["subset"]}
                )
                noninv_count = sum(
                    1
                    for r in _active
                    if frozenset(features[i] for i in r["subset"])
                    not in gt_invariant_subsets
                )
                scores_inv = [
                    r["score"]
                    for r in _active
                    if frozenset(features[i] for i in r["subset"])
                    in gt_invariant_subsets
                ]
                scores_noninv = [
                    r["score"]
                    for r in _active
                    if frozenset(features[i] for i in r["subset"])
                    not in gt_invariant_subsets
                ]
                ensemble_records.append(
                    {
                        "rep": rep,
                        "n_e": n_e,
                        "n_active": n_active,
                        "ir3_frac": ir3_count / n_active
                        if n_active > 0
                        else float("nan"),
                        "noninv_frac": noninv_count / n_active
                        if n_active > 0
                        else float("nan"),
                        "mean_score_inv": float(np.mean(scores_inv))
                        if scores_inv
                        else float("nan"),
                        "mean_score_noninv": float(np.mean(scores_noninv))
                        if scores_noninv
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
        pd.DataFrame(subset_records),
        pd.DataFrame(pvalue_records),
        pd.DataFrame(best_records),
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
    """Return (mean, lower, upper) clipped to [0, inf)."""
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
    ax.set_ylabel("Min. accuracy (worst env.)", fontsize=11)

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
            m, lo, hi = _mean_ci(vals)
            means.append(m)
            ci_lows.append(lo)
            ci_highs.append(hi)
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


def make_stable_blanket_plot(df_pv: pd.DataFrame, output_path: str) -> None:
    """Acceptance rate of {red, green, blue, vis_3} (p-value > 0.05) vs n_e."""
    sb_key = ",".join(sorted(STABLE_BLANKET_1B))
    n_e_values = sorted(df_pv["n_e"].unique())
    positions = list(range(len(n_e_values)))
    color = "#2ca02c"

    means, ci_lows, ci_highs = [], [], []
    for n_e in n_e_values:
        sub = df_pv[(df_pv["n_e"] == n_e) & (df_pv["subset"] == sb_key)]
        vals = (
            sub.groupby("rep")["p_value"]
            .first()
            .apply(lambda p: float(p > 0.05))
            .to_numpy()
        )
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
    ax.set_ylabel("Acceptance rate", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def make_avg_accepted_size_plot(df_pv: pd.DataFrame, output_path: str) -> None:
    """Mean size of accepted (p > 0.05) subsets vs n_e."""
    n_e_values = sorted(df_pv["n_e"].unique())
    positions = list(range(len(n_e_values)))
    color = "#4472c4"

    means, ci_lows, ci_highs = [], [], []
    for n_e in n_e_values:
        sub = df_pv[df_pv["n_e"] == n_e]
        size_per_rep = []
        for _, grp in sub.groupby("rep"):
            accepted = grp[grp["p_value"] > 0.05]
            size_per_rep.append(
                float(accepted["subset_size"].mean()) if len(accepted) > 0 else 0.0
            )
        vals = np.array(size_per_rep)
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
    ax.set_ylabel("Avg. size of accepted subsets", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def make_tpr_fpr_plot(df_pv: pd.DataFrame, output_path: str) -> None:
    """TPR and FPR at alpha=0.05 vs n_e, with dual y-axes."""
    n_e_values = sorted(df_pv["n_e"].unique())
    positions = list(range(len(n_e_values)))
    fpr_color = "#c75b5b"
    tpr_color = "#4472c4"

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


def make_pairwise_auc_plot(df_pv: pd.DataFrame, output_path: str) -> None:
    """Pairwise AUC (invariant vs non-invariant p-values) vs n_e."""
    n_e_values = sorted(df_pv["n_e"].unique())
    positions = list(range(len(n_e_values)))
    color = "#9467bd"

    means, ci_lows, ci_highs = [], [], []
    for n_e in n_e_values:
        sub = df_pv[df_pv["n_e"] == n_e]
        auc_per_rep = []
        for _, grp in sub.groupby("rep"):
            inv_pv = grp[grp["is_gt_invariant"]]["p_value"].dropna().to_numpy()
            noninv_pv = grp[~grp["is_gt_invariant"]]["p_value"].dropna().to_numpy()
            auc_per_rep.append(_pairwise_auc(inv_pv, noninv_pv))
        vals = np.array([v for v in auc_per_rep if not np.isnan(v)])
        m, lo, hi = _mean_ci(vals)
        means.append(m)
        ci_lows.append(lo)
        ci_highs.append(hi)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(positions, means, marker="o", color=color)
    ax.fill_between(positions, ci_lows, ci_highs, alpha=0.2, color=color)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in n_e_values])
    ax.set_xlabel("Observations per environment ($n_e$)", fontsize=11)
    ax.set_ylabel("Pairwise AUC", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def make_best_noninv_plot(df_best: pd.DataFrame, output_path: str) -> None:
    """Fraction of reps where 'best' subset is GT-non-invariant vs n_e."""
    n_e_values = sorted(df_best["n_e"].unique())
    positions = list(range(len(n_e_values)))
    color = "#c75b5b"

    means, ci_lows, ci_highs = [], [], []
    for n_e in n_e_values:
        vals = (
            (~df_best[df_best["n_e"] == n_e]["is_gt_invariant"])
            .to_numpy()
            .astype(float)
        )
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
    ax.set_ylabel("Fraction non-GT-invariant", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def make_ensemble_contamination_plot(df_ens: pd.DataFrame, output_path: str) -> None:
    """Fraction of ensemble members with ir_3 and fraction GT-non-invariant vs n_e."""
    n_e_values = sorted(df_ens["n_e"].unique())
    positions = list(range(len(n_e_values)))
    series = {
        "noninv_frac": ("#c75b5b", "GT-non-invariant fraction"),
        "ir3_frac": ("#ff7f0e", "Contains ir_3"),
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for col, (color, _) in series.items():
        means, ci_lows, ci_highs = [], [], []
        for n_e in n_e_values:
            vals = df_ens[df_ens["n_e"] == n_e][col].dropna().to_numpy()
            m, lo, hi = _mean_ci(vals)
            means.append(m)
            ci_lows.append(lo)
            ci_highs.append(hi)
        ax.plot(positions, means, marker="o", color=color)
        ax.fill_between(positions, ci_lows, ci_highs, alpha=0.2, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in n_e_values])
    ax.set_xlabel("Observations per environment ($n_e$)", fontsize=11)
    ax.set_ylabel("Fraction of ensemble members", fontsize=11)

    _add_legend(fig, series)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def make_score_comparison_plot(df_ens: pd.DataFrame, output_path: str) -> None:
    """Mean predictive score of GT-invariant vs GT-non-invariant ensemble members vs n_e."""
    n_e_values = sorted(df_ens["n_e"].unique())
    positions = list(range(len(n_e_values)))
    series = {
        "mean_score_inv": ("#1f77b4", "GT-invariant"),
        "mean_score_noninv": ("#c75b5b", "GT-non-invariant"),
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for col, (color, _) in series.items():
        means, ci_lows, ci_highs = [], [], []
        for n_e in n_e_values:
            vals = df_ens[df_ens["n_e"] == n_e][col].dropna().to_numpy()
            m, lo, hi = _mean_ci(vals)
            means.append(m)
            ci_lows.append(lo)
            ci_highs.append(hi)
        ax.plot(positions, means, marker="o", color=color)
        ax.fill_between(positions, ci_lows, ci_highs, alpha=0.2, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in n_e_values])
    ax.set_xlabel("Observations per environment ($n_e$)", fontsize=11)
    ax.set_ylabel("Mean predictive score", fontsize=11)

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
    nreps = args.n_reps

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    save_dir = os.path.join(repo_root, "results", DATASET)
    os.makedirs(save_dir, exist_ok=True)

    df_acc, df_subsets, df_pvalues, df_best, df_ensemble = run_experiment(
        n_reps=args.n_reps,
        n_e_values=args.n_e_values,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    acc_path = os.path.join(save_dir, "sample_size_1b.csv")
    subsets_path = os.path.join(save_dir, "sample_size_1b_subsets.csv")
    pvalues_path = os.path.join(save_dir, "sample_size_1b_pvalues.csv")
    best_path = os.path.join(save_dir, "sample_size_1b_best.csv")
    ensemble_path = os.path.join(save_dir, "sample_size_1b_ensemble.csv")
    df_acc.to_csv(acc_path, index=False)
    df_subsets.to_csv(subsets_path, index=False)
    df_pvalues.to_csv(pvalues_path, index=False)
    df_best.to_csv(best_path, index=False)
    df_ensemble.to_csv(ensemble_path, index=False)
    print(f"\nResults saved to {acc_path}")
    print(f"Subset counts saved to {subsets_path}")
    print(f"P-values saved to {pvalues_path}")
    print(f"Best subsets saved to {best_path}")
    print(f"Ensemble composition saved to {ensemble_path}")

    make_plot(df_acc, os.path.join(save_dir, f"sample_size_1b_{nreps}.pdf"))
    make_subset_plot(
        df_subsets, os.path.join(save_dir, f"sample_size_1b_subsets_{nreps}.pdf")
    )

    if len(df_pvalues) > 0:
        make_stable_blanket_plot(
            df_pvalues,
            os.path.join(save_dir, f"sample_size_1b_sb_acceptance_{nreps}.pdf"),
        )
        make_avg_accepted_size_plot(
            df_pvalues, os.path.join(save_dir, f"sample_size_1b_avg_size_{nreps}.pdf")
        )
        make_tpr_fpr_plot(
            df_pvalues, os.path.join(save_dir, f"sample_size_1b_tpr_fpr_{nreps}.pdf")
        )
        make_pairwise_auc_plot(
            df_pvalues, os.path.join(save_dir, f"sample_size_1b_auc_{nreps}.pdf")
        )
    else:
        print("WARNING: no p-value data collected (all_results_ not available).")

    if len(df_best) > 0:
        make_best_noninv_plot(
            df_best, os.path.join(save_dir, f"sample_size_1b_best_noninv_{nreps}.pdf")
        )
    if len(df_ensemble) > 0:
        make_ensemble_contamination_plot(
            df_ensemble,
            os.path.join(save_dir, f"sample_size_1b_ens_contamination_{nreps}.pdf"),
        )
        make_score_comparison_plot(
            df_ensemble,
            os.path.join(save_dir, f"sample_size_1b_score_comparison_{nreps}.pdf"),
        )


if __name__ == "__main__":
    main()
