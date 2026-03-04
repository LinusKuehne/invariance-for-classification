"""
Evaluate pairwise AUC of invariance tests and LOEO rankings for separating
ground-truth invariant from non-invariant subsets.

For each of N_REPS repetitions:
  1. Draw a random subsample of N_OBS_PER_ENV observations per training
     environment from a causal chambers dataset.
  2. Compute p-values for all invariance tests on all 2^d subsets
     (and LOEO regret rankings).
  3. Using the ground-truth invariant/non-invariant labels, compute
     the pairwise AUC: fraction of (invariant, non-invariant) pairs
     where the invariant subset receives a higher p-value (or score).

Results are averaged over repetitions with 95% t-confidence intervals.
Only training data is used - no test data is needed.

================================================================================
INVARIANCE TESTS
================================================================================
  Linear:
    - DeLong (LR)
    - Residual (LR)
    - TramGCM (LR)
    - LOEO Regret (LR, mean)

  Nonlinear:
    - DeLong (RF)
    - Residual (RF)
    - TramGCM (RF)
    - InvEnvPred (RF)
    - WGCM_est (xgb)
    - WGCM_fix (xgb)
    - LOEO Regret (RF, mean)

================================================================================
OUTPUT
================================================================================
Results are saved to: <repo_root>/results/<dataset>/

  - AUC_raw_<dataset>_n<n_obs>.csv      Per-rep, per-test AUC values
  - AUC_summary_<dataset>_n<n_obs>.csv   Mean ± 95% t-CI over repetitions

================================================================================
USAGE
================================================================================
    python evaluate_invariance_AUC.py --dataset 1a
    python evaluate_invariance_AUC.py --dataset 2 --n-obs 200 --n-reps 20 --workers 10
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

# Limit threads for OpenMP-based libraries to allow proper parallelization
# at the process level. Must be set before importing sklearn/xgboost.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from invariance_for_classification.invariance_tests import (
    DeLongTest,
    InvariantEnvironmentPredictionTest,
    InvariantResidualDistributionTest,
    TramGcmTest,
    WGCMTest,
)
from invariance_for_classification.rankings import loeo_regret

# =============================================================================
# global configuration
# =============================================================================

SEED = 42

DATASETS = ["1a", "1b", "2"]

NORMAL_COLS: dict[str, list[str]] = {
    "1a": ["Y", "red", "green", "blue", "ir_1", "vis_1", "ir_3", "vis_3", "E"],
    "1b": ["Y", "red", "green", "blue", "ir_1", "vis_1", "ir_3", "vis_3", "E"],
    "2": ["Y", "red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3", "E"],
}

# =============================================================================
# invariant subset ground truth
# =============================================================================


def _get_invariant_subsets(name: str, features: list[str]) -> set[frozenset[str]]:
    """Return set of ground-truth invariant subsets for a given dataset."""
    if name in ["1a", "1b"]:
        base_subsets = [
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
        optional_cols = [
            f for f in features if f not in {"red", "green", "blue", "vis_3", "ir_3"}
        ]
        optional_subsets = [frozenset()]
        for r in range(1, len(optional_cols) + 1):
            for combo in itertools.combinations(optional_cols, r):
                optional_subsets.append(frozenset(combo))
        invariant_subsets = set()
        for base in base_subsets:
            for addition in optional_subsets:
                invariant_subsets.add(base | addition)
        return invariant_subsets

    if name == "2":
        base_subsets = [frozenset({"red", "green", "blue"})]
        optional_cols = [
            f for f in features if f not in {"red", "green", "blue", "ir_3", "vis_3"}
        ]
        optional_subsets = [frozenset()]
        for r in range(1, len(optional_cols) + 1):
            for combo in itertools.combinations(optional_cols, r):
                optional_subsets.append(frozenset(combo))
        invariant_subsets = set()
        for base in base_subsets:
            for addition in optional_subsets:
                invariant_subsets.add(base | addition)
        return invariant_subsets

    return set()


# =============================================================================
# data loading helpers
# =============================================================================


def _load_train(dataset: str, data_dir: str | None = None) -> pd.DataFrame:
    """Load training CSV, filtered to NORMAL_COLS."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_path = os.path.join(data_dir, f"{dataset}_train.csv")
    if not os.path.exists(train_path):
        sys.exit(f"Training file not found: {train_path}")
    df = pd.read_csv(train_path)
    if dataset in NORMAL_COLS:
        cols = [c for c in NORMAL_COLS[dataset] if c in df.columns]
        df = df[cols]
    return df


def _subsample_train(
    df_train: pd.DataFrame, n_obs_per_env: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Subsample exactly n_obs_per_env observations per environment."""
    dfs = []
    for env in sorted(df_train["E"].unique()):
        env_df = df_train[df_train["E"] == env]
        if len(env_df) <= n_obs_per_env:
            dfs.append(env_df)
        else:
            idx = rng.choice(len(env_df), size=n_obs_per_env, replace=False)
            dfs.append(env_df.iloc[idx])
    return pd.concat(dfs, ignore_index=True)


def _get_all_subsets(features: list[str]) -> list[list[str]]:
    """Generate all 2^d subsets of features (including empty set)."""
    subsets = []
    for L in range(len(features) + 1):
        for subset in itertools.combinations(features, L):
            subsets.append(list(subset))
    return subsets


# =============================================================================
# test definitions
# =============================================================================

# (display_name, constructor_callable, is_loeo)
# LOEO entries are handled separately because they use a different interface.

TEST_CONFIGS_PVALUE: list[tuple[str, Any]] = [
    ("DeLong (LR)", lambda: DeLongTest(test_classifier_type="LR")),
    ("DeLong (RF)", lambda: DeLongTest(test_classifier_type="RF")),
    (
        "InvEnvPred (RF)",
        lambda: InvariantEnvironmentPredictionTest(test_classifier_type="RF"),
    ),
    (
        "Residual (LR)",
        lambda: InvariantResidualDistributionTest(test_classifier_type="LR"),
    ),
    (
        "Residual (RF)",
        lambda: InvariantResidualDistributionTest(test_classifier_type="RF"),
    ),
    ("TramGCM (LR)", lambda: TramGcmTest(test_classifier_type="LR")),
    ("TramGCM (RF)", lambda: TramGcmTest(test_classifier_type="RF")),
    ("WGCM_est (xgb)", lambda: WGCMTest(method="est", beta=0.5)),
    ("WGCM_fix (xgb)", lambda: WGCMTest(method="fix")),
]

LOEO_CONFIGS: list[tuple[str, Literal["RF", "LR"]]] = [
    # (display_name, classifier_type)
    ("LOEO Regret (LR)", "LR"),
    ("LOEO Regret (RF)", "RF"),
]


# =============================================================================
# pairwise AUC
# =============================================================================


def _pairwise_auc(inv_values: np.ndarray, noninv_values: np.ndarray) -> float:
    """
    Fraction of (inv, noninv) pairs where inv > noninv.

    Higher p-value / higher LOEO score ⇒ more invariant, so higher is better.
    Returns value in [0, 1]; 1.0 = perfect separation, 0.5 = random.
    """
    count = sum(
        1.0 if i > n else 0.5 if i == n else 0.0
        for i in inv_values
        for n in noninv_values
    )
    total = len(inv_values) * len(noninv_values)
    return count / total if total > 0 else np.nan


# =============================================================================
# worker functions (run inside child processes)
# =============================================================================


def _run_pvalue_test_on_all_subsets(
    test_name: str,
    df: pd.DataFrame,
    all_subsets: list[list[str]],
    invariant_subsets: set[frozenset[str]],
) -> dict[str, Any]:
    """
    Run one invariance test on all subsets and return the pairwise AUC.

    This function is the unit of parallelisation (one per test x rep).
    The test object is created inside the worker.

    Returns
    -------
    dict with keys 'test_name' and 'auc'.
    """
    # Look up the correct constructor
    constructor = None
    for name, ctor in TEST_CONFIGS_PVALUE:
        if name == test_name:
            constructor = ctor
            break
    if constructor is None:
        return {"test_name": test_name, "auc": np.nan}

    test = constructor()

    y = df["Y"].to_numpy()
    E = df["E"].to_numpy()

    inv_values = []
    noninv_values = []

    for subset in all_subsets:
        X = np.zeros((len(df), 0)) if not subset else df[subset].to_numpy()
        try:
            pval = test.test(X, y, E)
        except Exception:
            pval = np.nan

        is_inv = frozenset(subset) in invariant_subsets
        if np.isnan(pval):
            continue
        if is_inv:
            inv_values.append(pval)
        else:
            noninv_values.append(pval)

    auc = _pairwise_auc(np.array(inv_values), np.array(noninv_values))
    return {"test_name": test_name, "auc": auc}


def _run_loeo_on_all_subsets(
    display_name: str,
    classifier_type: Literal["RF", "LR"],
    df: pd.DataFrame,
    all_subsets: list[list[str]],
    invariant_subsets: set[frozenset[str]],
) -> dict[str, Any]:
    """
    Run LOEO regret ranking on all subsets and return pairwise AUC.

    Uses the 'mean' aggregation of per-environment regrets.
    """
    y = df["Y"].to_numpy()
    E = df["E"].to_numpy()

    inv_values = []
    noninv_values = []

    for subset in all_subsets:
        X = np.zeros((len(df), 0)) if not subset else df[subset].to_numpy()
        try:
            scores = loeo_regret(y, E, X, classifier_type=classifier_type)
            val = scores["mean"]
        except Exception:
            val = np.nan

        is_inv = frozenset(subset) in invariant_subsets
        if np.isnan(val):
            continue
        if is_inv:
            inv_values.append(val)
        else:
            noninv_values.append(val)

    auc = _pairwise_auc(np.array(inv_values), np.array(noninv_values))
    return {"test_name": display_name, "auc": auc}


# =============================================================================
# top-level wrapper executed per rep (to be called from the main process)
# =============================================================================


def _run_one_rep(
    rep: int,
    df_train_full: pd.DataFrame,
    n_obs_per_env: int,
    features: list[str],
    all_subsets: list[list[str]],
    invariant_subsets: set[frozenset[str]],
    n_workers: int,
) -> list[dict]:
    """
    Run all tests for one repetition and return a list of
    {rep, test_name, auc} dicts.
    """
    rep_seed = SEED + rep
    rng = np.random.default_rng(rep_seed)
    df_train = _subsample_train(df_train_full, n_obs_per_env, rng)

    results: list[dict] = []

    # --- p-value tests (parallelised over tests) ---
    test_names = [name for name, _ in TEST_CONFIGS_PVALUE]

    if n_workers == 1:
        for tname in test_names:
            r = _run_pvalue_test_on_all_subsets(
                tname, df_train, all_subsets, invariant_subsets
            )
            results.append({"rep": rep, **r})
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _run_pvalue_test_on_all_subsets,
                    tname,
                    df_train,
                    all_subsets,
                    invariant_subsets,
                ): tname
                for tname in test_names
            }
            for future in as_completed(futures):
                tname = futures[future]
                try:
                    r = future.result()
                    results.append({"rep": rep, **r})
                except Exception as e:
                    print(f"  Error in {tname}: {e}")
                    results.append({"rep": rep, "test_name": tname, "auc": np.nan})

    # --- LOEO regret (sequential – already fast per subset) ---
    for display_name, clf_type in LOEO_CONFIGS:
        r = _run_loeo_on_all_subsets(
            display_name, clf_type, df_train, all_subsets, invariant_subsets
        )
        results.append({"rep": rep, **r})

    return results


# =============================================================================
# summary
# =============================================================================


def _ci_half_width(values: np.ndarray, confidence: float = 0.95) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(t_crit * se)


def compute_summary(raw_path: str, summary_path: str) -> pd.DataFrame:
    """Read per-rep AUC CSV and compute mean ± 95% t-CI."""
    df = pd.read_csv(raw_path)
    records = []
    for test_name in sorted(df["test_name"].unique()):
        mask = df["test_name"] == test_name
        vals = df[mask]["auc"].dropna().to_numpy()
        records.append(
            {
                "test_name": test_name,
                "auc_mean": float(np.mean(vals)) if len(vals) > 0 else np.nan,
                "auc_ci": _ci_half_width(vals),
                "n_reps": len(vals),
            }
        )
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    return summary_df


# =============================================================================
# LaTeX table
# =============================================================================

# Mapping from test display names to LaTeX labels
TABLE_LINEAR: list[tuple[str, str]] = [
    ("Residual (LR)", r"\quad IRD(LR)"),
    ("TramGCM (LR)", r"\quad \textsc{tram}-GCM(LR)"),
    ("DeLong (LR)", r"\quad ITP(LR)"),
    ("LOEO Regret (LR)", r"\quad LOEO(LR)"),
]

TABLE_NONLINEAR: list[tuple[str, str]] = [
    ("Residual (RF)", r"\quad IRD(RF)"),
    ("TramGCM (RF)", r"\quad \textsc{tram}-GCM(RF)"),
    ("DeLong (RF)", r"\quad ITP(RF)"),
    ("InvEnvPred (RF)", r"\quad IEP(RF)"),
    ("WGCM_est (xgb)", r"\quad WGCM\textsubscript{est}"),
    ("WGCM_fix (xgb)", r"\quad WGCM\textsubscript{fix}"),
    ("LOEO Regret (RF)", r"\quad LOEO(RF)"),
]


def _fmt(mean: float, hw: float, decimals: int, bold: bool = False) -> str:
    inner = f"{mean:.{decimals}f} \\pm {hw:.{decimals}f}"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def _load_auc_data(results_dir: str, dataset: str, n_obs: int) -> dict[str, np.ndarray]:
    """Load raw AUC CSV and return {test_name: array_of_auc_per_rep}."""
    path = os.path.join(results_dir, dataset, f"AUC_raw_{dataset}_n{n_obs}.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping dataset {dataset}")
        return {}
    df = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for test_name in df["test_name"].unique():
        mask = df["test_name"] == test_name
        vals = df[mask]["auc"].dropna().to_numpy()
        out[test_name] = vals
    return out


def _cell(
    data: dict[str, dict[str, np.ndarray]],
    dataset: str,
    method: str,
    decimals: int,
    bold: bool = False,
) -> str:
    vals = data.get(dataset, {}).get(method)
    if vals is None or len(vals) == 0:
        return "---"
    mean = float(np.mean(vals))
    hw = _ci_half_width(vals)
    return _fmt(mean, hw, decimals, bold=bold)


def _find_best(
    data: dict[str, dict[str, np.ndarray]],
    csv_names: list[str],
) -> dict[str, set[str]]:
    """For each dataset, find methods with highest mean AUC."""
    best: dict[str, set[str]] = {}
    for ds in DATASETS:
        best_val = -1.0
        best_methods: set[str] = set()
        for name in csv_names:
            vals = data.get(ds, {}).get(name)
            if vals is None or len(vals) == 0:
                continue
            m = float(np.mean(vals))
            if m > best_val + 1e-12:
                best_val = m
                best_methods = {name}
            elif abs(m - best_val) < 1e-12:
                best_methods.add(name)
        best[ds] = best_methods
    return best


def build_auc_table(
    data: dict[str, dict[str, np.ndarray]], decimals: int, n_obs: int
) -> str:
    """Build a LaTeX table of pairwise AUC for invariance tests."""
    all_methods = [csv for csv, _ in TABLE_LINEAR + TABLE_NONLINEAR]
    best = _find_best(data, all_methods)

    def _row(csv_name: str, latex_name: str) -> list[str]:
        cells = " & ".join(
            _cell(
                data,
                ds,
                csv_name,
                decimals,
                bold=(csv_name in best.get(ds, set())),
            )
            for ds in DATASETS
        )
        return [f"    {latex_name}", f"      & {cells} \\\\"]

    lines: list[str] = []
    a = lines.append

    a(r"\begin{table}[t]")
    a(r"  \centering")
    a(r"  \begin{tabular}{@{}lccc@{}}")
    a(r"    \toprule")
    a(r"    Method")
    a(r"      & {Dataset 1a}")
    a(r"      & {Dataset 1b}")
    a(r"      & {Dataset 2} \\")
    a(r"    \midrule")
    a(r"    %")

    # --- Linear ---
    a(r"    % --- Linear invariance tests -------------------------------------------")
    a(r"    \multicolumn{4}{@{}l}{\textbf{Linear invariance tests}} \\[2pt]")
    for csv_name, latex_name in TABLE_LINEAR:
        lines.extend(_row(csv_name, latex_name))
    a(r"    %")

    # --- Nonlinear ---
    a(r"    \midrule")
    a(r"    % --- Nonlinear invariance tests ----------------------------------------")
    a(r"    \multicolumn{4}{@{}l}{\textbf{Nonlinear invariance tests}} \\[2pt]")
    for csv_name, latex_name in TABLE_NONLINEAR:
        lines.extend(_row(csv_name, latex_name))

    a(r"    \bottomrule")
    a(r"  \end{tabular}")
    a(r"  \caption{%")
    a(r"    Pairwise AUC for separating ground-truth invariant from non-invariant")
    a(r"    subsets.  Higher values indicate better discrimination.")
    a(r"    Entries show the mean $\pm$ half-width of a 95\% $t$-confidence")
    a(f"    interval over 20 repetitions with $n = {n_obs}$ observations per")
    a(r"    training environment.%")
    a(r"  }")
    a(r"  \label{tab:invariance-auc}")
    a(r"\end{table}")

    return "\n".join(lines)


def generate_latex_table(
    results_dir: str, output_path: str, decimals: int, n_obs: int
) -> None:
    """Load AUC data for all datasets and write the LaTeX table."""
    data: dict[str, dict[str, np.ndarray]] = {}
    for ds in DATASETS:
        data[ds] = _load_auc_data(results_dir, ds, n_obs)

    table = build_auc_table(data, decimals, n_obs)
    with open(output_path, "w") as f:
        f.write(table + "\n")

    print(f"\nLaTeX table written to {output_path}")
    print("\nPreview:\n")
    for line in table.splitlines()[:40]:
        print(line)
    if len(table.splitlines()) > 40:
        print("...")


# =============================================================================
# main
# =============================================================================


def main(
    dataset: str = "1a",
    n_obs_per_env: int = 200,
    n_reps: int = 20,
    n_workers: int = 10,
) -> None:
    """Run the pairwise AUC evaluation for one dataset."""

    # ── output directory ─────────────────────────────────────────────────
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    save_dir = os.path.join(repo_root, "results", dataset)
    os.makedirs(save_dir, exist_ok=True)

    raw_path = os.path.join(save_dir, f"AUC_raw_{dataset}_n{n_obs_per_env}.csv")
    summary_path = os.path.join(save_dir, f"AUC_summary_{dataset}_n{n_obs_per_env}.csv")

    # Remove stale output files
    for p in (raw_path, summary_path):
        if os.path.exists(p):
            os.remove(p)

    # ── load training data ───────────────────────────────────────────────
    df_train_full = _load_train(dataset)
    features = [c for c in df_train_full.columns if c not in ("Y", "E")]
    all_subsets = _get_all_subsets(features)
    invariant_subsets = _get_invariant_subsets(dataset, features)

    print("=" * 70)
    print("Invariance Test AUC Evaluation")
    print("=" * 70)
    print(f"Dataset          : {dataset}")
    print(f"Features ({len(features):d})     : {features}")
    print(f"Full train size  : {len(df_train_full)}")
    print(f"Number of subsets: {len(all_subsets)}")
    print(f"Invariant subsets: {len(invariant_subsets)}")
    print(f"N_OBS_PER_ENV    : {n_obs_per_env}")
    print(f"N_REPS           : {n_reps}")
    print(f"N_WORKERS        : {n_workers}")
    print(f"Save directory   : {save_dir}")

    all_test_names = [name for name, _ in TEST_CONFIGS_PVALUE] + [
        name for name, _ in LOEO_CONFIGS
    ]
    print(f"\nTests ({len(all_test_names)}):")
    for t in all_test_names:
        print(f"  - {t}")

    # ── main loop ────────────────────────────────────────────────────────
    t_start = time.time()

    for rep in range(n_reps):
        t_rep = time.time()
        print(f"\n{'─' * 70}")
        print(f"Repetition {rep + 1}/{n_reps}  (seed={SEED + rep})")
        print(f"{'─' * 70}")

        rep_results = _run_one_rep(
            rep=rep,
            df_train_full=df_train_full,
            n_obs_per_env=n_obs_per_env,
            features=features,
            all_subsets=all_subsets,
            invariant_subsets=invariant_subsets,
            n_workers=n_workers,
        )

        # Append incrementally
        df_rep = pd.DataFrame(rep_results)
        header = not os.path.exists(raw_path)
        df_rep.to_csv(raw_path, mode="a", header=header, index=False)

        # Print per-test AUC for this rep
        for row in sorted(rep_results, key=lambda r: r["test_name"]):
            print(f"  {row['test_name']:<25s}  AUC = {row['auc']:.4f}")

        print(f"  (rep time: {time.time() - t_rep:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All {n_reps} repetitions complete in {elapsed / 60:.1f} min")
    print(f"{'=' * 70}")

    # ── summary ──────────────────────────────────────────────────────────
    summary_df = compute_summary(raw_path, summary_path)

    print("\nSUMMARY (mean ± 95% CI)")
    print("─" * 60)
    print(f"{'Test':<30s}  {'AUC':>20s}")
    print("─" * 60)
    for _, r in summary_df.iterrows():
        auc_str = f"{r['auc_mean']:.4f} ± {r['auc_ci']:.4f}"
        print(f"  {r['test_name']:<28s}  {auc_str:>20s}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pairwise AUC of invariance tests on causal chambers datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="1a",
        choices=["1a", "1b", "2"],
        help="Dataset base name (default: 1a).",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=200,
        help="Number of observations per environment (default: 200).",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=20,
        help="Number of repetitions (default: 20).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers for test computation (default: 10).",
    )
    parser.add_argument(
        "--latex-only",
        action="store_true",
        help="Skip evaluation; only generate the LaTeX table from existing results.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places in the LaTeX table (default: 2).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    results_dir = os.path.join(repo_root, "results")
    latex_path = os.path.join(results_dir, "latex_table_AUC.txt")

    if args.latex_only:
        generate_latex_table(results_dir, latex_path, args.decimals, args.n_obs)
    else:
        main(
            dataset=args.dataset,
            n_obs_per_env=args.n_obs,
            n_reps=args.n_reps,
            n_workers=args.workers,
        )
