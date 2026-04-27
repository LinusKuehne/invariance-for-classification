"""
Evaluate pairwise AUC of invariance tests for separating
ground-truth invariant from non-invariant subsets, using pre-computed results.

This script parses the p-values / scores saved by evaluate_OOD_predictions.py
and computes the AUC without re-running any tests or re-subsampling data.

================================================================================
USAGE
================================================================================
    python evaluate_invariance_AUC_from_OOD.py --dataset 1a
    python evaluate_invariance_AUC_from_OOD.py --dataset 2 --n-obs 200
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# global configuration
# =============================================================================

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


# =============================================================================
# pairwise AUC
# =============================================================================


def _pairwise_auc(inv_values: np.ndarray, noninv_values: np.ndarray) -> float:
    """
    Fraction of (inv, noninv) pairs where inv > noninv.

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
# summary
# =============================================================================


def _ci_half_width(values: np.ndarray, confidence: float = 0.95) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(t_crit * se)


def compute_summary(df: pd.DataFrame, summary_path: str) -> pd.DataFrame:
    """Recompute summary stats and save."""
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
]

TABLE_NONLINEAR: list[tuple[str, str]] = [
    ("Residual (RF)", r"\quad IRD(RF)"),
    ("TramGCM (RF)", r"\quad \textsc{tram}-GCM(RF)"),
    ("DeLong (RF)", r"\quad ITP(RF)"),
    ("InvEnvPred (RF)", r"\quad IEP(RF)"),
    # ("WGCM_est", r"\quad WGCM\textsubscript{est}"),
    # ("WGCM_fix", r"\quad WGCM\textsubscript{fix}"),
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

    a(r"\begin{tabular}{@{}lccc@{}}")
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
) -> None:
    """Run the pairwise AUC evaluation using stored pvalues for one dataset."""

    # ── output directory ─────────────────────────────────────────────────
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    save_dir = os.path.join(repo_root, "results", dataset)
    os.makedirs(save_dir, exist_ok=True)

    pvalues_path = os.path.join(
        save_dir, f"OOD_sc_pvalues_{dataset}_n{n_obs_per_env}.csv"
    )
    if not os.path.exists(pvalues_path):
        print(f"Error: Could not find precomputed p-values at {pvalues_path}")
        print("Please run evaluate_OOD_predictions.py first to generate them.")
        return

    raw_path = os.path.join(save_dir, f"AUC_raw_{dataset}_n{n_obs_per_env}.csv")
    summary_path = os.path.join(save_dir, f"AUC_summary_{dataset}_n{n_obs_per_env}.csv")

    # ── load training data ───────────────────────────────────────────────
    df_train_full = _load_train(dataset)
    features = [c for c in df_train_full.columns if c not in ("Y", "E")]
    invariant_subsets = _get_invariant_subsets(dataset, features)

    print("=" * 70)
    print("Invariance Test AUC Evaluation (from precomputed P-values)")
    print("=" * 70)
    print(f"Dataset          : {dataset}")
    print(f"Features ({len(features):d})     : {features}")
    print(f"Invariant subsets: {len(invariant_subsets)}")
    print(f"N_OBS_PER_ENV    : {n_obs_per_env}")
    print(f"Results File     : {pvalues_path}")

    # ── processing ───────────────────────────────────────────────────────
    df_pvals = pd.read_csv(pvalues_path)

    results = []

    t_start = time.time()

    # We iterate over reps directly from the logged files
    all_reps = sorted(df_pvals["rep"].unique())
    n_reps = len(all_reps)
    for rep in all_reps:
        print(f"\n[{rep + 1}/{n_reps}] Repetition {rep}:")
        df_rep = df_pvals[df_pvals["rep"] == rep]

        for config in sorted(df_rep["sc_config"].unique()):
            df_config = df_rep[df_rep["sc_config"] == config]

            inv_vals = []
            noninv_vals = []

            for _, row in df_config.iterrows():
                subset_list = json.loads(row["subset"])
                subset = frozenset(subset_list)
                val = row["p_value_or_score"]

                if pd.isna(val):
                    continue

                if subset in invariant_subsets:
                    inv_vals.append(val)
                else:
                    noninv_vals.append(val)

            auc = _pairwise_auc(np.array(inv_vals), np.array(noninv_vals))

            # Map test names like 'Residual(LR)' -> 'Residual (LR)' for consistency with established labels
            test_name = config.replace("(", " (") if "(" in config else config

            results.append({"rep": rep, "test_name": test_name, "auc": auc})

            print(f"  {test_name:<25s}  AUC = {auc:.4f}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All {n_reps} repetitions evaluated in {elapsed:.1f} sec")
    print(f"{'=' * 70}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(raw_path, index=False)

    # ── summary ──────────────────────────────────────────────────────────
    summary_df = compute_summary(df_results, summary_path)

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
        description="Evaluate AUC directly from p-values generated by evaluate_OOD_predictions.",
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
        "--generate-latex-table",
        action="store_true",
        help="Skip subset evaluation; aggregate existing results across all datasets to generate the final LaTeX table.",
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
    latex_path = os.path.join(results_dir, "latex_table_AUC_from_existing.txt")

    if args.generate_latex_table:
        generate_latex_table(results_dir, latex_path, args.decimals, args.n_obs)
    else:
        main(
            dataset=args.dataset,
            n_obs_per_env=args.n_obs,
        )
