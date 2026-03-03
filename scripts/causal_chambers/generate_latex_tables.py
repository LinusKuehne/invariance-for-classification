"""
Generate LaTeX tables from OOD evaluation results.

Reads the raw per-rep CSV files produced by evaluate_OOD_predictions.py
for datasets 1a, 1b, and 2, and outputs two LaTeX tables:

  1. tab:main-results          - worst-case accuracy for all methods (ensemble)
  2. tab:appendix-ensemble-vs-best - ensemble vs. best-subset for SC methods

The metric is: minimum accuracy over test environments per rep, then
mean ± half-width of a 95% t-CI across reps.

================================================================================
USAGE
================================================================================
    python generate_latex_tables.py
    python generate_latex_tables.py --results-dir ../../results --output tables.txt
    python generate_latex_tables.py --decimals 3
    python generate_latex_tables.py --n-obs 1000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# configuration
# =============================================================================

DATASETS = ["1a", "1b", "2"]

# Mapping: (csv_method_name) → LaTeX display name
# For SC methods the csv name is like "SC Residual(RF), pred=RF"
# or "SC Residual(RF), pred=RF, best".

# Table 1: main results (ensemble only, test_clf == pred_clf)
# Ordered exactly as in the LaTeX template.
TABLE1_LINEAR: list[tuple[str, str]] = [
    ("SC Residual(LR), pred=LR", r"\quad SC-IRD(LR)"),
    ("SC TramGCM(LR), pred=LR", r"\quad SC-\textsc{tram}-GCM(LR)"),
    ("SC DeLong(LR), pred=LR", r"\quad SC-ITP(LR)"),
    ("SC LOEO(LR), pred=LR", r"\quad SC-LOEO(LR)"),
]

TABLE1_NONLINEAR: list[tuple[str, str]] = [
    ("SC Residual(RF), pred=RF", r"\quad SC-IRD(RF)"),
    ("SC TramGCM(RF), pred=RF", r"\quad SC-\textsc{tram}-GCM(RF)"),
    ("SC DeLong(RF), pred=RF", r"\quad SC-ITP(RF)"),
    ("SC LOEO(RF), pred=RF", r"\quad SC-LOEO(RF)"),
    ("SC InvEnvPred(RF), pred=RF", r"\quad SC-IEP(RF)"),
    ("SC WGCM_est, pred=RF", r"\quad SC-WGCM\textsubscript{est}"),
    ("SC WGCM_fix, pred=RF", r"\quad SC-WGCM\textsubscript{fix}"),
]

TABLE1_BASELINES: list[tuple[str, str]] = [
    ("ERM LR", r"\quad ERM-LR"),
    ("ERM RF", r"\quad ERM-RF"),
    ("IRM-Linear", r"\quad IRM-Linear"),
    ("IRM-NN", r"\quad IRM-NN"),
]

TABLE1_ORACLE: list[tuple[str, str]] = [
    ("Oracle LR", r"\quad Oracle-LR"),
    ("Oracle RF", r"\quad Oracle-RF"),
]

# Table 2: ensemble vs best (test_clf == pred_clf)
# Each entry produces two rows (ensemble + best).
TABLE2_LINEAR: list[tuple[str, str, str]] = [
    # (csv_ensemble, csv_best, latex_name)
    (
        "SC Residual(LR), pred=LR",
        "SC Residual(LR), pred=LR, best",
        r"\quad \multirow{2}{*}{SC-IRD(LR)}",
    ),
    (
        "SC TramGCM(LR), pred=LR",
        "SC TramGCM(LR), pred=LR, best",
        r"\quad \multirow{2}{*}{SC-\textsc{tram}-GCM(LR)}",
    ),
    (
        "SC DeLong(LR), pred=LR",
        "SC DeLong(LR), pred=LR, best",
        r"\quad \multirow{2}{*}{SC-ITP(LR)}",
    ),
    (
        "SC LOEO(LR), pred=LR",
        "SC LOEO(LR), pred=LR, best",
        r"\quad \multirow{2}{*}{SC-LOEO(LR)}",
    ),
]

TABLE2_NONLINEAR: list[tuple[str, str, str]] = [
    (
        "SC Residual(RF), pred=RF",
        "SC Residual(RF), pred=RF, best",
        r"\quad \multirow{2}{*}{SC-IRD(RF)}",
    ),
    (
        "SC TramGCM(RF), pred=RF",
        "SC TramGCM(RF), pred=RF, best",
        r"\quad \multirow{2}{*}{SC-\textsc{tram}-GCM(RF)}",
    ),
    (
        "SC DeLong(RF), pred=RF",
        "SC DeLong(RF), pred=RF, best",
        r"\quad \multirow{2}{*}{SC-ITP(RF)}",
    ),
    (
        "SC LOEO(RF), pred=RF",
        "SC LOEO(RF), pred=RF, best",
        r"\quad \multirow{2}{*}{SC-LOEO(RF)}",
    ),
    (
        "SC InvEnvPred(RF), pred=RF",
        "SC InvEnvPred(RF), pred=RF, best",
        r"\quad \multirow{2}{*}{SC-IEP(RF)}",
    ),
    (
        "SC WGCM_est, pred=RF",
        "SC WGCM_est, pred=RF, best",
        r"\quad \multirow{2}{*}{SC-WGCM\textsubscript{est}}",
    ),
    (
        "SC WGCM_fix, pred=RF",
        "SC WGCM_fix, pred=RF, best",
        r"\quad \multirow{2}{*}{SC-WGCM\textsubscript{fix}}",
    ),
]


# =============================================================================
# helpers
# =============================================================================


def _ci_half_width(values: np.ndarray, confidence: float = 0.95) -> float:
    """Half-width of a t-CI."""
    n = len(values)
    if n < 2:
        return float("nan")
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(t_crit * se)


def _load_min_acc(results_dir: str, dataset: str, n_obs: int) -> dict[str, np.ndarray]:
    """
    Load raw CSV and return {method: array_of_min_acc_per_rep}.

    The min is taken over test environments within each rep.
    """
    path = os.path.join(results_dir, dataset, f"OOD_raw_{dataset}_n{n_obs}.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping dataset {dataset}")
        return {}

    df = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for method in df["method"].unique():
        m = df[df["method"] == method]
        min_accs = m.groupby("rep")["accuracy"].min().to_numpy()
        out[method] = min_accs
    return out


def _fmt(mean: float, hw: float, decimals: int, bold: bool = False) -> str:
    """Format as $mean \\pm hw$ with given decimal places, optionally bold."""
    inner = f"{mean:.{decimals}f} \\pm {hw:.{decimals}f}"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def _mean_acc(
    data: dict[str, dict[str, np.ndarray]],
    dataset: str,
    method: str,
) -> float | None:
    """Return the mean worst-case accuracy for a method/dataset, or None."""
    vals = data.get(dataset, {}).get(method)
    if vals is None or len(vals) == 0:
        return None
    return float(np.mean(vals))


def _find_best_methods(
    data: dict[str, dict[str, np.ndarray]],
    csv_names: list[str],
) -> dict[str, set[str]]:
    """For each dataset, find which csv_names have the highest mean.

    Returns {dataset: set_of_best_csv_names}.
    """
    best: dict[str, set[str]] = {}
    for ds in DATASETS:
        best_val = -1.0
        best_methods: set[str] = set()
        for name in csv_names:
            m = _mean_acc(data, ds, name)
            if m is None:
                continue
            if m > best_val + 1e-12:
                best_val = m
                best_methods = {name}
            elif abs(m - best_val) < 1e-12:
                best_methods.add(name)
        best[ds] = best_methods
    return best


def _cell(
    data: dict[str, dict[str, np.ndarray]],
    dataset: str,
    method: str,
    decimals: int,
    bold: bool = False,
) -> str:
    """Produce one table cell (mean ± CI) or a placeholder."""
    ds = data.get(dataset, {})
    vals = ds.get(method)
    if vals is None or len(vals) == 0:
        return "---"
    mean = float(np.mean(vals))
    hw = _ci_half_width(vals)
    return _fmt(mean, hw, decimals, bold=bold)


# =============================================================================
# table builders
# =============================================================================


def build_table1(
    data: dict[str, dict[str, np.ndarray]], decimals: int, n_obs: int
) -> str:
    """Build Table 1: main results."""
    # Determine best per column (excluding oracle)
    all_non_oracle = [
        csv for csv, _ in TABLE1_LINEAR + TABLE1_NONLINEAR + TABLE1_BASELINES
    ]
    best = _find_best_methods(data, all_non_oracle)

    def _row(csv_name: str, latex_name: str) -> list[str]:
        cells = " & ".join(
            _cell(data, ds, csv_name, decimals, bold=(csv_name in best.get(ds, set())))
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

    # --- SC linear ---
    a(r"    % --- Stabilized classification (linear) --------------------------------")
    a(r"    \multicolumn{4}{@{}l}{\textbf{Stabilized classification (linear)}} \\[2pt]")
    for csv_name, latex_name in TABLE1_LINEAR:
        lines.extend(_row(csv_name, latex_name))
    a(r"    %")

    # --- SC nonlinear ---
    a(r"    \midrule")
    a(r"    % --- Stabilized classification (nonlinear) -----------------------------")
    a(
        r"    \multicolumn{4}{@{}l}{\textbf{Stabilized classification (nonlinear)}} \\[2pt]"
    )
    for csv_name, latex_name in TABLE1_NONLINEAR:
        lines.extend(_row(csv_name, latex_name))

    a(r"    %")

    # --- Baselines ---
    a(r"    \midrule")
    a(r"    % --- Baselines ---------------------------------------------------------")
    a(r"    \multicolumn{4}{@{}l}{\textbf{Baselines}} \\[2pt]")
    for csv_name, latex_name in TABLE1_BASELINES:
        lines.extend(_row(csv_name, latex_name))

    a(r"    %")

    # --- Oracle ---
    a(r"    \midrule")
    a(r"    % --- Oracle ------------------------------------------------------------")
    a(r"    \multicolumn{4}{@{}l}{\textbf{Oracle (stable blanket)}} \\[2pt]")
    for csv_name, latex_name in TABLE1_ORACLE:
        cells = " & ".join(_cell(data, ds, csv_name, decimals) for ds in DATASETS)
        a(f"    {latex_name}")
        a(f"      & {cells} \\\\")

    a(r"    \bottomrule")
    a(r"  \end{tabular}")
    a(r"  \caption{%")
    a(r"    Worst-case test accuracy (minimum over five test environments) for")
    a(r"    stabilized classification (SC) configurations and baselines.")
    a(r"    Entries show the mean $\pm$ half-width of a 95\% $t$-confidence")
    a(f"    interval over 20 repetitions with $n = {n_obs}$ observations per")
    a(r"    training environment.%")
    a(r"  }")
    a(r"  \label{tab:main-results}")
    a(r"\end{table}")

    return "\n".join(lines)


def _table2_row_pair(
    data: dict[str, dict[str, np.ndarray]],
    csv_ensemble: str,
    csv_best: str,
    latex_name: str,
    decimals: int,
    best_methods: dict[str, set[str]],
    spacing: str = r"\\[3pt]",
) -> list[str]:
    """Build the two-row block for one SC method in Table 2."""
    ens_cells = " & ".join(
        _cell(
            data,
            ds,
            csv_ensemble,
            decimals,
            bold=(csv_ensemble in best_methods.get(ds, set())),
        )
        for ds in DATASETS
    )
    best_cells = " & ".join(
        _cell(
            data,
            ds,
            csv_best,
            decimals,
            bold=(csv_best in best_methods.get(ds, set())),
        )
        for ds in DATASETS
    )
    return [
        f"    {latex_name}",
        "      & ensemble",
        f"        & {ens_cells} \\\\",
        "      & best subset",
        f"        & {best_cells} {spacing}",
    ]


def build_table2(
    data: dict[str, dict[str, np.ndarray]], decimals: int, n_obs: int
) -> str:
    """Build Table 2: ensemble vs best subset."""
    # Determine best per column across all ensemble + best methods
    all_t2_csv: list[str] = []
    for csv_ens, csv_best, _ in TABLE2_LINEAR + TABLE2_NONLINEAR:
        all_t2_csv.extend([csv_ens, csv_best])
    best = _find_best_methods(data, all_t2_csv)

    lines: list[str] = []
    a = lines.append

    a(r"\begin{table}[t]")
    a(r"  \centering")
    a(r"  \begin{tabular}{@{}llccc@{}}")
    a(r"    \toprule")
    a(r"    Method & Aggregation")
    a(r"      & {Dataset 1a}")
    a(r"      & {Dataset 1b}")
    a(r"      & {Dataset 2} \\")
    a(r"    \midrule")
    a(r"    %")

    # --- SC linear ---
    a(r"    % --- Stabilized classification (linear) --------------------------------")
    a(r"    \multicolumn{5}{@{}l}{\textbf{Stabilized classification (linear)}} \\[2pt]")
    for i, (csv_ens, csv_best, latex_name) in enumerate(TABLE2_LINEAR):
        sp = r"\\[4pt]" if i == len(TABLE2_LINEAR) - 1 else r"\\[3pt]"
        lines.extend(
            _table2_row_pair(data, csv_ens, csv_best, latex_name, decimals, best, sp)
        )

    a(r"    %")

    # --- SC nonlinear ---
    a(r"    % --- Stabilized classification (nonlinear) -----------------------------")
    a(r"    \midrule")
    a(
        r"    \multicolumn{5}{@{}l}{\textbf{Stabilized classification (nonlinear)}} \\[2pt]"
    )
    for i, (csv_ens, csv_best, latex_name) in enumerate(TABLE2_NONLINEAR):
        sp = r"\\" if i == len(TABLE2_NONLINEAR) - 1 else r"\\[3pt]"
        lines.extend(
            _table2_row_pair(data, csv_ens, csv_best, latex_name, decimals, best, sp)
        )

    a(r"    \bottomrule")
    a(r"  \end{tabular}")
    a(r"  \caption{%")
    a(r"    Ensemble versus single best invariant subset for all SC")
    a(r"    configurations from Table~\ref{tab:main-results}.  ``Ensemble''")
    a(r"    aggregates predictions across all predictive invariant subsets")
    a(r"    (the default SC method); ``best subset'' uses only the single")
    a(r"    most predictive invariant subset.  Entries show worst-case test")
    a(f"    accuracy (mean $\\pm$ 95\\% CI, 20 repetitions, $n = {n_obs}$).%")
    a(r"  }")
    a(r"  \label{tab:appendix-ensemble-vs-best}")
    a(r"\end{table}")

    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================


def main(results_dir: str, output: str, decimals: int, n_obs: int) -> None:
    # Load data for all datasets
    data: dict[str, dict[str, np.ndarray]] = {}
    for ds in DATASETS:
        data[ds] = _load_min_acc(results_dir, ds, n_obs)

    table1 = build_table1(data, decimals, n_obs)
    table2 = build_table2(data, decimals, n_obs)

    full = (
        "% =============================================================================\n"
        "% Table 1: Main results\n"
        "% =============================================================================\n"
        "\n"
        f"{table1}\n"
        "\n\n"
        "% =============================================================================\n"
        "% Table 2: Ensemble vs. best subset\n"
        "% =============================================================================\n"
        "\n"
        f"{table2}\n"
    )

    with open(output, "w") as f:
        f.write(full)

    print(f"Tables written to {output}")
    print("\nPreview (first 40 lines):\n")
    for line in full.splitlines()[:40]:
        print(line)
    print("...")


if __name__ == "__main__":
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    default_results = os.path.join(repo_root, "results")
    default_output = os.path.join(repo_root, "results", "latex_tables.txt")

    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from OOD evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        default=default_results,
        help=f"Directory containing per-dataset result folders (default: {default_results}).",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output file path (default: {default_output}).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places for mean and CI (default: 2).",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=200,
        help="Number of observations per environment used in the evaluation run (default: 200).",
    )
    args = parser.parse_args()
    main(
        results_dir=args.results_dir,
        output=args.output,
        decimals=args.decimals,
        n_obs=args.n_obs,
    )
