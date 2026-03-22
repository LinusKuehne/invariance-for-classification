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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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


def _load_subset_counts(
    results_dir: str, dataset: str, n_obs: int
) -> dict[str, tuple[float, float]]:
    """Load SC subset counts and return mean (n_invariant, n_predictive) per csv method name.

    Returns a dict mapping csv ensemble method names (e.g. "SC Residual(LR), pred=LR")
    to (mean_n_invariant, mean_n_predictive) averaged over reps.
    """
    path = os.path.join(results_dir, dataset, f"OOD_sc_subsets_{dataset}_n{n_obs}.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    if len(df) == 0:
        return {}

    out: dict[str, tuple[float, float]] = {}
    for config_name, grp in df.groupby("sc_config"):
        mean_inv = grp["n_invariant"].mean()
        mean_pred_rf = grp["n_predictive_RF"].mean()
        mean_pred_lr = grp["n_predictive_LR"].mean()
        # Map to the csv ensemble method names used in the tables
        for pred_clf, mean_pred in [("RF", mean_pred_rf), ("LR", mean_pred_lr)]:
            csv_name = f"SC {config_name}, pred={pred_clf}"
            out[csv_name] = (mean_inv, mean_pred)
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

    a(r"\begin{tabular}{@{}lccc@{}}")
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

    a(r"  \bottomrule")
    a(r"\end{tabular}")

    return "\n".join(lines)


def _table2_row_pair(
    data: dict[str, dict[str, np.ndarray]],
    subset_data: dict[str, dict[str, tuple[float, float]]],
    csv_ensemble: str,
    csv_best: str,
    latex_name: str,
    decimals: int,
    spacing: str = r"\\[3pt]",
) -> list[str]:
    """Build the two-row block for one SC method in Table 2."""

    def _ens_cell(ds: str) -> str:
        """Ensemble cell: accuracy ± CI with subset counts appended."""
        cell = _cell(data, ds, csv_ensemble, decimals)
        counts = subset_data.get(ds, {}).get(csv_ensemble)
        if counts is not None and cell != "---":
            n_inv, n_pred = counts
            cell += f" ({n_inv:.0f}/{n_pred:.0f})"
        return cell

    ens_cells = " & ".join(_ens_cell(ds) for ds in DATASETS)
    best_cells = " & ".join(_cell(data, ds, csv_best, decimals) for ds in DATASETS)
    return [
        f"    {latex_name}",
        "      & ensemble",
        f"        & {ens_cells} \\\\",
        "      & best subset",
        f"        & {best_cells} {spacing}",
    ]


def build_table2(
    data: dict[str, dict[str, np.ndarray]],
    subset_data: dict[str, dict[str, tuple[float, float]]],
    decimals: int,
    n_obs: int,
) -> str:
    """Build Table 2: ensemble vs best subset."""
    lines: list[str] = []
    a = lines.append

    a(r"\begin{tabular}{@{}lllll@{}}")
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
            _table2_row_pair(
                data, subset_data, csv_ens, csv_best, latex_name, decimals, sp
            )
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
            _table2_row_pair(
                data, subset_data, csv_ens, csv_best, latex_name, decimals, sp
            )
        )

    a(r"  \bottomrule")
    a(r"\end{tabular}")

    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================


# =============================================================================
# spider / radar charts
# =============================================================================

# Consistent color mapping: SC-TramGCM=blue, IRM=orange, ERM=red, Oracle=yellow
SPIDER_COLORS = ["#4472c4", "#e8943a", "#c75b5b", "#f0d890"]
SPIDER_ALPHA = 0.20

SPIDER_METHODS_LINEAR: list[tuple[str, str]] = [
    ("SC TramGCM(LR), pred=LR", "SC-TramGCM(LR)"),
    ("IRM-Linear", "IRM-Linear"),
    ("ERM LR", "ERM-LR"),
    ("Oracle LR", "Oracle-LR"),
]

SPIDER_METHODS_NONLINEAR: list[tuple[str, str]] = [
    ("SC TramGCM(RF), pred=RF", "SC-TramGCM(RF)"),
    ("IRM-NN", "IRM-NN"),
    ("ERM RF", "ERM-RF"),
    ("Oracle RF", "Oracle-RF"),
]


def build_spider_charts(
    env_data: dict[str, dict[str, dict[int, np.ndarray]]],
    methods: list[tuple[str, str]],
    output_path: str,
    grid_step: float = 0.1,
    grid_step_overrides: dict[str, float] | None = None,
) -> None:
    """Save three side-by-side radar charts (one per dataset) to *output_path*.

    *methods* is a list of (csv_name, display_label) pairs; colors are taken from
    SPIDER_COLORS in order.
    """
    dataset_labels = {"1a": "Dataset 1a", "1b": "Dataset 1b", "2": "Dataset 2"}

    fig, axes = plt.subplots(1, 3, subplot_kw=dict(polar=True), figsize=(13, 4.5))

    for ax, ds in zip(axes, DATASETS, strict=False):
        ds_data = env_data.get(ds, {})

        all_envs: set[int] = set()
        for csv_name, _ in methods:
            if csv_name in ds_data:
                all_envs.update(ds_data[csv_name].keys())
        envs = sorted(all_envs)
        N = len(envs)

        if N == 0:
            ax.set_visible(False)
            continue

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        # Y-axis: floor of min-mean to nearest step, ceiling at 1.0
        step = (grid_step_overrides or {}).get(ds, grid_step)
        all_means = [
            float(np.mean(arr))
            for csv_name, _ in methods
            for env in envs
            for arr in [ds_data.get(csv_name, {}).get(env)]
            if arr is not None and len(arr) > 0
        ]
        raw_min = min(all_means, default=0.5)
        raw_max = max(all_means, default=1.0)
        y_min = max(0.0, np.floor(raw_min / step) * step)
        y_max = min(1.0, np.ceil(raw_max / step) * step)
        tick_vals = np.arange(y_min, y_max + step / 2, step)

        for (csv_name, _), color in zip(methods, SPIDER_COLORS, strict=False):
            values = [
                float(np.mean(arr))
                if (arr := ds_data.get(csv_name, {}).get(env)) is not None
                and len(arr) > 0
                else float("nan")
                for env in envs
            ]
            values += values[:1]

            if all(np.isnan(v) for v in values[:-1]):
                continue

            ax.plot(angles, values, color=color, linewidth=2)
            ax.fill(angles, values, color=color, alpha=SPIDER_ALPHA)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([str(e) for e in envs], fontsize=15)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(tick_vals)
        decimals = 2 if step < 0.1 else 1
        ax.set_yticklabels(
            [f"{v:.{decimals}f}" for v in tick_vals], fontsize=13, color="grey"
        )
        ax.set_title(dataset_labels.get(ds, f"Dataset {ds}"), pad=15, fontsize=17)
        ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    legend_handles = [
        mpatches.Patch(color=color, label=label, alpha=0.8)
        for (_, label), color in zip(methods, SPIDER_COLORS, strict=False)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(methods),
        fontsize=15,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Spider charts written to {output_path}")


# =============================================================================
# per-environment accuracy table
# =============================================================================

# Methods to include in the per-environment table, in column order.
# Tuple: (csv_name, short LaTeX label)
ENV_TABLE_METHODS: list[tuple[str, str]] = [
    ("SC TramGCM(LR), pred=LR", r"SC-\textsc{tram}-GCM(LR)"),
    ("SC TramGCM(RF), pred=RF", r"SC-\textsc{tram}-GCM(RF)"),
    ("ERM LR", r"ERM-LR"),
    ("ERM RF", r"ERM-RF"),
    ("IRM-Linear", r"IRM-Linear"),
    ("IRM-NN", r"IRM-NN"),
    ("Oracle LR", r"Oracle-LR"),
    ("Oracle RF", r"Oracle-RF"),
]


def _load_env_acc(
    results_dir: str, dataset: str, n_obs: int
) -> dict[str, dict[int, np.ndarray]]:
    """
    Return {method: {env: array_of_acc_over_reps}}.
    """
    path = os.path.join(results_dir, dataset, f"OOD_raw_{dataset}_n{n_obs}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict[int, np.ndarray]] = {}
    for method in df["method"].unique():
        m = df[df["method"] == method]
        env_dict: dict[int, np.ndarray] = {}
        for env in sorted(m["env"].unique()):
            env_dict[int(env)] = m[m["env"] == env]["accuracy"].to_numpy()
        out[method] = env_dict
    return out


def _env_cell(
    env_data: dict[str, dict[int, np.ndarray]],
    method: str,
    env: int,
    decimals: int,
    bold: bool = False,
) -> str:
    """Produce one table cell (mean ± CI) for a given method and environment."""
    vals = env_data.get(method, {}).get(env)
    if vals is None or len(vals) == 0:
        return "---"
    mean = float(np.mean(vals))
    hw = _ci_half_width(vals)
    return _fmt(mean, hw, decimals, bold=bold)


def build_env_acc_table(
    env_data: dict[str, dict[int, np.ndarray]],
    dataset: str,
    decimals: int,
) -> str:
    """Build a per-environment accuracy table for one dataset."""
    # Collect all test environments present in the data
    all_envs: set[int] = set()
    for method_envs in env_data.values():
        all_envs.update(method_envs.keys())
    envs = sorted(all_envs)

    n_methods = len(ENV_TABLE_METHODS)
    col_spec = "@{}l" + "c" * n_methods + "@{}"

    lines: list[str] = []
    a = lines.append

    a(rf"\begin{{tabular}}{{{col_spec}}}")
    a(r"    \toprule")

    # Header row
    header_cells = " & ".join(
        rf"\rotatebox{{90}}{{\strut {lname}}}" for _, lname in ENV_TABLE_METHODS
    )
    a(rf"    Test env & {header_cells} \\")
    a(r"    \midrule")

    for env in envs:
        cells: list[str] = []
        for csv_name, _ in ENV_TABLE_METHODS:
            cells.append(_env_cell(env_data, csv_name, env, decimals, bold=False))

        row_cells = " & ".join(cells)
        a(rf"    {env} & {row_cells} \\")

    a(r"    \bottomrule")
    a(r"\end{tabular}")

    return "\n".join(lines)


def main(
    results_dir: str,
    output: str,
    decimals: int,
    n_obs: int,
    spider_output: str | None = None,
) -> None:
    # Load data for all datasets
    data: dict[str, dict[str, np.ndarray]] = {}
    subset_data: dict[str, dict[str, tuple[float, float]]] = {}
    env_data: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for ds in DATASETS:
        data[ds] = _load_min_acc(results_dir, ds, n_obs)
        subset_data[ds] = _load_subset_counts(results_dir, ds, n_obs)
        env_data[ds] = _load_env_acc(results_dir, ds, n_obs)

    table1 = build_table1(data, decimals, n_obs)
    table2 = build_table2(data, subset_data, decimals, n_obs)

    # Per-environment accuracy tables (one per dataset)
    env_tables: list[str] = []
    for ds in DATASETS:
        t = build_env_acc_table(env_data[ds], ds, decimals)
        env_tables.append(f"% --- Dataset {ds} ---\n{t}")
    env_tables_str = "\n\n".join(env_tables)

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
        "\n\n"
        "% =============================================================================\n"
        "% Tables 3a/3b/3c: Per-environment accuracy\n"
        "% =============================================================================\n"
        "\n"
        f"{env_tables_str}\n"
    )

    with open(output, "w") as f:
        f.write(full)

    print(f"Tables written to {output}")

    out_dir = os.path.dirname(output)
    if spider_output is None:
        spider_linear = os.path.join(out_dir, f"spider_charts_linear_n{n_obs}.pdf")
        spider_nonlinear = os.path.join(
            out_dir, f"spider_charts_nonlinear_n{n_obs}.pdf"
        )
    else:
        base, ext = os.path.splitext(spider_output)
        spider_linear = f"{base}_linear_n{n_obs}{ext}"
        spider_nonlinear = f"{base}_nonlinear_n{n_obs}{ext}"
    build_spider_charts(
        env_data,
        SPIDER_METHODS_LINEAR,
        spider_linear,
        grid_step_overrides={"1a": 0.05, "1b": 0.05, "2": 0.2},
    )
    build_spider_charts(
        env_data,
        SPIDER_METHODS_NONLINEAR,
        spider_nonlinear,
        grid_step_overrides={"1a": 0.05, "2": 0.2},
    )
    print("\nPreview (first 40 lines):\n")
    for line in full.splitlines()[:40]:
        print(line)
    print("...")


if __name__ == "__main__":
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    default_results = os.path.join(repo_root, "results")
    default_output = None

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
        help="Output file path (default: <results-dir>/latex_tables_n<n_obs>.txt).",
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
    parser.add_argument(
        "--spider-output",
        default=None,
        help="Path for the spider chart PDF/PNG (default: <results-dir>/spider_charts.pdf).",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.results_dir, f"latex_tables_n{args.n_obs}.txt")
    main(
        results_dir=args.results_dir,
        output=args.output,
        decimals=args.decimals,
        n_obs=args.n_obs,
        spider_output=args.spider_output,
    )
