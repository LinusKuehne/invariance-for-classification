"""
Sample-efficiency experiment: LOEO Regret vs TramGCM.

Compares the LOEO regret ranking (RF & LR, mean & min aggregation) against
the ranking induced by TramGCM p-values (RF & LR) in terms of pairwise AUC
for classifying subsets as invariant / non-invariant.

The comparison is performed at increasing numbers of observations per
environment (50, 100, 150, 200) to reveal sample-efficiency characteristics.

Usage
-----
    python sample_efficiency.py --dataset 1a
    python sample_efficiency.py --dataset 2 --samples 50 100 200 400
"""

import argparse
import itertools
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from invariance_for_classification.invariance_tests import TramGcmTest
from invariance_for_classification.rankings import loeo_regret

# Default number of parallel workers
N_WORKERS = 11

# ---------------------------------------------------------------------------
# Shared helpers (kept self-contained so the script is standalone)
# ---------------------------------------------------------------------------

# Column definitions – mirrors evaluate_all_tests.py NORMAL_COLS
NORMAL_COLS: dict[str, list[str]] = {
    "1a": ["Y", "red", "green", "blue", "ir_1", "vis_1", "ir_3", "vis_3", "E"],
    "1b": ["Y", "red", "green", "blue", "ir_1", "vis_1", "ir_3", "vis_3", "E"],
    "2": ["Y", "red", "green", "blue", "ir_2", "vis_2", "ir_3", "vis_3", "E"],
}


def _get_invariant_subsets(name: str, features: list[str]) -> set[frozenset[str]]:
    """Return ground-truth invariant subsets (same logic as evaluate_all_tests.py)."""
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
        optional_subsets: list[frozenset[str]] = [frozenset()]
        for r in range(1, len(optional_cols) + 1):
            for combo in itertools.combinations(optional_cols, r):
                optional_subsets.append(frozenset(combo))
        invariant_subsets: set[frozenset[str]] = set()
        for base in base_subsets:
            for opt in optional_subsets:
                invariant_subsets.add(base | opt)
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
            for opt in optional_subsets:
                invariant_subsets.add(base | opt)
        return invariant_subsets

    return set()


def _get_all_subsets(features: list[str]) -> list[list[str]]:
    """Generate all 2^d subsets of *features*."""
    subsets: list[list[str]] = []
    for L in range(len(features) + 1):
        for combo in itertools.combinations(features, L):
            subsets.append(list(combo))
    return subsets


def _pairwise_auc(inv_values: np.ndarray, noninv_values: np.ndarray) -> float:
    """Fraction of (inv, noninv) pairs where inv > noninv (ties count 0.5)."""
    count = sum(
        1.0 if i > n else 0.5 if i == n else 0.0
        for i in inv_values
        for n in noninv_values
    )
    total = len(inv_values) * len(noninv_values)
    return count / total if total > 0 else np.nan


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    dataset: str,
    n_per_env: int,
    data_dir: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str], list[list[str]], set[frozenset[str]]]:
    """
    Load a causal-chambers dataset and subsample to *n_per_env* per environment.

    Returns (df, features, all_subsets, invariant_subsets).
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    filename = f"{dataset}_train.csv"
    df = pd.read_csv(os.path.join(data_dir, filename))

    # Column filter (NORMAL_COLS)
    cols = [c for c in NORMAL_COLS[dataset] if c in df.columns]
    df = df[cols]

    # Subsample per environment
    parts = []
    for _, grp in df.groupby("E"):
        parts.append(grp.sample(n=min(n_per_env, len(grp)), random_state=random_state))
    df = pd.concat(parts, ignore_index=True)

    features = [c for c in df.columns if c not in ("Y", "E")]
    all_subsets = _get_all_subsets(features)
    invariant_subsets = _get_invariant_subsets(dataset, features)

    return df, features, all_subsets, invariant_subsets


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def _loeo_one_subset(
    subset: list[str],
    df: pd.DataFrame,
    classifier_type: str,
) -> dict[str, float]:
    """Compute LOEO regret for a single subset (picklable top-level function)."""
    y = df["Y"].to_numpy()
    E = df["E"].to_numpy()
    X = np.zeros((len(df), 0)) if not subset else df[subset].to_numpy()
    try:
        return loeo_regret(y, E, X, classifier_type=classifier_type)  # type: ignore[arg-type]
    except Exception as exc:
        print(f"  LOEO ({classifier_type}) failed on {subset}: {exc}")
        return {"mean": np.nan, "min": np.nan}


def _tramgcm_one_subset(
    subset: list[str],
    df: pd.DataFrame,
    classifier_type: str,
) -> float:
    """Compute TramGCM p-value for a single subset (picklable top-level function)."""
    test = TramGcmTest(test_classifier_type=classifier_type)  # type: ignore[arg-type]
    y = df["Y"].to_numpy()
    E = df["E"].to_numpy()
    X = np.zeros((len(df), 0)) if not subset else df[subset].to_numpy()
    try:
        return test.test(X, y, E)
    except Exception as exc:
        print(f"  TramGCM ({classifier_type}) failed on {subset}: {exc}")
        return np.nan


def _run_loeo_regret(
    df: pd.DataFrame,
    all_subsets: list[list[str]],
    classifier_type: Literal["RF", "LR"],
    n_workers: int = N_WORKERS,
) -> dict[str, list[float]]:
    """Run LOEO regret on every subset (parallelized over subsets)."""
    results: list[tuple[int, dict[str, float]]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_loeo_one_subset, s, df, classifier_type): i
            for i, s in enumerate(all_subsets)
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"LOEO({classifier_type})",
            leave=False,
        ):
            results.append((futures[fut], fut.result()))
    results.sort(key=lambda x: x[0])
    return {
        "mean": [r["mean"] for _, r in results],
        "min": [r["min"] for _, r in results],
    }


def _run_tramgcm(
    df: pd.DataFrame,
    all_subsets: list[list[str]],
    classifier_type: Literal["RF", "LR"],
    n_workers: int = N_WORKERS,
) -> list[float]:
    """Run TramGCM test on every subset (parallelized over subsets)."""
    results: list[tuple[int, float]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_tramgcm_one_subset, s, df, classifier_type): i
            for i, s in enumerate(all_subsets)
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"TramGCM({classifier_type})",
            leave=False,
        ):
            results.append((futures[fut], fut.result()))
    results.sort(key=lambda x: x[0])
    return [v for _, v in results]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    dataset: str,
    sample_sizes: list[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run the sample-efficiency experiment.

    Returns a DataFrame with columns:
        n_per_env, method, pairwise_auc
    """
    rows: list[dict] = []

    for n_per_env in sample_sizes:
        print(f"\n{'=' * 60}")
        print(f"  n_per_env = {n_per_env}")
        print(f"{'=' * 60}")

        df, features, all_subsets, invariant_subsets = load_data(
            dataset, n_per_env, random_state=random_state
        )
        n_subsets = len(all_subsets)
        is_inv = np.array([frozenset(s) in invariant_subsets for s in all_subsets])

        # --- LOEO Regret (RF & LR) ---
        for clf in ("RF", "LR"):
            print(f"  Running LOEO Regret ({clf}) on {n_subsets} subsets …")
            loeo_scores = _run_loeo_regret(df, all_subsets, classifier_type=clf)
            for agg in ("mean", "min"):
                vals = np.array(loeo_scores[agg])
                inv_vals = vals[is_inv]
                noninv_vals = vals[~is_inv]
                auc = _pairwise_auc(inv_vals, noninv_vals)
                method_name = f"LOEO Regret ({clf}, {agg})"
                rows.append(
                    {"n_per_env": n_per_env, "method": method_name, "pairwise_auc": auc}
                )
                print(f"    {method_name}: AUC = {auc:.3f}")

        # --- TramGCM (RF & LR) ---
        for clf in ("RF", "LR"):
            print(f"  Running TramGCM ({clf}) on {n_subsets} subsets …")
            pvals = np.array(_run_tramgcm(df, all_subsets, classifier_type=clf))
            inv_vals = pvals[is_inv]
            noninv_vals = pvals[~is_inv]
            auc = _pairwise_auc(inv_vals, noninv_vals)
            method_name = f"TramGCM ({clf})"
            rows.append(
                {"n_per_env": n_per_env, "method": method_name, "pairwise_auc": auc}
            )
            print(f"    {method_name}: AUC = {auc:.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# One colour per method (colour-blind friendly palette)
METHOD_COLORS: dict[str, str] = {
    "LOEO Regret (RF, mean)": "#1f77b4",  # blue
    "LOEO Regret (RF, min)": "#aec7e8",  # light blue
    "LOEO Regret (LR, mean)": "#ff7f0e",  # orange
    "LOEO Regret (LR, min)": "#ffbb78",  # light orange
    "TramGCM (RF)": "#2ca02c",  # green
    "TramGCM (LR)": "#d62728",  # red
}

METHOD_MARKERS: dict[str, str] = {
    "LOEO Regret (RF, mean)": "o",
    "LOEO Regret (RF, min)": "s",
    "LOEO Regret (LR, mean)": "^",
    "LOEO Regret (LR, min)": "D",
    "TramGCM (RF)": "P",
    "TramGCM (LR)": "X",
}


def plot_results(
    results: pd.DataFrame,
    dataset: str,
    save_path: str | None = None,
) -> None:
    """Line plot: n_per_env (x) vs pairwise AUC (y), one line per method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in results["method"].unique():
        sub = results[results["method"] == method].sort_values("n_per_env")
        color = METHOD_COLORS.get(method, "grey")
        marker = METHOD_MARKERS.get(method, "o")
        ax.plot(
            sub["n_per_env"],
            sub["pairwise_auc"],
            marker=marker,
            label=method,
            color=color,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Observations per environment", fontsize=12)
    ax.set_ylabel("Pairwise AUC", fontsize=12)
    ax.set_title(
        f"Sample Efficiency: LOEO Regret vs TramGCM (dataset {dataset})",
        fontsize=13,
    )
    ax.axhline(y=0.5, color="grey", linestyle=":", linewidth=1, label="Random (0.5)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample-efficiency comparison: LOEO Regret vs TramGCM"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="1a",
        choices=["1a", "1b", "2"],
        help="Causal-chambers dataset to use (default: 1a)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="*",
        default=[50, 100, 150, 200],
        help="List of per-environment sample sizes (default: 50 100 150 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sub-sampling (default: 42)",
    )
    args = parser.parse_args()

    # Determine save directory
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    save_dir = os.path.join(repo_root, "results", args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("SAMPLE-EFFICIENCY EXPERIMENT")
    print("=" * 60)
    print(f"Dataset      : {args.dataset}")
    print(f"Sample sizes : {args.samples}")
    print(f"Seed         : {args.seed}")
    print(f"Save dir     : {save_dir}")

    results = run_experiment(
        dataset=args.dataset,
        sample_sizes=args.samples,
        random_state=args.seed,
    )

    # Save results table
    csv_path = os.path.join(save_dir, f"sample_efficiency_{args.dataset}.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(results.to_string(index=False))

    # Save plot
    plot_path = os.path.join(save_dir, f"sample_efficiency_{args.dataset}.pdf")
    plot_results(results, dataset=args.dataset, save_path=plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
