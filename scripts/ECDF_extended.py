"""
Extended ECDF simulation comparing linear vs non-linear link functions.

Generates data from the simple DGP using both logistic_linear and logistic_complex
link functions, then computes ECDFs for the invariance tests on invariant sets.

This helps evaluate whether the tests maintain proper level under model misspecification
(when the true relationship is non-linear but LR-based tests assume linearity).
"""

import warnings
from functools import partial
from multiprocessing import Pool
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from invariance_for_classification.generate_data.synthetic_DGP import generate_scm_data
from invariance_for_classification.invariance_tests import (
    DeLongTest,
    InvariantResidualDistributionTest,
    TramGcmTest,
)

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

SEED = 42
N_REPS = 100
N_PER_ENV = 150
N_CORES = 10

# Invariant sets to test
SET_S1 = [0]  # X1 only
SET_S13 = [0, 2]  # X1 and X3


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------


def run_single_iteration(
    iteration: int,
    seed_offset: int,
    model: Literal["logistic_linear", "logistic_complex"],
) -> dict:
    """Run a single simulation iteration."""
    seed = seed_offset + iteration
    df = generate_scm_data(n_per_env=N_PER_ENV, seed=seed, model=model)

    X = df[["X1", "X2", "X3"]].values
    y = df["Y"].values
    E = df["E"].values

    tests = {
        "DeLong_LR": DeLongTest(test_classifier_type="LR"),
        "DeLong_RF": DeLongTest(test_classifier_type="RF"),
        "TramGCM_LR": TramGcmTest(test_classifier_type="LR"),
        "TramGCM_RF": TramGcmTest(test_classifier_type="RF"),
        "Residual_LR": InvariantResidualDistributionTest(test_classifier_type="LR"),
        "Residual_RF": InvariantResidualDistributionTest(test_classifier_type="RF"),
    }

    results = {}

    # Test S = {1} (X1 only)
    X_S1 = X[:, SET_S1]
    for test_name, test in tests.items():
        try:
            p_val = test.test(X_S1, y, E)
        except Exception:
            p_val = np.nan
        results[f"S1_{test_name}"] = p_val

    # Test S = {1,3} (X1 and X3)
    X_S13 = X[:, SET_S13]
    for test_name, test in tests.items():
        try:
            p_val = test.test(X_S13, y, E)
        except Exception:
            p_val = np.nan
        results[f"S13_{test_name}"] = p_val

    return results


def run_simulation(
    n_reps: int,
    n_cores: int,
    seed: int,
    model: Literal["logistic_linear", "logistic_complex"],
) -> pd.DataFrame:
    """Run simulation in parallel."""
    run_iter = partial(run_single_iteration, seed_offset=seed, model=model)

    with Pool(processes=n_cores) as pool:
        results = list(
            tqdm(
                pool.imap(run_iter, range(n_reps)),
                total=n_reps,
                desc=f"Running ({model})",
            )
        )

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_ecdf(df: pd.DataFrame, set_name: str, title: str, ax) -> None:
    """Plot ECDF of p-values for a given set."""
    test_colors = {"DeLong": "C0", "TramGCM": "C1", "Residual": "C2"}
    line_styles = {"LR": "-", "RF": ":"}

    for test_type, color in test_colors.items():
        for clf_type, linestyle in line_styles.items():
            col_name = f"{set_name}_{test_type}_{clf_type}"
            if col_name in df.columns:
                p_values = df[col_name].dropna().sort_values()
                ecdf = np.arange(1, len(p_values) + 1) / len(p_values)
                ax.step(
                    p_values,
                    ecdf,
                    where="post",
                    label=f"{test_type} ({clf_type})",
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                )

    ax.plot([0, 1], [0, 1], "k-", linewidth=0.8, alpha=0.5, label="Uniform")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def create_comparison_plot(
    results_linear: pd.DataFrame,
    results_complex: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Create 2x2 comparison plot: rows = link function, cols = subset."""
    _, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: logistic_linear
    plot_ecdf(results_linear, "S13", r"Linear: $S = \{X_1, X_3\}$", axes[0, 0])
    plot_ecdf(results_linear, "S1", r"Linear: $S = \{X_1\}$", axes[0, 1])

    # Bottom row: logistic_complex
    plot_ecdf(results_complex, "S13", r"Complex: $S = \{X_1, X_3\}$", axes[1, 0])
    plot_ecdf(results_complex, "S1", r"Complex: $S = \{X_1\}$", axes[1, 1])

    plt.suptitle("ECDF Comparison: Linear vs Non-linear Link Function", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")


def print_level_summary(
    results_linear: pd.DataFrame,
    results_complex: pd.DataFrame,
    alpha: float = 0.05,
) -> None:
    """Print summary of empirical rejection rates (should be ≤ alpha for level)."""
    print("\n" + "=" * 70)
    print(f"LEVEL SUMMARY: Rejection rates at α = {alpha}")
    print("(For valid level, rejection rate should be ≤ α)")
    print("=" * 70)

    for model_name, df in [("Linear", results_linear), ("Complex", results_complex)]:
        print(f"\n{model_name} Link Function:")
        print("-" * 50)
        for col in sorted(df.columns):
            p_vals = df[col].dropna()
            rej_rate = (p_vals < alpha).mean()
            marker = "✓" if rej_rate <= alpha + 0.02 else "✗"  # small tolerance
            print(f"  {col}: {rej_rate:.3f} {marker}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Running extended ECDF simulation with {N_REPS} reps on {N_CORES} cores...")
    print("Comparing logistic_linear vs logistic_complex link functions")
    print("Testing invariant sets: S1 = {X1}, S13 = {X1, X3}")

    # Run simulations for both link functions
    print("\n" + "-" * 40)
    results_linear = run_simulation(N_REPS, N_CORES, SEED, model="logistic_linear")

    print("\n" + "-" * 40)
    results_complex = run_simulation(N_REPS, N_CORES, SEED, model="logistic_complex")

    # Print summary
    print_level_summary(results_linear, results_complex)

    # Create comparison plot
    create_comparison_plot(
        results_linear,
        results_complex,
        save_path="ecdf_comparison.pdf",
    )

    # Also save individual plots
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plot_ecdf(results_linear, "S13", r"Linear: $S = \{X_1, X_3\}$", axes[0])
    # plot_ecdf(results_linear, "S1", r"Linear: $S = \{X_1\}$", axes[1])
    # plt.tight_layout()
    # plt.savefig("ecdf_linear.pdf", dpi=150, bbox_inches="tight")

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plot_ecdf(results_complex, "S13", r"Complex: $S = \{X_1, X_3\}$", axes[0])
    # plot_ecdf(results_complex, "S1", r"Complex: $S = \{X_1\}$", axes[1])
    # plt.tight_layout()
    # plt.savefig("ecdf_complex.pdf", dpi=150, bbox_inches="tight")
