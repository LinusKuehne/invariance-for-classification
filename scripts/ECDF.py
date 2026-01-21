"""
Test level ECDF simulation script for invariance tests.

Generates data from the synthetic DGP and computes p-values using all available
invariance tests for invariant sets S = {1} (X1 only) and S = {1,3} (X1, X3).
Plots the empirical CDF (ECDF) of these p-values. If the tests are level,
the ECDFs should lie on or below the diagonal.
"""

import warnings
from functools import partial
from multiprocessing import Pool

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

# suppress warnings during parallel execution
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

SEED = 1
N_REPS = 500  # number of simulation repetitions
N_PER_ENV = 200  # observations per environment (5 envs => 5*n total)
N_CORES = 13  # number of parallel workers

# Invariant sets to test (column indices: X1=0, X2=1, X3=2)
# S = {1} in R notation => X1 => index 0
# S = {1,3} in R notation => X1, X3 => indices 0, 2
SET_S1 = [0]  # X1 only
SET_S13 = [0, 2]  # X1 and X3


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def run_single_iteration(iteration: int, seed_offset: int = 0) -> dict:
    """
    Run a single simulation iteration.

    Parameters
    ----------
    iteration : int
        Iteration index (used for seeding)
    seed_offset : int
        Base seed offset

    Returns
    -------
    dict
        Dictionary containing p-values for all tests and both sets
    """
    # generate data with unique seed per iteration
    seed = seed_offset + iteration
    df = generate_scm_data(n_per_env=N_PER_ENV, seed=seed)

    X = df[["X1", "X2", "X3"]].values
    y = df["Y"].values
    E = df["E"].values

    # initialize all tests (both LR and RF versions)
    tests = {
        "DeLong_LR": DeLongTest(test_classifier_type="LR"),
        "DeLong_RF": DeLongTest(test_classifier_type="RF"),
        "TramGCM_LR": TramGcmTest(test_classifier_type="LR"),
        "TramGCM_RF": TramGcmTest(test_classifier_type="RF"),
        "Residual_LR": InvariantResidualDistributionTest(test_classifier_type="LR"),
        "Residual_RF": InvariantResidualDistributionTest(test_classifier_type="RF"),
    }

    results = {}

    # test S = {1} (X1 only, index 0)
    X_S1 = X[:, SET_S1]
    for test_name, test in tests.items():
        try:
            p_val = test.test(X_S1, y, E)
        except Exception:
            p_val = np.nan
        results[f"S1_{test_name}"] = p_val

    # test S = {1,3} (X1 and X3, indices 0, 2)
    X_S13 = X[:, SET_S13]
    for test_name, test in tests.items():
        try:
            p_val = test.test(X_S13, y, E)
        except Exception:
            p_val = np.nan
        results[f"S13_{test_name}"] = p_val

    return results


def run_simulation_parallel(n_reps: int, n_cores: int, seed: int) -> pd.DataFrame:
    """
    Run the simulation in parallel.

    Parameters
    ----------
    n_reps : int
        Number of repetitions
    n_cores : int
        Number of parallel workers
    seed : int
        Base seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with p-values for all tests and iterations
    """
    # create partial function with fixed seed offset
    run_iter = partial(run_single_iteration, seed_offset=seed)

    # run in parallel with progress bar
    with Pool(processes=n_cores) as pool:
        results = list(
            tqdm(
                pool.imap(run_iter, range(n_reps)),
                total=n_reps,
                desc="Running simulations",
            )
        )

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------


def plot_ecdf(df: pd.DataFrame, set_name: str, title: str, ax) -> None:
    """
    Plot ECDF of p-values for a given set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with p-values
    set_name : str
        Prefix for column names (e.g., "S1" or "S13")
    title : str
        Plot title
    ax : plt.Axes
        Matplotlib axes to plot on
    """
    # define test types and their colors
    test_colors = {
        "DeLong": "C0",
        "TramGCM": "C1",
        "Residual": "C2",
    }

    # line styles for LR vs RF
    line_styles = {
        "LR": "-",  # solid
        "RF": ":",  # dotted
    }

    # plot each test
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

    # plot diagonal (uniform distribution reference)
    ax.plot([0, 1], [0, 1], "k-", linewidth=0.8, alpha=0.5, label="Uniform")

    ax.set_xlabel("p-value")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def create_ecdf_plot(results_df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Create the combined ECDF plot.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with simulation results
    save_path : str, optional
        Path to save the plot
    """
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    # plot for S = {1,3} (left panel, like R script)
    plot_ecdf(results_df, "S13", r"Subset $S = \{X_1, X_3\}$", axes[0])

    # plot for S = {1} (right panel)
    plot_ecdf(results_df, "S1", r"Subset $S = \{X_1\}$", axes[1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # plt.show()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Running ECDF simulation with {N_REPS} repetitions on {N_CORES} cores...")
    print("Testing invariant sets: S1 = {X1}, S13 = {X1, X3}")

    # run simulation
    results_df = run_simulation_parallel(N_REPS, N_CORES, SEED)

    # print summary statistics
    # print("\nSummary of p-values (mean ± std):")
    # for col in results_df.columns:
    #     mean_val = results_df[col].mean()
    #     std_val = results_df[col].std()
    #     print(f"  {col}: {mean_val:.3f} ± {std_val:.3f}")

    # save results
    # results_df.to_csv("ecdf_results.csv", index=False)
    # print("\nResults saved to ecdf_results.csv")

    # create and save plot
    create_ecdf_plot(results_df, save_path="ecdf_standard.pdf")
