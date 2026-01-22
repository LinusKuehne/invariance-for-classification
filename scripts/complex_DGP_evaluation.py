"""
Evaluation script for invariance tests using the complex DGP.

The complex DGP has the following causal structure:
    X1 -> X2
    X1, X2 -> Y
    Y -> X3 <- X5
    Y, X3, X7 -> X4
    Y, X4 -> X6
    X2 <- E -> X4

For Y to be d-separated from E given X_S:
    - Must INCLUDE {X1, X2}
    - Must NOT include X4, X6

Invariant sets:
    - {X1, X2}
    - {X1, X2, X3}
    - {X1, X2, X5}
    - {X1, X2, X7}
    - {X1, X2, X3, X5}
    - {X1, X2, X3, X7}
    - {X1, X2, X5, X7}
    - {X1, X2, X3, X5, X7}

This script:
1. Generates all 2^7 = 128 subsets of {X1, ..., X7}
2. Computes p-values for each subset using all available tests
3. Plots ordered p-values as a bar chart (green=invariant, red=non-invariant)
"""

import itertools
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

from invariance_for_classification.generate_data.complex_DGP import (
    generate_complex_scm_data,
)
from invariance_for_classification.invariance_tests import (
    DeLongTest,
    InvarianceTest,
    InvariantResidualDistributionTest,
    TramGcmTest,
)

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

SEED = 42
N_REPS = 20
N_PER_ENV = 500
N_ENVS = 5
N_CORES = 12
ALPHA = 0.05

# global settings for classifier type and link function
CLASSIFIER_TYPE: Literal["LR", "RF"] = "RF"
LINK_FUNCTION: Literal["logistic_linear", "logistic_complex"] = "logistic_linear"

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def get_all_subsets(features: list[str]) -> list[list[str]]:
    """Generate all 2^n subsets of features."""
    subsets = []
    for L in range(len(features) + 1):
        for subset in itertools.combinations(features, L):
            subsets.append(list(subset))
    return subsets


def is_invariant(subset: list[str]) -> bool:
    """
    Check if a subset is invariant (Y d-separated from E given X_S)."""
    has_x1 = "X1" in subset
    has_x2 = "X2" in subset
    has_x4 = "X4" in subset
    has_x6 = "X6" in subset
    return has_x1 and has_x2 and (not has_x4) and (not has_x6)


def subset_to_str(subset: list[str]) -> str:
    """Convert subset to string representation."""
    if not subset:
        return "Empty"
    nums = [s.replace("X", "") for s in subset]
    return ",".join(sorted(nums, key=int))


def get_all_tests(classifier_type: str) -> dict[str, InvarianceTest]:
    """Create all test instances with the given classifier type."""
    return {
        "DeLong": DeLongTest(test_classifier_type=classifier_type),
        "Residual": InvariantResidualDistributionTest(
            test_classifier_type=classifier_type
        ),
        "TramGCM": TramGcmTest(test_classifier_type=classifier_type),
    }


# -----------------------------------------------------------------------------
# main sim
# -----------------------------------------------------------------------------


def run_single_rep(
    rep: int,
    n_per_env: int,
    n_envs: int,
    seed: int,
    link_function: Literal["logistic_linear", "logistic_complex"],
    classifier_type: str,
) -> list[dict]:
    """
    Run a single repetition: compute p-values for all subsets and all tests.
    This function is called in parallel.
    """
    subsets = get_all_subsets(FEATURES)
    tests = get_all_tests(classifier_type)

    # generate data for this repetition
    data = generate_complex_scm_data(
        n_train=n_per_env,
        n_test=n_per_env,
        model=link_function,
        int_strength_train=1.0,
        int_strength_test=1.5,
        num_noise=0,
        n_envs=n_envs,
        seed=seed + rep,
    )
    df = data["sample_train"]
    y = df["Y"].to_numpy()
    E = pd.factorize(df["Env"])[0]

    results = []
    for subset in subsets:
        # get X matrix for this subset
        if not subset:
            X = np.zeros((len(y), 0))
        else:
            X = df[subset].to_numpy()

        # compute p-value for each test
        for test_name, test in tests.items():
            try:
                pval = test.test(X, y, E)
            except Exception:
                pval = np.nan

            results.append(
                {
                    "rep": rep,
                    "test": test_name,
                    "subset": tuple(subset),
                    "subset_str": subset_to_str(subset),
                    "is_invariant": is_invariant(subset),
                    "pvalue": pval,
                }
            )

    return results


def run_experiment(
    n_reps: int = N_REPS,
    n_per_env: int = N_PER_ENV,
    n_envs: int = N_ENVS,
    seed: int = SEED,
    link_function: Literal["logistic_linear", "logistic_complex"] = LINK_FUNCTION,
    classifier_type: str = CLASSIFIER_TYPE,
    n_cores: int = N_CORES,
) -> pd.DataFrame:
    """
    Run the experiment: compute p-values for all subsets and all tests.
    Parallelized over repetitions.

    Returns DataFrame with columns: rep, test, subset, subset_str, is_invariant, pvalue
    """
    # create partial function with fixed parameters
    worker = partial(
        run_single_rep,
        n_per_env=n_per_env,
        n_envs=n_envs,
        seed=seed,
        link_function=link_function,
        classifier_type=classifier_type,
    )

    # run in parallel
    print(f"Running {n_reps} repetitions with {n_cores} cores...")
    with Pool(n_cores) as pool:
        all_results = list(
            tqdm(pool.imap(worker, range(n_reps)), total=n_reps, desc="Repetitions")
        )

    results = [item for sublist in all_results for item in sublist]

    return pd.DataFrame(results)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate p-values across repetitions (take mean) for each test."""
    agg = (
        df.groupby(["test", "subset", "subset_str", "is_invariant"])
        .agg(
            pvalue_mean=("pvalue", "mean"),
            pvalue_std=("pvalue", "std"),
        )
        .reset_index()
    )
    return agg


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_ordered_pvalues(
    df: pd.DataFrame,
    classifier_type: str = CLASSIFIER_TYPE,
    link_function: str = LINK_FUNCTION,
    save_path: str | None = None,
) -> None:
    """
    Plot ordered p-values as thin bars for each test (subplots).
    Green = invariant sets, Red = non-invariant sets.
    """
    tests = df["test"].unique()
    n_tests = len(tests)

    # create figure with subplots
    fig, axes = plt.subplots(1, n_tests, figsize=(6 * n_tests, 5), sharey=True)
    if n_tests == 1:
        axes = [axes]

    for ax, test_name in zip(axes, tests, strict=True):
        # filter for this test
        df_test = df[df["test"] == test_name].copy()

        # sort by p-value
        df_sorted = df_test.sort_values("pvalue_mean").reset_index(drop=True)

        n_subsets = len(df_sorted)
        x = np.arange(n_subsets)

        # colors based on invariance
        colors = ["green" if inv else "red" for inv in df_sorted["is_invariant"]]

        # plot thin bars
        ax.bar(x, df_sorted["pvalue_mean"], width=0.8, color=colors, alpha=0.7)

        # add significance threshold line
        ax.axhline(y=ALPHA, color="black", linestyle=":", linewidth=1.5)

        # styling
        ax.set_xlim(-0.5, n_subsets - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"Subsets (n={n_subsets})", fontsize=11)
        ax.set_title(f"{test_name} Test", fontsize=12)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("p-value", fontsize=12)

    # add shared legend
    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="Invariant (X1,X2 ∈ S, X4,X6 ∉ S)"),
        Patch(facecolor="red", alpha=0.7, label="Non-invariant"),
        Line2D(
            [0], [0], color="black", linestyle=":", linewidth=1.5, label=f"α = {ALPHA}"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )

    # add overall title
    fig.suptitle(
        f"Complex DGP - Classifier: {classifier_type}, Link: {link_function}",
        fontsize=14,
        y=1.08,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics including power for each test."""
    tests = df["test"].unique()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # overall counts (same for all tests)
    df_first_test = df[df["test"] == tests[0]]
    invariant = df_first_test[df_first_test["is_invariant"]]
    non_invariant = df_first_test[~df_first_test["is_invariant"]]

    print(f"Total subsets: {len(df_first_test)}")
    print(f"Invariant subsets: {len(invariant)}")
    print(f"Non-invariant subsets: {len(non_invariant)}")

    for test_name in tests:
        df_test = df[df["test"] == test_name]
        inv = df_test[df_test["is_invariant"]]
        non_inv = df_test[~df_test["is_invariant"]]

        print(f"\n--- {test_name} Test ---")

        # level: rejection rate for invariant sets (should be ≤ α)
        inv_rejected = (inv["pvalue_mean"] < ALPHA).sum()
        level = inv_rejected / len(inv) if len(inv) > 0 else 0
        print("  Level (rejection rate for invariant sets):")
        print(f"    Rejected: {inv_rejected}/{len(inv)} ({100 * level:.1f}%)")
        print(f"    Expected: ≤ {100 * ALPHA:.1f}%")

        # power: rejection rate for non-invariant sets (should be high)
        noninv_rejected = (non_inv["pvalue_mean"] < ALPHA).sum()
        power = noninv_rejected / len(non_inv) if len(non_inv) > 0 else 0
        print("  Power (rejection rate for non-invariant sets):")
        print(f"    Rejected: {noninv_rejected}/{len(non_inv)} ({100 * power:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("COMPLEX DGP EVALUATION - All 2^7 Subsets")
    print("=" * 60)
    print(f"Features: {FEATURES}")
    print("Invariant criterion: X1,X2 ∈ S and X4,X6 ∉ S")
    print(f"Classifier type: {CLASSIFIER_TYPE}")
    print(f"Link function: {LINK_FUNCTION}")
    print(f"Repetitions: {N_REPS}")
    print(f"Observations per env: {N_PER_ENV}")
    print(f"Number of environments: {N_ENVS}")
    print(f"Parallel cores: {N_CORES}")

    print("\nRunning experiment...")
    results_df = run_experiment(
        link_function=LINK_FUNCTION,
        classifier_type=CLASSIFIER_TYPE,
    )

    agg_df = aggregate_results(results_df)

    print_summary(agg_df)

    plot_ordered_pvalues(
        agg_df,
        classifier_type=CLASSIFIER_TYPE,
        link_function=LINK_FUNCTION,
        save_path="complex_dgp_pvalues.pdf",
    )
