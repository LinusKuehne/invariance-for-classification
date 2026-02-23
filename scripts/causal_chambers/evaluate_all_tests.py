"""
Comprehensive evaluation script for invariance tests on causal chambers datasets.

This script:
1. Reads in a dataset and retrieves ground-truth invariant/non-invariant subsets
2. Performs diagnostics (pairplot, distribution analysis, correlation matrices)
3. Computes p-values for all invariance tests on all 2^d subsets (parallelized)
4. Evaluates test quality with plots and various metrics
5. Additionally evaluates LOEO regret ranking (scores instead of p-values)

================================================================================
COMMAND LINE FLAGS
================================================================================

--dataset <name>
    Name of the dataset CSV file (with or without .csv extension).
    The file should be in the data/ subdirectory.
    Examples:
        python evaluate_all_tests.py --dataset simple
        python evaluate_all_tests.py --dataset simple.csv

--workers <int>
    Number of parallel workers for test computation. Default: 10
    Example:
        python evaluate_all_tests.py --dataset simple --workers 8

--tests <test1> <test2> ...
    Only run specific tests. Use lowercase test names (partial matching).
    If not specified, all tests are run.

    Available test names (use partial matching, e.g., "delong" matches all DeLong variants):
      - CRT (HGBT)           - Conditional Randomization Test
      - CRT (RF)
      - DeLong (HGBT)        - DeLong AUC comparison test
      - DeLong (LR)
      - DeLong (RF)
      - InvEnvPred (HGBT)    - Invariant Environment Prediction test
      - InvEnvPred (RF)
      - Residual (HGBT)      - Invariant Residual Distribution test
      - Residual (LR)
      - Residual (RF)
      - TramGCM (HGBT)       - Tram-GCM test
      - TramGCM (LR)
      - TramGCM (RF)
      - WGCM_est (xgb)       - Weighted GCM (estimated weights)
      - WGCM_fix (xgb)       - Weighted GCM (fixed weights)

    Examples:
        python evaluate_all_tests.py --dataset simple --tests residual tramgcm
        python evaluate_all_tests.py --dataset simple --tests crt  # runs all CRT variants

================================================================================
OUTPUT
================================================================================

Results are saved to: <repo_root>/results/<dataset_name>/
    - results_<dataset>.csv     : Raw p-values/scores for all test-subset pairs
    - metrics_<dataset>.csv     : Computed metrics (FPR, TPR, average and min/max among invariant/non-invariant)
    - ordered_pvalues_<dataset>.pdf : Ordered p-value plots for each test
    - comparison_<dataset>.pdf  : Summary comparison of all tests
    - diagnostics_*.pdf         : Data diagnostic plots

================================================================================
EXAMPLES
================================================================================

# Full evaluation with all tests
python evaluate_all_tests.py --dataset simple

# Quick evaluation with only a fast test
python evaluate_all_tests.py --dataset simple --tests delong
"""

import argparse
import itertools
import os

# Limit threads for OpenMP-based libraries (HGBT, XGBoost) to allow proper
# parallelization at the process level. Must be set before importing sklearn/xgboost.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

from invariance_for_classification.invariance_tests import (
    ConditionalRandomizationTest,
    DeLongTest,
    InvariantEnvironmentPredictionTest,
    InvariantResidualDistributionTest,
    TramGcmTest,
    WGCMTest,
)
from invariance_for_classification.rankings import loeo_regret

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DatasetInfo:
    """Container for dataset information."""

    df: pd.DataFrame
    features: list[str]
    all_subsets: list[list[str]]
    invariant_subsets: set[
        frozenset[str]
    ]  # set of frozensets for order-independent lookup


@dataclass
class TestResult:
    """Container for a single test result."""

    test_name: str
    subset: tuple[str, ...]
    subset_str: str
    is_invariant: bool
    value: float  # p-value for tests, score for LOEO regret ranking


# =============================================================================
# Data Loading and Subset Generation
# =============================================================================


def get_all_subsets(features: list[str]) -> list[list[str]]:
    """Generate all 2^d subsets of features."""
    subsets = []
    for L in range(len(features) + 1):
        for subset in itertools.combinations(features, L):
            subsets.append(list(subset))
    return subsets


def subset_to_str(subset: list[str]) -> str:
    """Convert subset to string representation."""
    if not subset:
        return "∅"
    return "{" + ", ".join(sorted(subset)) + "}"


def subset_to_short_str(subset: list[str]) -> str:
    """Convert subset to short string (just variable numbers/names)."""
    if not subset:
        return "∅"
    # Try to extract numbers if variables are like X1, X2, etc.
    nums = []
    for s in subset:
        if s.startswith("X") and s[1:].isdigit():
            nums.append(s[1:])
        else:
            nums.append(s)
    return ",".join(
        sorted(nums, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
    )


def _get_invariant_subsets(name: str, features: list[str]) -> set[frozenset[str]]:
    """
    Return set of invariant subsets for a given dataset.

    Add new datasets here as they are created.
    Uses frozenset for order-independent membership checking.
    """
    if name == "simple.csv":
        return {
            frozenset({"X1"}),
            frozenset({"X1", "X3"}),
        }

    if name in ["1a_small_train.csv", "1b_small_train.csv"]:
        return {
            frozenset(),  # empty set
            frozenset({"red"}),
            frozenset({"green"}),
            frozenset({"blue"}),
            frozenset({"red", "green"}),
            frozenset({"red", "blue"}),
            frozenset({"green", "blue"}),
            frozenset({"red", "green", "blue"}),
            frozenset({"red", "green", "blue", "vis_3"}),
        }

    if name in ["1a_train.csv", "1b_train.csv"]:
        # Same as dataset_1_small, but each subset can optionally include ir_1 and/or vis_1
        base_subsets = [
            frozenset(),  # empty set
            frozenset({"red"}),
            frozenset({"green"}),
            frozenset({"blue"}),
            frozenset({"red", "green"}),
            frozenset({"red", "blue"}),
            frozenset({"green", "blue"}),
            frozenset({"red", "green", "blue"}),
            frozenset({"red", "green", "blue", "vis_3"}),
        ]
        # For each base subset, create variants with optional ir_1 and/or vis_1
        invariant_subsets = set()
        optional_additions = [
            frozenset(),
            frozenset({"ir_1"}),
            frozenset({"vis_1"}),
            frozenset({"ir_1", "vis_1"}),
        ]
        for base in base_subsets:
            for addition in optional_additions:
                invariant_subsets.add(base | addition)
        return invariant_subsets

    if name in ["2_small_train.csv"]:
        return {
            frozenset({"red", "green", "blue"}),
        }

    if name in ["2_train.csv"]:
        return {
            frozenset({"red", "green", "blue"}),
            frozenset({"red", "green", "blue", "vis_1"}),
            frozenset({"red", "green", "blue", "ir_2"}),
            frozenset({"red", "green", "blue", "vis_2"}),
            frozenset({"red", "green", "blue", "vis_1", "ir_2"}),
            frozenset({"red", "green", "blue", "vis_1", "vis_2"}),
            frozenset({"red", "green", "blue", "ir_2", "vis_2"}),
            frozenset({"red", "green", "blue", "vis_1", "ir_2", "vis_2"}),
        }

    # default: empty set (no known invariant subsets)
    return set()


def get_data(name: str, data_dir: str | None = None) -> DatasetInfo:
    """
    Load dataset and return DatasetInfo with ground-truth invariant subsets.

    Parameters
    ----------
    name : str
        Name of the CSV file (e.g., "simple" or "simple.csv")
    data_dir : str or None
        Directory containing the data files. If None, uses the data/ subdirectory
        relative to this script.

    Returns
    -------
    DatasetInfo
        Container with df, features, all_subsets, and invariant_subsets
    """
    # Ensure .csv extension
    if not name.endswith(".csv"):
        name = name + ".csv"

    # Use script-relative path if data_dir not specified
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")

    filepath = os.path.join(data_dir, name)
    df = pd.read_csv(filepath)

    # Extract features (all columns except Y and E)
    features = [col for col in df.columns if col not in ["Y", "E"]]

    all_subsets = get_all_subsets(features)

    # Define invariant subsets for known datasets
    invariant_subsets = _get_invariant_subsets(name, features)

    return DatasetInfo(
        df=df,
        features=features,
        all_subsets=all_subsets,
        invariant_subsets=invariant_subsets,
    )


# =============================================================================
# Diagnostics
# =============================================================================


def run_diagnostics(
    data: DatasetInfo, save_dir: str | None = None, dataset_name: str = ""
) -> None:
    """
    Run diagnostic analyses on the dataset.

    Parameters
    ----------
    data : DatasetInfo
        Dataset information
    save_dir : str or None
        Directory to save plots. If None, plots are displayed.
    dataset_name : str
        Name of dataset for output file naming
    """
    # Prefix for output files
    prefix = f"{dataset_name}_" if dataset_name else ""
    df = data.df
    features = data.features

    print("\n" + "=" * 60)
    print("DATASET DIAGNOSTICS")
    print("=" * 60)

    # Basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {features}")
    print(f"Number of environments: {df['E'].nunique()}")
    print("Samples per environment:")
    print(df["E"].value_counts().sort_index().to_string())

    # Y distribution per environment
    print("\nTarget (Y) distribution by environment:")
    for env in sorted(df["E"].unique()):
        subset_df = df[df["E"] == env]
        y_mean = subset_df["Y"].mean()
        print(f"  E={env}: mean={y_mean:.3f}, n={len(subset_df)}")

    # Correlation matrix
    print("\nCorrelation matrix (features + Y):")
    corr_cols = features + ["Y"]
    print(df[corr_cols].corr().round(3).to_string())

    # Plot 1: Pairplot
    print("\nGenerating pairplot...")
    fig_pairplot = sns.pairplot(
        df,
        vars=features + ["Y"],
        hue="E",
        palette="husl",
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 20},
    )
    fig_pairplot.figure.suptitle(
        "Pairplot of Features and Target by Environment", y=1.02
    )

    if save_dir:
        filepath = os.path.join(save_dir, f"{prefix}diagnostic_pairplot.pdf")
        fig_pairplot.savefig(filepath, bbox_inches="tight")
        print(f"Saved: {filepath}")
    else:
        plt.show()
    plt.close()

    # Plot 2: Feature distributions by environment
    print("Generating feature distribution plots...")
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(features):
        sns.kdeplot(
            data=df, x=feat, hue="E", common_norm=False, ax=axes[i], palette="husl"
        )
        axes[i].set_title(f"Distribution of {feat}")

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.suptitle("Feature Distributions by Environment", y=1.02)

    if save_dir:
        filepath = os.path.join(
            save_dir, f"{prefix}diagnostic_feature_distributions.pdf"
        )
        fig.savefig(filepath, bbox_inches="tight")
        print(f"Saved: {filepath}")
    else:
        plt.show()
    plt.close()

    # Plot 3: Heatmap of correlations by environment
    print("Generating correlation heatmaps by environment...")
    envs = sorted(df["E"].unique())
    n_envs = len(envs)

    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 4))
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs, strict=True):
        env_df = df[df["E"] == env][corr_cols]
        corr = env_df.corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"E={env}")

    plt.tight_layout()
    fig.suptitle("Correlation Matrices by Environment", y=1.02)

    if save_dir:
        filepath = os.path.join(save_dir, f"{prefix}diagnostic_correlations.pdf")
        fig.savefig(filepath, bbox_inches="tight")
        print(f"Saved: {filepath}")
    else:
        plt.show()
    plt.close()

    print("\nDiagnostics complete!")


# =============================================================================
# Test Computation (Parallelized)
# =============================================================================


def get_all_tests(test_filter: list[str] | None = None) -> dict[str, Any]:
    """
    Create all test instances with all supported classifier types.

    Each test is instantiated with each classifier type it supports.
    The naming convention is "TestName (ClassifierType)".
    Tests are returned sorted by test name then classifier type.

    Parameters
    ----------
    test_filter : list[str] or None
        If provided, only include tests whose names contain any of these strings
        (case-insensitive). E.g., ["delong", "wgcm"] will include only DeLong
        and WGCM tests.

    Returns
    -------
    dict[str, InvarianceTest]
        Dictionary mapping test names (with classifier type) to test instances
    """
    tests = {}

    # CRT: supports RF, HGBT
    for clf in ["HGBT", "RF"]:
        tests[f"CRT ({clf})"] = ConditionalRandomizationTest(
            test_classifier_type=clf,
        )

    # DeLong: supports RF, HGBT, LR
    for clf in ["HGBT", "LR", "RF"]:
        tests[f"DeLong ({clf})"] = DeLongTest(test_classifier_type=clf)

    # InvEnvPred: supports RF, HGBT
    for clf in ["HGBT", "RF"]:
        tests[f"InvEnvPred ({clf})"] = InvariantEnvironmentPredictionTest(
            test_classifier_type=clf
        )

    # Residual: supports RF, HGBT, LR
    for clf in ["HGBT", "LR", "RF"]:
        tests[f"Residual ({clf})"] = InvariantResidualDistributionTest(
            test_classifier_type=clf
        )

    # TramGCM: supports RF, HGBT, LR
    for clf in ["HGBT", "LR", "RF"]:
        tests[f"TramGCM ({clf})"] = TramGcmTest(test_classifier_type=clf)

    # WGCM uses xgboost internally, two methods: "est" and "fix"
    # Uses default parameters
    tests["WGCM_est (xgb)"] = WGCMTest(method="est", beta=0.5)
    tests["WGCM_fix (xgb)"] = WGCMTest(method="fix")

    # Filter tests if requested
    if test_filter:
        filter_lower = [f.lower() for f in test_filter]
        filtered_tests = {}
        for name, test in tests.items():
            name_lower = name.lower()
            if any(f in name_lower for f in filter_lower):
                filtered_tests[name] = test
        tests = filtered_tests

    # Sort by test name for consistent ordering
    return dict(sorted(tests.items()))


def _compute_test_for_subset(
    test_name: str,
    test: Any,
    subset: list[str],
    df: pd.DataFrame,
    invariant_subsets: set[frozenset[str]],
) -> TestResult:
    """
    Compute p-value for a single test and subset.

    This function is called in parallel.
    """
    y = df["Y"].to_numpy()
    E = df["E"].to_numpy()

    if not subset:
        X = np.zeros((len(df), 0))
    else:
        X = df[subset].to_numpy()

    try:
        pval = test.test(X, y, E)
    except Exception as e:
        print(f"Warning: {test_name} failed on subset {subset}: {e}")
        pval = np.nan

    subset_frozen = frozenset(subset)
    is_invariant = subset_frozen in invariant_subsets

    return TestResult(
        test_name=test_name,
        subset=tuple(sorted(subset)),  # Store as sorted tuple for display
        subset_str=subset_to_str(subset),
        is_invariant=is_invariant,
        value=pval,
    )


def _compute_loeo_regret_for_subset(
    subset: list[str],
    df: pd.DataFrame,
    invariant_subsets: set[frozenset[str]],
    classifier_type: Literal["RF", "HGBT"] = "RF",
) -> list[TestResult]:
    """
    Compute LOEO regret scores for a single subset.
    Returns two TestResult objects: one for 'mean' and one for 'min'.
    """
    y = df["Y"].to_numpy()
    E = df["E"].to_numpy()

    if not subset:
        X = np.zeros((len(df), 0))
    else:
        X = df[subset].to_numpy()

    try:
        scores = loeo_regret(y, E, X, classifier_type=classifier_type)
    except Exception as e:
        print(
            f"Warning: LOEO regret ({classifier_type}) failed on subset {subset}: {e}"
        )
        scores = {"mean": np.nan, "min": np.nan}

    subset_frozen = frozenset(subset)
    is_invariant = subset_frozen in invariant_subsets
    subset_tuple = tuple(sorted(subset))
    subset_string = subset_to_str(subset)

    results = []
    # Create result for mean
    results.append(
        TestResult(
            test_name=f"LOEO_Regret ({classifier_type}, mean)",
            subset=subset_tuple,
            subset_str=subset_string,
            is_invariant=is_invariant,
            value=scores["mean"],
        )
    )
    # Create result for min
    results.append(
        TestResult(
            test_name=f"LOEO_Regret ({classifier_type}, min)",
            subset=subset_tuple,
            subset_str=subset_string,
            is_invariant=is_invariant,
            value=scores["min"],
        )
    )

    return results


def _run_single_test(
    test_name: str,
    df: pd.DataFrame,
    all_subsets: list[list[str]],
    invariant_subsets: set[frozenset[str]],
) -> list[TestResult]:
    """
    Run a single test on all subsets. This is the unit of parallelization.
    """
    # Create test instance inside the worker (required for multiprocessing)
    tests = get_all_tests()  # No filter needed here, we're passed specific test_name
    test = tests[test_name]

    results = []
    for subset in all_subsets:
        result = _compute_test_for_subset(
            test_name, test, subset, df, invariant_subsets
        )
        results.append(result)

    return results


def run_all_tests_parallel(
    data: DatasetInfo,
    n_workers: int | None = None,
    test_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run all invariance tests on all subsets, parallelized over tests.

    Parameters
    ----------
    data : DatasetInfo
        Dataset information
    n_workers : int or None
        Number of parallel workers. None for automatic.

    Returns
    -------
    pd.DataFrame
        Results with columns: test_name, subset, subset_str, is_invariant, value
    """
    test_names = list(get_all_tests(test_filter=test_filter).keys())

    print(
        f"\nRunning {len(test_names)} test configurations on {len(data.all_subsets)} subsets..."
    )
    print(f"Tests: {test_names}")

    # Prepare the worker function
    worker = partial(
        _run_single_test,
        df=data.df,
        all_subsets=data.all_subsets,
        invariant_subsets=data.invariant_subsets,
    )

    all_results = []

    # Parallelize over tests
    if n_workers == 1:
        # Sequential execution (useful for debugging)
        for test_name in tqdm(test_names, desc="Tests"):
            results = worker(test_name)
            all_results.extend(results)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(worker, test_name): test_name
                for test_name in test_names
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Tests"):
                test_name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"\nError in test {test_name}: {e}")

    # Also run LOEO regret ranking with both classifier types
    print("\nRunning LOEO Regret ranking (RF and HGBT)...")
    for clf_type in ["HGBT", "RF"]:
        for subset in tqdm(data.all_subsets, desc=f"LOEO_Regret ({clf_type})"):
            results = _compute_loeo_regret_for_subset(
                subset,
                data.df,
                data.invariant_subsets,
                classifier_type=clf_type,  # pyright: ignore[reportArgumentType]
            )
            all_results.extend(results)

    # Convert to DataFrame
    df_results = pd.DataFrame(
        [
            {
                "test_name": r.test_name,
                "subset": r.subset,
                "subset_str": r.subset_str,
                "is_invariant": r.is_invariant,
                "value": r.value,
            }
            for r in all_results
        ]
    )

    return df_results


# =============================================================================
# Evaluation Metrics
# =============================================================================


def _pairwise_auc(inv_values: np.ndarray, noninv_values: np.ndarray) -> float:
    """
    Compute pairwise AUC: fraction of (inv, noninv) pairs where inv > noninv.

    For both p-value tests (higher p = more invariant) and ranking tests
    (higher score = more invariant), this measures how well the test
    separates invariant from non-invariant subsets.

    Returns a value in [0, 1]. A value of 1.0 means perfect separation
    (all invariant values exceed all non-invariant values). 0.5 means
    random performance.
    """
    count = sum(
        1.0 if i > n else 0.5 if i == n else 0.0
        for i in inv_values
        for n in noninv_values
    )
    total = len(inv_values) * len(noninv_values)
    return count / total if total > 0 else np.nan


def compute_metrics(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute evaluation metrics for each test.

    Metrics computed:
    - FPR: false positive rate (rejection rate for invariant sets, should be ≤ α)
    - TPR: true positive rate (rejection rate for non-invariant sets, should be high)
    - avg_pval_inv: average p-value for invariant sets (should be high)
    - avg_pval_noninv: average p-value for non-invariant sets (should be low)
    - min_pval_inv: minimum p-value among invariant sets
    - max_pval_noninv: maximum p-value among non-invariant sets

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame with columns test_name, is_invariant, value
    alpha : float
        Significance level for rejection

    Returns
    -------
    pd.DataFrame
        Metrics for each test
    """
    metrics_list = []

    for test_name in df["test_name"].unique():
        test_df = df[df["test_name"] == test_name]

        inv = test_df[test_df["is_invariant"]]
        noninv = test_df[~test_df["is_invariant"]]

        # Skip if no valid values
        inv_values = inv["value"].dropna()
        noninv_values = noninv["value"].dropna()

        if len(inv_values) == 0 or len(noninv_values) == 0:
            continue

        # For LOEO regret ranking, higher is better (more invariant)
        # For p-value tests, higher p-value means more invariant
        is_ranking_test = test_name.startswith("LOEO_Regret")

        # Pairwise AUC: fraction of (inv, noninv) pairs where inv ranks higher
        auc = _pairwise_auc(inv_values.values, noninv_values.values)

        if not is_ranking_test:
            # P-value based metrics
            n_inv = len(inv_values)
            n_noninv = len(noninv_values)

            # FPR: false positive rate (rejection rate for invariant sets)
            inv_rejected = (inv_values < alpha).sum()
            fpr = inv_rejected / n_inv if n_inv > 0 else np.nan

            # TPR: true positive rate (rejection rate for non-invariant sets)
            noninv_rejected = (noninv_values < alpha).sum()
            tpr = noninv_rejected / n_noninv if n_noninv > 0 else np.nan

            # Average p-values
            avg_pval_inv = inv_values.mean()
            avg_pval_noninv = noninv_values.mean()

            # Min/max p-values (for critical subset analysis)
            min_pval_inv = inv_values.min()
            max_pval_noninv = noninv_values.max()

            metrics_list.append(
                {
                    "test_name": test_name,
                    "FPR": fpr,
                    "TPR": tpr,
                    "avg_value_inv": avg_pval_inv,
                    "avg_value_noninv": avg_pval_noninv,
                    "min_pval_inv": min_pval_inv,
                    "max_pval_noninv": max_pval_noninv,
                    "pairwise_auc": auc,
                }
            )
        else:
            avg_score_inv = inv_values.mean()
            avg_score_noninv = noninv_values.mean()

            metrics_list.append(
                {
                    "test_name": test_name,
                    "FPR": np.nan,  # Not applicable for ranking
                    "TPR": np.nan,
                    "avg_value_inv": avg_score_inv,
                    "avg_value_noninv": avg_score_noninv,
                    "min_pval_inv": np.nan,
                    "max_pval_noninv": np.nan,
                    "pairwise_auc": auc,
                }
            )

    return pd.DataFrame(metrics_list)


def print_metrics_summary(metrics_df: pd.DataFrame, alpha: float = 0.05) -> None:
    """Print a formatted summary of the metrics."""
    print("\n" + "=" * 110)
    print("TEST EVALUATION METRICS")
    print("=" * 110)
    print(f"\nSignificance level α = {alpha}")

    if len(metrics_df) == 0:
        print(
            "\nNo metrics to display (need both invariant and non-invariant subsets)."
        )
        return

    # Separate p-value tests from ranking
    pval_tests = metrics_df[
        ~metrics_df["test_name"].str.startswith("LOEO_Regret")
    ].copy()
    ranking = metrics_df[metrics_df["test_name"].str.startswith("LOEO_Regret")].copy()

    if len(pval_tests) > 0:
        print("\n--- P-Value Tests ---")
        # Sort by test name for consistent ordering (same test with different models together)
        pval_tests = pval_tests.sort_values("test_name")

        # Dynamic column width based on longest test name
        max_name_len = max(len(name) for name in pval_tests["test_name"])
        name_width = max(max_name_len, 20)

        header = f"{'Test':<{name_width}} | FPR    | TPR    | Avg P(inv) | Avg P(non) | Min P(inv) | Max P(non) | AUC   "
        print(f"\n{header}")
        print("-" * len(header))
        for _, row in pval_tests.iterrows():
            print(
                f"{row['test_name']:<{name_width}} | "
                f"{row['FPR']:6.3f} | "
                f"{row['TPR']:6.3f} | "
                f"{row['avg_value_inv']:10.3f} | "
                f"{row['avg_value_noninv']:10.3f} | "
                f"{row['min_pval_inv']:10.3f} | "
                f"{row['max_pval_noninv']:10.3f} | "
                f"{row['pairwise_auc']:5.3f} | "
            )

        print("\nInterpretation:")
        print("  - FPR: False Positive Rate (should be ≤ α)")
        print("  - TPR: True Positive Rate (should be high, detects non-invariant)")
        print("  - Avg P(inv): Should be high (correctly not rejecting invariant)")
        print("  - Avg P(non): Should be low (correctly rejecting non-invariant)")
        print("  - Min P(inv): Minimum p-value among invariant sets (should be > α)")
        print(
            "  - Max P(non): Maximum p-value among non-invariant sets (should be < α ideally)"
        )
        print(
            "  - AUC: Pairwise AUC (fraction of inv/noninv pairs correctly ranked; 1.0 = perfect)"
        )

    if len(ranking) > 0:
        print("\n--- LOEO Regret Ranking ---")
        for _, row in ranking.iterrows():
            print(f"\n{row['test_name']}:")
            print(f"  Avg Score (invariant):     {row['avg_value_inv']:.6f}")
            print(f"  Avg Score (non-invariant): {row['avg_value_noninv']:.6f}")
            print(f"  Pairwise AUC:             {row['pairwise_auc']:.3f}")
        print("\nNote: Higher scores indicate more invariance")


# =============================================================================
# Plotting
# =============================================================================


def plot_ordered_values(
    df: pd.DataFrame,
    alpha: float = 0.05,
    save_path: str | None = None,
    max_subsets_for_labels: int = 32,
) -> None:
    """
    Plot ordered p-values/scores as bars for each test.

    Green = invariant sets, Red = non-invariant sets.
    """
    # Separate p-value tests from LOEO ranking
    pval_df = df[~df["test_name"].str.startswith("LOEO_Regret")]
    ranking_df = df[df["test_name"].str.startswith("LOEO_Regret")]

    # Get unique test names (sorted)
    pval_tests = sorted(pval_df["test_name"].unique())
    ranking_tests = sorted(ranking_df["test_name"].unique())
    n_pval_tests = len(pval_tests)
    n_ranking_tests = len(ranking_tests)

    # Calculate layout
    total_plots = n_pval_tests + n_ranking_tests
    n_cols = min(4, total_plots)
    n_rows = (total_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    ax_idx = 0

    # Plot p-value tests
    for test_name in pval_tests:
        ax = axes[ax_idx]
        test_data = pval_df[pval_df["test_name"] == test_name].copy()
        test_data = test_data.sort_values("value").reset_index(drop=True)

        n_subsets = len(test_data)
        x = np.arange(n_subsets)
        colors = ["green" if inv else "red" for inv in test_data["is_invariant"]]

        ax.bar(x, test_data["value"], width=0.8, color=colors, alpha=0.7)
        ax.axhline(
            y=alpha, color="black", linestyle=":", linewidth=1.5, label=f"α={alpha}"
        )

        ax.set_xlim(-0.5, n_subsets - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{test_name}", fontsize=11)
        ax.set_ylabel("p-value")
        ax.grid(True, alpha=0.3, axis="y")

        # Add x-axis labels if few enough subsets
        if n_subsets <= max_subsets_for_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(
                [subset_to_short_str(list(s)) for s in test_data["subset"]],
                rotation=90,
                fontsize=7,
            )
        else:
            ax.set_xticks([])
            ax.set_xlabel(f"Subsets (n={n_subsets})")

        ax_idx += 1

    # Plot LOEO ranking tests
    for test_name in ranking_tests:
        ax = axes[ax_idx]
        ranking_data = ranking_df[ranking_df["test_name"] == test_name].copy()
        # Sort by value (higher = more invariant, so sort ascending to have invariant on right)
        ranking_data = ranking_data.sort_values("value", ascending=True).reset_index(
            drop=True
        )

        n_subsets = len(ranking_data)
        x = np.arange(n_subsets)
        colors = ["green" if inv else "red" for inv in ranking_data["is_invariant"]]

        ax.bar(x, ranking_data["value"], width=0.8, color=colors, alpha=0.7)

        ax.set_xlim(-0.5, n_subsets - 0.5)
        ax.set_title(f"{test_name}", fontsize=11)
        ax.set_ylabel("Score (higher = more invariant)")
        ax.grid(True, alpha=0.3, axis="y")

        if n_subsets <= max_subsets_for_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(
                [subset_to_short_str(list(s)) for s in ranking_data["subset"]],
                rotation=90,
                fontsize=7,
            )
        else:
            ax.set_xticks([])
            ax.set_xlabel(f"Subsets (n={n_subsets})")

        ax_idx += 1

    # Hide unused axes
    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)

    # Add legend
    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="Invariant (ground truth)"),
        Patch(facecolor="red", alpha=0.7, label="Non-invariant"),
        Line2D(
            [0], [0], color="black", linestyle=":", linewidth=1.5, label=f"α = {alpha}"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison_summary(
    metrics_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Plot a summary comparison of all tests.
    """
    # Filter to p-value tests only
    pval_metrics = metrics_df[
        ~metrics_df["test_name"].str.startswith("LOEO_Regret")
    ].copy()

    if len(pval_metrics) == 0:
        print("No p-value tests to compare")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Sort by test name for consistent ordering (same test with different models together)
    pval_metrics = pval_metrics.sort_values("test_name")
    test_names = pval_metrics["test_name"].tolist()
    x = np.arange(len(test_names))

    # Plot 1: FPR vs TPR
    ax = axes[0]
    width = 0.35
    ax.bar(x - width / 2, pval_metrics["FPR"], width, label="FPR", color="salmon")
    ax.bar(x + width / 2, pval_metrics["TPR"], width, label="TPR", color="steelblue")
    ax.axhline(y=0.05, color="black", linestyle=":", linewidth=1, label="α=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("FPR vs TPR")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Plot 2: Average p-values
    ax = axes[1]
    ax.bar(
        x - width / 2,
        pval_metrics["avg_value_inv"],
        width,
        label="Invariant",
        color="green",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        pval_metrics["avg_value_noninv"],
        width,
        label="Non-invariant",
        color="red",
        alpha=0.7,
    )
    ax.axhline(y=0.05, color="black", linestyle=":", linewidth=1, label="α=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.set_ylabel("Average p-value")
    ax.set_title("Average P-values by Ground Truth")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Plot 3: Min p-value (invariant) vs Max p-value (non-invariant)
    ax = axes[2]
    ax.bar(
        x - width / 2,
        pval_metrics["min_pval_inv"],
        width,
        label="Min P (invariant)",
        color="green",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        pval_metrics["max_pval_noninv"],
        width,
        label="Max P (non-inv)",
        color="red",
        alpha=0.7,
    )
    ax.axhline(y=0.05, color="black", linestyle=":", linewidth=1, label="α=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.set_ylabel("P-value")
    ax.set_title("Critical P-values (min invariant, max non-invariant)")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Plot 4: Pairwise AUC (all tests including ranking)
    ax = axes[3]
    all_metrics = metrics_df.dropna(subset=["pairwise_auc"]).sort_values("test_name")
    all_names = all_metrics["test_name"].tolist()
    x_all = np.arange(len(all_names))
    bar_colors = [
        "mediumpurple" if name.startswith("LOEO_Regret") else "steelblue"
        for name in all_names
    ]
    ax.bar(x_all, all_metrics["pairwise_auc"], color=bar_colors, alpha=0.8)
    ax.axhline(y=0.5, color="black", linestyle=":", linewidth=1, label="Random (0.5)")
    ax.set_xticks(x_all)
    ax.set_xticklabels(all_names, rotation=45, ha="right")
    ax.set_ylabel("Pairwise AUC")
    ax.set_title("Pairwise AUC (invariant vs non-invariant)")
    legend_elements_auc = [
        Patch(facecolor="steelblue", alpha=0.8, label="P-value tests"),
        Patch(facecolor="mediumpurple", alpha=0.8, label="Ranking tests"),
        Line2D(
            [0], [0], color="black", linestyle=":", linewidth=1, label="Random (0.5)"
        ),
    ]
    ax.legend(handles=legend_elements_auc, fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# Main
# =============================================================================


def main(
    dataset: str = "simple.csv",
    n_workers: int | None = None,
    save_dir: str | None = None,
    test_filter: list[str] | None = None,
):
    """
    Main evaluation pipeline.

    Parameters
    ----------
    dataset : str
        Name of the dataset CSV file
    n_workers : int or None
        Number of parallel workers
    save_dir : str or None
        Directory to save results. If None, uses results/<dataset>/.
    test_filter : list[str] or None
        Filter tests by name (case-insensitive partial match).
    """
    alpha = 0.05  # Fixed significance level
    # Extract dataset basename for naming output files
    dataset_base = dataset.replace(".csv", "")

    # Setup save_dir: default to results/<dataset>/ at repo root
    if save_dir is None:
        # Get repo root (two levels up from this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        save_dir = os.path.join(repo_root, "results", dataset_base)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("INVARIANCE TEST EVALUATION")
    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Significance level: {alpha}")

    # Load data
    print("\nLoading data...")
    data = get_data(dataset)
    print(f"Features: {data.features}")
    print(f"Number of subsets: {len(data.all_subsets)}")
    print(f"Invariant subsets: {len(data.invariant_subsets)}")
    for s in sorted(data.invariant_subsets):
        print(f"  - {subset_to_str(list(s))}")

    # Diagnostics
    run_diagnostics(data, save_dir=save_dir, dataset_name=dataset_base)

    # Run tests
    results_df = run_all_tests_parallel(
        data,
        n_workers=n_workers,
        test_filter=test_filter,
    )

    # Save raw results
    results_path = os.path.join(save_dir, f"results_{dataset_base}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nRaw results saved to {results_path}")

    # Compute metrics
    metrics_df = compute_metrics(results_df, alpha=alpha)
    print_metrics_summary(metrics_df, alpha=alpha)

    # Save metrics
    metrics_path = os.path.join(save_dir, f"metrics_{dataset_base}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    # Plots
    print("\nGenerating plots...")

    plot_ordered_values(
        results_df,
        alpha=alpha,
        save_path=os.path.join(save_dir, f"ordered_pvalues_{dataset_base}.pdf"),
    )

    plot_comparison_summary(
        metrics_df,
        save_path=os.path.join(save_dir, f"comparison_{dataset_base}.pdf"),
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return results_df, metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate invariance tests on causal chambers datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default="simple.csv", help="Dataset CSV file name"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel workers"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--tests",
        type=str,
        nargs="*",
        default=None,
        help="Filter tests by name (case-insensitive partial match). E.g., --tests delong wgcm",
    )

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        n_workers=args.workers,
        save_dir=args.save_dir,
        test_filter=args.tests,
    )
