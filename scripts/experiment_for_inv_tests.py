import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from invariance_for_classification.generate_data.synthetic_DGP import generate_scm_data
from invariance_for_classification.invariance_tests import (
    DeLongTest,
    InvariantResidualDistributionTest,
    TramGcmTest,
)


def get_subsets(features):
    subsets = []
    for L in range(0, len(features) + 1):
        for subset in itertools.combinations(features, L):
            subsets.append(list(subset))
    return subsets


def subset_to_str(subset):
    if not subset:
        return "Empty"
    # Mapping X1->1, X2->2 etc for shorter labels on plot as typically seen in papers
    # Assumes input format "X1", "X2" etc
    nums = [s.replace("X", "") for s in subset]
    return ",".join(sorted(nums))


def main():
    n_draws = 5
    features = ["X1", "X2", "X3"]
    target = "Y"
    env_col = "E"

    subsets = get_subsets(features)
    subset_labels = [subset_to_str(s) for s in subsets]

    tests = {
        "Residual Test (RF)": InvariantResidualDistributionTest(
            test_classifier_type="RF"
        ),
        "Tram-GCM (RF)": TramGcmTest(test_classifier_type="RF"),
        "DeLong Test (RF)": DeLongTest(test_classifier_type="RF"),
    }

    # Store results: results[test_name][subset_idx] = [pval_draw1, pval_draw2, ...]
    results = {name: [[] for _ in range(len(subsets))] for name in tests}

    print(f"Running experiment with {n_draws} draws...")

    for draw in range(n_draws):
        print(f"Draw {draw + 1}/{n_draws}")
        # Generate new data with different seed
        df = generate_scm_data(n_per_env=500, seed=42 + draw)

        for i, subset in enumerate(tqdm(subsets, leave=False)):
            if not subset:
                X = np.zeros((len(df), 0))
            else:
                X = df[subset].to_numpy()

            y = df[target].to_numpy()
            E = df[env_col].to_numpy()

            for test_name, test_class in tests.items():
                try:
                    pval = test_class.test(X, y, E)
                except Exception as e:
                    print(f"Error in {test_name} for set {subset_labels[i]}: {e}")
                    pval = 0.0  # Indicate failure

                results[test_name][i].append(pval)

    # Compute stats
    means = {name: [] for name in tests}
    stds = {name: [] for name in tests}

    for name in tests:
        for i in range(len(subsets)):
            vals = results[name][i]
            means[name].append(np.mean(vals))
            stds[name].append(np.std(vals))

    # Plotting
    n_tests = len(tests)
    fig, axes = plt.subplots(1, n_tests, figsize=(6 * n_tests, 5), sharey=True)

    if n_tests == 1:
        axes = [axes]

    for ax, test_name in zip(axes, tests.keys(), strict=True):
        pvals = means[test_name]
        errors = stds[test_name]

        x_pos = np.arange(len(subset_labels))

        # Plot bars with error bars
        ax.bar(
            x_pos,
            pvals,
            yerr=errors,
            align="center",
            alpha=0.7,
            capsize=10,
            color="lightgreen",
            error_kw={"elinewidth": 2, "capthick": 2},
        )

        ax.set_ylabel("Average p-value")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subset_labels)
        ax.set_title(test_name)
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

        # Highlight significant thresholds
        ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "invariance_test_results.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
