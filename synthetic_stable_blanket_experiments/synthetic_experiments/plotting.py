from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["parents", "stable_blanket", "all_variables"]
LABELS = {
    "signed_error": r"Adversary minimizes $\mathbb{E}[Y-f_S(X_S)]$",
    "mse": r"Adversary maximizes MSE",
}


def save_plots(results: pd.DataFrame, output_dir: str | Path, attack_mode: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_label = "Intervention bound" if attack_mode == "bound" else "Adversary cost"
    x_col = "sweep_value"
    filename_prefix = "mse_vs_bound" if attack_mode == "bound" else "mse_vs_cost"

    for objective, group in results.groupby("objective"):
        stats = (
            group.groupby(["method", x_col])["attacked_test_mse"]
            .agg(["mean", "std", "count"]).reset_index()
        )
        stats["std"] = stats["std"].fillna(0.0)
        stats["sem"] = stats["std"] / stats["count"].clip(lower=1) ** 0.5
        stats["ci95_radius"] = 1.96 * stats["sem"]

        plt.figure(figsize=(7, 4.5))
        for method in METHOD_ORDER:
            sub = stats[stats["method"] == method].sort_values(x_col)
            if sub.empty:
                continue
            plt.plot(sub[x_col], sub["mean"], marker="o", label=method)
            plt.fill_between(
                sub[x_col],
                sub["mean"] - sub["ci95_radius"],
                sub["mean"] + sub["ci95_radius"],
                alpha=0.2,
            )
            clean = group[group["method"] == method]["clean_test_mse"].mean()
            plt.axhline(clean, linestyle="--", alpha=0.35)
        plt.xlabel(x_label)
        plt.ylabel("Own-adversary test MSE")
        plt.title(f"{LABELS.get(objective, objective)} (mean +/- 95% CI)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_prefix}_{objective}.png", dpi=180)
        plt.close()
