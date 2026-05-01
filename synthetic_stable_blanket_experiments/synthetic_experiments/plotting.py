from __future__ import annotations

from pathlib import Path

from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = ["parents", "stable_blanket", "all_variables"]
METHOD_COLORS = {
    "parents": "#0072B2",
    "stable_blanket": "#D55E00",
    "all_variables": "#009E73",
}
METHOD_MARKERS = {
    "parents": "o",
    "stable_blanket": "s",
    "all_variables": "^",
}
TRAIN_SIZE_LINESTYLES = ["--", "-.", "-"]
TRAIN_SIZE_SHADE_WEIGHTS = [0.55, 0.25, 0.0]
LABELS = {
    "signed_error": r"Adversary minimizes $\mathbb{E}[Y-f_S(X_S)]$",
    "mse": r"Adversary maximizes MSE",
    "prediction": r"Adversary minimizes $\mathbb{E}[f_S(X_S)]$",
}


def _blend_with_white(color: str, weight: float) -> tuple[float, float, float]:
    rgb = mcolors.to_rgb(color)
    return tuple((1.0 - weight) * channel + weight for channel in rgb)


def _save_figure(filename: Path) -> None:
    plt.savefig(filename, dpi=180)
    plt.savefig(filename.with_suffix(".pdf"))


def save_plots(
    results: pd.DataFrame,
    output_dir: str | Path,
    attack_mode: str,
    max_sweep_value: float | None = None,
    endpoint_train_sizes_only: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_label = "Intervention bound" if attack_mode == "bound" else "Adversary cost"
    x_col = "sweep_value"
    filename_prefix = "mse_vs_bound" if attack_mode == "bound" else "mse_vs_cost"
    if max_sweep_value is not None:
        results = results[results[x_col] <= max_sweep_value]
    has_train_sweep = (
        "train_size" in results.columns and results["train_size"].nunique() > 1
    )

    for objective, group in results.groupby("objective"):
        if has_train_sweep:
            _save_train_size_sweep_plot(
                group=group,
                x_col=x_col,
                x_label=x_label,
                filename=output_dir / f"{filename_prefix}_{objective}.png",
                objective=objective,
                endpoint_train_sizes_only=endpoint_train_sizes_only,
            )
            continue

        stats = (
            group.groupby(["method", x_col])["attacked_test_mse"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        stats["std"] = stats["std"].fillna(0.0)
        stats["sem"] = stats["std"] / stats["count"].clip(lower=1) ** 0.5
        stats["ci95_radius"] = 1.96 * stats["sem"]

        plt.figure(figsize=(7, 4.5))
        for method in METHOD_ORDER:
            sub = stats[stats["method"] == method].sort_values(x_col)
            if sub.empty:
                continue
            color = METHOD_COLORS[method]
            plt.plot(
                sub[x_col],
                sub["mean"],
                color=color,
                marker=METHOD_MARKERS[method],
                label=method,
            )
            plt.fill_between(
                sub[x_col],
                sub["mean"] - sub["ci95_radius"],
                sub["mean"] + sub["ci95_radius"],
                color=color,
                alpha=0.2,
            )
            clean = group[group["method"] == method]["clean_test_mse"].mean()
            plt.axhline(clean, color=color, linestyle=":", alpha=0.35)
        plt.xlabel(x_label)
        plt.ylabel("Own-adversary test MSE")
        plt.title(f"{LABELS.get(objective, objective)} (mean +/- 95% CI)")
        plt.legend()
        plt.tight_layout()
        _save_figure(output_dir / f"{filename_prefix}_{objective}.png")
        plt.close()


def _save_train_size_sweep_plot(
    *,
    group: pd.DataFrame,
    x_col: str,
    x_label: str,
    filename: Path,
    objective: str,
    endpoint_train_sizes_only: bool,
) -> None:
    stats = (
        group.groupby(["method", "train_size", x_col])["attacked_test_mse"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats["sem"] = stats["std"] / stats["count"].clip(lower=1) ** 0.5
    stats["ci95_radius"] = 1.96 * stats["sem"]

    train_sizes = sorted(stats["train_size"].unique())
    if endpoint_train_sizes_only and len(train_sizes) > 2:
        train_sizes = [train_sizes[0], train_sizes[-1]]
    style_by_train_size = {
        train_size: TRAIN_SIZE_LINESTYLES[idx % len(TRAIN_SIZE_LINESTYLES)]
        for idx, train_size in enumerate(train_sizes)
    }
    shade_by_train_size = {
        train_size: TRAIN_SIZE_SHADE_WEIGHTS[idx % len(TRAIN_SIZE_SHADE_WEIGHTS)]
        for idx, train_size in enumerate(train_sizes)
    }
    plt.figure(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        for train_size in train_sizes:
            sub = stats[
                (stats["method"] == method) & (stats["train_size"] == train_size)
            ].sort_values(x_col)
            if sub.empty:
                continue
            label = f"{method}, n={int(train_size)}"
            color = _blend_with_white(
                METHOD_COLORS[method], shade_by_train_size[train_size]
            )
            plt.plot(
                sub[x_col],
                sub["mean"],
                color=color,
                linestyle=style_by_train_size[train_size],
                marker=METHOD_MARKERS[method],
                label=label,
            )
            plt.fill_between(
                sub[x_col],
                sub["mean"] - sub["ci95_radius"],
                sub["mean"] + sub["ci95_radius"],
                color=color,
                alpha=0.12,
            )
    plt.xlabel(x_label)
    plt.ylabel("Own-adversary test MSE")
    plt.title(f"{LABELS.get(objective, objective)} (mean +/- 95% CI)")
    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle="-",
            label=method,
        )
        for method in METHOD_ORDER
    ]
    train_size_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=style_by_train_size[train_size],
            label=f"n={int(train_size)}",
        )
        for train_size in train_sizes
    ]
    method_legend = plt.legend(
        handles=method_handles, title="Method", fontsize="small", loc="upper left"
    )
    plt.gca().add_artist(method_legend)
    plt.legend(
        handles=train_size_handles,
        title="Train size",
        fontsize="small",
        loc="upper center",
    )
    plt.tight_layout()
    _save_figure(filename)
    plt.close()
