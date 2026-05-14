from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from synthetic_experiments.plotting import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METHOD_ORDER,
    TRAIN_SIZE_LINESTYLES,
    TRAIN_SIZE_SHADE_WEIGHTS,
    _blend_with_white,
)

MAX_SWEEP_VALUE = 1.0


@dataclass(frozen=True)
class PanelSpec:
    title: str
    output_dir: Path
    objective: str
    endpoint_train_sizes_only: bool = False


def _read_results(output_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(output_dir / "results_per_run.csv")
    return results[results["sweep_value"] <= MAX_SWEEP_VALUE]


def _stats(group: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    stats = (
        group.groupby(group_cols)["attacked_test_mse"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats["sem"] = stats["std"] / stats["count"].clip(lower=1) ** 0.5
    stats["ci95_radius"] = 1.96 * stats["sem"]
    return stats


def _plot_standard_panel(ax: plt.Axes, spec: PanelSpec) -> None:
    results = _read_results(spec.output_dir)
    group = results[results["objective"] == spec.objective]
    stats = _stats(group, ["method", "sweep_value"])

    for method in METHOD_ORDER:
        sub = stats[stats["method"] == method].sort_values("sweep_value")
        if sub.empty:
            continue
        color = METHOD_COLORS[method]
        ax.plot(
            sub["sweep_value"],
            sub["mean"],
            color=color,
            marker=METHOD_MARKERS[method],
            markersize=4,
            linewidth=1.4,
            label=METHOD_LABELS[method],
        )
        ax.fill_between(
            sub["sweep_value"],
            sub["mean"] - sub["ci95_radius"],
            sub["mean"] + sub["ci95_radius"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    if spec.title:
        ax.set_title(spec.title, fontsize=9)
    ax.set_xlabel("intervention bound", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)


def _plot_train_sweep_panel(ax: plt.Axes, spec: PanelSpec) -> None:
    results = _read_results(spec.output_dir)
    group = results[results["objective"] == spec.objective]
    stats = _stats(group, ["method", "train_size", "sweep_value"])

    train_sizes = sorted(stats["train_size"].unique())
    if spec.endpoint_train_sizes_only and len(train_sizes) > 2:
        train_sizes = [train_sizes[0], train_sizes[-1]]
    style_by_train_size = {
        train_size: TRAIN_SIZE_LINESTYLES[idx % len(TRAIN_SIZE_LINESTYLES)]
        for idx, train_size in enumerate(train_sizes)
    }
    shade_by_train_size = {
        train_size: TRAIN_SIZE_SHADE_WEIGHTS[idx % len(TRAIN_SIZE_SHADE_WEIGHTS)]
        for idx, train_size in enumerate(train_sizes)
    }

    for method in METHOD_ORDER:
        for train_size in train_sizes:
            sub = stats[
                (stats["method"] == method) & (stats["train_size"] == train_size)
            ].sort_values("sweep_value")
            if sub.empty:
                continue
            color = _blend_with_white(
                METHOD_COLORS[method], shade_by_train_size[train_size]
            )
            ax.plot(
                sub["sweep_value"],
                sub["mean"],
                color=color,
                linestyle=style_by_train_size[train_size],
                marker=METHOD_MARKERS[method],
                markersize=4,
                linewidth=1.3,
            )
            ax.fill_between(
                sub["sweep_value"],
                sub["mean"] - sub["ci95_radius"],
                sub["mean"] + sub["ci95_radius"],
                color=color,
                alpha=0.10,
                linewidth=0,
            )

    if spec.title:
        ax.set_title(spec.title, fontsize=9)
    ax.set_xlabel("intervention bound", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)


def _add_method_legend(
    fig: plt.Figure, fontsize: float = 7, markersize: float = 4
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle="-",
            markersize=markersize,
            linewidth=1.6,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=fontsize,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _make_two_panel_figure(
    output_path: Path,
    left: PanelSpec,
    right: PanelSpec,
    *,
    left_train_sweep: bool = False,
    right_train_sweep: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.35), sharey=False)
    plotters = [
        _plot_train_sweep_panel if left_train_sweep else _plot_standard_panel,
        _plot_train_sweep_panel if right_train_sweep else _plot_standard_panel,
    ]
    for ax, spec, plotter in zip(axes, [left, right], plotters, strict=False):
        plotter(ax, spec)
    axes[0].set_ylabel("deployment MSE", fontsize=8)
    _add_method_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_figure(fig, output_path)


def _make_multi_panel_figure(
    output_path: Path,
    specs: list[PanelSpec],
    *,
    shape: tuple[int, int],
    figsize: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(*shape, figsize=figsize, sharey=False)
    flat_axes = list(axes.ravel())
    for ax, spec in zip(flat_axes, specs, strict=False):
        _plot_standard_panel(ax, spec)
    for row_idx in range(shape[0]):
        flat_axes[row_idx * shape[1]].set_ylabel("deployment MSE", fontsize=8)
    _add_method_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_figure(fig, output_path)


def _make_grouped_row_figure(output_path: Path, specs: list[PanelSpec]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.45), sharey=False)
    for ax, spec in zip(axes, specs, strict=False):
        _plot_standard_panel(ax, spec)
        ax.title.set_fontsize(11)
        ax.xaxis.label.set_fontsize(10)
        ax.tick_params(axis="both", labelsize=9)
    axes[0].set_ylabel("deployment MSE", fontsize=10)

    for label, x_position in [
        ("Linear-Gaussian SCM", 0.285),
        ("Nonlinear SCM", 0.725),
    ]:
        fig.text(
            x_position,
            0.965,
            label,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    _add_method_legend(fig, fontsize=9, markersize=5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    _save_figure(fig, output_path)


def _make_grouped_two_panel_figure(
    output_path: Path,
    left: PanelSpec,
    right: PanelSpec,
    *,
    left_header: str,
    right_header: str,
    left_train_sweep: bool = False,
    right_train_sweep: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.45), sharey=False)
    plotters = [
        _plot_train_sweep_panel if left_train_sweep else _plot_standard_panel,
        _plot_train_sweep_panel if right_train_sweep else _plot_standard_panel,
    ]
    for ax, spec, plotter in zip(axes, [left, right], plotters, strict=False):
        plotter(ax, spec)
    axes[0].set_ylabel("deployment MSE", fontsize=8)

    if left_train_sweep:
        results = _read_results(left.output_dir)
        group = results[results["objective"] == left.objective]
        train_sizes = sorted(group["train_size"].unique())
        if left.endpoint_train_sizes_only and len(train_sizes) > 2:
            train_sizes = [train_sizes[0], train_sizes[-1]]
        style_by_size = {
            ts: TRAIN_SIZE_LINESTYLES[idx % len(TRAIN_SIZE_LINESTYLES)]
            for idx, ts in enumerate(train_sizes)
        }
        train_size_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=style_by_size[ts],
                linewidth=1.3,
                label=f"n = {ts:,}",
            )
            for ts in train_sizes
        ]
        axes[0].legend(
            handles=train_size_handles,
            loc="upper center",
            fontsize=6,
            frameon=False,
            ncol=1,
        )

    for label, x_position in [(left_header, 0.28), (right_header, 0.74)]:
        fig.text(
            x_position,
            0.965,
            label,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    _add_method_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    _save_figure(fig, output_path)


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "paper_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    _make_two_panel_figure(
        output_dir / "lineargaussian_mse_prediction.png",
        PanelSpec("linear-Gaussian, MSE follower", root / "outputs_lingauss", "mse"),
        PanelSpec(
            "linear-Gaussian, prediction follower",
            root / "outputs_lingauss",
            "prediction",
        ),
    )
    _make_two_panel_figure(
        output_dir / "standard_mse_prediction.png",
        PanelSpec("nonlinear, MSE follower", root / "outputs_standard", "mse"),
        PanelSpec(
            "nonlinear, prediction follower",
            root / "outputs_standard",
            "prediction",
        ),
    )
    _make_grouped_two_panel_figure(
        output_dir / "sweep_and_x4_prediction.png",
        PanelSpec(
            "",
            root / "outputs_train-size-sweep",
            "prediction",
        ),
        PanelSpec(
            "",
            root / "outputs_x4-uses-x1-x3",
            "prediction",
        ),
        left_header="Train-size sweep",
        right_header=r"$X_4$ attack uses $X_1,X_3$",
        left_train_sweep=True,
    )

    first_two_figure_specs = [
        PanelSpec("MSE attack", root / "outputs_lingauss", "mse"),
        PanelSpec(
            "prediction attack",
            root / "outputs_lingauss",
            "prediction",
        ),
        PanelSpec("MSE attack", root / "outputs_standard", "mse"),
        PanelSpec(
            "prediction attack",
            root / "outputs_standard",
            "prediction",
        ),
    ]
    _make_multi_panel_figure(
        output_dir / "lineargaussian_standard_grid.png",
        first_two_figure_specs,
        shape=(2, 2),
        figsize=(6.2, 4.4),
    )
    _make_multi_panel_figure(
        output_dir / "lineargaussian_standard_row.png",
        first_two_figure_specs,
        shape=(1, 4),
        figsize=(9.0, 2.25),
    )
    _make_grouped_row_figure(
        output_dir / "lineargaussian_standard_row_grouped.png",
        first_two_figure_specs,
    )
    print(f"Saved paper figures to {output_dir}")


if __name__ == "__main__":
    main()
