import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def load_data(dataset: str) -> pd.DataFrame:
    """Load the dataset (no subsampling, use all data)."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    file_path = os.path.join(data_dir, f"{dataset}_train.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    return pd.read_csv(file_path)


def plot_combined(df_d_lin, df_d_nonlin, filename):
    plt.rcParams.update(
        {
            "font.size": 22,
            "text.usetex": True,
            "font.family": "sans-serif",
            "text.latex.preamble": r"\usepackage{amsfonts} \usepackage{amsmath}",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, ds in zip(
        axes,
        [df_d_lin, df_d_nonlin],
        ["Dataset D-lin", "Dataset D-nonlin"],
        strict=False,
    ):
        df_plot = df.copy()
        df_plot["Y_cat"] = df_plot["Y"].astype(str)

        # Primary axis: Density
        sns.kdeplot(
            data=df_plot,
            x="vis_3",
            hue="Y_cat",
            fill=True,
            alpha=0.3,
            ax=ax,
            common_norm=False,
            legend=False,
            palette={"0": "#1f77b4", "1": "#ff7f0e"},
        )
        ax.set_ylabel("Scaled class-conditional density")
        # Drop density tick labels
        ax.set_yticks([])
        ax.set_xlabel(r"$\mathtt{vis\_3}$")

        # Secondary axis: P(Y=1 | vis_3)
        ax2 = ax.twinx()
        df_plot["vis_3_bin"] = pd.qcut(df_plot["vis_3"], q=20, duplicates="drop")
        df_plot["vis_3_bin_mid"] = (
            df_plot["vis_3_bin"].apply(lambda x: x.mid).astype(float)
        )

        prob_label = r"$\hat{\mathbb{P}}(Y=1 \mid \mathtt{vis\_3})$"
        sns.lineplot(
            data=df_plot,
            x="vis_3_bin_mid",
            y="Y",
            ax=ax2,
            color="black",
            errorbar=None,
            linewidth=2,
            legend=False,
        )

        ax2.set_ylabel(prob_label, color="black")
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.set_ylim(-0.05, 1.05)

        ax.set_title(ds, pad=15, fontsize=20)

    # Custom unified Legend
    legend_handles = [
        mpatches.Patch(color="#1f77b4", label=r"$Y = 0$", alpha=0.3),
        mpatches.Patch(color="#ff7f0e", label=r"$Y = 1$", alpha=0.3),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            label=r"$\hat{\mathbb{P}}(Y=1 \mid \mathtt{vis\_3})$",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=22,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved: {filename}")


def main():
    print("Loading data...")
    df_d_lin = load_data("d_lin")
    df_d_nonlin = load_data("d_nonlin")

    print("Generating figures...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(base_dir))
    results_dir = os.path.join(repo_root, "results")

    plot_path = os.path.join(results_dir, "combined_density_prob_vis3.png")

    plot_combined(
        df_d_lin,
        df_d_nonlin,
        filename=plot_path,
    )


if __name__ == "__main__":
    main()
