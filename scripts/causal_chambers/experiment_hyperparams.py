"""
Hyperparameter sensitivity analysis for Stabilized Classification (SC).

For each dataset and repetition we compute p-values and fitted RF models for
all 2^d feature subsets once (using TramGCM + RF with alpha_inv=0 so every
subset is scored).  Then sweep over (alpha_inv, alpha_pred) pairs:

  1. Filter invariant subsets by alpha_inv threshold.
  2. Determine S_max = best-scoring invariant subset.
  3. Compute the bootstrap predictive cutoff at quantile alpha_pred
     (cached by S_max so the bootstrap runs at most once per unique S_max
     per repetition — typically 1-3 bootstraps instead of 7x7=49).
  4. Filter active subsets and ensemble-predict.

Usage
-----
    python experiment_hyperparams.py --n-reps 5 --n-jobs 8
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from evaluate_OOD_predictions import (
    NORMAL_COLS,
    _extract_positive_proba,
    _load_train_test,
    _per_env_metrics,
    _subsample_train,
)
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from invariance_for_classification import StabilizedClassificationClassifier
from invariance_for_classification.estimators._stabilized import (
    _bootstrap_predictiveness_worker,
)

N_BOOTSTRAP = 250


# ---------------------------------------------------------------------------
# bootstrap helper
# ---------------------------------------------------------------------------


def _bootstrap_cutoff_scores(
    X,
    y,
    environment,
    S_max,
    seed,
    n_bootstrap,
    n_jobs,
    model_random_state,
):
    """Return n_bootstrap out-of-bag log-loss scores for feature subset S_max.

    Uses the same worker logic as StabilizedClassificationClassifier._compute_cutoff:
    - bootstrap randomness comes from the sampled indices (seeded by `seed`)
    - RF model randomness is fixed by `model_random_state` for all bootstrap draws
    """

    pred_classifier = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        random_state=model_random_state,
        n_jobs=1,
    )

    rng_main = np.random.RandomState(seed)
    seeds = rng_main.randint(0, np.iinfo(np.int32).max, size=n_bootstrap)
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_predictiveness_worker)(
            s,
            X,
            y,
            list(S_max),
            pred_classifier,
            "mean",
            environment,
        )
        for s in seeds
    )
    return [s for s in scores if s is not None]


# ---------------------------------------------------------------------------
# threshold application
# ---------------------------------------------------------------------------


def _apply_thresholds(
    all_results,
    X_train,
    y_enc,
    E_train,
    X_test,
    a_inv,
    a_pred,
    bootstrap_cache,
    seed,
    n_jobs,
):
    """Apply thresholds and return (proba, n_invariant, n_predictive)."""
    # 1. Filter by invariance threshold
    invariant = [r for r in all_results if r["p_value"] >= a_inv]
    if not invariant:  # fallback: best p-value
        invariant = [max(all_results, key=lambda r: r["p_value"])]

    # 2. Best-scoring invariant subset
    S_max = tuple(max(invariant, key=lambda r: r["RF_score"])["subset"])

    # 3. Bootstrap predictive cutoff (cached per S_max)
    if S_max not in bootstrap_cache:
        bootstrap_cache[S_max] = _bootstrap_cutoff_scores(
            X_train,
            y_enc,
            E_train,
            S_max,
            seed,
            N_BOOTSTRAP,
            n_jobs,
            model_random_state=seed,
        )
    bs_scores = bootstrap_cache[S_max]
    cutoff = (
        np.quantile(a=np.asarray(bs_scores, dtype=float), q=a_pred)
        if bs_scores
        else -np.inf
    )

    # 4. Filter active subsets
    active = [r for r in invariant if r["RF_score"] >= cutoff]
    if not active:  # fallback: best-scoring invariant subset
        active = [max(invariant, key=lambda r: r["RF_score"])]

    # 5. Ensemble predict
    n_samples = X_test.shape[0]
    sum_proba = np.zeros((n_samples, 2))
    for r in active:
        subset = r["subset"]
        X_S = X_test[:, subset] if len(subset) > 0 else np.zeros((n_samples, 0))
        sum_proba += r["RF_model"].predict_proba(X_S)
    return sum_proba / len(active), len(invariant), len(active)


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------


def run_experiment(datasets, alpha_invs, alpha_preds, n_obs, n_reps, n_jobs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for dataset in datasets:
        print(f"=== Dataset {dataset} ===")

        # _load_train_test already filters to NORMAL_COLS[dataset]
        df_train, df_test = _load_train_test(dataset)
        feature_cols = [c for c in NORMAL_COLS[dataset] if c not in ("Y", "E")]

        X_test_all = df_test[feature_cols].to_numpy()
        y_test_all = df_test["Y"].to_numpy()
        E_test_all = df_test["E"].to_numpy()

        acc_lists = {
            (a_inv, a_pred): [] for a_inv in alpha_invs for a_pred in alpha_preds
        }
        inv_count_lists = {
            (a_inv, a_pred): [] for a_inv in alpha_invs for a_pred in alpha_preds
        }
        pred_count_lists = {
            (a_inv, a_pred): [] for a_inv in alpha_invs for a_pred in alpha_preds
        }

        for rep in tqdm(range(n_reps), desc="  reps"):
            seed = 42 + rep
            rng = np.random.default_rng(seed)
            df_sub = _subsample_train(df_train, n_obs, rng)

            X_train = df_sub[feature_cols].to_numpy()
            y_train = df_sub["Y"].to_numpy()
            E_train = df_sub["E"].to_numpy()

            # ── Precompute p-values + RF models for all 2^d subsets (once per rep) ──
            # alpha_inv=0.0 ensures every subset is scored and fitted.
            # n_bootstrap=1 minimises wasted computation in SC.fit() — we redo
            # the bootstrap ourselves below with the full N_BOOTSTRAP samples.
            clf = StabilizedClassificationClassifier(
                alpha_inv=0.0,
                alpha_pred=0.0,
                invariance_test="tram_gcm",
                test_classifier_type="RF",
                pred_classifier_type="RF",
                n_jobs=n_jobs,
                verbose=0,
                random_state=seed,
                n_bootstrap=1,
            )
            clf.fit(X_train, y_train, environment=E_train)

            all_results = clf.all_results_
            # y_enc matches what the workers used (label-encoded inside SC.fit)
            y_enc = clf.le_.transform(y_train)

            # Bootstrap cache shared across all (a_inv, a_pred) for this rep.
            # Key: tuple(S_max).  Typically 1-3 unique S_max values per rep.
            bootstrap_cache: dict[tuple, list[float]] = {}

            for a_inv in alpha_invs:
                for a_pred in alpha_preds:
                    proba, n_invariant, n_predictive = _apply_thresholds(
                        all_results,
                        X_train,
                        y_enc,
                        E_train,
                        X_test_all,
                        a_inv,
                        a_pred,
                        bootstrap_cache,
                        seed,
                        n_jobs,
                    )
                    y_prob_pos = _extract_positive_proba(proba)
                    y_pred_enc = (proba[:, 1] >= 0.5).astype(int)
                    y_pred = clf.le_.inverse_transform(y_pred_enc)
                    per_env = _per_env_metrics(
                        y_test_all, y_prob_pos, y_pred, E_test_all
                    )
                    acc_lists[(a_inv, a_pred)].append(
                        min(r["accuracy"] for r in per_env)
                    )
                    inv_count_lists[(a_inv, a_pred)].append(n_invariant)
                    pred_count_lists[(a_inv, a_pred)].append(n_predictive)

        results = [
            {
                "alpha_inv": a_inv,
                "alpha_pred": a_pred,
                "worst_case_accuracy": np.mean(accs),
                "n_invariant": float(np.mean(inv_count_lists[(a_inv, a_pred)])),
                "n_predictive": float(np.mean(pred_count_lists[(a_inv, a_pred)])),
            }
            for (a_inv, a_pred), accs in acc_lists.items()
        ]
        df_res = pd.DataFrame(results)
        df_res.to_csv(
            os.path.join(out_dir, f"heatmap_results_{dataset}.csv"), index=False
        )

        # ── Heatmap ──
        heatmap_data = df_res.pivot(
            index="alpha_pred", columns="alpha_inv", values="worst_case_accuracy"
        )
        heatmap_data = heatmap_data.sort_index(ascending=False)
        inv_count_data = df_res.pivot(
            index="alpha_pred", columns="alpha_inv", values="n_invariant"
        )
        pred_count_data = df_res.pivot(
            index="alpha_pred", columns="alpha_inv", values="n_predictive"
        )
        inv_count_data = inv_count_data.reindex(
            index=heatmap_data.index, columns=heatmap_data.columns
        )
        pred_count_data = pred_count_data.reindex(
            index=heatmap_data.index, columns=heatmap_data.columns
        )

        acc_values = heatmap_data.to_numpy(dtype=float)
        inv_values = inv_count_data.to_numpy(dtype=float)
        pred_values = pred_count_data.to_numpy(dtype=float)
        annotations = np.empty(heatmap_data.shape, dtype=object)
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                acc = acc_values[i, j]
                n_inv = int(np.rint(inv_values[i, j]))
                n_pred = int(np.rint(pred_values[i, j]))
                annotations[i, j] = f"{acc:.3f}\n({n_inv},{n_pred})"

        _, ax = plt.subplots(figsize=(9, 6))
        vmin = heatmap_data.values.min()
        vmax = heatmap_data.values.max()
        hm = sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            annot=annotations,
            fmt="",
            annot_kws={"size": 16},
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Worst-case accuracy"},
        )
        colorbar = hm.collections[0].colorbar if hm.collections else None
        if colorbar is not None:
            colorbar.set_label("Worst-case accuracy", size=19)
            colorbar.ax.tick_params(labelsize=15)

        ax.set_title(f"Dataset {dataset}", fontsize=23)
        ax.set_xlabel(r"$\alpha_{\mathrm{inv}}$", fontsize=19)
        ax.set_ylabel(r"$\alpha_{\mathrm{class}}$", fontsize=19)
        ax.tick_params(axis="both", labelsize=16)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"heatmap_{dataset}.png")
        plt.savefig(out_path, dpi=450)
        plt.close()
        print(f"Saved heatmap for dataset {dataset} → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-obs", type=int, default=200, help="Observations per environment"
    )
    parser.add_argument(
        "--n-reps", type=int, default=25, help="Repetitions to average over"
    )
    parser.add_argument("--n-jobs", type=int, default=12, help="Parallel jobs")
    parser.add_argument(
        "--out-dir", type=str, default="results/sensitivity", help="Output directory"
    )
    args = parser.parse_args()

    alpha_invs = [0.005, 0.01, 0.05, 0.1, 0.2]
    alpha_preds = [0.005, 0.01, 0.05, 0.1, 0.2]
    datasets = ["1a", "1b", "2"]

    run_experiment(
        datasets,
        alpha_invs,
        alpha_preds,
        args.n_obs,
        args.n_reps,
        args.n_jobs,
        args.out_dir,
    )
