"""
Runtime analysis for the causal chambers OOD experiments.

This script is separate from evaluate_OOD_predictions.py so the main OOD
evaluation stays on the faster code path.

It measures per-repetition wall-clock runtimes for the methods that appear in
the single-dataset runtime table:
  - SC TramGCM(LR), pred=LR
  - SC TramGCM(RF), pred=RF
  - ERM LR / RF
  - IRM-Linear / IRM-NN
  - Oracle LR / RF

The runtime output is saved to:
  <repo_root>/results/<dataset>/OOD_runtime_<dataset>_n<n_obs>.csv

This file is then consumed by generate_latex_tables.py.

USAGE
=====
    python runtime_analysis.py --dataset d_nonlin
    python runtime_analysis.py --dataset d_nonlin --n-obs 200 --n-reps 20 --n-jobs 10
"""

from __future__ import annotations

import argparse
import os
import time

# Keep thread-limited behavior aligned with the main OOD evaluation script.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from evaluate_OOD_predictions import (
    SEED,
    STABLE_BLANKETS,
    _append_to_csv,
    _load_train_test,
    _run_erm_lr,
    _run_erm_rf,
    _run_irm_linear,
    _run_irm_nn,
    _run_oracle_lr,
    _run_oracle_rf,
    _subsample_train,
)
from scipy import stats

from invariance_for_classification import StabilizedClassificationClassifier

RUNTIME_SC_CONFIGS: list[tuple[str, str, str | None, str]] = [
    ("TramGCM(LR)", "tram_gcm", "LR", "LR"),
    ("TramGCM(RF)", "tram_gcm", "RF", "RF"),
]


def _ci_half_width(values: np.ndarray, confidence: float = 0.95) -> float:
    """Half-width of a t-CI."""
    n = len(values)
    if n < 2:
        return float("nan")
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(t_crit * se)


def _run_sc_runtime(
    config_name: str,
    invariance_test: str,
    test_classifier_type: str | None,
    pred_classifier_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    rep: int,
    rep_seed: int,
    n_jobs: int,
    verbose: bool = False,
) -> dict:
    """Measure end-to-end runtime for one SC row in the runtime table.

    The timing includes fitting and both ensemble/best prediction calls on the
    test data, but only a single runtime row is written for the SC method.
    """
    kwargs: dict = {
        "alpha_inv": 0.05,
        "alpha_pred": 0.05,
        "pred_classifier_type": pred_classifier_type,
        "invariance_test": invariance_test,
        "pred_scoring": "mean",
        "n_bootstrap": 250,
        "verbose": 1 if verbose else 0,
        "n_jobs": n_jobs,
        "random_state": rep_seed,
    }
    if test_classifier_type is not None:
        kwargs["test_classifier_type"] = test_classifier_type

    t0 = time.time()
    clf = StabilizedClassificationClassifier(**kwargs)
    clf.fit(X_train, y_train, environment=E_train)

    # Include both prediction modes because the main results use both SC and IMP,
    # even though the runtime table only reports the SC row.
    for method in ["ensemble", "best"]:
        _ = clf.predict_proba(
            X_test,
            pred_classifier_type=pred_classifier_type,
            method=method,
        )
        _ = clf.predict(
            X_test,
            pred_classifier_type=pred_classifier_type,
            method=method,
        )

    return {
        "rep": rep,
        "method": f"SC {config_name}, pred={pred_classifier_type}",
        "runtime_seconds": time.time() - t0,
    }


def _print_runtime_summary(runtime_path: str) -> None:
    """Print mean ± 95% CI runtime summary."""
    df = pd.read_csv(runtime_path)
    print("\nRUNTIME SUMMARY (mean ± 95% CI)")
    print("─" * 70)
    print(f"{'Method':<40s}  {'Runtime (s)':>20s}")
    print("─" * 70)

    for method in sorted(df["method"].unique()):
        vals = df[df["method"] == method]["runtime_seconds"].to_numpy()
        mean = float(np.mean(vals))
        ci = _ci_half_width(vals)
        print(f"  {method:<38s}  {mean:>9.3f}±{ci:<9.3f}")
    print()


def main(
    dataset: str = "d_nonlin",
    n_obs_per_env: int = 200,
    n_reps: int = 20,
    n_jobs: int = 10,
    verbose: bool = False,
) -> None:
    """Run the runtime analysis."""
    try:
        import torch

        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    save_dir = os.path.join(repo_root, "results", dataset)
    os.makedirs(save_dir, exist_ok=True)

    runtime_path = os.path.join(save_dir, f"OOD_runtime_{dataset}_n{n_obs_per_env}.csv")
    if os.path.exists(runtime_path):
        os.remove(runtime_path)

    df_train_full, df_test = _load_train_test(dataset)
    features = [c for c in df_train_full.columns if c not in ("Y", "E")]

    print("=" * 70)
    print("Runtime Analysis")
    print("=" * 70)
    print(f"Dataset          : {dataset}")
    print(f"Features ({len(features):d})     : {features}")
    print(f"Full train size  : {len(df_train_full)}")
    print(f"Test size        : {len(df_test)}")
    print(f"N_OBS_PER_ENV    : {n_obs_per_env}")
    print(f"N_REPS           : {n_reps}")
    print(f"N_JOBS           : {n_jobs}")
    print(f"Save directory   : {save_dir}")

    X_test = df_test[features].to_numpy()
    y_test = df_test["Y"].to_numpy().astype(int)
    E_test = df_test["E"].to_numpy()

    stable_blanket = STABLE_BLANKETS.get(dataset)
    stable_indices: list[int] | None = None
    if stable_blanket is not None:
        stable_indices = [features.index(f) for f in stable_blanket if f in features]
        print(f"Stable blanket   : {stable_blanket} -> indices {stable_indices}")
    else:
        print("Stable blanket   : (unknown)")

    t_start_total = time.time()

    for rep in range(n_reps):
        rep_seed = SEED + rep
        rng = np.random.default_rng(rep_seed)

        try:
            import torch

            torch.manual_seed(rep_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rep_seed)
        except ImportError:
            pass

        print(f"\n{'─' * 70}")
        print(f"Repetition {rep + 1}/{n_reps}  (seed={rep_seed})")
        print(f"{'─' * 70}")

        df_train = _subsample_train(df_train_full, n_obs_per_env, rng)
        X_train = df_train[features].to_numpy()
        y_train = df_train["Y"].to_numpy().astype(int)
        E_train = df_train["E"].to_numpy()

        print(
            f"  Train subsample: {len(df_train)} obs "
            f"(envs: {sorted(df_train['E'].unique())})"
        )

        for cfg_name, inv_test, test_clf, pred_clf in RUNTIME_SC_CONFIGS:
            print(
                f"  Running SC {cfg_name}, pred={pred_clf} ...",
                end=" ",
                flush=True,
            )
            row = _run_sc_runtime(
                config_name=cfg_name,
                invariance_test=inv_test,
                test_classifier_type=test_clf,
                pred_classifier_type=pred_clf,
                X_train=X_train,
                y_train=y_train,
                E_train=E_train,
                X_test=X_test,
                rep=rep,
                rep_seed=rep_seed,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            _append_to_csv(pd.DataFrame([row]), runtime_path)
            print(f"done ({row['runtime_seconds']:.1f}s)")

        baseline_tasks = [
            (
                "ERM RF",
                _run_erm_rf,
                dict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    E_test=E_test,
                    rep=rep,
                    rep_seed=rep_seed,
                    n_jobs=n_jobs,
                ),
            ),
            (
                "ERM LR",
                _run_erm_lr,
                dict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    E_test=E_test,
                    rep=rep,
                    rep_seed=rep_seed,
                ),
            ),
        ]

        if stable_indices is not None and len(stable_indices) > 0:
            baseline_tasks.extend(
                [
                    (
                        "Oracle RF",
                        _run_oracle_rf,
                        dict(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            E_test=E_test,
                            stable_indices=stable_indices,
                            rep=rep,
                            rep_seed=rep_seed,
                            n_jobs=n_jobs,
                        ),
                    ),
                    (
                        "Oracle LR",
                        _run_oracle_lr,
                        dict(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            E_test=E_test,
                            stable_indices=stable_indices,
                            rep=rep,
                            rep_seed=rep_seed,
                        ),
                    ),
                ]
            )

        baseline_tasks.extend(
            [
                (
                    "IRM-NN",
                    _run_irm_nn,
                    dict(
                        X_train=X_train,
                        y_train=y_train,
                        E_train=E_train,
                        X_test=X_test,
                        y_test=y_test,
                        E_test=E_test,
                        rep=rep,
                        rep_seed=rep_seed,
                        n_jobs=n_jobs,
                        verbose=verbose,
                    ),
                ),
                (
                    "IRM-Linear",
                    _run_irm_linear,
                    dict(
                        X_train=X_train,
                        y_train=y_train,
                        E_train=E_train,
                        X_test=X_test,
                        y_test=y_test,
                        E_test=E_test,
                        rep=rep,
                        rep_seed=rep_seed,
                        n_jobs=n_jobs,
                        verbose=verbose,
                    ),
                ),
            ]
        )

        for bname, bfunc, bkwargs in baseline_tasks:
            t0 = time.time()
            print(f"  Running {bname} ...", end=" ", flush=True)
            _ = bfunc(**bkwargs)
            elapsed = time.time() - t0
            _append_to_csv(
                pd.DataFrame(
                    [{"rep": rep, "method": bname, "runtime_seconds": elapsed}]
                ),
                runtime_path,
            )
            print(f"done ({elapsed:.1f}s)")

    elapsed_total = time.time() - t_start_total
    print(f"\n{'=' * 70}")
    print(f"All {n_reps} repetitions complete in {elapsed_total / 60:.1f} min")
    print(f"{'=' * 70}")

    _print_runtime_summary(runtime_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runtime analysis for causal chambers OOD methods.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="d_nonlin",
        choices=["d_lin", "d_nonlin", "d_spur"],
        help="Dataset base name (default: d_nonlin).",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=200,
        help="Number of observations per environment in training subsample (default: 200).",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=25,
        help="Number of repetitions (default: 25).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=15,
        help="Number of parallel workers for internal parallelism (default: 12).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for individual methods.",
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        n_obs_per_env=args.n_obs,
        n_reps=args.n_reps,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
